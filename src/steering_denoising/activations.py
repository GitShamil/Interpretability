"""Sharded storage for GPT-2 activations and attention masks."""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from steering_denoising.exceptions import CompatibilityError
from steering_denoising.utils import atomic_write_json

FORMAT_VERSION = 1
_DTYPES = {"float16": np.dtype(np.float16), "float32": np.dtype(np.float32)}


@dataclass(frozen=True, slots=True)
class ActivationShard:
    activations_file: str
    mask_file: str
    sequences: int
    sequence_length: int
    d_model: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ActivationManifest:
    format_version: int
    model_name: str
    model_revision: str | None
    tokenizer_name: str
    tokenizer_revision: str | None
    module_path: str
    layer: int
    dtype: str
    sequences: int
    valid_tokens: int
    sequence_length: int
    d_model: int
    shards: tuple[ActivationShard, ...]
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: str | Path) -> ActivationManifest:
        with (Path(directory) / "manifest.json").open(encoding="utf-8") as handle:
            raw = json.load(handle)
        shards = tuple(ActivationShard(**item) for item in raw.pop("shards"))
        manifest = cls(shards=shards, **raw)
        if manifest.format_version != FORMAT_VERSION:
            raise CompatibilityError("Unsupported activation store format.")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivationStoreWriter:
    def __init__(
        self,
        directory: str | Path,
        *,
        model_name: str,
        model_revision: str | None,
        tokenizer_name: str,
        tokenizer_revision: str | None,
        module_path: str,
        layer: int,
        storage_dtype: str = "float16",
        shard_sequences: int = 256,
        source: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.directory = Path(directory)
        if (self.directory / "manifest.json").exists() and not overwrite:
            raise FileExistsError(f"Activation store already exists at {self.directory}.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dtype = _DTYPES[storage_dtype]
        self.shard_sequences = shard_sequences
        self.metadata = {
            "model_name": model_name,
            "model_revision": model_revision,
            "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "module_path": module_path,
            "layer": layer,
            "dtype": storage_dtype,
            "source": source or {},
        }
        self.activation_buffer: list[np.ndarray] = []
        self.mask_buffer: list[np.ndarray] = []
        self.buffered_sequences = 0
        self.sequence_length: int | None = None
        self.d_model: int | None = None
        self.valid_tokens = 0
        self.sequences = 0
        self.shards: list[ActivationShard] = []

    def add(self, activations: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if activations.ndim != 3 or attention_mask.shape != activations.shape[:2]:
            raise ValueError("Expected activations [batch, sequence, d_model] and matching mask.")
        values = activations.detach().cpu().float().numpy()
        masks = attention_mask.detach().cpu().bool().numpy()
        _, sequence_length, d_model = values.shape
        if self.sequence_length is None:
            self.sequence_length, self.d_model = sequence_length, d_model
        elif (sequence_length, d_model) != (self.sequence_length, self.d_model):
            raise ValueError("All activation batches must have the same shape.")

        cursor = 0
        while cursor < len(values):
            count = min(self.shard_sequences - self.buffered_sequences, len(values) - cursor)
            self.activation_buffer.append(values[cursor : cursor + count])
            self.mask_buffer.append(masks[cursor : cursor + count])
            self.buffered_sequences += count
            self.sequences += count
            self.valid_tokens += int(masks[cursor : cursor + count].sum())
            cursor += count
            if self.buffered_sequences == self.shard_sequences:
                self._flush()

    def finalize(self) -> ActivationManifest:
        if self.buffered_sequences:
            self._flush()
        if self.sequence_length is None or self.d_model is None:
            raise RuntimeError("No activations were collected.")
        manifest = ActivationManifest(
            format_version=FORMAT_VERSION,
            sequences=self.sequences,
            valid_tokens=self.valid_tokens,
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            shards=tuple(self.shards),
            **self.metadata,
        )
        atomic_write_json(self.directory / "manifest.json", manifest.to_dict())
        return manifest

    def _flush(self) -> None:
        index = len(self.shards)
        activations = np.concatenate(self.activation_buffer).astype(self.dtype, copy=False)
        masks = np.concatenate(self.mask_buffer).astype(np.bool_, copy=False)
        activations_file = f"activations-{index:05d}.npy"
        mask_file = f"mask-{index:05d}.npy"
        np.save(self.directory / activations_file, activations, allow_pickle=False)
        np.save(self.directory / mask_file, masks, allow_pickle=False)
        self.shards.append(
            ActivationShard(
                activations_file=activations_file,
                mask_file=mask_file,
                sequences=len(activations),
                sequence_length=activations.shape[1],
                d_model=activations.shape[2],
                sha256="",
            )
        )
        self.activation_buffer.clear()
        self.mask_buffer.clear()
        self.buffered_sequences = 0


class ActivationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest = ActivationManifest.load(self.directory)
        self.ends: list[int] = []
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        total = 0
        for shard in self.manifest.shards:
            total += shard.sequences
            self.ends.append(total)

    def __len__(self) -> int:
        return self.manifest.sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        activations, masks = self._open_shard(shard_index)
        return {
            "activation": torch.from_numpy(np.array(activations[index - start], copy=True)),
            "attention_mask": torch.from_numpy(np.array(masks[index - start], copy=True)),
        }

    def iter_shards(
        self, *, chunk_sequences: int = 32
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for shard_index in range(len(self.manifest.shards)):
            activations, masks = self._open_shard(shard_index)
            for start in range(0, len(activations), chunk_sequences):
                stop = start + chunk_sequences
                yield (
                    torch.from_numpy(np.array(activations[start:stop], copy=True)),
                    torch.from_numpy(np.array(masks[start:stop], copy=True)),
                )

    def _open_shard(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index not in self.cache:
            shard = self.manifest.shards[index]
            activations = np.load(
                self.directory / shard.activations_file, mmap_mode="r", allow_pickle=False
            )
            masks = np.load(self.directory / shard.mask_file, mmap_mode="r", allow_pickle=False)
            expected = (shard.sequences, shard.sequence_length, shard.d_model)
            if activations.shape != expected or masks.shape != expected[:2]:
                raise CompatibilityError(f"Invalid activation shard {index} shape.")
            self.cache[index] = activations, masks
        return self.cache[index]
