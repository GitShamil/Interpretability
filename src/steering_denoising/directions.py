"""SAE direction loading and validation split handling."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from steering_denoising.exceptions import CompatibilityError

DirectionId: TypeAlias = str | int


class DirectionFormatError(ValueError):
    pass


class DirectionBank:
    def __init__(
        self,
        vectors: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
        ids: Sequence[DirectionId] | None = None,
    ) -> None:
        matrix = torch.as_tensor(vectors, dtype=torch.float32, device="cpu").clone()
        if matrix.ndim != 2:
            raise DirectionFormatError("Directions must have shape [features, d_model].")
        norms = matrix.double().norm(dim=1)
        if not torch.isfinite(matrix).all() or (norms == 0).any():
            raise DirectionFormatError("Directions must be finite and non-zero.")
        matrix = (matrix.double() / norms[:, None]).float()
        matrix[matrix == 0] = 0.0

        self._vectors = matrix.contiguous()
        self.ids = tuple(range(len(matrix))) if ids is None else tuple(ids)
        if len(self.ids) != len(matrix) or len(set(self.ids)) != len(self.ids):
            raise DirectionFormatError("Direction IDs must be unique and match the matrix.")
        self._indices = {direction_id: index for index, direction_id in enumerate(self.ids)}
        fingerprints = tuple(_vector_hash(row) for row in self._vectors)
        self.bank_hash = _bank_hash(self.ids, fingerprints, self.d_model)

    @classmethod
    def load(cls, path: str | Path, *, expected_dim: int | None = None) -> DirectionBank:
        matrix = np.load(Path(path), allow_pickle=False)
        bank = cls(matrix)
        if expected_dim is not None and bank.d_model != expected_dim:
            raise DirectionFormatError(f"Expected d_model={expected_dim}, got {bank.d_model}.")
        return bank

    @property
    def vectors(self) -> torch.Tensor:
        return self._vectors.clone()

    @property
    def d_model(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def fingerprint(self) -> str:
        return self.bank_hash

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, direction_id: object) -> bool:
        return direction_id in self._indices

    def get(self, direction_id: DirectionId) -> torch.Tensor:
        return self._vectors[self._indices[direction_id]].clone()

    def subset_ids(self, ids: Sequence[DirectionId]) -> DirectionBank:
        selected = tuple(ids)
        return DirectionBank(
            self._vectors[[self._indices[direction_id] for direction_id in selected]],
            selected,
        )


@dataclass(frozen=True, slots=True)
class DirectionSplitManifest:
    bank_hash: str
    train_ids: tuple[DirectionId, ...]
    validation_ids: tuple[DirectionId, ...]

    @classmethod
    def create(
        cls,
        bank: DirectionBank,
        *,
        validation_ids: Sequence[DirectionId] | None = None,
    ) -> DirectionSplitManifest:
        if not validation_ids:
            raise DirectionFormatError("validation_direction_ids.json must not be empty.")
        held_out = tuple(validation_ids)
        if len(set(held_out)) != len(held_out) or any(item not in bank for item in held_out):
            raise DirectionFormatError("Validation direction IDs must be unique and known.")
        train = tuple(direction_id for direction_id in bank.ids if direction_id not in held_out)
        return cls(bank.bank_hash, train, held_out)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_hash": self.bank_hash,
            "validation_ids": list(self.validation_ids),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, bank: DirectionBank) -> DirectionSplitManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls.create(bank, validation_ids=payload["validation_ids"])
        if payload["bank_hash"] != bank.bank_hash:
            raise CompatibilityError("Direction split was created for another SAE bank.")
        return manifest


def _vector_hash(vector: torch.Tensor) -> str:
    values = vector.numpy().astype("<f4")
    digest = hashlib.sha256()
    digest.update(b"steering-direction-v1\0")
    digest.update(struct.pack(">Q", len(values)))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _bank_hash(ids: Sequence[DirectionId], fingerprints: Sequence[str], d_model: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"steering-direction-bank-v1\0")
    digest.update(struct.pack(">QQ", len(ids), d_model))
    digest.update(b"l2\0")
    for direction_id, fingerprint in zip(ids, fingerprints, strict=True):
        encoded = _json_bytes(
            {
                "type": "integer" if isinstance(direction_id, int) else "string",
                "value": direction_id,
            }
        )
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        digest.update(bytes.fromhex(fingerprint))
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
