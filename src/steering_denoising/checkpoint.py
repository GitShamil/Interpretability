"""Denoiser checkpoint loading and saving."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from steering_denoising.denoiser import ActivationStats, ResidualDenoiser
from steering_denoising.exceptions import CompatibilityError

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint.pt"


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    format_version: int
    method: str
    model_name: str
    model_revision: str | None
    module_path: str
    layer: int
    d_model: int
    activation_fingerprint: str
    direction_fingerprint: str | None
    direction_split_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointMetadata:
        return cls(**dict(value))


@dataclass(slots=True)
class LoadedCheckpoint:
    metadata: CheckpointMetadata
    model_state: dict[str, torch.Tensor]
    stats: ActivationStats
    resolved_config: dict[str, Any]


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def direction_bank_fingerprint(direction_bank: object | None) -> str | None:
    if direction_bank is None:
        return None
    fingerprint = getattr(direction_bank, "fingerprint", None)
    if fingerprint:
        return str(fingerprint() if callable(fingerprint) else fingerprint)
    vectors = direction_bank.vectors
    ids = direction_bank.ids
    return canonical_fingerprint(
        {
            "ids": ids,
            "shape": tuple(vectors.shape),
            "sha256": hashlib.sha256(vectors.cpu().numpy().tobytes()).hexdigest(),
        }
    )


def activation_manifest_fingerprint(manifest: object) -> str:
    return canonical_fingerprint(manifest.to_dict())  # type: ignore[attr-defined]


class CheckpointManager:
    def __init__(self, output_dir: str | Path, metadata: CheckpointMetadata) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = metadata

    def path_for(self, tag: str) -> Path:
        return self.output_dir / tag

    def save(
        self,
        tag: str,
        *,
        model: ResidualDenoiser,
        resolved_config: Mapping[str, Any] | None = None,
    ) -> Path:
        checkpoint_dir = self.path_for(tag)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "metadata": self.metadata.to_dict(),
            "model_state": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "stats": model.stats().to_dict(),
            "resolved_config": dict(resolved_config or {}),
        }
        temporary = checkpoint_dir / f"{CHECKPOINT_FILENAME}.tmp"
        torch.save(payload, temporary)
        temporary.replace(checkpoint_dir / CHECKPOINT_FILENAME)
        return checkpoint_dir


def load_checkpoint(
    path: str | Path,
    *,
    expected: CheckpointMetadata | Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    checkpoint_file = Path(path)
    if checkpoint_file.is_dir():
        checkpoint_file /= CHECKPOINT_FILENAME
    payload = torch.load(checkpoint_file, map_location=map_location, weights_only=True)
    metadata = CheckpointMetadata.from_dict(payload["metadata"])
    if expected is not None:
        values = expected.to_dict() if isinstance(expected, CheckpointMetadata) else expected
        mismatches = [name for name, value in values.items() if getattr(metadata, name) != value]
        if mismatches:
            raise CompatibilityError(f"Incompatible checkpoint fields: {mismatches}")
    stats = ActivationStats.from_dict(payload["stats"])
    return LoadedCheckpoint(
        metadata=metadata,
        model_state=payload["model_state"],
        stats=stats,
        resolved_config=dict(payload["resolved_config"]),
    )


def restore_checkpoint(loaded: LoadedCheckpoint, *, model: ResidualDenoiser) -> None:
    if loaded.metadata.d_model != model.d_model:
        raise CompatibilityError("Checkpoint and denoiser dimensions differ.")
    model.load_state_dict(loaded.model_state, strict=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    return value
