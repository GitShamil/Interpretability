from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from steering_denoising.checkpoint import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    CheckpointMetadata,
    load_checkpoint,
    restore_checkpoint,
)
from steering_denoising.config import DenoiserConfig, ExperimentConfig
from steering_denoising.denoiser import ActivationStats, ResidualDenoiser
from steering_denoising.exceptions import CompatibilityError, LeakageError
from steering_denoising.training import (
    deterministic_sequence_split,
    fit_training_stats,
    train_denoiser,
)


class TensorActivationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, activations: torch.Tensor, mask: torch.Tensor) -> None:
        self.activations = activations
        self.mask = mask

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "activation": self.activations[index],
            "attention_mask": self.mask[index],
        }


@dataclass
class TinyDirectionBank:
    vectors: torch.Tensor
    ids: tuple[str, ...]

    @property
    def d_model(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return self.vectors.shape[0]


def _dataset(sequence_count: int = 12, d_model: int = 4) -> TensorActivationDataset:
    generator = torch.Generator().manual_seed(91)
    activations = torch.randn(sequence_count, 3, d_model, generator=generator)
    mask = torch.ones(sequence_count, 3, dtype=torch.bool)
    mask[:, -1] = torch.arange(sequence_count) % 3 != 0
    activations[~mask] = 1_000_000.0
    return TensorActivationDataset(activations, mask)


def _config(output_dir: Path, *, method: str = "gaussian"):
    config = ExperimentConfig()
    config.model.name_or_path = "synthetic-model"
    config.model.device = "cpu"
    config.denoiser = DenoiserConfig(
        width=8,
        width_multiplier=1.0,
        depth=1,
        expansion=1,
    )
    config.training.method = method  # type: ignore[assignment]
    config.training.output_dir = str(output_dir)
    config.training.seed = 123
    config.training.epochs = 1
    config.training.batch_size = 3
    config.training.learning_rate = 1e-3
    config.training.validation_fraction = 0.25
    config.corruption.gaussian_probability = 1.0
    config.corruption.structured_probability = 0.0
    config.corruption.clean_probability = 0.0
    config.corruption.gaussian_strength_min = 0.1
    config.corruption.gaussian_strength_max = 0.1
    return config


def test_sequence_split_and_train_only_statistics() -> None:
    split = deterministic_sequence_split(8, 0.25, 9)
    activations = torch.zeros(8, 2, 2)
    mask = torch.ones(8, 2, dtype=torch.bool)
    for index in split.train_indices:
        activations[index] = float(index + 1)
    for index in split.validation_indices:
        activations[index] = 100_000.0
    activations[split.train_indices[0], 1] = -1_000_000.0
    mask[split.train_indices[0], 1] = False

    stats = fit_training_stats(
        TensorActivationDataset(activations, mask), split.train_indices, batch_size=2
    )
    expected = activations[list(split.train_indices)][mask[list(split.train_indices)]]

    assert set(split.train_indices).isdisjoint(split.validation_indices)
    assert stats.count == expected.shape[0]
    assert torch.allclose(stats.mean, expected.mean(dim=0))


def test_checkpoint_round_trip_and_compatibility(tmp_path: Path) -> None:
    stats = ActivationStats(
        mean=torch.zeros(4),
        scale=torch.ones(4),
        activation_rms=1.0,
        count=12,
        scale_floor=1e-5,
    )
    config = DenoiserConfig(width=8, depth=1, expansion=1)
    model = ResidualDenoiser(4, stats, config)
    metadata = CheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        method="gaussian",
        model_name="synthetic-model",
        model_revision="revision-a",
        module_path="blocks.5",
        layer=5,
        d_model=4,
        activation_fingerprint="activations-a",
        direction_fingerprint=None,
    )
    path = CheckpointManager(tmp_path, metadata).save("best", model=model)
    loaded = load_checkpoint(path, expected=metadata)
    clone = ResidualDenoiser(4, stats, config)
    restore_checkpoint(loaded, model=clone)

    for expected, actual in zip(model.parameters(), clone.parameters(), strict=True):
        assert torch.equal(expected, actual)
    with pytest.raises(CompatibilityError, match="layer"):
        load_checkpoint(path, expected={"layer": 6})


def test_gaussian_training_writes_best_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset()
    result = train_denoiser(_config(tmp_path / "run"), dataset=dataset)

    assert result.global_step > 0
    assert math.isfinite(result.best_validation_loss)
    assert (result.best_checkpoint / CHECKPOINT_FILENAME).is_file()
    assert len(result.history_path.read_text().splitlines()) == 1


def test_structured_training_uses_only_train_directions(tmp_path: Path) -> None:
    config = _config(tmp_path / "structured", method="structured")
    config.corruption.gaussian_probability = 0.0
    config.corruption.structured_probability = 1.0
    bank = TinyDirectionBank(torch.eye(4)[:2], ("train-0", "train-1"))

    with pytest.raises(LeakageError, match="train SAE split"):
        train_denoiser(
            config,
            dataset=_dataset(),
            direction_bank=bank,
        )

    result = train_denoiser(
        config,
        dataset=_dataset(),
        direction_bank=bank,
        direction_split_fingerprint="a" * 64,
    )
    assert result.metadata.direction_split_fingerprint == "a" * 64
