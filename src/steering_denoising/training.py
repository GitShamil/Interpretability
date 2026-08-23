"""Training loop for Gaussian and structured denoisers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from steering_denoising.activations import ActivationDataset
from steering_denoising.checkpoint import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    CheckpointMetadata,
    activation_manifest_fingerprint,
    direction_bank_fingerprint,
)
from steering_denoising.config import ExperimentConfig, save_resolved_config
from steering_denoising.corruption import ActivationCorruptor, masked_reconstruction_loss
from steering_denoising.denoiser import ActivationStats, ResidualDenoiser
from steering_denoising.exceptions import CompatibilityError, LeakageError
from steering_denoising.utils import derived_seed, make_generator, resolve_device, seed_everything


class DirectionBankLike(Protocol):
    vectors: torch.Tensor
    ids: tuple[str | int, ...]
    d_model: int

    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SequenceSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    seed: int


@dataclass(slots=True)
class TrainingResult:
    model: ResidualDenoiser
    stats: ActivationStats
    split: SequenceSplit
    metadata: CheckpointMetadata
    global_step: int
    best_validation_loss: float
    best_checkpoint: Path
    history_path: Path


def deterministic_sequence_split(
    sequence_count: int, validation_fraction: float, seed: int
) -> SequenceSplit:
    validation_count = min(
        sequence_count - 1,
        max(1, round(sequence_count * validation_fraction)),
    )
    ranked = sorted(
        range(sequence_count),
        key=lambda index: derived_seed(seed, "sequence-split", index),
    )
    validation = tuple(sorted(ranked[:validation_count]))
    training = tuple(sorted(ranked[validation_count:]))
    return SequenceSplit(training, validation, seed)


def fit_training_stats(
    dataset: Dataset[dict[str, torch.Tensor]],
    train_indices: Iterable[int],
    *,
    scale_floor: float = 1e-5,
    batch_size: int = 64,
    show_progress: bool = False,
) -> ActivationStats:
    loader = DataLoader(
        Subset(dataset, tuple(train_indices)),
        batch_size=batch_size,
        shuffle=False,
    )
    batches = (
        tqdm(loader, desc="activation stats", unit="batch", leave=False)
        if show_progress
        else loader
    )
    return ActivationStats.fit(
        (_batch_tensors(batch) for batch in batches), scale_floor=scale_floor
    )


def train_denoiser(
    config: ExperimentConfig,
    *,
    dataset: Dataset[dict[str, torch.Tensor]] | None = None,
    direction_bank: DirectionBankLike | None = None,
    direction_split_fingerprint: str | None = None,
    device: str | torch.device | None = None,
) -> TrainingResult:
    config.validate()
    seed_everything(config.training.seed)
    if dataset is None:
        activation_dir = config.resolve_path(config.data.activation_dir)
        dataset = ActivationDataset(activation_dir)

    output_dir = config.resolve_path(config.training.output_dir)
    if (output_dir / "best" / CHECKPOINT_FILENAME).exists():
        raise FileExistsError(f"A trained checkpoint already exists in {output_dir}.")

    _validate_activation_manifest(config, dataset)
    structured = config.training.method == "structured"
    if structured and (direction_bank is None or not direction_split_fingerprint):
        raise LeakageError("Structured training requires the train SAE split.")

    split = deterministic_sequence_split(
        len(dataset), config.training.validation_fraction, config.training.seed
    )
    stats = fit_training_stats(
        dataset,
        split.train_indices,
        batch_size=max(64, config.training.batch_size),
        show_progress=True,
    )
    training_device = (
        torch.device(device) if device is not None else resolve_device(config.model.device)
    )
    model = ResidualDenoiser(stats.d_model, stats, config.denoiser).to(training_device)
    corruptor = ActivationCorruptor(
        config.corruption,
        stats,
        direction_bank if structured else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    metadata = build_checkpoint_metadata(
        config,
        dataset,
        model,
        direction_bank if structured else None,
        split,
        direction_split_fingerprint,
    )
    manager = CheckpointManager(output_dir, metadata)
    history_path = output_dir / "history.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    save_resolved_config(config, output_dir / "config.resolved.json")

    train_data = Subset(dataset, split.train_indices)
    validation_data = Subset(dataset, split.validation_indices)
    best_loss = float("inf")
    global_step = 0

    batches_per_epoch = (
        len(train_data) + config.training.batch_size - 1
    ) // config.training.batch_size
    with tqdm(
        total=config.training.epochs * batches_per_epoch,
        desc=f"train {config.training.method}",
        unit="batch",
        dynamic_ncols=True,
    ) as progress:
        for epoch in range(config.training.epochs):
            loader = DataLoader(
                train_data,
                batch_size=config.training.batch_size,
                shuffle=True,
                generator=make_generator(
                    derived_seed(config.training.seed, "train-loader", epoch), "cpu"
                ),
            )
            model.train()
            train_loss = 0.0
            train_tokens = 0
            for batch in loader:
                clean, mask = _move_batch(batch, training_device)
                generator = make_generator(
                    derived_seed(config.training.seed, "corruption", global_step),
                    training_device,
                )
                corrupted = corruptor(clean, mask, generator=generator)
                loss = masked_reconstruction_loss(model(corrupted), clean, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tokens = int(mask.sum())
                train_loss += float(loss.detach()) * tokens
                train_tokens += tokens
                global_step += 1
                progress.update()
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{config.training.epochs}",
                    train_loss=f"{train_loss / train_tokens:.4f}",
                )

            validation_loss = evaluate_denoiser(
                model,
                validation_data,
                corruptor,
                config=config,
                device=training_device,
            )
            average_train_loss = train_loss / train_tokens
            improved = validation_loss < best_loss
            if improved:
                best_loss = validation_loss
                manager.save(
                    "best",
                    model=model,
                    resolved_config=config.as_resolved_dict(),
                )
            progress.set_postfix(
                epoch=f"{epoch + 1}/{config.training.epochs}",
                train_loss=f"{average_train_loss:.4f}",
                validation_loss=f"{validation_loss:.4f}",
            )
            _append_history(
                history_path,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": average_train_loss,
                    "validation_loss": validation_loss,
                    "best": improved,
                },
            )

    return TrainingResult(
        model=model,
        stats=stats,
        split=split,
        metadata=metadata,
        global_step=global_step,
        best_validation_loss=best_loss,
        best_checkpoint=manager.path_for("best"),
        history_path=history_path,
    )


def evaluate_denoiser(
    model: ResidualDenoiser,
    dataset: Dataset[dict[str, torch.Tensor]],
    corruptor: ActivationCorruptor,
    *,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            clean, mask = _move_batch(batch, device)
            generator = make_generator(
                derived_seed(config.training.seed, "validation", batch_index),
                device,
            )
            corrupted = corruptor(clean, mask, generator=generator)
            loss = masked_reconstruction_loss(model(corrupted), clean, mask)
            tokens = int(mask.sum())
            total_loss += float(loss) * tokens
            total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def build_checkpoint_metadata(
    config: ExperimentConfig,
    dataset: Dataset[dict[str, torch.Tensor]],
    model: ResidualDenoiser,
    direction_bank: DirectionBankLike | None,
    split: SequenceSplit,
    direction_split_fingerprint: str | None,
) -> CheckpointMetadata:
    manifest = getattr(dataset, "manifest", None)
    if manifest is None:
        model_name = config.model.name_or_path
        model_revision = config.model.revision
        module_path = f"transformer.h.{config.model.layer}"
        layer = config.model.layer
        activation_fingerprint = _dataset_fingerprint(dataset)
    else:
        model_name = manifest.model_name
        model_revision = manifest.model_revision
        module_path = manifest.module_path
        layer = manifest.layer
        activation_fingerprint = activation_manifest_fingerprint(manifest)

    return CheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        method=config.training.method,
        model_name=model_name,
        model_revision=model_revision,
        module_path=module_path,
        layer=layer,
        d_model=model.d_model,
        activation_fingerprint=activation_fingerprint,
        direction_fingerprint=direction_bank_fingerprint(direction_bank),
        direction_split_fingerprint=direction_split_fingerprint or "",
    )


def _batch_tensors(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["activation"], batch["attention_mask"]


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    clean, mask = _batch_tensors(batch)
    return clean.to(device=device, dtype=torch.float32), mask.to(device=device, dtype=torch.bool)


def _validate_activation_manifest(
    config: ExperimentConfig, dataset: Dataset[dict[str, torch.Tensor]]
) -> None:
    manifest = getattr(dataset, "manifest", None)
    if manifest is None:
        return
    if manifest.model_name != config.model.name_or_path or manifest.layer != config.model.layer:
        raise CompatibilityError("Activations were collected from another model or layer.")


def _dataset_fingerprint(dataset: Dataset[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        activation, mask = _batch_tensors(dataset[index])
        digest.update(activation.contiguous().numpy().tobytes())
        digest.update(mask.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _append_history(path: Path, event: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
