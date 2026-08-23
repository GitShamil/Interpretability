"""Generation evaluation for baseline, Gaussian, and structured steering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from steering_denoising.activations import ActivationDataset
from steering_denoising.analysis import EvaluationRecord, summarize_records, write_summary_csv
from steering_denoising.checkpoint import (
    CheckpointMetadata,
    activation_manifest_fingerprint,
    load_checkpoint,
    restore_checkpoint,
)
from steering_denoising.config import DenoiserConfig, ExperimentConfig
from steering_denoising.denoiser import ActivationStats, ResidualDenoiser
from steering_denoising.directions import DirectionBank, DirectionId, DirectionSplitManifest
from steering_denoising.exceptions import CompatibilityError
from steering_denoising.hf import LoadedCausalLM, load_causal_lm, read_text_records
from steering_denoising.intervention import SteeringOperator
from steering_denoising.metrics import trim_token_ids
from steering_denoising.training import deterministic_sequence_split, fit_training_stats
from steering_denoising.utils import derived_seed


@dataclass(slots=True)
class _Method:
    name: str
    denoiser: ResidualDenoiser | None


@dataclass(frozen=True, slots=True)
class _LoadedDenoiser:
    model: ResidualDenoiser
    metadata: CheckpointMetadata
    stats: ActivationStats


@dataclass(frozen=True, slots=True)
class _PromptBatch:
    start: int
    prompts: tuple[str, ...]
    encoded: dict[str, torch.Tensor]


def run_evaluation(config: ExperimentConfig, *, overwrite: bool = False) -> list[dict[str, Any]]:
    """Generate and aggregate distinct-n for held-out directions."""

    output_dir = config.resolve_path(config.evaluation.output_dir)
    output_markers = ("records.jsonl", "summary.csv")
    if not overwrite and any((output_dir / name).exists() for name in output_markers):
        raise FileExistsError(
            f"Evaluation artifacts already exist in {output_dir}; pass overwrite=True "
            "or choose a new output_dir."
        )
    loaded = load_causal_lm(config.model)
    activation_dir = config.resolve_path(config.data.activation_dir)
    activation_dataset = ActivationDataset(activation_dir)
    _assert_activation_compatible(loaded, activation_dataset, config)

    direction_path = config.resolve_path(config.directions.tensor_path)
    bank = DirectionBank.load(
        direction_path,
        expected_dim=activation_dataset.manifest.d_model,
    )
    split_path = config.resolve_path(config.directions.split_path)
    split = DirectionSplitManifest.load(split_path, bank=bank)
    direction_ids: Sequence[DirectionId] = split.validation_ids
    if not direction_ids:
        raise ValueError("Validation direction split is empty.")
    denoisers = _load_requested_denoisers(config, loaded, activation_dataset)
    reference_stats = _reference_stats(denoisers, activation_dataset, config)
    _audit_checkpoint_protocol(denoisers, bank, split)
    method = _resolve_method(config.evaluation.method, denoisers)

    prompts_path = config.resolve_path(config.data.eval_prompts_path)
    prompts = read_text_records(prompts_path)
    if len(prompts) != 30:
        raise ValueError("Evaluation requires exactly 30 prompts.")
    prompt_batches = _prepare_prompt_batches(loaded, prompts, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    temporary_records = records_path.with_suffix(records_path.suffix + ".tmp")
    summary: list[dict[str, Any]] = []
    stream_complete = False
    try:
        with temporary_records.open("w", encoding="utf-8") as record_stream:
            for direction_id in direction_ids:
                direction = bank.get(direction_id)
                for alpha in config.evaluation.alphas:
                    condition_records = _evaluate_condition(
                        loaded,
                        prompt_batches,
                        direction_id=direction_id,
                        direction=direction,
                        alpha=alpha,
                        method=method,
                        stats=reference_stats,
                        config=config,
                    )
                    summary.append(summarize_records(condition_records))
                    for record in condition_records:
                        record_stream.write(
                            json.dumps(record.to_export_dict(), ensure_ascii=False) + "\n"
                        )
        temporary_records.replace(records_path)
        stream_complete = True
    finally:
        if not stream_complete:
            temporary_records.unlink(missing_ok=True)

    if not summary:
        raise RuntimeError("Evaluation unexpectedly produced no summary rows.")
    write_summary_csv(output_dir / "summary.csv", summary)
    return summary


def _load_requested_denoisers(
    config: ExperimentConfig,
    loaded: LoadedCausalLM,
    activations: ActivationDataset,
) -> dict[str, _LoadedDenoiser]:
    requested = {config.evaluation.method}
    result: dict[str, _LoadedDenoiser] = {}
    specs = {
        "gaussian": config.evaluation.gaussian_checkpoint,
        "structured": config.evaluation.structured_checkpoint,
    }
    for method, path_value in specs.items():
        if method not in requested:
            continue
        path = config.resolve_path(path_value)
        checkpoint = load_checkpoint(
            path,
            expected={
                "method": method,
                "model_name": config.model.name_or_path,
                "model_revision": loaded.model_revision,
                "module_path": loaded.module_path,
                "layer": config.model.layer,
                "d_model": activations.manifest.d_model,
                "activation_fingerprint": activation_manifest_fingerprint(activations.manifest),
            },
            map_location="cpu",
        )
        if checkpoint.resolved_config is None or not isinstance(
            checkpoint.resolved_config.get("denoiser"), Mapping
        ):
            raise CompatibilityError("Checkpoint is missing its resolved denoiser config.")
        denoiser_config = DenoiserConfig(**checkpoint.resolved_config["denoiser"])
        denoiser = ResidualDenoiser(checkpoint.metadata.d_model, checkpoint.stats, denoiser_config)
        restore_checkpoint(checkpoint, model=denoiser)
        denoiser.to(device=loaded.device)
        denoiser.eval()
        result[method] = _LoadedDenoiser(
            model=denoiser,
            metadata=checkpoint.metadata,
            stats=checkpoint.stats,
        )
    return result


def _reference_stats(
    denoisers: Mapping[str, _LoadedDenoiser],
    activations: ActivationDataset,
    config: ExperimentConfig,
) -> ActivationStats:
    if "gaussian" in denoisers:
        return denoisers["gaussian"].stats
    if "structured" in denoisers:
        return denoisers["structured"].stats
    # The store contains training-corpus activations, so this does not inspect
    # evaluation prompts/directions. It is only needed for a baseline-only run.
    split = deterministic_sequence_split(
        len(activations), config.training.validation_fraction, config.training.seed
    )
    return fit_training_stats(
        activations,
        split.train_indices,
        batch_size=max(config.training.batch_size, 64),
    )


def _audit_checkpoint_protocol(
    denoisers: Mapping[str, _LoadedDenoiser],
    bank: DirectionBank,
    split: DirectionSplitManifest,
) -> None:
    structured = denoisers.get("structured")
    if structured is None:
        return
    metadata = structured.metadata
    expected_train_fingerprint = bank.subset_ids(split.train_ids).bank_hash
    if metadata.direction_fingerprint != expected_train_fingerprint:
        raise CompatibilityError(
            "Structured checkpoint direction bank does not match the split's train subset."
        )
    if metadata.direction_split_fingerprint != split.fingerprint:
        raise CompatibilityError(
            "Structured checkpoint was trained under a different direction split manifest."
        )


def _resolve_method(name: str, denoisers: Mapping[str, _LoadedDenoiser]) -> _Method:
    if name == "baseline":
        return _Method(name=name, denoiser=None)
    try:
        return _Method(name=name, denoiser=denoisers[name].model)
    except KeyError as error:
        raise CompatibilityError(f"Missing checkpoint for requested method {name}.") from error


def _prepare_prompt_batches(
    loaded: LoadedCausalLM,
    prompts: Sequence[str],
    config: ExperimentConfig,
) -> list[_PromptBatch]:
    """Tokenize prompts once on CPU and reuse them for every paired condition."""

    generation = config.evaluation.generation
    prompt_limit = _context_limit(loaded.model) - generation.max_new_tokens
    if prompt_limit < 1:
        raise ValueError("max_new_tokens leaves no room in the model context window.")
    batches: list[_PromptBatch] = []
    for start in range(0, len(prompts), generation.batch_size):
        batch_prompts = tuple(prompts[start : start + generation.batch_size])
        encoded_value = loaded.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=prompt_limit,
            return_tensors="pt",
        )
        encoded = {
            key: value.detach().cpu()
            for key, value in encoded_value.items()
            if torch.is_tensor(value)
        }
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise CompatibilityError(
                "Tokenizer output must contain tensor input_ids and attention_mask."
            )
        batches.append(_PromptBatch(start=start, prompts=batch_prompts, encoded=encoded))
    return batches


def _evaluate_condition(
    loaded: LoadedCausalLM,
    prompt_batches: Sequence[_PromptBatch],
    *,
    direction_id: DirectionId,
    direction: torch.Tensor,
    alpha: float,
    method: _Method,
    stats: ActivationStats,
    config: ExperimentConfig,
) -> list[EvaluationRecord]:
    generation = config.evaluation.generation
    results: list[EvaluationRecord] = []
    operator = (
        SteeringOperator.baseline(
            direction,
            alpha,
            activation_rms=stats.activation_rms,
        )
        if method.denoiser is None
        else SteeringOperator.denoised(
            direction,
            alpha,
            method.denoiser,
            activation_rms=stats.activation_rms,
        )
    )
    eos_token_id = _effective_eos_token_id(loaded)
    for batch in prompt_batches:
        start = batch.start
        batch_prompts = batch.prompts
        encoded = {key: value.to(loaded.device) for key, value in batch.encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        # Common random numbers: every direction/method/alpha condition for the
        # same prompt batch starts from the same sampling stream.
        pair_seed = derived_seed(config.evaluation.seed, start, "generation")
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": True,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "pad_token_id": loaded.tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
        }
        with (
            _fork_seeded_rng(pair_seed, loaded.device),
            operator.hook(
                loaded.model,
                target_module=loaded.module,
            ),
            torch.inference_mode(),
        ):
            sequences = loaded.model.generate(**encoded, **generate_kwargs)

        trimmed_tokens = [
            trim_token_ids(row[prompt_width:].tolist(), eos_token_id=eos_token_id)
            for row in sequences
        ]
        for prompt, tokens in zip(batch_prompts, trimmed_tokens, strict=True):
            continuation = loaded.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(
                EvaluationRecord(
                    method=method.name,
                    direction_id=str(direction_id),
                    alpha=float(alpha),
                    text=prompt + continuation,
                    continuation_token_ids=tuple(tokens),
                )
            )
    return results


def _assert_activation_compatible(
    loaded: LoadedCausalLM, activations: ActivationDataset, config: ExperimentConfig
) -> None:
    manifest = activations.manifest
    mismatches = []
    if manifest.model_name != config.model.name_or_path:
        mismatches.append("model_name")
    if manifest.model_revision != loaded.model_revision:
        mismatches.append("model_revision")
    if manifest.tokenizer_name != loaded.tokenizer_name:
        mismatches.append("tokenizer_name")
    if manifest.tokenizer_revision != loaded.tokenizer_revision:
        mismatches.append("tokenizer_revision")
    if manifest.module_path != loaded.module_path:
        mismatches.append("module_path")
    if manifest.layer != config.model.layer:
        mismatches.append("layer")
    model_width = getattr(loaded.model.config, "hidden_size", None) or getattr(
        loaded.model.config, "n_embd", None
    )
    if model_width is not None and manifest.d_model != model_width:
        mismatches.append("d_model")
    if mismatches:
        raise CompatibilityError(f"Activation store mismatch: {', '.join(mismatches)}.")


def _effective_eos_token_id(loaded: LoadedCausalLM) -> int | list[int] | None:
    generation_config = getattr(loaded.model, "generation_config", None)
    candidate = getattr(generation_config, "eos_token_id", None)
    if candidate is None:
        candidate = loaded.tokenizer.eos_token_id
    if isinstance(candidate, int):
        return candidate
    return list(candidate) if candidate else None


def _context_limit(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 2048


@contextmanager
def _fork_seeded_rng(seed: int, device: torch.device):
    """Pair stochastic decoding without relying on version-specific HF kwargs."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    mps_state = None
    if device.type == "mps" and hasattr(torch, "mps"):
        mps_state = torch.mps.get_rng_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # torch.manual_seed also mutates every CUDA generator. Seed the CPU
            # default directly so unrelated GPUs remain untouched.
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
