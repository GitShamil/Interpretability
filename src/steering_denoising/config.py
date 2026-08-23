"""Small TOML configuration for the GPT-2 experiment."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

from steering_denoising.exceptions import ConfigurationError


@dataclass(slots=True)
class ModelConfig:
    name_or_path: str = "gpt2"
    revision: str | None = "main"
    layer: int = 5
    device: str = "auto"
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = "float32"


@dataclass(slots=True)
class DataConfig:
    train_text_path: str = "../data/train.txt"
    eval_prompts_path: str = "../data/prompts.txt"
    activation_dir: str = "../artifacts/activations/gpt2-layer5"
    max_length: int = 256
    collection_batch_size: int = 8
    shard_sequences: int = 256
    max_train_tokens: int = 1_000_000


@dataclass(slots=True)
class DirectionConfig:
    tensor_path: str = "../data/sae_directions.npy"
    split_path: str = "../artifacts/directions/gpt2-layer5-split.json"
    validation_ids_path: str = "../data/validation_direction_ids.json"


@dataclass(slots=True)
class DenoiserConfig:
    width: int | None = None
    width_multiplier: float = 2.0
    depth: int = 2
    expansion: int = 2


@dataclass(slots=True)
class CorruptionConfig:
    gaussian_probability: float = 0.8
    structured_probability: float = 0.0
    clean_probability: float = 0.2
    gaussian_strength_min: float = 0.02
    gaussian_strength_max: float = 1.5
    structured_strength_min: float = 0.05
    structured_strength_max: float = 1.5


@dataclass(slots=True)
class TrainingConfig:
    method: Literal["gaussian", "structured"] = "gaussian"
    output_dir: str = "../artifacts/checkpoints/gaussian"
    seed: int = 42
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.05


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    batch_size: int = 4


@dataclass(slots=True)
class EvaluationConfig:
    output_dir: str = "../artifacts/evaluation/gpt2-layer5"
    gaussian_checkpoint: str = "../artifacts/checkpoints/gaussian/best"
    structured_checkpoint: str = "../artifacts/checkpoints/structured/best"
    method: Literal["baseline", "gaussian", "structured"] = "baseline"
    alphas: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 1.5])
    seed: int = 123
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    directions: DirectionConfig = field(default_factory=DirectionConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    _base_dir: Path = field(default=Path.cwd(), init=False, repr=False)

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self._base_dir / path).resolve()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("_base_dir", None)
        return result

    def as_resolved_dict(self) -> dict[str, Any]:
        result = self.as_dict()
        path_fields = {
            "data": ("train_text_path", "eval_prompts_path", "activation_dir"),
            "directions": ("tensor_path", "split_path", "validation_ids_path"),
            "training": ("output_dir",),
            "evaluation": ("output_dir", "gaussian_checkpoint", "structured_checkpoint"),
        }
        for section, names in path_fields.items():
            for name in names:
                result[section][name] = str(self.resolve_path(result[section][name]))
        return result

    def validate(self) -> None:
        if self.model.layer < 0:
            raise ConfigurationError("model.layer must be non-negative.")
        if (
            min(
                self.data.max_length,
                self.data.collection_batch_size,
                self.data.shard_sequences,
                self.data.max_train_tokens,
                self.training.epochs,
                self.training.batch_size,
                self.evaluation.generation.max_new_tokens,
                self.evaluation.generation.batch_size,
            )
            < 1
        ):
            raise ConfigurationError(
                "Lengths, token limits, epochs and batch sizes must be positive."
            )
        probabilities = (
            self.corruption.gaussian_probability,
            self.corruption.structured_probability,
            self.corruption.clean_probability,
        )
        if any(probability < 0 for probability in probabilities) or sum(probabilities) <= 0:
            raise ConfigurationError("Corruption probabilities must be non-negative and sum > 0.")
        if self.training.method == "gaussian" and self.corruption.structured_probability:
            raise ConfigurationError("Gaussian training cannot use structured corruption.")
        if self.training.method == "structured" and not self.corruption.structured_probability:
            raise ConfigurationError("Structured training requires structured corruption.")
        for minimum, maximum in (
            (self.corruption.gaussian_strength_min, self.corruption.gaussian_strength_max),
            (self.corruption.structured_strength_min, self.corruption.structured_strength_max),
        ):
            if minimum < 0 or maximum < minimum:
                raise ConfigurationError("Corruption strength range is invalid.")
        if self.denoiser.depth < 0 or self.denoiser.expansion < 1:
            raise ConfigurationError("Denoiser depth/expansion is invalid.")
        if self.denoiser.width is not None and self.denoiser.width < 1:
            raise ConfigurationError("Denoiser width must be positive.")
        if self.denoiser.width_multiplier <= 0:
            raise ConfigurationError("Denoiser width multiplier must be positive.")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ConfigurationError("Learning rate or weight decay is invalid.")
        if not 0 < self.training.validation_fraction < 0.5:
            raise ConfigurationError("Validation fraction must be between 0 and 0.5.")
        if not self.evaluation.alphas or any(
            not math.isfinite(alpha) or alpha < 0 for alpha in self.evaluation.alphas
        ):
            raise ConfigurationError("Evaluation alphas must be finite and non-negative.")
        generation = self.evaluation.generation
        if generation.temperature <= 0 or not 0 < generation.top_p <= 1:
            raise ConfigurationError("Generation temperature/top_p is invalid.")


ConfigType = TypeVar("ConfigType")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".toml":
        raise ConfigurationError("Configuration must be a TOML file.")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError("Top-level configuration must be a table.")

    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ConfigurationError(f"Unknown top-level fields: {sorted(unknown)}")
    sections: dict[str, Any] = {}
    for name, section_type in _SECTION_TYPES.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ConfigurationError(f"[{name}] must be a table.")
        if name == "evaluation":
            values = dict(values)
            generation = values.pop("generation", {})
            values["generation"] = _construct(GenerationConfig, generation, "evaluation.generation")
        sections[name] = _construct(section_type, values, name)
    config = ExperimentConfig(**sections)
    config._base_dir = config_path.parent
    config.validate()
    return config


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.as_resolved_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _construct(cls: type[ConfigType], values: dict[str, Any], path: str) -> ConfigType:
    unknown = set(values) - {item.name for item in fields(cls) if item.init}
    if unknown:
        raise ConfigurationError(f"Unknown fields in [{path}]: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as error:
        raise ConfigurationError(f"Invalid [{path}] config: {error}") from error


_SECTION_TYPES: dict[str, type[Any]] = {
    "model": ModelConfig,
    "data": DataConfig,
    "directions": DirectionConfig,
    "denoiser": DenoiserConfig,
    "corruption": CorruptionConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
}
