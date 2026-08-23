"""Hugging Face integration kept behind lazy imports.

The core denoiser and its tests do not require Transformers. Install the ``lm``
extra for collection and generation: ``pip install -e '.[lm]'``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any

import torch
from torch import nn

from steering_denoising.activations import ActivationManifest, ActivationStoreWriter
from steering_denoising.config import ExperimentConfig, ModelConfig
from steering_denoising.exceptions import CompatibilityError
from steering_denoising.utils import resolve_device, resolve_dtype, sha256_file


@dataclass(slots=True)
class LoadedCausalLM:
    model: nn.Module
    tokenizer: Any
    module: nn.Module
    module_path: str
    device: torch.device
    dtype: torch.dtype
    model_revision: str | None
    tokenizer_name: str
    tokenizer_revision: str | None


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "This command needs Hugging Face Transformers. Install with `pip install -e '.[lm]'`."
        ) from error
    return AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(config: ModelConfig) -> LoadedCausalLM:
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    tokenizer_name = config.name_or_path
    tokenizer_revision = config.revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise CompatibilityError("Decoder-only tokenizer needs an EOS or PAD token.")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "revision": config.revision,
    }
    # Transformers 5 renamed torch_dtype -> dtype; v4 does not consistently
    # accept the new spelling across architectures.
    try:
        import transformers

        major_version = int(transformers.__version__.split(".", maxsplit=1)[0])
    except (ImportError, ValueError):  # pragma: no cover - already guarded above.
        major_version = 4
    kwargs["dtype" if major_version >= 5 else "torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(config.name_or_path, **kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    module_path, module = resolve_intervention_module(model, config.layer)
    # Prefer the resolved immutable commit over a floating request such as
    # ``main``. This value is persisted in activation/checkpoint manifests.
    revision = getattr(model.config, "_commit_hash", None) or config.revision
    resolved_tokenizer_revision = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or tokenizer_revision
    )
    return LoadedCausalLM(
        model=model,
        tokenizer=tokenizer,
        module=module,
        module_path=module_path,
        device=device,
        dtype=dtype,
        model_revision=revision,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=resolved_tokenizer_revision,
    )


def resolve_intervention_module(
    model: nn.Module,
    layer: int,
) -> tuple[str, nn.Module]:
    """Return GPT-2 block ``layer`` (zero-based)."""

    module_path = f"transformer.h.{layer}"
    try:
        return module_path, model.get_submodule(module_path)
    except (AttributeError, KeyError) as error:
        raise CompatibilityError(f"GPT-2 block {module_path!r} was not found.") from error


def extract_hidden(output: Any) -> torch.Tensor:
    """Extract resid-post hidden states from HF block output across v4/v5 ABIs."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise CompatibilityError(
        f"Unsupported transformer block output {type(output).__name__}; expected Tensor/tuple."
    )


class HiddenCapture:
    """One-hook capture utility that never mutates model output."""

    def __init__(self, module: nn.Module, *, detach: bool = True) -> None:
        self.module = module
        self.detach = detach
        self.value: torch.Tensor | None = None
        self._handle: Any | None = None

    def __enter__(self) -> HiddenCapture:
        if self._handle is not None:
            raise RuntimeError("HiddenCapture cannot be nested or reused while active.")

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = extract_hidden(output)
            self.value = value.detach() if self.detach else value

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._handle is not None
        self._handle.remove()
        self._handle = None

    def require(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("Target activation module was not called.")
        return self.value


def read_text_records(path: str | Path) -> list[str]:
    return list(iter_text_records(path))


def iter_text_records(path: str | Path) -> Iterator[str]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield text


def collect_activations(
    config: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> ActivationManifest:
    """Collect resid-post activations from the training text file."""

    loaded = load_causal_lm(config.model)
    text_path = config.resolve_path(config.data.train_text_path)
    text_iterator = iter_text_records(text_path)
    try:
        first_text = next(text_iterator)
    except StopIteration:
        raise ValueError(f"No non-empty texts found at {text_path}.") from None
    texts = chain((first_text,), text_iterator)
    target_dir = Path(output_dir) if output_dir else config.resolve_path(config.data.activation_dir)
    writer = ActivationStoreWriter(
        target_dir,
        model_name=config.model.name_or_path,
        model_revision=loaded.model_revision,
        tokenizer_name=config.model.name_or_path,
        tokenizer_revision=loaded.tokenizer_revision,
        module_path=loaded.module_path,
        layer=config.model.layer,
        storage_dtype="float16",
        shard_sequences=config.data.shard_sequences,
        source={
            "path": str(text_path),
            "sha256": sha256_file(text_path),
            "requested_model_revision": config.model.revision,
        },
        overwrite=overwrite,
    )
    valid_tokens = 0
    batch_size = config.data.collection_batch_size
    with torch.inference_mode():
        for batch_texts in _batched(texts, batch_size):
            encoded = loaded.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=config.data.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_mask = encoded.pop("special_tokens_mask")
            encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
            encoded["position_ids"] = position_ids_from_attention_mask(encoded["attention_mask"])
            with HiddenCapture(loaded.module) as capture:
                loaded.model(**encoded, use_cache=False)
            activations = capture.require()
            mask = encoded["attention_mask"].bool()
            mask = mask & ~special_mask.to(device=loaded.device, dtype=torch.bool)
            max_tokens = config.data.max_train_tokens
            if max_tokens is not None and valid_tokens + int(mask.sum()) > max_tokens:
                remaining = max_tokens - valid_tokens
                if remaining <= 0:
                    break
                flat_valid = mask.flatten().nonzero(as_tuple=False).flatten()
                mask_flat = torch.zeros_like(mask, dtype=torch.bool).flatten()
                mask_flat[flat_valid[:remaining]] = True
                mask = mask_flat.reshape_as(mask)
            writer.add(activations, mask)
            valid_tokens += int(mask.sum())
            if (
                config.data.max_train_tokens is not None
                and valid_tokens >= config.data.max_train_tokens
            ):
                break
    return writer.finalize()


def _batched(values: Iterator[str], size: int) -> Iterator[list[str]]:
    while batch := list(islice(values, size)):
        yield batch


@contextmanager
def temporarily_eval(module: nn.Module) -> Iterator[None]:
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Natural zero-based positions for left- or right-padded decoder batches."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence].")
    mask = attention_mask.to(dtype=torch.long)
    positions = mask.cumsum(dim=-1) - 1
    return positions.masked_fill(mask == 0, 0)
