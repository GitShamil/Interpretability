"""Activation steering applied through one GPT-2 block hook."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch
from torch import nn


def _hidden(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError("A transformer block must return a tensor or a tuple starting with one.")


def replace_hidden_output(output: object, hidden: torch.Tensor) -> object:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError("Unsupported transformer block output.")


class SteeringOperator:
    def __init__(
        self,
        direction: torch.Tensor,
        alpha: float,
        *,
        denoiser: nn.Module | None = None,
        activation_rms: float | None = None,
    ) -> None:
        if direction.ndim != 1 or not direction.norm():
            raise ValueError("direction must be a non-zero vector.")
        direction = direction.detach().float()
        if activation_rms is None:
            shift = float(alpha) * direction
        else:
            direction_rms = direction.square().mean().sqrt()
            shift = float(alpha) * float(activation_rms) * direction / direction_rms
        self.direction = direction
        self.alpha = float(alpha)
        self.denoiser = denoiser
        self.shift = shift
        self._shift_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    def baseline(
        cls,
        direction: torch.Tensor,
        alpha: float,
        *,
        activation_rms: float | None = None,
    ) -> SteeringOperator:
        return cls(direction, alpha, activation_rms=activation_rms)

    @classmethod
    def denoised(
        cls,
        direction: torch.Tensor,
        alpha: float,
        denoiser: nn.Module,
        *,
        activation_rms: float | None = None,
    ) -> SteeringOperator:
        return cls(
            direction,
            alpha,
            denoiser=denoiser,
            activation_rms=activation_rms,
        )

    @property
    def is_exact_noop(self) -> bool:
        return self.denoiser is None and self.alpha == 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.is_exact_noop:
            return hidden
        key = (hidden.device, hidden.dtype)
        if key not in self._shift_cache:
            self._shift_cache[key] = self.shift.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + self._shift_cache[key]
        return steered if self.denoiser is None else self.denoiser(steered)

    def hook(self, model: nn.Module, *, target_module: nn.Module) -> SteeringHook:
        return SteeringHook(model, target_module, self)


class SteeringHook(AbstractContextManager["SteeringHook"]):
    def __init__(
        self,
        model: nn.Module,
        target_module: nn.Module,
        operator: SteeringOperator,
    ) -> None:
        self.model = model
        self.target_module = target_module
        self.operator = operator
        self.attention_mask: torch.Tensor | None = None
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> SteeringHook:
        self.handles = [
            self.model.register_forward_pre_hook(self._capture_mask, with_kwargs=True),
            self.target_module.register_forward_hook(self._apply),
        ]
        return self

    def __exit__(self, *args: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture_mask(
        self,
        _module: nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        mask = kwargs.get("attention_mask")
        self.attention_mask = mask if torch.is_tensor(mask) else None

    def _apply(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> object:
        if self.operator.is_exact_noop:
            return output
        hidden = _hidden(output)
        updates = self.operator(hidden)
        if self.attention_mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        else:
            mask = self.attention_mask[:, -hidden.shape[1] :].to(hidden.device).bool()
        return replace_hidden_output(output, torch.where(mask[..., None], updates, hidden))
