"""Gaussian and SAE-direction corruptions used to train the denoisers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from steering_denoising.config import CorruptionConfig
from steering_denoising.denoiser import ActivationStats

if TYPE_CHECKING:
    from steering_denoising.directions import DirectionBank


class ActivationCorruptor:
    CLEAN = 0
    GAUSSIAN = 1
    STRUCTURED = 2

    def __init__(
        self,
        config: CorruptionConfig,
        stats: ActivationStats,
        direction_bank: DirectionBank | None = None,
    ) -> None:
        if config.structured_probability and direction_bank is None:
            raise ValueError("Structured corruption requires train SAE directions.")
        self.config = config
        self.activation_rms = stats.activation_rms
        self.direction_bank = direction_bank
        probabilities = torch.tensor(
            [
                config.clean_probability,
                config.gaussian_probability,
                config.structured_probability,
            ]
        )
        self.probabilities = probabilities / probabilities.sum()
        self.direction_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def __call__(
        self,
        clean: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        batch = clean.shape[0]
        device = clean.device
        kinds = torch.multinomial(
            self.probabilities.to(device).expand(batch, -1),
            1,
            generator=generator,
        ).squeeze(1)
        result = clean.clone()

        gaussian_rows = kinds == self.GAUSSIAN
        if gaussian_rows.any():
            source = clean[gaussian_rows]
            strengths = _sample_strength(
                len(source),
                self.config.gaussian_strength_min,
                self.config.gaussian_strength_max,
                device,
                generator,
            )
            noise = torch.randn(source.shape, device=device, dtype=clean.dtype, generator=generator)
            result[gaussian_rows] = source + noise * strengths[:, None, None] * self.activation_rms

        structured_rows = kinds == self.STRUCTURED
        if structured_rows.any():
            assert self.direction_bank is not None
            count = int(structured_rows.sum())
            strengths = _sample_strength(
                count,
                self.config.structured_strength_min,
                self.config.structured_strength_max,
                device,
                generator,
            )
            indices = torch.randint(
                len(self.direction_bank), (count,), device=device, generator=generator
            )
            directions = self._directions(device, clean.dtype)[indices]
            directions /= directions.float().square().mean(-1, keepdim=True).sqrt().to(clean.dtype)
            signs = torch.randint(0, 2, (count,), device=device, generator=generator) * 2 - 1
            shift = directions * strengths[:, None] * signs[:, None] * self.activation_rms
            result[structured_rows] = clean[structured_rows] + shift[:, None, :]

        return torch.where(attention_mask[..., None].bool(), result, clean)

    def _directions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self.direction_cache:
            assert self.direction_bank is not None
            self.direction_cache[key] = self.direction_bank.vectors.to(device=device, dtype=dtype)
        return self.direction_cache[key]


def _sample_strength(
    size: int,
    minimum: float,
    maximum: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if minimum == maximum:
        return torch.full((size,), minimum, device=device)
    uniform = torch.rand(size, device=device, generator=generator)
    if minimum == 0:
        return uniform * maximum
    return torch.exp(math.log(minimum) + uniform * math.log(maximum / minimum))


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    per_token_mse = (prediction.float() - target.float()).square().mean(-1)
    mask = attention_mask.to(device=prediction.device, dtype=torch.float32)
    return (per_token_mse * mask).sum() / mask.sum()
