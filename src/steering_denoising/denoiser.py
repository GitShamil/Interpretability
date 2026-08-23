"""Token-local residual MLP and activation normalization statistics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from steering_denoising.activations import ActivationDataset
from steering_denoising.config import DenoiserConfig


@dataclass(frozen=True, slots=True)
class ActivationStats:
    mean: torch.Tensor
    scale: torch.Tensor
    activation_rms: float
    count: int
    scale_floor: float = 1e-5

    @property
    def d_model(self) -> int:
        return self.mean.numel()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.cpu().tolist(),
            "scale": self.scale.cpu().tolist(),
            "activation_rms": self.activation_rms,
            "count": self.count,
            "scale_floor": self.scale_floor,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationStats:
        return cls(
            mean=torch.tensor(value["mean"], dtype=torch.float32),
            scale=torch.tensor(value["scale"], dtype=torch.float32),
            activation_rms=float(value["activation_rms"]),
            count=int(value["count"]),
            scale_floor=float(value.get("scale_floor", 1e-5)),
        )

    @classmethod
    def fit(
        cls,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        scale_floor: float = 1e-5,
    ) -> ActivationStats:
        total: torch.Tensor | None = None
        squared_total: torch.Tensor | None = None
        token_rms: list[torch.Tensor] = []
        count = 0
        for activations, mask in batches:
            valid = activations.detach().cpu().double()[mask.detach().cpu().bool()]
            if not len(valid):
                continue
            if total is None:
                total = torch.zeros(valid.shape[-1], dtype=torch.float64)
                squared_total = torch.zeros_like(total)
            total += valid.sum(0)
            assert squared_total is not None
            squared_total += valid.square().sum(0)
            token_rms.append(valid.square().mean(-1).sqrt().float())
            count += len(valid)
        if total is None or squared_total is None:
            raise ValueError("No valid activation tokens.")
        mean = total / count
        scale = (squared_total / count - mean.square()).clamp_min(0).sqrt()
        return cls(
            mean=mean.float(),
            scale=scale.clamp_min(scale_floor).float(),
            activation_rms=float(torch.cat(token_rms).median()),
            count=count,
            scale_floor=scale_floor,
        )

    @classmethod
    def fit_dataset(cls, dataset: ActivationDataset) -> ActivationStats:
        return cls.fit(dataset.iter_shards())


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.network = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.SiLU(),
            nn.Linear(width * expansion, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(self.norm(inputs))


class ResidualDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int,
        stats: ActivationStats,
        config: DenoiserConfig | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config or DenoiserConfig()
        width = self.config.width or round(d_model * self.config.width_multiplier)
        self.register_buffer("activation_mean", stats.mean.float().clone())
        self.register_buffer("activation_scale", stats.scale.float().clone())
        self.register_buffer("activation_rms", torch.tensor(stats.activation_rms))
        self.stats_count = stats.count
        self.stats_scale_floor = stats.scale_floor
        self.input_projection = nn.Sequential(nn.Linear(d_model, width), nn.SiLU())
        self.blocks = nn.ModuleList(
            _ResidualBlock(width, self.config.expansion) for _ in range(self.config.depth)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        mean = self.activation_mean.to(inputs.device)
        scale = self.activation_scale.to(inputs.device)
        hidden = self.input_projection((inputs.float() - mean) / scale)
        for block in self.blocks:
            hidden = block(hidden)
        correction = scale * self.output_projection(self.output_norm(hidden))
        return inputs + correction.to(dtype)

    def stats(self) -> ActivationStats:
        return ActivationStats(
            mean=self.activation_mean.cpu(),
            scale=self.activation_scale.cpu(),
            activation_rms=float(self.activation_rms),
            count=self.stats_count,
            scale_floor=self.stats_scale_floor,
        )
