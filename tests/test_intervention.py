from __future__ import annotations

import pytest
import torch
from torch import nn

from steering_denoising.intervention import SteeringOperator


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Identity()

    def forward(
        self, hidden: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del attention_mask
        return self.block(hidden)


def test_baseline_and_rms_scaling() -> None:
    direction = torch.tensor([3.0, 4.0])
    hidden = torch.zeros(1, 2)
    torch.testing.assert_close(
        SteeringOperator.baseline(direction, 0.5)(hidden),
        hidden + 0.5 * direction,
    )
    torch.testing.assert_close(
        SteeringOperator.baseline(direction, 0.5, activation_rms=2.0)(hidden),
        hidden + direction / direction.square().mean().sqrt(),
    )


def test_denoiser_runs_after_steering() -> None:
    denoiser = lambda value: value + 2  # noqa: E731
    operator = SteeringOperator.denoised(torch.tensor([1.0, 0.0]), 1.5, denoiser)
    torch.testing.assert_close(operator(torch.zeros(1, 2)), torch.tensor([[3.5, 2.0]]))


def test_hook_respects_padding_and_cached_mask_suffix() -> None:
    model = TinyModel()
    operator = SteeringOperator.baseline(torch.tensor([1.0, 0.0]), 1.0)
    hidden = torch.zeros(1, 2, 2)
    full_mask = torch.tensor([[1, 1, 0, 1]])

    with operator.hook(model, target_module=model.block):
        output = model(hidden, attention_mask=full_mask)

    torch.testing.assert_close(output, torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]))
    assert not model.block._forward_hooks


def test_tiny_huggingface_gpt2_generation_smoke() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        vocab_size=19,
        n_positions=16,
        n_embd=8,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    operator = SteeringOperator.baseline(torch.arange(1, 9).float(), 0.01)

    with operator.hook(model, target_module=model.transformer.h[0]):
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=2,
            use_cache=True,
        )

    assert generated.shape == (1, 5)
