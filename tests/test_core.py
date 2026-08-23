from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from steering_denoising.activations import ActivationDataset, ActivationStoreWriter
from steering_denoising.analysis import EvaluationRecord, summarize_records
from steering_denoising.cli import build_parser
from steering_denoising.config import (
    ConfigurationError,
    CorruptionConfig,
    ExperimentConfig,
    load_config,
)
from steering_denoising.corruption import ActivationCorruptor
from steering_denoising.denoiser import ActivationStats
from steering_denoising.directions import (
    DirectionBank,
)
from steering_denoising.evaluation import _effective_eos_token_id, _fork_seeded_rng
from steering_denoising.exceptions import CompatibilityError
from steering_denoising.hf import (
    extract_hidden,
    position_ids_from_attention_mask,
    read_text_records,
    resolve_intervention_module,
)
from steering_denoising.metrics import distinct_n_micro, trim_token_ids


def test_all_committed_configs_load_and_resolve_relative_paths() -> None:
    for path in Path("configs").glob("*.toml"):
        config = load_config(path)
        assert config.resolve_path(config.data.activation_dir).is_absolute()
    structured = load_config("configs/train_structured.toml")
    assert structured.training.method == "structured"
    assert structured.corruption.structured_probability > 0


def test_config_rejects_unknown_and_method_corruption_mismatch(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("surprise = 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unknown top-level"):
        load_config(bad)
    config = ExperimentConfig()
    config.corruption.structured_probability = 0.1
    with pytest.raises(ConfigurationError, match="Gaussian training"):
        config.validate()


def test_activation_store_round_trip_and_masked_stats(tmp_path: Path) -> None:
    writer = ActivationStoreWriter(
        tmp_path,
        model_name="toy",
        model_revision="abc",
        tokenizer_name="toy",
        tokenizer_revision="abc",
        module_path="transformer.h.0",
        layer=0,
        storage_dtype="float32",
        shard_sequences=2,
    )
    values = torch.tensor(
        [
            [[1.0, 2.0], [100.0, 100.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0], [100.0, 100.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
        ]
    )
    mask = torch.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    writer.add(values[:1], mask[:1])
    writer.add(values[1:], mask[1:])
    manifest = writer.finalize()
    assert manifest.sequences == 3
    assert manifest.valid_tokens == 7
    assert len(manifest.shards) == 2
    dataset = ActivationDataset(tmp_path)
    assert torch.equal(dataset[2]["activation"], values[2])
    stats = ActivationStats.fit_dataset(dataset)
    expected = values[mask]
    assert stats.count == 7
    assert torch.allclose(stats.mean, expected.mean(0), atol=1e-6)


def test_structured_corruption_is_sequence_shared_and_padding_safe() -> None:
    stats = ActivationStats(
        mean=torch.zeros(3),
        scale=torch.ones(3),
        activation_rms=2.0,
        count=10,
        scale_floor=1e-5,
    )
    bank = DirectionBank(torch.tensor([[3.0, 0.0, 0.0]]))
    config = CorruptionConfig(
        gaussian_probability=0.0,
        structured_probability=1.0,
        clean_probability=0.0,
        structured_strength_min=0.5,
        structured_strength_max=0.5,
    )
    clean = torch.zeros(2, 3, 3)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    corrupted = ActivationCorruptor(config, stats, bank)(
        clean, mask, generator=torch.Generator().manual_seed(4)
    )
    assert torch.equal(corrupted[0, 0], corrupted[0, 1])
    assert torch.equal(corrupted[0, 2], clean[0, 2])
    perturbation_rms = corrupted[1, 0].square().mean().sqrt()
    assert perturbation_rms.item() == pytest.approx(1.0, rel=1e-6)


def test_distinct_micro_and_eos_trimming() -> None:
    assert distinct_n_micro([[1, 2, 1], [1, 2]], 2) == pytest.approx(2 / 3)
    assert trim_token_ids([3, 0, 4, 1, 0], eos_token_id=1) == [3, 0, 4]
    assert trim_token_ids([3, 5, 4], eos_token_id=[5, 6]) == [3]


def test_effective_eos_prefers_model_generation_config() -> None:
    loaded = SimpleNamespace(
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 7, 2])),
        tokenizer=SimpleNamespace(eos_token_id=9),
    )
    assert _effective_eos_token_id(loaded) == [2, 7, 2]


def test_seed_fork_is_paired_and_restores_global_rng() -> None:
    torch.manual_seed(99)
    state = torch.random.get_rng_state().clone()
    with _fork_seeded_rng(7, torch.device("cpu")):
        first = torch.rand(4)
    assert torch.equal(torch.random.get_rng_state(), state)
    with _fork_seeded_rng(7, torch.device("cpu")):
        second = torch.rand(4)
    assert torch.equal(first, second)


def test_analysis_outputs_only_distinct_micro_metrics() -> None:
    records = [
        EvaluationRecord(
            method="baseline",
            direction_id="v",
            alpha=0.5,
            text="first",
            continuation_token_ids=(1, 2, 1),
        ),
        EvaluationRecord(
            method="baseline",
            direction_id="v",
            alpha=0.5,
            text="second",
            continuation_token_ids=(1, 2),
        ),
    ]
    assert summarize_records(records) == {
        "method": "baseline",
        "direction_id": "v",
        "alpha": 0.5,
        "distinct_1_micro": pytest.approx(2 / 5),
        "distinct_2_micro": pytest.approx(2 / 3),
        "distinct_3_micro": pytest.approx(1.0),
    }


def test_evaluation_record_export_contains_only_judging_fields() -> None:
    record = EvaluationRecord(
        method="baseline",
        direction_id="42",
        alpha=0.5,
        text="The city was quiet.",
        continuation_token_ids=(1, 2),
    )

    assert record.to_export_dict() == {
        "method": "baseline",
        "direction_id": "42",
        "alpha": 0.5,
        "text": "The city was quiet.",
    }


def test_hf_output_adapter_text_readers_and_cli(tmp_path: Path) -> None:
    hidden = torch.randn(2, 3, 4)
    assert extract_hidden(hidden) is hidden
    assert extract_hidden((hidden, object())) is hidden
    text_path = tmp_path / "records.txt"
    text_path.write_text("one\n\ntwo\n", encoding="utf-8")
    assert read_text_records(text_path) == ["one", "two"]
    args = build_parser().parse_args(["train", "--config", "run.toml"])
    assert args.command == "train"
    positions = position_ids_from_attention_mask(torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]))
    assert positions.tolist() == [[0, 0, 0, 1], [0, 0, 1, 2]]
    with pytest.raises(CompatibilityError, match="was not found"):
        resolve_intervention_module(torch.nn.Module(), 5)
