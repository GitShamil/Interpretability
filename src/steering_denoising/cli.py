"""Command-line interface for the experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from steering_denoising import __version__
from steering_denoising.activations import ActivationDataset
from steering_denoising.config import ExperimentConfig, load_config
from steering_denoising.directions import DirectionBank, DirectionId, DirectionSplitManifest
from steering_denoising.evaluation import run_evaluation
from steering_denoising.hf import collect_activations
from steering_denoising.training import train_denoiser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steerlab",
        description="Activation-denoising experiments for language-model steering.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="Collect resid-post activations.")
    _add_config_argument(collect)
    collect.add_argument("--overwrite", action="store_true")

    split = commands.add_parser(
        "split-directions", help="Create a deterministic held-out direction split."
    )
    _add_config_argument(split)
    split.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="Train one configured denoiser.")
    _add_config_argument(train)

    evaluate = commands.add_parser("evaluate", help="Evaluate one steering method.")
    _add_config_argument(evaluate)
    evaluate.add_argument(
        "--method",
        choices=("baseline", "gaussian", "structured"),
        required=True,
    )
    evaluate.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        config = load_config(args.config)
        manifest = collect_activations(config, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "activation_dir": str(config.resolve_path(config.data.activation_dir)),
                    "valid_tokens": manifest.valid_tokens,
                    "sequences": manifest.sequences,
                    "d_model": manifest.d_model,
                    "module_path": manifest.module_path,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "split-directions":
        config = load_config(args.config)
        split = _create_direction_split(config, overwrite=args.overwrite)
        print(json.dumps(split.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "train":
        config = load_config(args.config)
        training_bank, split_fingerprint = _load_training_direction_bank(config)
        result = train_denoiser(
            config,
            direction_bank=training_bank,
            direction_split_fingerprint=split_fingerprint,
        )
        print(
            json.dumps(
                {
                    "method": result.metadata.method,
                    "steps": result.global_step,
                    "best_validation_loss": result.best_validation_loss,
                    "best_checkpoint": str(result.best_checkpoint),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate":
        config = load_config(args.config)
        output_root = config.resolve_path(config.evaluation.output_dir)
        config.evaluation.output_dir = str(output_root / args.method)
        config.evaluation.method = args.method
        summary = run_evaluation(config, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "output_dir": str(config.resolve_path(config.evaluation.output_dir)),
                    "summary_points": len(summary),
                },
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a TOML config.")


def _load_bank(config: ExperimentConfig) -> DirectionBank:
    activation_dir = config.resolve_path(config.data.activation_dir)
    activations = ActivationDataset(activation_dir)
    direction_path = config.resolve_path(config.directions.tensor_path)
    return DirectionBank.load(
        direction_path,
        expected_dim=activations.manifest.d_model,
    )


def _create_direction_split(config: ExperimentConfig, *, overwrite: bool) -> DirectionSplitManifest:
    bank = _load_bank(config)
    split_path = config.resolve_path(config.directions.split_path)
    if split_path.exists() and not overwrite:
        raise FileExistsError(f"Direction split already exists at {split_path}.")
    validation_ids = _read_ids(config.resolve_path(config.directions.validation_ids_path))
    split = DirectionSplitManifest.create(bank, validation_ids=validation_ids)
    split.save(split_path)
    return split


def _load_training_direction_bank(
    config: ExperimentConfig,
) -> tuple[DirectionBank | None, str | None]:
    if config.training.method == "gaussian":
        return None, None
    bank = _load_bank(config)
    split_path = config.resolve_path(config.directions.split_path)
    split = DirectionSplitManifest.load(split_path, bank=bank)
    return bank.subset_ids(split.train_ids), split.fingerprint


def _read_ids(path: Path) -> list[DirectionId]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{path} must contain a JSON list of integer SAE IDs.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
