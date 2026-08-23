"""Aggregation for steering evaluations."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from steering_denoising.metrics import distinct_n_micro


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    method: str
    direction_id: str
    alpha: float
    text: str
    continuation_token_ids: tuple[int, ...]

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "direction_id": self.direction_id,
            "alpha": self.alpha,
            "text": self.text,
        }


def summarize_records(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize empty evaluation records.")
    first = records[0]
    token_sequences = [record.continuation_token_ids for record in records]
    return {
        "method": first.method,
        "direction_id": first.direction_id,
        "alpha": first.alpha,
        "distinct_1_micro": distinct_n_micro(token_sequences, 1),
        "distinct_2_micro": distinct_n_micro(token_sequences, 2),
        "distinct_3_micro": distinct_n_micro(token_sequences, 3),
    }


def write_summary_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evaluation summary.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
