"""Metrics used by steering evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def trim_token_ids(
    token_ids: Sequence[int], *, eos_token_id: int | Sequence[int] | None
) -> list[int]:
    if eos_token_id is None:
        return [int(token_id) for token_id in token_ids]
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
    result: list[int] = []
    for token_id in token_ids:
        if int(token_id) in eos_ids:
            break
        result.append(int(token_id))
    return result


def distinct_n_micro(token_sequences: Iterable[Sequence[int]], n: int) -> float:
    unique: set[tuple[int, ...]] = set()
    total = 0
    for sequence in token_sequences:
        for index in range(len(sequence) - n + 1):
            unique.add(tuple(sequence[index : index + n]))
            total += 1
    return len(unique) / total if total else 0.0
