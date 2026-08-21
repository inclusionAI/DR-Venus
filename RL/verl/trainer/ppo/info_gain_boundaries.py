"""Validation helpers for explicit IG turn-boundary positions."""

from __future__ import annotations

from collections.abc import Sequence


def boundary_rows(
    positions_by_sample: Sequence[Sequence[int]], *, batch_size: int, sequence_length: int
) -> list[list[bool]]:
    if len(positions_by_sample) != batch_size:
        raise ValueError(
            f"IG boundary batch has {len(positions_by_sample)} rows, expected {batch_size}"
        )
    rows: list[list[bool]] = []
    for sample_index, positions in enumerate(positions_by_sample):
        row = [False] * sequence_length
        for raw_position in positions:
            if isinstance(raw_position, bool) or not isinstance(raw_position, int):
                raise ValueError(f"IG boundary for sample {sample_index} must be an integer")
            if not 0 <= raw_position < sequence_length:
                raise ValueError(
                    f"IG boundary {raw_position} for sample {sample_index} is outside [0, {sequence_length})"
                )
            row[raw_position] = True
        rows.append(row)
    return rows
