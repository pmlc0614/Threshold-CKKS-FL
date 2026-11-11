"""Deterministic dropout schedules for federated protocol experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DropoutScheduleEntry:
    """One deterministic post-upload dropout event."""

    round_id: int
    dropped_client_id: int
    survivor_client_ids: tuple[int, ...]


def build_dropout_schedule(
    *,
    total_rounds: int,
    dropout_round_count: int,
    client_count: int,
    seed: int,
) -> tuple[DropoutScheduleEntry, ...]:
    """Preselect dropout rounds and one dropped silo without global RNG use."""

    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive")
    if client_count <= 1:
        raise ValueError("dropout requires at least two clients")
    if not 0 <= dropout_round_count <= total_rounds:
        raise ValueError(
            "dropout_round_count must be between zero and total_rounds"
        )
    generator = random.Random(int(seed))
    round_ids = sorted(
        generator.sample(range(total_rounds), dropout_round_count)
    )
    result = []
    for round_id in round_ids:
        dropped_client_id = generator.randrange(client_count)
        result.append(
            DropoutScheduleEntry(
                round_id=round_id,
                dropped_client_id=dropped_client_id,
                survivor_client_ids=tuple(
                    client_id
                    for client_id in range(client_count)
                    if client_id != dropped_client_id
                ),
            )
        )
    return tuple(result)
