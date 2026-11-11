"""Wall-clock helpers for sequential and logical parallel FL timing."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any


def synchronized_perf_counter(device: Any = None) -> float:
    """Read a wall clock after completing queued work on a CUDA device."""

    if device is not None and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
    return time.perf_counter()


def maximum_phase_seconds(values: Sequence[float]) -> float:
    """Return the critical-path time for one parallel client phase."""

    if not values:
        return 0.0
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("phase timings must be finite and non-negative")
    return max(values)


def estimated_barrier_parallel_seconds(
    *,
    client_phases: Sequence[Sequence[float]],
    server_seconds: float,
) -> float:
    """Add server work and the slowest client in every barrier phase."""

    if not math.isfinite(server_seconds) or server_seconds < 0.0:
        raise ValueError("server_seconds must be finite and non-negative")
    return server_seconds + sum(
        maximum_phase_seconds(phase) for phase in client_phases
    )
