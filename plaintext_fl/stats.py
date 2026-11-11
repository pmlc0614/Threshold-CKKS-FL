"""Numerical statistics for model states and updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .state import (
    StateEntry,
    layout_numel,
    validate_layout,
)


def _as_cpu_float64(
    values: Any,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
    else:
        tensor = torch.as_tensor(values)

    tensor = (
        tensor.reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )

    if tensor.numel() == 0:
        raise ValueError(
            "cannot calculate statistics for an empty vector"
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            "statistics input contains NaN or infinity"
        )

    return tensor


def tensor_stats(
    values: Any,
) -> dict[str, int | float]:
    """Calculate descriptive statistics for a vector."""

    tensor = _as_cpu_float64(values)

    absolute = tensor.abs()

    quantiles = torch.quantile(
        tensor,
        torch.tensor(
            [0.01, 0.50, 0.99],
            dtype=torch.float64,
        ),
    )

    return {
        "count": int(tensor.numel()),
        "bytes_float32": int(
            tensor.numel() * 4
        ),
        "bytes_float64": int(
            tensor.numel() * 8
        ),
        "mean": float(
            tensor.mean().item()
        ),
        "std": float(
            tensor.std(
                unbiased=False
            ).item()
        ),
        "min": float(
            tensor.min().item()
        ),
        "max": float(
            tensor.max().item()
        ),
        "max_abs": float(
            absolute.max().item()
        ),
        "l1": float(
            absolute.sum().item()
        ),
        "l2": float(
            torch.linalg.vector_norm(
                tensor,
                ord=2,
            ).item()
        ),
        "zero_fraction": float(
            (tensor == 0)
            .to(torch.float64)
            .mean()
            .item()
        ),
        "q01": float(
            quantiles[0].item()
        ),
        "q50": float(
            quantiles[1].item()
        ),
        "q99": float(
            quantiles[2].item()
        ),
    }


def difference_stats(
    reference: Any,
    candidate: Any,
) -> dict[str, float | None]:
    """Measure candidate error relative to reference."""

    reference_tensor = _as_cpu_float64(
        reference
    )
    candidate_tensor = _as_cpu_float64(
        candidate
    )

    if reference_tensor.shape != candidate_tensor.shape:
        raise ValueError(
            "reference and candidate must have equal shapes"
        )

    error = (
        candidate_tensor - reference_tensor
    )

    reference_l2 = torch.linalg.vector_norm(
        reference_tensor,
        ord=2,
    )

    candidate_l2 = torch.linalg.vector_norm(
        candidate_tensor,
        ord=2,
    )

    error_l2 = torch.linalg.vector_norm(
        error,
        ord=2,
    )

    denominator = (
        reference_l2 * candidate_l2
    )

    if denominator.item() > 0:
        cosine_similarity: float | None = float(
            torch.dot(
                reference_tensor,
                candidate_tensor,
            ).div(denominator).item()
        )
    elif error_l2.item() == 0:
        cosine_similarity = 1.0
    else:
        cosine_similarity = None

    if reference_l2.item() > 0:
        relative_l2_error: float | None = float(
            error_l2.div(
                reference_l2
            ).item()
        )
    elif error_l2.item() == 0:
        relative_l2_error = 0.0
    else:
        relative_l2_error = None

    return {
        "rmse": float(
            torch.sqrt(
                torch.mean(error.square())
            ).item()
        ),
        "max_abs_error": float(
            error.abs().max().item()
        ),
        "error_l2": float(
            error_l2.item()
        ),
        "reference_l2": float(
            reference_l2.item()
        ),
        "candidate_l2": float(
            candidate_l2.item()
        ),
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
    }


def statewise_stats(
    flat_values: Any,
    layout: Sequence[StateEntry],
) -> dict[str, dict[str, Any]]:
    """Calculate statistics for every state entry."""

    validate_layout(layout)

    flat_tensor = _as_cpu_float64(
        flat_values
    )

    expected_numel = layout_numel(
        layout
    )

    if flat_tensor.numel() != expected_numel:
        raise ValueError(
            f"flat vector has {flat_tensor.numel()} values; "
            f"expected {expected_numel}"
        )

    result: dict[str, dict[str, Any]] = {}

    for entry in layout:
        piece = flat_tensor[
            entry.offset:
            entry.offset + entry.numel
        ]

        result[entry.name] = {
            "shape": list(entry.shape),
            "offset": entry.offset,
            "is_parameter": entry.is_parameter,
            **tensor_stats(piece),
        }

    return result


def model_storage_stats(
    model: nn.Module,
    layout: Sequence[StateEntry],
) -> dict[str, int]:
    """Return model parameter and state storage sizes."""

    validate_layout(layout)

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    float_state_numel = layout_numel(
        layout
    )

    nonfloat_state_numel = sum(
        tensor.numel()
        for tensor in model.state_dict().values()
        if not tensor.is_floating_point()
    )

    return {
        "trainable_parameter_count": int(
            trainable_parameter_count
        ),
        "total_parameter_count": int(
            total_parameter_count
        ),
        "float_state_numel": int(
            float_state_numel
        ),
        "float_state_bytes_float32": int(
            float_state_numel * 4
        ),
        "float_state_bytes_float64": int(
            float_state_numel * 8
        ),
        "nonfloat_state_numel": int(
            nonfloat_state_numel
        ),
        "state_entry_count": int(
            len(layout)
        ),
    }


def write_json(
    path: str | Path,
    document: dict[str, Any],
) -> None:
    """Write a JSON document atomically."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(destination)


def append_jsonl(
    path: str | Path,
    record: dict[str, Any],
) -> None:
    """Append one JSON object to a JSON Lines file."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )

    with destination.open(
        mode="a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            serialized + "\n"
        )
        handle.flush()