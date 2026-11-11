"""Flatten and restore floating-point PyTorch model state.

Only floating-point entries in model.state_dict() are included.
Integer buffers such as BatchNorm num_batches_tracked are excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


LAYOUT_SCHEMA = "plaintext-fl-state-layout-v1"


@dataclass(frozen=True)
class StateEntry:
    """Description of one floating-point state_dict entry."""

    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int
    torch_dtype: str
    is_parameter: bool


StateLayout = tuple[StateEntry, ...]


def build_float_state_layout(
    model: nn.Module,
) -> StateLayout:
    """Build a deterministic layout from model.state_dict() order."""

    parameter_names = {
        name
        for name, _parameter in model.named_parameters()
    }

    entries: list[StateEntry] = []
    offset = 0

    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue

        entry = StateEntry(
            name=name,
            shape=tuple(tensor.shape),
            offset=offset,
            numel=tensor.numel(),
            torch_dtype=str(tensor.dtype),
            is_parameter=name in parameter_names,
        )

        entries.append(entry)
        offset += tensor.numel()

    layout = tuple(entries)
    validate_layout(layout)

    return layout


def validate_layout(
    layout: Sequence[StateEntry],
) -> None:
    """Validate offsets, shapes and unique state names."""

    if len(layout) == 0:
        raise ValueError(
            "state layout must contain at least one entry"
        )

    expected_offset = 0
    names: set[str] = set()

    for entry in layout:
        if not entry.name:
            raise ValueError(
                "state entry name must not be empty"
            )

        if entry.name in names:
            raise ValueError(
                f"duplicate state entry: {entry.name}"
            )

        names.add(entry.name)

        if entry.offset != expected_offset:
            raise ValueError(
                f"invalid offset for {entry.name}: "
                f"expected {expected_offset}, "
                f"got {entry.offset}"
            )

        if entry.numel <= 0:
            raise ValueError(
                f"entry {entry.name} must have positive numel"
            )

        shape_numel = math.prod(entry.shape)

        if shape_numel != entry.numel:
            raise ValueError(
                f"shape/numel mismatch for {entry.name}: "
                f"shape contains {shape_numel} values, "
                f"but numel is {entry.numel}"
            )

        expected_offset += entry.numel


def layout_numel(
    layout: Sequence[StateEntry],
) -> int:
    validate_layout(layout)

    return sum(
        entry.numel
        for entry in layout
    )


def compact_parameter_layout(
    layout: Sequence[StateEntry],
) -> StateLayout:
    """Return trainable parameters reindexed into one contiguous layout.

    BatchNorm running statistics use their own dense auxiliary path in the
    v17 protocol. They must not occupy positions or alter chunk boundaries in
    the sparse trainable-parameter vector.
    """

    validate_layout(layout)
    entries: list[StateEntry] = []
    offset = 0
    for entry in layout:
        if not entry.is_parameter:
            continue
        entries.append(replace(entry, offset=offset))
        offset += entry.numel
    if not entries:
        raise ValueError("state layout contains no trainable parameters")
    result = tuple(entries)
    validate_layout(result)
    return result


def _canonical_layout_payload(
    layout: Sequence[StateEntry],
) -> str:
    validate_layout(layout)

    entries = [
        asdict(entry)
        for entry in layout
    ]

    return json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def layout_sha256(
    layout: Sequence[StateEntry],
) -> str:
    """Return a deterministic hash of the state layout."""

    payload = _canonical_layout_payload(
        layout
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def layout_to_document(
    layout: Sequence[StateEntry],
    model_name: str = "resnet20-cifar10",
) -> dict[str, Any]:
    validate_layout(layout)

    return {
        "schema": LAYOUT_SCHEMA,
        "model": model_name,
        "total_numel": layout_numel(layout),
        "sha256": layout_sha256(layout),
        "entries": [
            asdict(entry)
            for entry in layout
        ],
    }


def write_layout_json(
    path: str | Path,
    layout: Sequence[StateEntry],
    model_name: str = "resnet20-cifar10",
) -> None:
    """Write model layout atomically."""

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = layout_to_document(
        layout=layout,
        model_name=model_name,
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
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(destination)


def read_layout_json(
    path: str | Path,
) -> StateLayout:
    """Read and validate a previously written model layout."""

    document = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if document.get("schema") != LAYOUT_SCHEMA:
        raise ValueError(
            "unsupported model layout schema"
        )

    raw_entries = document.get("entries")

    if not isinstance(raw_entries, list):
        raise ValueError(
            "model layout entries must be a list"
        )

    entries: list[StateEntry] = []

    for raw_entry in raw_entries:
        entries.append(
            StateEntry(
                name=str(raw_entry["name"]),
                shape=tuple(
                    int(dimension)
                    for dimension
                    in raw_entry["shape"]
                ),
                offset=int(raw_entry["offset"]),
                numel=int(raw_entry["numel"]),
                torch_dtype=str(
                    raw_entry["torch_dtype"]
                ),
                is_parameter=bool(
                    raw_entry["is_parameter"]
                ),
            )
        )

    layout = tuple(entries)
    validate_layout(layout)

    expected_numel = layout_numel(layout)
    recorded_numel = int(
        document["total_numel"]
    )

    if recorded_numel != expected_numel:
        raise ValueError(
            "model layout total_numel mismatch"
        )

    expected_hash = layout_sha256(layout)
    recorded_hash = str(document["sha256"])

    if recorded_hash != expected_hash:
        raise ValueError(
            "model layout SHA-256 mismatch"
        )

    return layout


@torch.no_grad()
def flatten_float_state(
    model: nn.Module,
    layout: Sequence[StateEntry],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Flatten floating-point model state into one vector.

    The default output is a CPU float32 tensor. This gives a stable
    representation independent of whether model training occurred on
    CPU or CUDA.
    """

    validate_layout(layout)

    state = model.state_dict()
    destination_device = torch.device(device)
    flattened_parts: list[torch.Tensor] = []

    for entry in layout:
        if entry.name not in state:
            raise KeyError(
                f"model has no state entry named {entry.name}"
            )

        tensor = state[entry.name]

        if not tensor.is_floating_point():
            raise TypeError(
                f"state entry {entry.name} is not floating point"
            )

        if tuple(tensor.shape) != entry.shape:
            raise ValueError(
                f"shape mismatch for {entry.name}: "
                f"expected {entry.shape}, "
                f"got {tuple(tensor.shape)}"
            )

        if tensor.numel() != entry.numel:
            raise ValueError(
                f"numel mismatch for {entry.name}"
            )

        flattened_parts.append(
            tensor.detach()
            .reshape(-1)
            .to(
                device=destination_device,
                dtype=dtype,
            )
        )

    flattened = torch.cat(
        flattened_parts,
        dim=0,
    )

    expected_numel = layout_numel(layout)

    if flattened.numel() != expected_numel:
        raise RuntimeError(
            "flattened model state has an unexpected size"
        )

    return flattened.contiguous()


@torch.no_grad()
def load_float_state(
    model: nn.Module,
    flat_state: torch.Tensor,
    layout: Sequence[StateEntry],
) -> None:
    """Load a flat floating-point vector into a model.

    Non-floating-point state entries remain unchanged.
    """

    validate_layout(layout)

    flat = flat_state.detach().reshape(-1)

    expected_numel = layout_numel(layout)

    if flat.numel() != expected_numel:
        raise ValueError(
            f"flat state has {flat.numel()} values; "
            f"expected {expected_numel}"
        )

    current_state = model.state_dict()
    new_state = dict(current_state)

    for entry in layout:
        if entry.name not in current_state:
            raise KeyError(
                f"model has no state entry named {entry.name}"
            )

        target_tensor = current_state[entry.name]

        piece = flat[
            entry.offset:
            entry.offset + entry.numel
        ]

        piece = piece.reshape(entry.shape)

        new_state[entry.name] = piece.to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        )

    model.load_state_dict(
        new_state,
        strict=True,
    )


@torch.no_grad()
def compute_model_delta(
    local_model: nn.Module,
    global_model: nn.Module,
    layout: Sequence[StateEntry],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return local_state - global_state."""

    local_flat = flatten_float_state(
        model=local_model,
        layout=layout,
        device=device,
        dtype=torch.float32,
    )

    global_flat = flatten_float_state(
        model=global_model,
        layout=layout,
        device=device,
        dtype=torch.float32,
    )

    return (
        local_flat - global_flat
    ).contiguous()


@torch.no_grad()
def apply_model_delta(
    model: nn.Module,
    delta: torch.Tensor,
    layout: Sequence[StateEntry],
    scale: float = 1.0,
) -> None:
    """Apply model_state += scale * delta."""

    if not math.isfinite(scale):
        raise ValueError(
            "delta scale must be finite"
        )

    flat_delta = delta.detach().reshape(-1)

    expected_numel = layout_numel(layout)

    if flat_delta.numel() != expected_numel:
        raise ValueError(
            f"delta has {flat_delta.numel()} values; "
            f"expected {expected_numel}"
        )

    current_flat = flatten_float_state(
        model=model,
        layout=layout,
        device=flat_delta.device,
        dtype=torch.float32,
    )

    updated_flat = (
        current_flat
        + float(scale) * flat_delta.to(
            dtype=torch.float32
        )
    )

    load_float_state(
        model=model,
        flat_state=updated_flat,
        layout=layout,
    )


@torch.no_grad()
def clone_nonfloat_state(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Clone integer and boolean state entries.

    For ResNet-20 this primarily contains BatchNorm
    num_batches_tracked buffers.
    """

    result: dict[str, torch.Tensor] = {}

    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            continue

        result[name] = (
            tensor.detach()
            .cpu()
            .clone()
        )

    return result

@torch.no_grad()
def zero_nonparameter_entries(
    flat_values: torch.Tensor,
    layout: Sequence[StateEntry],
) -> torch.Tensor:
    """Set floating-point buffer entries to zero.

    Trainable parameters remain unchanged. This prevents BatchNorm
    running_mean and running_var from entering sparsification and
    error-feedback.
    """

    validate_layout(layout)

    flat = (
        flat_values.detach()
        .reshape(-1)
        .clone()
    )

    expected_numel = layout_numel(layout)

    if flat.numel() != expected_numel:
        raise ValueError(
            f"flat vector has {flat.numel()} values; "
            f"expected {expected_numel}"
        )

    for entry in layout:
        if entry.is_parameter:
            continue

        flat[
            entry.offset:
            entry.offset + entry.numel
        ].zero_()

    return flat.contiguous()


@torch.no_grad()
def clone_float_buffers(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Clone floating-point model buffers onto CPU.

    For ResNet-20 these are primarily BatchNorm running_mean and
    running_var. Integer num_batches_tracked buffers are excluded.
    """

    return {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }


@torch.no_grad()
def load_weighted_average_float_buffers(
    model: nn.Module,
    client_buffers: Sequence[
        dict[str, torch.Tensor]
    ],
    sample_counts: Sequence[int],
) -> None:
    """Load the exact sample-weighted average of client buffers."""

    if len(client_buffers) == 0:
        raise ValueError(
            "client_buffers must not be empty"
        )

    if len(client_buffers) != len(sample_counts):
        raise ValueError(
            "client_buffers and sample_counts "
            "must have equal lengths"
        )

    if any(
        int(count) <= 0
        for count in sample_counts
    ):
        raise ValueError(
            "all sample counts must be positive"
        )

    target_buffers = {
        name: buffer
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }

    expected_names = set(
        target_buffers.keys()
    )

    for client_id, buffers in enumerate(
        client_buffers
    ):
        if set(buffers.keys()) != expected_names:
            raise ValueError(
                f"client {client_id} float-buffer "
                "names do not match the global model"
            )

    total_samples = float(
        sum(
            int(count)
            for count in sample_counts
        )
    )

    for name, target in target_buffers.items():
        weighted_sum = torch.zeros(
            target.shape,
            dtype=torch.float64,
            device="cpu",
        )

        for client_id, (
            buffers,
            count,
        ) in enumerate(
            zip(
                client_buffers,
                sample_counts,
            )
        ):
            source = (
                buffers[name]
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )

            if source.shape != target.shape:
                raise ValueError(
                    f"client {client_id} buffer "
                    f"{name} has an invalid shape"
                )

            if not torch.isfinite(source).all():
                raise ValueError(
                    f"client {client_id} buffer "
                    f"{name} contains NaN or Inf"
                )

            weighted_sum.add_(
                source,
                alpha=float(count),
            )

        averaged = (
            weighted_sum / total_samples
        )

        if (
            name.endswith("running_var")
            and torch.any(averaged < 0)
        ):
            raise RuntimeError(
                f"aggregated {name} contains "
                "a negative variance"
            )

        target.copy_(
            averaged.to(
                device=target.device,
                dtype=target.dtype,
            )
        )
