"""Aggregated-update statistics and noise experiments."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .state import (
    StateEntry,
    layout_numel,
    validate_layout,
)


def extract_parameter_values(
    flat_values: torch.Tensor,
    layout: Sequence[StateEntry],
) -> torch.Tensor:
    """Extract trainable parameter coordinates from a flat state."""

    validate_layout(layout)

    flat = (
        flat_values.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
    )

    expected_numel = layout_numel(
        layout
    )

    if flat.numel() != expected_numel:
        raise ValueError(
            f"flat vector has {flat.numel()} values; "
            f"expected {expected_numel}"
        )

    parts = [
        flat[
            entry.offset:
            entry.offset + entry.numel
        ]
        for entry in layout
        if entry.is_parameter
    ]

    if len(parts) == 0:
        raise ValueError(
            "layout contains no trainable parameters"
        )

    return torch.cat(
        parts,
        dim=0,
    ).contiguous()


def build_selected_parameter_mask(
    layout: Sequence[StateEntry],
    chunk_size: int,
    selected_chunk_ids: Sequence[int],
) -> torch.Tensor:
    """Return mask for parameters inside selected chunks."""

    validate_layout(layout)

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    total_numel = layout_numel(
        layout
    )

    total_chunk_count = math.ceil(
        total_numel / chunk_size
    )

    mask = torch.zeros(
        total_numel,
        dtype=torch.bool,
        device="cpu",
    )

    for chunk_id in selected_chunk_ids:
        chunk_id = int(chunk_id)

        if not 0 <= chunk_id < total_chunk_count:
            raise ValueError(
                f"invalid chunk ID: {chunk_id}"
            )

        begin = chunk_id * chunk_size
        end = min(
            begin + chunk_size,
            total_numel,
        )

        mask[begin:end] = True

    # Exclude buffers such as BatchNorm running means and variances.
    for entry in layout:
        if entry.is_parameter:
            continue

        mask[
            entry.offset:
            entry.offset + entry.numel
        ] = False

    if not torch.any(mask):
        raise ValueError(
            "selected chunks contain no parameter coordinates"
        )

    return mask.contiguous()


def resolve_gaussian_noise_std(
    reference_values: torch.Tensor,
    absolute_std: float | None,
    relative_std_ratio: float | None,
) -> float:
    """Resolve an absolute or update-relative noise sigma."""

    if (
        absolute_std is not None
        and relative_std_ratio is not None
    ):
        raise ValueError(
            "absolute and relative noise cannot "
            "be enabled simultaneously"
        )

    if absolute_std is not None:
        sigma = float(absolute_std)

        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError(
                "absolute noise std must be finite "
                "and non-negative"
            )

        return sigma

    if relative_std_ratio is not None:
        ratio = float(
            relative_std_ratio
        )

        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(
                "relative noise ratio must be finite "
                "and non-negative"
            )

        reference = (
            reference_values.detach()
            .reshape(-1)
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        reference_std = float(
            reference.std(
                unbiased=False
            ).item()
        )

        return ratio * reference_std

    return 0.0


def gaussian_noise_std_from_exponent(
    exponent: float | None,
    reference_bound: float | None = 1.0,
) -> float:
    """Return ``reference_bound * 2**(-exponent)``.

    A missing exponent disables noise and does not require a bound.
    """

    if exponent is None:
        return 0.0

    value = float(exponent)

    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "noise exponent must be finite and non-negative"
        )

    if reference_bound is None:
        raise ValueError(
            "reference bound is required when noise is enabled"
        )

    bound = float(reference_bound)

    if not math.isfinite(bound) or bound <= 0:
        raise ValueError(
            "reference bound must be finite and positive"
        )

    return bound * math.pow(2.0, -value)


def add_masked_gaussian_noise(
    flat_values: torch.Tensor,
    coordinate_mask: torch.Tensor,
    standard_deviation: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add reproducible Gaussian noise at masked coordinates."""

    sigma = float(
        standard_deviation
    )

    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(
            "standard_deviation must be finite "
            "and non-negative"
        )

    values = (
        flat_values.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )

    mask = (
        coordinate_mask.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.bool,
        )
        .contiguous()
    )

    if values.numel() != mask.numel():
        raise ValueError(
            "values and mask must have equal dimensions"
        )

    active_count = int(
        mask.sum().item()
    )

    if active_count == 0:
        raise ValueError(
            "noise mask must select at least one coordinate"
        )

    noise = torch.zeros_like(
        values
    )

    if sigma > 0:
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            int(seed)
        )

        selected_noise = torch.randn(
            active_count,
            dtype=torch.float64,
            device="cpu",
            generator=generator,
        )

        selected_noise *= sigma

        noise[mask] = selected_noise

    return (
        values + noise,
        noise,
    )


def buffer_delta_vector(
    previous_buffers: dict[str, torch.Tensor],
    current_buffers: dict[str, torch.Tensor],
    name_suffix: str,
) -> torch.Tensor:
    """Flatten current - previous for matching buffers."""

    previous_names = set(
        previous_buffers.keys()
    )

    current_names = set(
        current_buffers.keys()
    )

    if previous_names != current_names:
        raise ValueError(
            "buffer dictionaries have different names"
        )

    matching_names = sorted(
        name
        for name in previous_names
        if name.endswith(name_suffix)
    )

    if len(matching_names) == 0:
        raise ValueError(
            f"no buffers end with {name_suffix}"
        )

    parts = []

    for name in matching_names:
        previous = (
            previous_buffers[name]
            .detach()
            .reshape(-1)
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        current = (
            current_buffers[name]
            .detach()
            .reshape(-1)
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        if previous.shape != current.shape:
            raise ValueError(
                f"buffer shape mismatch: {name}"
            )

        delta = current - previous

        if not torch.isfinite(delta).all():
            raise ValueError(
                f"buffer delta contains NaN or Inf: {name}"
            )

        parts.append(
            delta
        )

    return torch.cat(
        parts,
        dim=0,
    ).contiguous()


def buffer_values_vector(
    buffers: dict[str, torch.Tensor],
    name_suffix: str,
) -> torch.Tensor:
    """Flatten matching floating-point buffers in sorted-name order."""

    matching_names = sorted(
        name
        for name in buffers
        if name.endswith(name_suffix)
    )

    if len(matching_names) == 0:
        raise ValueError(
            f"no buffers end with {name_suffix}"
        )

    parts = []

    for name in matching_names:
        values = (
            buffers[name]
            .detach()
            .reshape(-1)
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        if not torch.isfinite(values).all():
            raise ValueError(
                f"buffer contains NaN or Inf: {name}"
            )

        parts.append(values)

    return torch.cat(
        parts,
        dim=0,
    ).contiguous()


@torch.no_grad()
def apply_buffer_delta_vector(
    model: nn.Module,
    previous_buffers: dict[str, torch.Tensor],
    delta: torch.Tensor,
    name_suffix: str,
) -> None:
    """Set matching model buffers to ``previous + delta``.

    The ordering is identical to :func:`buffer_delta_vector`: buffer
    names are sorted and every matching tensor is flattened in-place.
    """

    target_buffers = {
        name: buffer
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }

    matching_names = sorted(
        name
        for name in previous_buffers
        if name.endswith(name_suffix)
    )

    if len(matching_names) == 0:
        raise ValueError(
            f"no buffers end with {name_suffix}"
        )

    if not set(matching_names).issubset(
        target_buffers.keys()
    ):
        raise ValueError(
            "previous buffers do not match model buffers"
        )

    flat_delta = (
        delta.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )

    expected_numel = sum(
        int(previous_buffers[name].numel())
        for name in matching_names
    )

    if flat_delta.numel() != expected_numel:
        raise ValueError(
            f"delta has {flat_delta.numel()} values; "
            f"expected {expected_numel}"
        )

    if not torch.isfinite(flat_delta).all():
        raise ValueError(
            "buffer delta contains NaN or Inf"
        )

    offset = 0

    for name in matching_names:
        target = target_buffers[name]
        previous = (
            previous_buffers[name]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        if previous.shape != target.shape:
            raise ValueError(
                f"buffer shape mismatch: {name}"
            )

        numel = int(previous.numel())
        update = flat_delta[
            offset:offset + numel
        ].reshape(previous.shape)

        target.copy_(
            (previous + update).to(
                device=target.device,
                dtype=target.dtype,
            )
        )

        offset += numel

