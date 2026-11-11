"""Chunk-wise top-k sparsification and Gaussian perturbation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from .chunk_layout import chunk_count, selected_chunk_count_from_ratio


@dataclass(frozen=True)
class SparseSelection:
    """Top-k chunks selected from one flat model update.

    All tensors are stored on CPU to simplify serialization and
    comparison with the OpenFHE reference implementation.
    """

    total_numel: int
    chunk_size: int
    total_chunk_count: int

    # Shape: [selected_chunk_count]
    # dtype: torch.int64
    chunk_ids: torch.Tensor

    # Shape: [selected_chunk_count, chunk_size]
    values: torch.Tensor

    # Squared L2 score for every chunk.
    # Shape: [total_chunk_count]
    # dtype: torch.float64
    scores: torch.Tensor

    @property
    def selected_chunk_count(self) -> int:
        return int(self.chunk_ids.numel())

    @property
    def wire_value_numel(self) -> int:
        """Number of values transmitted including final padding."""

        return (
            self.selected_chunk_count
            * self.chunk_size
        )

    @property
    def selected_valid_numel(self) -> int:
        """Number of selected coordinates excluding final padding."""

        total = 0

        for chunk_id in self.chunk_ids.tolist():
            begin = int(chunk_id) * self.chunk_size
            remaining = self.total_numel - begin

            total += min(
                self.chunk_size,
                max(remaining, 0),
            )

        return total

    def validate(self) -> None:
        if self.total_numel <= 0:
            raise ValueError(
                "total_numel must be positive"
            )

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive"
            )

        expected_chunk_count = chunk_count(
            total_numel=self.total_numel,
            chunk_size=self.chunk_size,
        )

        if self.total_chunk_count != expected_chunk_count:
            raise ValueError(
                "total_chunk_count does not match "
                "total_numel and chunk_size"
            )

        if self.chunk_ids.device.type != "cpu":
            raise ValueError(
                "chunk_ids must be stored on CPU"
            )

        if self.values.device.type != "cpu":
            raise ValueError(
                "selected values must be stored on CPU"
            )

        if self.scores.device.type != "cpu":
            raise ValueError(
                "chunk scores must be stored on CPU"
            )

        if self.chunk_ids.dtype != torch.int64:
            raise TypeError(
                "chunk_ids must use torch.int64"
            )

        if self.chunk_ids.ndim != 1:
            raise ValueError(
                "chunk_ids must be one-dimensional"
            )

        if self.chunk_ids.numel() == 0:
            raise ValueError(
                "at least one chunk must be selected"
            )

        expected_values_shape = (
            self.chunk_ids.numel(),
            self.chunk_size,
        )

        if tuple(self.values.shape) != expected_values_shape:
            raise ValueError(
                "selected values have an invalid shape: "
                f"expected {expected_values_shape}, "
                f"got {tuple(self.values.shape)}"
            )

        if not self.values.is_floating_point():
            raise TypeError(
                "selected values must be floating point"
            )

        if tuple(self.scores.shape) != (
            self.total_chunk_count,
        ):
            raise ValueError(
                "chunk scores have an invalid shape"
            )

        if self.scores.dtype != torch.float64:
            raise TypeError(
                "chunk scores must use torch.float64"
            )

        unique_chunk_ids = torch.unique(
            self.chunk_ids
        )

        if unique_chunk_ids.numel() != self.chunk_ids.numel():
            raise ValueError(
                "selected chunk IDs must be unique"
            )

        if torch.any(self.chunk_ids < 0):
            raise ValueError(
                "chunk IDs must be non-negative"
            )

        if torch.any(
            self.chunk_ids >= self.total_chunk_count
        ):
            raise ValueError(
                "selected chunk ID is outside the model"
            )

        if not torch.isfinite(self.values).all():
            raise ValueError(
                "selected values contain NaN or infinity"
            )

        if not torch.isfinite(self.scores).all():
            raise ValueError(
                "chunk scores contain NaN or infinity"
            )

    def to_dense(
        self,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Restore selected chunks into a zero-filled dense vector."""

        self.validate()

        output_dtype = (
            self.values.dtype
            if dtype is None
            else dtype
        )

        padded = torch.zeros(
            self.total_chunk_count * self.chunk_size,
            dtype=output_dtype,
            device="cpu",
        )

        chunk_matrix = padded.reshape(
            self.total_chunk_count,
            self.chunk_size,
        )

        chunk_matrix[
            self.chunk_ids
        ] = self.values.to(
            dtype=output_dtype
        )

        return padded[
            :self.total_numel
        ].contiguous()

    def metadata(self) -> dict[str, Any]:
        self.validate()

        return {
            "total_numel": self.total_numel,
            "chunk_size": self.chunk_size,
            "total_chunk_count": self.total_chunk_count,
            "selected_chunk_count": self.selected_chunk_count,
            "selected_valid_numel": self.selected_valid_numel,
            "wire_value_numel": self.wire_value_numel,
            "chunk_ids": self.chunk_ids.tolist(),
        }


def _prepare_flat_update(
    update: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(update, torch.Tensor):
        raise TypeError(
            "update must be a torch.Tensor"
        )

    if not update.is_floating_point():
        raise TypeError(
            "update must be floating point"
        )

    flat = (
        update.detach()
        .reshape(-1)
        .to(device="cpu")
        .contiguous()
    )

    if flat.numel() == 0:
        raise ValueError(
            "update must not be empty"
        )

    if not torch.isfinite(flat).all():
        raise ValueError(
            "update contains NaN or infinity"
        )

    return flat


def _padded_chunk_matrix_copy(
    flat: torch.Tensor,
    *,
    total_chunk_count: int,
    chunk_size: int,
) -> torch.Tensor:
    """Copy a CPU vector into a mutable padded chunk matrix."""

    padded_numel = total_chunk_count * chunk_size
    if flat.numel() == padded_numel:
        return flat.clone().reshape(total_chunk_count, chunk_size)

    padded = torch.zeros(
        padded_numel,
        dtype=flat.dtype,
        device="cpu",
    )
    padded[: flat.numel()] = flat
    return padded.reshape(total_chunk_count, chunk_size)


def compute_chunk_scores(
    update: torch.Tensor,
    chunk_size: int,
    timing: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded chunk matrix and squared L2 scores.

    The score is calculated in float64:

        score_l = sum_j update_l[j]^2
    """

    if timing is not None:
        timing.clear()
        total_started = time.perf_counter()
        prepare_started = total_started

    flat = _prepare_flat_update(update)

    if timing is not None:
        timing["input_prepare_seconds"] = (
            time.perf_counter() - prepare_started
        )
        score_started = time.perf_counter()

    total_chunks = chunk_count(
        total_numel=flat.numel(),
        chunk_size=chunk_size,
    )

    padded_numel = (
        total_chunks * chunk_size
    )

    padded = torch.zeros(
        padded_numel,
        dtype=flat.dtype,
        device="cpu",
    )

    padded[:flat.numel()] = flat

    chunk_matrix = padded.reshape(
        total_chunks,
        chunk_size,
    )

    score_input = chunk_matrix.to(
        dtype=torch.float64
    )

    scores = torch.sum(
        score_input.square(),
        dim=1,
    )

    if timing is not None:
        timing["score_seconds"] = time.perf_counter() - score_started
        timing["total_seconds"] = time.perf_counter() - total_started

    return (
        chunk_matrix.contiguous(),
        scores.contiguous(),
    )


def select_topk_chunks(
    update: torch.Tensor,
    chunk_size: int,
    selected_chunk_count: int,
    timing: dict[str, float] | None = None,
) -> SparseSelection:
    """Select chunks by squared L2 norm.

    Deterministic ordering matches the OpenFHE implementation:

    1. score descending
    2. smaller chunk ID first when scores are equal
    """

    if timing is not None:
        timing.clear()
        total_started = time.perf_counter()
        prepare_started = total_started

    flat = _prepare_flat_update(update)

    if timing is not None:
        timing["selection_input_prepare_seconds"] = (
            time.perf_counter() - prepare_started
        )

    if selected_chunk_count <= 0:
        raise ValueError(
            "selected_chunk_count must be positive"
        )

    score_timing: dict[str, float] | None = (
        {} if timing is not None else None
    )
    chunk_matrix, scores = compute_chunk_scores(
        update=flat,
        chunk_size=chunk_size,
        timing=score_timing,
    )

    if timing is not None:
        if score_timing is None:
            raise RuntimeError("chunk score timing was not collected")
        timing["score_input_prepare_seconds"] = score_timing[
            "input_prepare_seconds"
        ]
        timing["chunk_score_seconds"] = score_timing["score_seconds"]
        topk_started = time.perf_counter()

    total_chunks = int(
        chunk_matrix.shape[0]
    )

    if selected_chunk_count > total_chunks:
        raise ValueError(
            "selected_chunk_count cannot exceed "
            "the total number of chunks"
        )

    # scores are already ordered by ascending chunk ID.
    # A stable descending sort therefore preserves smaller IDs
    # when two scores are identical.
    sorted_chunk_ids = torch.argsort(
        scores,
        descending=True,
        stable=True,
    )

    selected_ids = (
        sorted_chunk_ids[
            :selected_chunk_count
        ]
        .to(dtype=torch.int64)
        .contiguous()
    )

    selected_values = (
        chunk_matrix[
            selected_ids
        ]
        .clone()
        .contiguous()
    )

    if timing is not None:
        timing["topk_seconds"] = time.perf_counter() - topk_started

    selection = SparseSelection(
        total_numel=int(flat.numel()),
        chunk_size=chunk_size,
        total_chunk_count=total_chunks,
        chunk_ids=selected_ids,
        values=selected_values,
        scores=scores.to(
            dtype=torch.float64
        ),
    )

    if timing is not None:
        validation_started = time.perf_counter()

    selection.validate()

    if timing is not None:
        timing["selection_validation_seconds"] = (
            time.perf_counter() - validation_started
        )
        timing["total_seconds"] = time.perf_counter() - total_started

    return selection


def compression_residual(
    original_update: torch.Tensor,
    selection: SparseSelection,
) -> torch.Tensor:
    """Return original_update - sparse_selected_update."""

    original = _prepare_flat_update(
        original_update
    )

    selection.validate()

    if original.numel() != selection.total_numel:
        raise ValueError(
            "original update and sparse selection "
            "have different dimensions"
        )

    residual_chunks = _padded_chunk_matrix_copy(
        original,
        total_chunk_count=selection.total_chunk_count,
        chunk_size=selection.chunk_size,
    )
    residual_chunks.index_add_(
        0,
        selection.chunk_ids,
        selection.values.to(dtype=original.dtype),
        alpha=-1.0,
    )

    return residual_chunks.reshape(-1)[: selection.total_numel].contiguous()


def restore_rejected_chunks_to_residual(
    residual: torch.Tensor,
    selection: SparseSelection,
    accepted_chunk_ids: list[int] | tuple[int, ...] | set[int],
) -> torch.Tensor:
    """Return residual with cohort-rejected selected chunks restored.

    Top-k compression removes every selected chunk from the local
    residual. If the server subsequently rejects a chunk because its
    cohort is too small, that chunk did not contribute to the global
    update and therefore remains compression error.
    """

    restored = _prepare_flat_update(
        residual
    )

    selection.validate()

    if restored.numel() != selection.total_numel:
        raise ValueError(
            "residual and sparse selection have different dimensions"
        )

    accepted = {
        int(chunk_id)
        for chunk_id in accepted_chunk_ids
    }

    if any(
        chunk_id < 0
        or chunk_id >= selection.total_chunk_count
        for chunk_id in accepted
    ):
        raise ValueError(
            "accepted chunk ID is outside the model"
        )

    accepted_mask = torch.zeros(
        selection.total_chunk_count,
        dtype=torch.bool,
        device="cpu",
    )
    if accepted:
        accepted_mask[
            torch.tensor(
                tuple(accepted),
                dtype=torch.int64,
                device="cpu",
            )
        ] = True

    rejected_rows = ~accepted_mask[selection.chunk_ids]
    restored_chunks = _padded_chunk_matrix_copy(
        restored,
        total_chunk_count=selection.total_chunk_count,
        chunk_size=selection.chunk_size,
    )
    restored_chunks.index_add_(
        0,
        selection.chunk_ids[rejected_rows],
        selection.values[rejected_rows].to(dtype=restored.dtype),
    )

    return restored_chunks.reshape(-1)[: selection.total_numel].contiguous()


def compress_with_error_feedback(
    update: torch.Tensor,
    previous_residual: torch.Tensor | None,
    chunk_size: int,
    selected_chunk_count: int,
    residual_beta: float = 1.0,
    timing: dict[str, float] | None = None,
) -> tuple[
    SparseSelection,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply error feedback and top-k chunk selection.

    compression_input =
        update + residual_beta * previous_residual

    new_residual =
        compression_input - sparse(compression_input)

    Returns:
        selection
        new_residual
        compression_input
    """

    if timing is not None:
        timing.clear()
        total_started = time.perf_counter()

    if not math.isfinite(residual_beta):
        raise ValueError(
            "residual_beta must be finite"
        )

    if residual_beta < 0:
        raise ValueError(
            "residual_beta must be non-negative"
        )

    if timing is not None:
        prepare_started = time.perf_counter()

    flat_update = _prepare_flat_update(
        update
    )

    if previous_residual is None:
        flat_residual = torch.zeros_like(
            flat_update
        )
    else:
        flat_residual = _prepare_flat_update(
            previous_residual
        )

        if flat_residual.numel() != flat_update.numel():
            raise ValueError(
                "update and previous residual "
                "must have equal dimensions"
            )

        flat_residual = flat_residual.to(
            dtype=flat_update.dtype
        )

    compression_input = (
        flat_update
        + float(residual_beta) * flat_residual
    ).contiguous()

    if timing is not None:
        timing["compression_prepare_seconds"] = (
            time.perf_counter() - prepare_started
        )

    selection_timing: dict[str, float] | None = (
        {} if timing is not None else None
    )
    selection = select_topk_chunks(
        update=compression_input,
        chunk_size=chunk_size,
        selected_chunk_count=selected_chunk_count,
        timing=selection_timing,
    )

    if timing is not None:
        if selection_timing is None:
            raise RuntimeError("selection timing was not collected")
        for name, value in selection_timing.items():
            timing[name] = value
        residual_started = time.perf_counter()

    new_residual = compression_residual(
        original_update=compression_input,
        selection=selection,
    )

    if timing is not None:
        timing["residual_seconds"] = (
            time.perf_counter() - residual_started
        )
        timing["total_seconds"] = time.perf_counter() - total_started

    return (
        selection,
        new_residual,
        compression_input,
    )


def add_gaussian_noise(
    selection: SparseSelection,
    reference_update: torch.Tensor,
    std_ratio: float,
    seed: int,
) -> tuple[SparseSelection, float]:
    """Add Gaussian noise to the selected chunk values.

    sigma = std_ratio * std(reference_update)

    reference_update should be the complete local model delta before
    sparsification.
    """

    selection.validate()

    if not math.isfinite(std_ratio):
        raise ValueError(
            "std_ratio must be finite"
        )

    if std_ratio < 0:
        raise ValueError(
            "std_ratio must be non-negative"
        )

    reference = _prepare_flat_update(
        reference_update
    ).to(
        dtype=torch.float64
    )

    reference_std = float(
        reference.std(
            unbiased=False
        ).item()
    )

    sigma = (
        reference_std
        * float(std_ratio)
    )

    if sigma == 0.0:
        return (
            SparseSelection(
                total_numel=selection.total_numel,
                chunk_size=selection.chunk_size,
                total_chunk_count=selection.total_chunk_count,
                chunk_ids=selection.chunk_ids.clone(),
                values=selection.values.clone(),
                scores=selection.scores.clone(),
            ),
            0.0,
        )

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(seed)

    noise = torch.randn(
        selection.values.shape,
        dtype=torch.float64,
        device="cpu",
        generator=generator,
    )

    noise = noise * sigma

    noisy_values = (
        selection.values.to(
            dtype=torch.float64
        )
        + noise
    ).to(
        dtype=selection.values.dtype
    )

    noisy_selection = SparseSelection(
        total_numel=selection.total_numel,
        chunk_size=selection.chunk_size,
        total_chunk_count=selection.total_chunk_count,
        chunk_ids=selection.chunk_ids.clone(),
        values=noisy_values.contiguous(),
        scores=selection.scores.clone(),
    )

    noisy_selection.validate()

    return (
        noisy_selection,
        sigma,
    )

def effective_sparsity_metrics(
    *,
    accepted_contribution_count: int,
    client_count: int,
    selected_chunk_count: int,
    total_chunk_count: int,
) -> dict[str, float | int]:
    """Return client-contribution-based effective sparsity metrics.

    ``rho_eff`` counts only client/chunk contributions whose chunk passed the
    minimum-cohort filter.  It is deliberately different from the ratio of
    globally accepted chunk IDs, which is a union and may exceed the requested
    per-client selection ratio.
    """

    for name, value in (
        ("accepted_contribution_count", accepted_contribution_count),
        ("client_count", client_count),
        ("selected_chunk_count", selected_chunk_count),
        ("total_chunk_count", total_chunk_count),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if total_chunk_count <= 0:
        raise ValueError("total_chunk_count must be positive")
    if not 1 <= selected_chunk_count <= total_chunk_count:
        raise ValueError(
            "selected_chunk_count must be between one and total_chunk_count"
        )

    candidate_contribution_count = client_count * selected_chunk_count
    if not 0 <= accepted_contribution_count <= candidate_contribution_count:
        raise ValueError(
            "accepted_contribution_count must be between zero and the "
            "candidate contribution count"
        )

    return {
        "candidate_contribution_count": candidate_contribution_count,
        "accepted_contribution_count": accepted_contribution_count,
        "rho_selected": selected_chunk_count / total_chunk_count,
        "rho_eff": (
            accepted_contribution_count
            / (client_count * total_chunk_count)
        ),
        "cohort_retention_ratio": (
            accepted_contribution_count
            / candidate_contribution_count
        ),
    }

