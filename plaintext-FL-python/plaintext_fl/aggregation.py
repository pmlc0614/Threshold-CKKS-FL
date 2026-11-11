"""Plaintext sparse aggregation matching the OpenFHE semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .protocol_contract import (
    RegisteredClientWeight,
    quantize_registered_client_weights,
)

from .sparsify import SparseSelection


@dataclass(frozen=True)
class ClientSparseUpdate:
    """Sparse update produced by one client in one round."""

    client_id: int
    round_id: int
    num_examples: int
    selection: SparseSelection

    def validate(self) -> None:
        if self.client_id < 0:
            raise ValueError(
                "client_id must be non-negative"
            )

        if self.round_id < 0:
            raise ValueError(
                "round_id must be non-negative"
            )

        if self.num_examples <= 0:
            raise ValueError(
                "num_examples must be positive"
            )

        self.selection.validate()


@dataclass(frozen=True)
class AggregationResult:
    """Result of one plaintext sparse aggregation."""

    # float64 dense aggregate.
    dense_update: torch.Tensor

    # Sorted accepted chunk IDs.
    accepted_chunk_ids: tuple[int, ...]

    # Number of clients selecting every chunk.
    cohort_sizes: torch.Tensor

    # Integer alpha_i values in input client order.
    alphas: torch.Tensor

    weight_denominator: int
    minimum_cohort_size: int
    round_id: int
    client_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.dense_update.device.type != "cpu":
            raise ValueError(
                "dense aggregate must be stored on CPU"
            )

        if self.dense_update.dtype != torch.float64:
            raise TypeError(
                "dense aggregate must use torch.float64"
            )

        if self.dense_update.ndim != 1:
            raise ValueError(
                "dense aggregate must be one-dimensional"
            )

        if self.cohort_sizes.dtype != torch.int64:
            raise TypeError(
                "cohort_sizes must use torch.int64"
            )

        if self.cohort_sizes.ndim != 1:
            raise ValueError(
                "cohort_sizes must be one-dimensional"
            )

        if self.alphas.dtype != torch.int64:
            raise TypeError(
                "alphas must use torch.int64"
            )

        if self.alphas.ndim != 1:
            raise ValueError(
                "alphas must be one-dimensional"
            )

        if self.alphas.numel() != len(self.client_ids):
            raise ValueError(
                "alpha count does not match client count"
            )

        if int(self.alphas.sum().item()) != self.weight_denominator:
            raise ValueError(
                "alphas must sum to weight_denominator"
            )

        if self.minimum_cohort_size <= 0:
            raise ValueError(
                "minimum_cohort_size must be positive"
            )


@dataclass(frozen=True)
class SparseCommunicationCounts:
    """Exact cohort counts used for accepted-only transmission.

    A contribution is one client selecting one chunk.  Candidate
    contributions announce only their chunk labels.  A chunk-value
    payload is sent later only for accepted contributions.
    """

    union_chunk_count: int
    accepted_chunk_count: int
    rejected_chunk_count: int
    candidate_contribution_count: int
    accepted_contribution_count: int
    rejected_contribution_count: int


def sparse_communication_counts(
    cohort_sizes: torch.Tensor,
    minimum_cohort_size: int,
) -> SparseCommunicationCounts:
    """Return exact candidate and accepted contribution counts."""

    if cohort_sizes.dtype != torch.int64:
        raise TypeError(
            "cohort_sizes must use torch.int64"
        )

    if cohort_sizes.ndim != 1:
        raise ValueError(
            "cohort_sizes must be one-dimensional"
        )

    if torch.any(cohort_sizes < 0):
        raise ValueError(
            "cohort_sizes must be non-negative"
        )

    if minimum_cohort_size <= 0:
        raise ValueError(
            "minimum_cohort_size must be positive"
        )

    union_mask = cohort_sizes > 0
    accepted_mask = (
        cohort_sizes >= minimum_cohort_size
    )

    union_chunk_count = int(
        union_mask.sum().item()
    )
    accepted_chunk_count = int(
        accepted_mask.sum().item()
    )
    candidate_contribution_count = int(
        cohort_sizes.sum().item()
    )
    accepted_contribution_count = int(
        cohort_sizes[accepted_mask].sum().item()
    )

    return SparseCommunicationCounts(
        union_chunk_count=union_chunk_count,
        accepted_chunk_count=accepted_chunk_count,
        rejected_chunk_count=(
            union_chunk_count
            - accepted_chunk_count
        ),
        candidate_contribution_count=(
            candidate_contribution_count
        ),
        accepted_contribution_count=(
            accepted_contribution_count
        ),
        rejected_contribution_count=(
            candidate_contribution_count
            - accepted_contribution_count
        ),
    )


def quantize_client_weights(
    example_counts: Sequence[int],
    denominator: int = 128,
) -> torch.Tensor:
    """Convert sample counts into integer alpha_i weights.

    Uses the largest-remainder method so that:

        sum_i alpha_i = denominator

    Ties are resolved using the smaller client position.
    """

    if denominator <= 0:
        raise ValueError(
            "denominator must be positive"
        )

    if len(example_counts) == 0:
        raise ValueError(
            "example_counts must not be empty"
        )

    alphas = quantize_registered_client_weights(
        tuple(
            RegisteredClientWeight(
                client_id=position,
                example_count=int(count),
            )
            for position, count in enumerate(example_counts)
        ),
        denominator=denominator,
    )

    return torch.tensor(
        alphas,
        dtype=torch.int64,
        device="cpu",
    ).contiguous()


def uniform_client_weights(
    client_count: int,
    denominator: int = 128,
) -> torch.Tensor:
    """Return approximately uniform integer client weights."""

    if client_count <= 0:
        raise ValueError(
            "client_count must be positive"
        )

    return quantize_client_weights(
        example_counts=[
            1
            for _ in range(client_count)
        ],
        denominator=denominator,
    )


def _validate_client_updates(
    updates: Sequence[ClientSparseUpdate],
) -> None:
    if len(updates) == 0:
        raise ValueError(
            "at least one client update is required"
        )

    client_ids: set[int] = set()

    first = updates[0]
    expected_round_id = first.round_id
    expected_numel = (
        first.selection.total_numel
    )
    expected_chunk_size = (
        first.selection.chunk_size
    )
    expected_chunk_count = (
        first.selection.total_chunk_count
    )

    for update in updates:
        update.validate()

        if update.client_id in client_ids:
            raise ValueError(
                f"duplicate client ID: {update.client_id}"
            )

        client_ids.add(update.client_id)

        if update.round_id != expected_round_id:
            raise ValueError(
                "all client updates must have "
                "the same round_id"
            )

        if (
            update.selection.total_numel
            != expected_numel
        ):
            raise ValueError(
                "all client updates must have "
                "the same model dimension"
            )

        if (
            update.selection.chunk_size
            != expected_chunk_size
        ):
            raise ValueError(
                "all client updates must have "
                "the same chunk_size"
            )

        if (
            update.selection.total_chunk_count
            != expected_chunk_count
        ):
            raise ValueError(
                "all client updates must have "
                "the same total chunk count"
            )


def aggregate_sparse_updates(
    updates: Sequence[ClientSparseUpdate],
    minimum_cohort_size: int = 3,
    weight_denominator: int = 128,
    alphas: torch.Tensor | None = None,
) -> AggregationResult:
    """Aggregate sparse client updates.

    For each accepted chunk:

        aggregate_chunk =
            sum_i alpha_i * client_chunk_i
            / weight_denominator

    A chunk is accepted only when at least
    minimum_cohort_size clients selected it.

    Client weights are not renormalized per chunk.
    This matches the OpenFHE reference semantics.
    """

    _validate_client_updates(updates)

    if minimum_cohort_size <= 0:
        raise ValueError(
            "minimum_cohort_size must be positive"
        )

    if minimum_cohort_size > len(updates):
        raise ValueError(
            "minimum_cohort_size cannot exceed "
            "the number of clients"
        )

    if weight_denominator <= 0:
        raise ValueError(
            "weight_denominator must be positive"
        )

    if alphas is None:
        alpha_values = quantize_client_weights(
            example_counts=[
                update.num_examples
                for update in updates
            ],
            denominator=weight_denominator,
        )
    else:
        alpha_values = (
            alphas.detach()
            .reshape(-1)
            .to(
                device="cpu",
                dtype=torch.int64,
            )
            .contiguous()
        )

        if alpha_values.numel() != len(updates):
            raise ValueError(
                "alpha count must match client count"
            )

        if torch.any(alpha_values <= 0):
            raise ValueError(
                "alphas must be positive"
            )

        if (
            int(alpha_values.sum().item())
            != weight_denominator
        ):
            raise ValueError(
                "alphas must sum to weight_denominator"
            )

    first_selection = updates[0].selection

    total_numel = (
        first_selection.total_numel
    )
    chunk_size = (
        first_selection.chunk_size
    )
    total_chunk_count = (
        first_selection.total_chunk_count
    )

    cohort_sizes = torch.zeros(
        total_chunk_count,
        dtype=torch.int64,
        device="cpu",
    )

    padded_aggregate = torch.zeros(
        total_chunk_count * chunk_size,
        dtype=torch.float64,
        device="cpu",
    )

    aggregate_chunks = padded_aggregate.reshape(
        total_chunk_count,
        chunk_size,
    )

    # Each client has unique chunk IDs, so one index_add_ per client preserves
    # the original client-order accumulation while removing the Python loop
    # over individual chunks.
    for client_position, update in enumerate(updates):
        selection = update.selection
        chunk_ids = selection.chunk_ids
        cohort_sizes.index_add_(
            0,
            chunk_ids,
            torch.ones_like(chunk_ids),
        )
        weighted_values = selection.values.to(
            dtype=torch.float64,
            device="cpu",
        ) * int(alpha_values[client_position].item())
        aggregate_chunks.index_add_(
            0,
            chunk_ids,
            weighted_values,
        )

    accepted_mask = cohort_sizes >= minimum_cohort_size
    aggregate_chunks[~accepted_mask] = 0.0
    aggregate_chunks.div_(float(weight_denominator))
    accepted_chunk_ids = tuple(
        int(chunk_id)
        for chunk_id in torch.nonzero(
            accepted_mask,
            as_tuple=False,
        ).reshape(-1).tolist()
    )

    dense_update = padded_aggregate[
        :total_numel
    ].contiguous()

    result = AggregationResult(
        dense_update=dense_update,
        accepted_chunk_ids=accepted_chunk_ids,
        cohort_sizes=cohort_sizes,
        alphas=alpha_values,
        weight_denominator=weight_denominator,
        minimum_cohort_size=minimum_cohort_size,
        round_id=updates[0].round_id,
        client_ids=tuple(
            update.client_id
            for update in updates
        ),
    )

    result.validate()

    return result


def aggregate_dense_updates(
    dense_updates: Sequence[torch.Tensor],
    alphas: torch.Tensor,
    weight_denominator: int,
) -> torch.Tensor:
    """Dense weighted aggregation used as a reference."""

    if len(dense_updates) == 0:
        raise ValueError(
            "at least one dense update is required"
        )

    if weight_denominator <= 0:
        raise ValueError(
            "weight_denominator must be positive"
        )

    alpha_values = (
        alphas.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.int64,
        )
    )

    if alpha_values.numel() != len(dense_updates):
        raise ValueError(
            "alpha count must match update count"
        )

    if torch.any(alpha_values <= 0):
        raise ValueError(
            "alphas must be positive"
        )

    if (
        int(alpha_values.sum().item())
        != weight_denominator
    ):
        raise ValueError(
            "alphas must sum to weight_denominator"
        )

    flattened_updates = [
        update.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        for update in dense_updates
    ]

    expected_numel = (
        flattened_updates[0].numel()
    )

    for update in flattened_updates:
        if update.numel() != expected_numel:
            raise ValueError(
                "all dense updates must have equal dimensions"
            )

        if not torch.isfinite(update).all():
            raise ValueError(
                "dense update contains NaN or infinity"
            )

    result = torch.zeros(
        expected_numel,
        dtype=torch.float64,
        device="cpu",
    )

    for client_position, update in enumerate(
        flattened_updates
    ):
        result += (
            int(
                alpha_values[
                    client_position
                ].item()
            )
            * update
        )

    return (
        result / weight_denominator
    ).contiguous()

def aggregate_dense_fedavg(
    dense_updates: Sequence[torch.Tensor],
    sample_counts: Sequence[int],
) -> torch.Tensor:
    """FedAvg dense model updates with sample_counts[i]/sum(sample_counts). Accumulate in CPU float64 and return float32 to reduce numerical error."""
    if len(dense_updates) == 0:
        raise ValueError("dense_updates must not be empty")

    if len(dense_updates) != len(sample_counts):
        raise ValueError(
            "dense_updates and sample_counts must have the same length"
        )

    if any(count <= 0 for count in sample_counts):
        raise ValueError("all sample_counts must be positive")

    reference_shape = dense_updates[0].shape
    reference_numel = dense_updates[0].numel()

    for index, update in enumerate(dense_updates):
        if update.numel() != reference_numel:
            raise ValueError(
                f"dense update {index} has {update.numel()} elements, "
                f"expected {reference_numel}"
            )

        if update.shape != reference_shape:
            raise ValueError(
                f"dense update {index} has shape {tuple(update.shape)}, "
                f"expected {tuple(reference_shape)}"
            )

        if not torch.isfinite(update).all():
            raise ValueError(f"dense update {index} contains NaN or Inf")

    total_samples = sum(sample_counts)

    aggregated = torch.zeros(
        reference_shape,
        dtype=torch.float64,
        device="cpu",
    )

    for update, count in zip(
        dense_updates,
        sample_counts,
    ):
        aggregated.add_(
            update.detach().to(
                device="cpu",
                dtype=torch.float64,
            ),
            alpha=float(count),
        )

    return (
        aggregated / float(total_samples)
    ).contiguous()

    return aggregated.contiguous()

def aggregate_sparse_fedavg(
    updates: Sequence[ClientSparseUpdate],
) -> torch.Tensor:
    """Exact sample-weighted FedAvg for sparse client updates.

    Unselected coordinates are treated as zero. Every selected chunk
    participates in aggregation; no cohort-size rejection and no
    alpha/denominator quantization are applied.
    """

    _validate_client_updates(updates)

    first_selection = updates[0].selection
    padded_aggregate = torch.zeros(
        first_selection.total_chunk_count * first_selection.chunk_size,
        dtype=torch.float64,
        device="cpu",
    )
    aggregate_chunks = padded_aggregate.reshape(
        first_selection.total_chunk_count,
        first_selection.chunk_size,
    )

    total_samples = 0
    for update in updates:
        selection = update.selection
        total_samples += update.num_examples
        weighted_values = selection.values.to(
            dtype=torch.float64,
            device="cpu",
        ) * float(update.num_examples)
        aggregate_chunks.index_add_(
            0,
            selection.chunk_ids,
            weighted_values,
        )

    return (
        padded_aggregate[: first_selection.total_numel]
        .div(float(total_samples))
        .contiguous()
    )
