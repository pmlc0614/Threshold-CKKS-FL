"""Encrypted CIFAR-10/100 SPARSE federated training.

Python owns data loading, local training, sparsification, minimum-cohort
orchestration, and model state.  The injected crypto engine owns the v17
encrypted path.  The command-line entry point uses the pybind11-backed
``NativeCryptoEngine``; the round function remains engine-neutral so its
training and state semantics can be tested against the plaintext reference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .communication import (
    cumulative_communication,
    measure_aborted_sparse_round_communication,
    measure_full_online_round_communication,
    measure_online_round_communication,
    measure_setup_communication,
)
from .crypto_engine import (
    AdmissionPlan,
    CryptoEngineSetup,
    FederatedCryptoEngine,
    PostRequestDropoutError,
)
from .crypto_parameter_profile import (
    CryptoParameterProfile,
    load_crypto_parameter_profile,
)
from .data import (
    build_client_datasets,
    build_client_loaders,
    build_test_loader,
    client_sample_counts,
    load_or_create_partitions,
    load_cifar10,
    load_cifar100,
)
from .federated import partition_document
from .model import resnet20, resnet32
from .model_adapter import (
    ModelProtocolLayouts,
    apply_layout_delta,
    apply_decrypted_result,
    build_model_protocol_layouts,
    candidate_from_models,
    compute_layout_delta,
    flatten_layout,
)
from .native_engine import NativeCryptoEngine
from .plaintext_engine import PlaintextCryptoEngine
from .protocol_contract import RegisteredClientWeight
from .round_orchestrator import (
    ClientFullRoundCandidate,
    ClientRoundCandidate,
    FederatedRoundOrchestrator,
    OrchestratedRoundResult,
)
from .sparsify import (
    SparseSelection,
    compress_with_error_feedback,
    effective_sparsity_metrics,
    restore_rejected_chunks_to_residual,
)
from .stats import write_json
from .timing import (
    estimated_barrier_parallel_seconds,
    maximum_phase_seconds,
    synchronized_perf_counter,
)
from .training import (
    EvaluationMetrics,
    build_optimizer,
    evaluate,
    learning_rate_at_step,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


SUPPORTED_ENCRYPTED_EXPERIMENTS = {
    ("cifar10", "resnet20"): "resnet20-cifar10",
    ("cifar100", "resnet32"): "resnet32-cifar100",
}

_CRYPTO_STAGE_NAMES = (
    "encryption",
    "client_to_collective_key_switch",
    "rlwe_aggregation",
    "r2m",
    "mlwe_aggregation",
    "m2r",
    "partial_decryption",
    "decoding",
)
from .dropout import build_dropout_schedule


def _validate_encrypted_experiment(
    *,
    dataset_name: str,
    model_name: str,
    profile: CryptoParameterProfile,
) -> str:
    profile_model_name = SUPPORTED_ENCRYPTED_EXPERIMENTS.get(
        (dataset_name, model_name)
    )
    if profile_model_name is None:
        supported = ", ".join(
            f"{dataset}/{model}"
            for dataset, model in sorted(SUPPORTED_ENCRYPTED_EXPERIMENTS)
        )
        raise ValueError(
            "unsupported encrypted dataset/model pair: "
            f"{dataset_name}/{model_name}; supported pairs: {supported}"
        )
    if profile.model_name != profile_model_name:
        raise ValueError(
            "crypto parameter profile model_name conflicts with the selected "
            f"experiment: {profile.model_name} != {profile_model_name}"
        )
    return profile_model_name


def _load_encrypted_dataset(
    *,
    dataset_name: str,
    data_dir: str,
    download: bool,
):
    if dataset_name == "cifar10":
        return load_cifar10(data_dir=data_dir, download=download)
    if dataset_name == "cifar100":
        return load_cifar100(data_dir=data_dir, download=download)
    raise ValueError(f"unsupported encrypted dataset: {dataset_name}")


def _build_encrypted_model(model_name: str) -> nn.Module:
    if model_name == "resnet20":
        return resnet20()
    if model_name == "resnet32":
        return resnet32()
    raise ValueError(f"unsupported encrypted model: {model_name}")


@dataclass(frozen=True)
class EncryptedSparseRoundConfig:
    """Training and sparsification inputs for exactly one encrypted round."""

    chunk_size: int
    selected_chunk_count: int
    minimum_cohort_size: int
    local_epochs: int = 1
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    minimum_learning_rate: float = 0.0
    schedule: str = "cosine"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    residual_beta: float = 1.0
    aggregate_update_bound: float | None = None
    aggregate_auxiliary_update_bound: float | None = None
    round_id: int = 0
    protocol_mode: str = "SPARSE"
    # Explicitly opt into privileged reference calculations for diagnostics.
    compute_plaintext_reference: bool = False

    def validate(self, client_count: int) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.selected_chunk_count <= 0:
            raise ValueError("selected_chunk_count must be positive")
        if not 1 <= self.minimum_cohort_size <= client_count:
            raise ValueError(
                "minimum_cohort_size must be between one and client count"
            )
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError(
                "minimum_learning_rate must be between zero and learning_rate"
            )
        if self.schedule not in {"constant", "cosine"}:
            raise ValueError("schedule must be constant or cosine")
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.protocol_mode not in {"SPARSE", "FULL"}:
            raise ValueError("protocol_mode must be SPARSE or FULL")
        for name, value in (
            ("aggregate_update_bound", self.aggregate_update_bound),
            (
                "aggregate_auxiliary_update_bound",
                self.aggregate_auxiliary_update_bound,
            ),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ClientEncryptedRoundMetrics:
    client_id: int
    example_count: int
    average_train_loss: float
    selected_chunk_ids: tuple[int, ...]
    rejected_chunk_ids: tuple[int, ...]
    selected_max_abs: float
    running_mean_max_abs: float
    running_variance_max_abs: float
    processing_seconds: float


@dataclass(frozen=True)
class EncryptedSparseRoundOutcome:
    """Model, residual, admission, and timing state after a committed round."""

    result: OrchestratedRoundResult
    layouts: ModelProtocolLayouts
    residuals: tuple[torch.Tensor, ...]
    clients: tuple[ClientEncryptedRoundMetrics, ...]
    plaintext_aggregate: AggregateReference | None
    plaintext_aggregate_maxima: AggregateStreamMaxima | None
    decrypted_aggregate_maxima: AggregateStreamMaxima
    learning_rate: float
    setup_seconds: float
    estimated_parallel_setup_seconds: float
    setup_parallel_timing: dict
    local_training_seconds: float
    encrypted_path_seconds: float
    client_local_processing_seconds: tuple[float, ...]
    estimated_parallel_local_training_seconds: float
    estimated_parallel_encrypted_path_seconds: float
    parallel_timing: dict


@dataclass(frozen=True)
class AbortedEncryptedRoundAttempt:
    logical_round_index: int
    native_round_id: int
    setup_seconds: float
    estimated_parallel_setup_seconds: float
    setup_parallel_timing: dict
    local_training_seconds: float
    estimated_parallel_local_training_seconds: float
    encrypted_path_wall_seconds: float
    native_traffic: dict | None


class EncryptedRoundAborted(PostRequestDropoutError):
    def __init__(self, attempt: AbortedEncryptedRoundAttempt) -> None:
        super().__init__(
            f"native round {attempt.native_round_id} aborted after final request"
        )
        self.attempt = attempt


@dataclass(frozen=True)
class EncryptedSparseTrainingOutcome:
    """Memory-bounded result for rounds sharing setup and model state."""

    completed_round_ids: tuple[int, ...]
    final_round: EncryptedSparseRoundOutcome

    @property
    def completed_rounds(self) -> int:
        return len(self.completed_round_ids)

    @property
    def residuals(self) -> tuple[torch.Tensor, ...]:
        return self.final_round.residuals

    @property
    def layouts(self) -> ModelProtocolLayouts:
        return self.final_round.layouts


@dataclass(frozen=True)
class PairedEncryptedSparseRoundOutcome:
    """Both model trajectories after one controlled comparison round."""

    reference: EncryptedSparseRoundOutcome
    encrypted: EncryptedSparseRoundOutcome
    reference_residuals: tuple[torch.Tensor, ...]
    encrypted_residuals: tuple[torch.Tensor, ...]


def encrypted_round_sparsity_metrics(
    outcome: EncryptedSparseRoundOutcome,
    *,
    total_chunk_count: int,
) -> dict[str, float | int]:
    """Measure post-cohort sparsity from client/chunk contributions."""

    if not outcome.clients:
        raise ValueError("encrypted round must contain at least one client")

    selected_counts = {
        len(client.selected_chunk_ids)
        for client in outcome.clients
    }
    if len(selected_counts) != 1:
        raise ValueError("all clients must select the same number of chunks")
    selected_chunk_count = selected_counts.pop()
    if selected_chunk_count == 0:
        contribution_count = len(outcome.clients) * total_chunk_count
        return {
            "candidate_contribution_count": contribution_count,
            "accepted_contribution_count": contribution_count,
            "rho_selected": 1.0,
            "rho_eff": 1.0,
            "cohort_retention_ratio": 1.0,
            "rejected_contribution_count": 0,
        }

    accepted_contribution_count = 0
    for client in outcome.clients:
        selected = set(client.selected_chunk_ids)
        rejected = set(client.rejected_chunk_ids)
        if len(selected) != len(client.selected_chunk_ids):
            raise ValueError("client selected chunk IDs must be unique")
        if not rejected.issubset(selected):
            raise ValueError("rejected chunk IDs must be a subset of selected IDs")
        accepted_contribution_count += len(selected - rejected)

    metrics = effective_sparsity_metrics(
        accepted_contribution_count=accepted_contribution_count,
        client_count=len(outcome.clients),
        selected_chunk_count=selected_chunk_count,
        total_chunk_count=total_chunk_count,
    )
    return {
        **metrics,
        "rejected_contribution_count": (
            int(metrics["candidate_contribution_count"])
            - accepted_contribution_count
        ),
    }


def _maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


@dataclass(frozen=True)
class AggregateStreamMaxima:
    parameter: float
    running_mean: float
    running_variance: float


@dataclass(frozen=True)
class AggregateReference:
    parameter: np.ndarray
    running_mean: np.ndarray
    running_variance: np.ndarray

    @property
    def maxima(self) -> AggregateStreamMaxima:
        return AggregateStreamMaxima(
            parameter=_maximum_absolute(self.parameter),
            running_mean=_maximum_absolute(self.running_mean),
            running_variance=_maximum_absolute(self.running_variance),
        )


def _require_aggregate_within_bound(
    *,
    name: str,
    observed: float,
    bound: float | None,
) -> None:
    if bound is not None and observed > bound:
        raise ValueError(
            f"{name} exceeds the configured aggregate bound: "
            f"{observed:.9g} > {bound:.9g}"
        )


def _aggregate_admitted_reference(
    *,
    setup: CryptoEngineSetup,
    admission: AdmissionPlan,
    candidates: Sequence[ClientRoundCandidate],
) -> AggregateReference:
    candidate_by_client = {
        candidate.client_id: candidate for candidate in candidates
    }
    alpha_by_client = dict(zip(setup.client_ids, setup.alphas, strict=True))
    parameter = np.zeros(
        setup.parameter_layout.total_numel,
        dtype=np.float64,
    )
    running_mean = np.zeros(
        setup.running_mean_layout.total_numel,
        dtype=np.float64,
    )
    running_variance = np.zeros(
        setup.running_variance_layout.total_numel,
        dtype=np.float64,
    )
    for client_id in setup.client_ids:
        candidate = candidate_by_client[client_id]
        alpha = alpha_by_client[client_id]
        row_by_chunk = {
            chunk_id: row
            for row, chunk_id in enumerate(
                candidate.selected_actual_chunk_ids
            )
        }
        accepted_ids = admission.for_client(
            client_id
        ).accepted_actual_chunk_ids
        for chunk_id in accepted_ids:
            begin = chunk_id * setup.chunk_size
            end = min(begin + setup.chunk_size, parameter.size)
            valid_count = end - begin
            parameter[begin:end] += (
                alpha
                * candidate.selected_values[
                    row_by_chunk[chunk_id], :valid_count
                ].astype(np.float64)
            )
        running_mean += alpha * candidate.running_mean_delta.astype(np.float64)
        running_variance += (
            alpha * candidate.running_variance_delta.astype(np.float64)
        )
    denominator = float(setup.weight_denominator)
    return AggregateReference(
        parameter=parameter / denominator,
        running_mean=running_mean / denominator,
        running_variance=running_variance / denominator,
    )


def apply_aggregate_reference(
    model: nn.Module,
    layouts: ModelProtocolLayouts,
    aggregate: AggregateReference,
) -> None:
    """Apply the canonical float64 reference through the float32 model boundary."""

    for layout, values in (
        (layouts.parameter, aggregate.parameter),
        (layouts.running_mean, aggregate.running_mean),
        (layouts.running_variance, aggregate.running_variance),
    ):
        apply_layout_delta(
            model,
            layout,
            np.ascontiguousarray(values, dtype=np.float32),
        )


def _aggregate_full_reference(
    *,
    setup: CryptoEngineSetup,
    candidates: Sequence[ClientFullRoundCandidate],
    survivor_client_ids: Sequence[int] | None = None,
) -> AggregateReference:
    candidate_by_client = {
        candidate.client_id: candidate for candidate in candidates
    }
    alpha_by_client = dict(zip(setup.client_ids, setup.alphas, strict=True))
    parameter = np.zeros(setup.parameter_layout.total_numel, dtype=np.float64)
    running_mean = np.zeros(
        setup.running_mean_layout.total_numel, dtype=np.float64
    )
    running_variance = np.zeros(
        setup.running_variance_layout.total_numel, dtype=np.float64
    )
    for client_id in (setup.client_ids if survivor_client_ids is None else survivor_client_ids):
        candidate = candidate_by_client[client_id]
        alpha = alpha_by_client[client_id]
        parameter += alpha * candidate.parameter_delta.astype(np.float64)
        running_mean += alpha * candidate.running_mean_delta.astype(np.float64)
        running_variance += (
            alpha * candidate.running_variance_delta.astype(np.float64)
        )
    denominator = float(setup.weight_denominator)
    return AggregateReference(
        parameter=parameter / denominator,
        running_mean=running_mean / denominator,
        running_variance=running_variance / denominator,
    )


def _validate_aggregate_maxima(
    maxima: AggregateStreamMaxima,
    config: EncryptedSparseRoundConfig,
    *,
    prefix: str,
) -> None:
    _require_aggregate_within_bound(
        name=f"{prefix} parameter delta",
        observed=maxima.parameter,
        bound=config.aggregate_update_bound,
    )
    _require_aggregate_within_bound(
        name=f"{prefix} running_mean delta",
        observed=maxima.running_mean,
        bound=config.aggregate_auxiliary_update_bound,
    )
    _require_aggregate_within_bound(
        name=f"{prefix} running_variance delta",
        observed=maxima.running_variance,
        bound=config.aggregate_auxiliary_update_bound,
    )


def _build_setup_request(
    *,
    layouts: ModelProtocolLayouts,
    sample_counts: Sequence[int],
    config: EncryptedSparseRoundConfig,
) -> CryptoEngineSetup:
    return CryptoEngineSetup(
        setup_id=bytes(16),
        parameter_layout=layouts.parameter,
        running_mean_layout=layouts.running_mean,
        running_variance_layout=layouts.running_variance,
        clients=tuple(
            RegisteredClientWeight(client_id, int(example_count))
            for client_id, example_count in enumerate(sample_counts)
        ),
        chunk_size=config.chunk_size,
        minimum_cohort_size=config.minimum_cohort_size,
        protocol_mode=config.protocol_mode,
    )


def _require_compatible_installed_setup(
    installed: CryptoEngineSetup,
    requested: CryptoEngineSetup,
) -> None:
    """Reject accidental layout/client/profile changes between rounds."""

    comparable_fields = (
        "parameter_layout",
        "running_mean_layout",
        "running_variance_layout",
        "clients",
        "chunk_size",
        "minimum_cohort_size",
        "weight_denominator",
        "protocol_mode",
    )
    for name in comparable_fields:
        if getattr(installed, name) != getattr(requested, name):
            raise ValueError(
                f"round input is incompatible with installed setup: {name}"
            )


def _estimate_parallel_encrypted_path(
    *,
    result: OrchestratedRoundResult,
    native_diagnostics: dict | None,
) -> tuple[float, dict]:
    """Replace sequential client phases by their barrier-critical maxima."""

    timing = result.timing
    if timing is None or native_diagnostics is None:
        return 0.0, {"available": False}

    native_encryption = tuple(
        float(value)
        for value in native_diagnostics.get("client_encryption_seconds", ())
    )
    native_key_switch = tuple(
        float(value)
        for value in native_diagnostics.get(
            "client_key_switch_seconds",
            (0.0,) * len(native_encryption),
        )
    )
    native_partial = tuple(
        float(value)
        for value in native_diagnostics.get(
            "client_partial_decryption_seconds",
            (),
        )
    )
    survivor_collective_key_generation = tuple(
        float(value)
        for value in native_diagnostics.get(
            "survivor_collective_key_generation_seconds",
            (0.0,) * len(native_encryption),
        )
    )
    submission = timing.client_value_submission_seconds
    selection = timing.client_selection_submission_seconds
    client_count = len(submission)
    full_mode = len(selection) == 0
    if not (
        client_count
        and (full_mode or len(selection) == client_count)
        and len(native_encryption) == client_count
        and len(native_key_switch) == client_count
        and len(native_partial) == client_count
        and len(survivor_collective_key_generation) == client_count
    ):
        return 0.0, {"available": False}

    native_sequential = float(
        native_diagnostics["crypto_sequential_seconds"]
    )
    native_server = float(native_diagnostics["crypto_server_seconds"])
    native_call_wall_seconds = (
        timing.survivor_transition_seconds + timing.finalize_seconds
    )
    native_wrapper_seconds = max(
        0.0,
        native_call_wall_seconds - native_sequential,
    )
    non_finalize_server_seconds = max(
        0.0,
        timing.total_seconds
        - sum(selection)
        - sum(submission)
        - timing.survivor_transition_seconds
        - timing.finalize_seconds,
    )
    server_seconds = (
        non_finalize_server_seconds
        + native_wrapper_seconds
        + native_server
    )
    encryption_and_submission = tuple(
        submit_seconds
        + encryption_seconds
        + collective_key_seconds
        + key_switch_seconds
        for (
            submit_seconds,
            encryption_seconds,
            collective_key_seconds,
            key_switch_seconds,
        ) in zip(
            submission,
            native_encryption,
            survivor_collective_key_generation,
            native_key_switch,
            strict=True,
        )
    )
    selection_phase = (
        tuple(0.0 for _ in range(client_count))
        if full_mode
        else selection
    )
    uncapped_estimated = estimated_barrier_parallel_seconds(
        client_phases=(
            selection_phase,
            encryption_and_submission,
            native_partial,
        ),
        server_seconds=server_seconds,
    )
    # A barrier critical path replaces sequential client sums with phase
    # maxima.  It therefore cannot exceed the measured sequential encrypted
    # path.  Keep this invariant even when an older native module reports
    # additive per-block work durations instead of client wall times.
    estimated = min(uncapped_estimated, timing.total_seconds)
    return estimated, {
        "available": True,
        "model": (
            "barrier-critical-path-independent-clients-v3"
            if native_diagnostics.get("crypto_timing_model") == "client-batch-service-v3"
            else "barrier-critical-path-independent-clients-v2"
        ),
        "executor_wait_excluded_from_client_times": (
            native_diagnostics.get("crypto_timing_model") == "client-batch-service-v3"
        ),
        "network_latency_included": False,
        "client_selection_submission_seconds": list(selection_phase),
        "client_value_preparation_submission_seconds": list(submission),
        "native_client_encryption_seconds": list(native_encryption),
        "native_client_key_switch_seconds": list(native_key_switch),
        "native_survivor_collective_key_generation_seconds": list(
            survivor_collective_key_generation
        ),
        "native_client_partial_decryption_seconds": list(native_partial),
        "maximum_selection_submission_seconds": maximum_phase_seconds(
            selection_phase
        ),
        "maximum_encryption_and_submission_seconds": maximum_phase_seconds(
            encryption_and_submission
        ),
        "maximum_partial_decryption_seconds": maximum_phase_seconds(
            native_partial
        ),
        "server_seconds": server_seconds,
        "orchestrator_sequential_seconds": timing.total_seconds,
        "native_crypto_sequential_seconds": native_sequential,
        "survivor_transition_seconds": timing.survivor_transition_seconds,
        "finalize_seconds": timing.finalize_seconds,
        "native_wrapper_seconds": native_wrapper_seconds,
        "uncapped_barrier_estimate_seconds": uncapped_estimated,
        "sequential_encrypted_path_upper_bound_seconds": timing.total_seconds,
        "capped_to_sequential_encrypted_path": (
            uncapped_estimated > timing.total_seconds
        ),
        "estimated_parallel_encrypted_path_seconds": estimated,
    }


def _estimate_parallel_setup(
    *,
    engine: FederatedCryptoEngine,
    setup_seconds: float,
    incurred_this_round: bool,
) -> tuple[float, dict]:
    """Convert native setup phases into a barrier-critical-path estimate."""

    diagnostics = getattr(engine, "setup_diagnostics", None)
    if diagnostics is None:
        one_time_parallel = setup_seconds
        document = {
            "available": False,
            "model": "sequential-wall-clock-fallback",
            "network_latency_included": False,
            "actual_setup_wall_seconds": setup_seconds,
            "estimated_parallel_setup_seconds": one_time_parallel,
        }
    else:
        native_sequential = float(diagnostics["sequential_seconds"])
        native_parallel = float(diagnostics["estimated_parallel_seconds"])
        python_wrapper = max(0.0, setup_seconds - native_sequential)
        uncapped_parallel = native_parallel + python_wrapper
        one_time_parallel = min(uncapped_parallel, setup_seconds)
        document = {
            "available": True,
            **diagnostics,
            "native_estimated_parallel_seconds": native_parallel,
            "python_wrapper_seconds": python_wrapper,
            "actual_setup_wall_seconds": setup_seconds,
            "uncapped_estimated_parallel_setup_seconds": uncapped_parallel,
            "capped_to_actual_setup_wall": (
                uncapped_parallel > setup_seconds
            ),
            "estimated_parallel_setup_seconds": one_time_parallel,
        }
    this_round = one_time_parallel if incurred_this_round else 0.0
    document["incurred_this_round"] = incurred_this_round
    document["estimated_parallel_setup_seconds_this_round"] = this_round
    return this_round, document


def run_encrypted_sparse_round(
    *,
    global_model: nn.Module,
    client_loaders: Sequence,
    sample_counts: Sequence[int],
    engine: FederatedCryptoEngine,
    config: EncryptedSparseRoundConfig,
    device: torch.device,
    layouts: ModelProtocolLayouts | None = None,
    residuals: Sequence[torch.Tensor] | None = None,
    training_model: nn.Module | None = None,
    survivor_client_ids: Sequence[int] | None = None,
    logical_round_index: int | None = None,
    post_request_dropouts: Sequence[int] = (),
) -> EncryptedSparseRoundOutcome:
    """Train clients sequentially and commit one engine-backed SPARSE round.

    ``training_model`` optionally fixes the model from which local updates and
    chunk selections are produced while ``global_model`` receives the decrypted
    result.  This is the reference-locked comparison primitive; ordinary and
    free-running calls omit it.
    """

    client_count = len(client_loaders)
    config.validate(client_count)
    if config.protocol_mode != "SPARSE":
        raise ValueError("run_encrypted_sparse_round requires SPARSE mode")
    if len(sample_counts) != client_count or client_count == 0:
        raise ValueError(
            "client loaders and sample counts must be non-empty and align"
        )
    if any(int(count) <= 0 for count in sample_counts):
        raise ValueError("all client sample counts must be positive")
    if survivor_client_ids is not None:
        survivors = tuple(int(value) for value in survivor_client_ids)
        if len(set(survivors)) != len(survivors):
            raise ValueError("survivor client IDs must be unique")
        if not set(survivors).issubset(set(range(client_count))):
            raise ValueError("survivor client IDs must be valid client indices")
        if len(survivors) < config.minimum_cohort_size:
            raise ValueError("survivor count is below minimum cohort size")

    active_training_model = training_model or global_model
    active_layouts = layouts or build_model_protocol_layouts(
        active_training_model
    )
    active_layouts.validate()
    total_chunks = (
        active_layouts.parameter.total_numel + config.chunk_size - 1
    ) // config.chunk_size
    if config.selected_chunk_count > total_chunks:
        raise ValueError("selected_chunk_count exceeds parameter chunk count")

    if residuals is None:
        working_residuals = [
            torch.zeros(
                active_layouts.parameter.total_numel,
                dtype=torch.float32,
                device="cpu",
            )
            for _ in range(client_count)
        ]
    else:
        if len(residuals) != client_count:
            raise ValueError("residual count does not match client count")
        working_residuals = [
            residual.detach()
            .reshape(-1)
            .to(device="cpu", dtype=torch.float32)
            .clone()
            for residual in residuals
        ]
        if any(
            residual.numel() != active_layouts.parameter.total_numel
            for residual in working_residuals
        ):
            raise ValueError("residual dimension does not match parameter layout")

    setup_request = _build_setup_request(
        layouts=active_layouts,
        sample_counts=sample_counts,
        config=config,
    )
    orchestrator = FederatedRoundOrchestrator(engine)
    installed_setup = engine.setup_config
    setup_incurred = installed_setup is None
    if installed_setup is None:
        setup_started = time.perf_counter()
        orchestrator.setup(setup_request)
        setup_seconds = time.perf_counter() - setup_started
    else:
        _require_compatible_installed_setup(installed_setup, setup_request)
        setup_seconds = 0.0

    estimated_parallel_setup_seconds, setup_parallel_timing = (
        _estimate_parallel_setup(
            engine=engine,
            setup_seconds=setup_seconds,
            incurred_this_round=setup_incurred,
        )
    )

    candidates = []
    selections: list[SparseSelection] = []
    client_metrics: list[ClientEncryptedRoundMetrics] = []
    client_local_processing_seconds: list[float] = []
    local_started = synchronized_perf_counter(device)

    for client_id, (loader, example_count) in enumerate(
        zip(client_loaders, sample_counts, strict=True)
    ):
        client_started = synchronized_perf_counter(device)
        local_model = copy.deepcopy(active_training_model).to(device)
        optimizer = build_optimizer(
            model=local_model,
            optimizer_name=config.optimizer,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        weighted_loss = 0.0
        trained_samples = 0
        for _ in range(config.local_epochs):
            metrics = train_one_epoch(
                model=local_model,
                loader=loader,
                optimizer=optimizer,
                device=device,
            )
            if metrics.sample_count != int(example_count):
                raise RuntimeError(
                    f"client {client_id} loader produced "
                    f"{metrics.sample_count} samples, expected {example_count}"
                )
            weighted_loss += metrics.average_loss * metrics.sample_count
            trained_samples += metrics.sample_count

        parameter_delta = compute_layout_delta(
            local_model,
            active_training_model,
            active_layouts.parameter,
        )
        selection, new_residual, _ = compress_with_error_feedback(
            update=torch.from_numpy(parameter_delta.copy()),
            previous_residual=working_residuals[client_id],
            chunk_size=config.chunk_size,
            selected_chunk_count=config.selected_chunk_count,
            residual_beta=config.residual_beta,
        )
        candidate = candidate_from_models(
            client_id=client_id,
            selection=selection,
            local_model=local_model,
            global_model=active_training_model,
            layouts=active_layouts,
        )

        selected_max_abs = _maximum_absolute(candidate.selected_values)
        running_mean_max_abs = _maximum_absolute(candidate.running_mean_delta)
        running_variance_max_abs = _maximum_absolute(
            candidate.running_variance_delta
        )
        client_processing_seconds = (
            synchronized_perf_counter(device) - client_started
        )
        client_local_processing_seconds.append(client_processing_seconds)
        candidates.append(candidate)
        selections.append(selection)
        working_residuals[client_id] = new_residual
        client_metrics.append(
            ClientEncryptedRoundMetrics(
                client_id=client_id,
                example_count=int(example_count),
                average_train_loss=weighted_loss / trained_samples,
                selected_chunk_ids=tuple(
                    int(value) for value in selection.chunk_ids
                ),
                rejected_chunk_ids=(),
                selected_max_abs=selected_max_abs,
                running_mean_max_abs=running_mean_max_abs,
                running_variance_max_abs=running_variance_max_abs,
                processing_seconds=client_processing_seconds,
            )
        )
        del optimizer
        del local_model

    local_training_seconds = synchronized_perf_counter(device) - local_started
    encrypted_started = synchronized_perf_counter(device)
    installed_setup = engine.setup_config
    if installed_setup is None:
        raise RuntimeError("crypto engine did not retain setup metadata")
    plaintext_maxima: AggregateStreamMaxima | None = None
    plaintext_reference: AggregateReference | None = None
    decrypted_maxima: AggregateStreamMaxima | None = None

    def validate_before_encryption(admission: AdmissionPlan) -> None:
        nonlocal plaintext_maxima, plaintext_reference
        if not config.compute_plaintext_reference:
            return
        reference = _aggregate_admitted_reference(
            setup=installed_setup,
            admission=admission,
            candidates=candidates,
        )
        plaintext_reference = reference
        plaintext_maxima = reference.maxima
        _validate_aggregate_maxima(
            plaintext_maxima,
            config,
            prefix="plaintext reference aggregate",
        )

    def validate_and_apply(round_result: OrchestratedRoundResult) -> None:
        nonlocal decrypted_maxima
        decrypted = round_result.decrypted
        decrypted_maxima = AggregateStreamMaxima(
            parameter=_maximum_absolute(decrypted.parameter_delta),
            running_mean=_maximum_absolute(decrypted.running_mean_delta),
            running_variance=_maximum_absolute(
                decrypted.running_variance_delta
            ),
        )
        _validate_aggregate_maxima(
            decrypted_maxima,
            config,
            prefix="decrypted aggregate",
        )
        apply_decrypted_result(
            global_model,
            active_layouts,
            decrypted,
        )

    try:
        result = orchestrator.run_round(
            config.round_id,
            candidates,
            apply_result=validate_and_apply,
            before_value_submission=(
                validate_before_encryption
                if survivor_client_ids is None
                else None
            ),
            survivor_client_ids=survivor_client_ids,
            after_survivor_transition=(
                validate_before_encryption
                if survivor_client_ids is not None
                else None
            ),
            logical_round_index=logical_round_index,
            post_request_dropouts=post_request_dropouts,
        )
    except PostRequestDropoutError as error:
        traffic = getattr(engine, "last_aborted_round_traffic", None)
        if traffic is not None:
            traffic = dict(traffic)
            if int(traffic["round_id"]) != config.round_id:
                raise RuntimeError("late-abort traffic round ID mismatch") from error
        raise EncryptedRoundAborted(
            AbortedEncryptedRoundAttempt(
                logical_round_index=(
                    config.round_id
                    if logical_round_index is None
                    else logical_round_index
                ),
                native_round_id=config.round_id,
                setup_seconds=setup_seconds,
                estimated_parallel_setup_seconds=(
                    estimated_parallel_setup_seconds
                ),
                setup_parallel_timing=setup_parallel_timing,
                local_training_seconds=local_training_seconds,
                estimated_parallel_local_training_seconds=(
                    maximum_phase_seconds(client_local_processing_seconds)
                ),
                encrypted_path_wall_seconds=(
                    synchronized_perf_counter(device) - encrypted_started
                ),
                native_traffic=traffic,
            )
        ) from error
    encrypted_path_seconds = (
        synchronized_perf_counter(device) - encrypted_started
    )
    if decrypted_maxima is None or (
        config.compute_plaintext_reference
        and (plaintext_reference is None or plaintext_maxima is None)
    ):
        raise RuntimeError("aggregate validation did not complete")

    finalized_metrics: list[ClientEncryptedRoundMetrics] = []
    for client_id, (selection, metrics) in enumerate(
        zip(selections, client_metrics, strict=True)
    ):
        rejected = result.rejected_for_client(client_id)
        working_residuals[client_id] = restore_rejected_chunks_to_residual(
            residual=working_residuals[client_id],
            selection=selection,
            accepted_chunk_ids=result.admission.for_client(
                client_id
            ).accepted_actual_chunk_ids,
        )
        if client_id in result.dropped_client_ids:
            working_residuals[client_id] = torch.zeros_like(working_residuals[client_id])
        finalized_metrics.append(
            ClientEncryptedRoundMetrics(
                client_id=metrics.client_id,
                example_count=metrics.example_count,
                average_train_loss=metrics.average_train_loss,
                selected_chunk_ids=metrics.selected_chunk_ids,
                rejected_chunk_ids=rejected,
                selected_max_abs=metrics.selected_max_abs,
                running_mean_max_abs=metrics.running_mean_max_abs,
                running_variance_max_abs=metrics.running_variance_max_abs,
                processing_seconds=metrics.processing_seconds,
            )
        )

    native_diagnostics = getattr(engine, "last_diagnostics", None)
    estimated_parallel_encrypted_path_seconds, parallel_timing = (
        _estimate_parallel_encrypted_path(
            result=result,
            native_diagnostics=native_diagnostics,
        )
    )
    if not parallel_timing["available"]:
        estimated_parallel_encrypted_path_seconds = encrypted_path_seconds
    estimated_parallel_local_training_seconds = maximum_phase_seconds(
        client_local_processing_seconds
    )

    return EncryptedSparseRoundOutcome(
        result=result,
        layouts=active_layouts,
        residuals=tuple(residual.contiguous() for residual in working_residuals),
        clients=tuple(finalized_metrics),
        plaintext_aggregate=plaintext_reference,
        plaintext_aggregate_maxima=plaintext_maxima,
        decrypted_aggregate_maxima=decrypted_maxima,
        learning_rate=config.learning_rate,
        setup_seconds=setup_seconds,
        estimated_parallel_setup_seconds=estimated_parallel_setup_seconds,
        setup_parallel_timing=setup_parallel_timing,
        local_training_seconds=local_training_seconds,
        encrypted_path_seconds=encrypted_path_seconds,
        client_local_processing_seconds=tuple(
            client_local_processing_seconds
        ),
        estimated_parallel_local_training_seconds=(
            estimated_parallel_local_training_seconds
        ),
        estimated_parallel_encrypted_path_seconds=(
            estimated_parallel_encrypted_path_seconds
        ),
        parallel_timing=parallel_timing,
    )


def run_encrypted_full_round(
    *,
    global_model: nn.Module,
    client_loaders: Sequence,
    sample_counts: Sequence[int],
    engine: FederatedCryptoEngine,
    config: EncryptedSparseRoundConfig,
    device: torch.device,
    layouts: ModelProtocolLayouts | None = None,
    training_model: nn.Module | None = None,
    survivor_client_ids: Sequence[int] | None = None,
) -> EncryptedSparseRoundOutcome:
    """Train and commit one direct-RLWE FULL round without sparse routing."""

    client_count = len(client_loaders)
    config.validate(client_count)
    if config.protocol_mode != "FULL":
        raise ValueError("run_encrypted_full_round requires FULL mode")
    if len(sample_counts) != client_count or client_count == 0:
        raise ValueError("client loaders and sample counts must align")
    if any(int(count) <= 0 for count in sample_counts):
        raise ValueError("all client sample counts must be positive")

    active_training_model = training_model or global_model
    active_layouts = layouts or build_model_protocol_layouts(
        active_training_model
    )
    active_layouts.validate()
    setup_request = _build_setup_request(
        layouts=active_layouts,
        sample_counts=sample_counts,
        config=config,
    )
    orchestrator = FederatedRoundOrchestrator(engine)
    installed_setup = engine.setup_config
    setup_incurred = installed_setup is None
    if installed_setup is None:
        setup_started = time.perf_counter()
        orchestrator.setup(setup_request)
        setup_seconds = time.perf_counter() - setup_started
    else:
        _require_compatible_installed_setup(installed_setup, setup_request)
        setup_seconds = 0.0
    estimated_parallel_setup_seconds, setup_parallel_timing = (
        _estimate_parallel_setup(
            engine=engine,
            setup_seconds=setup_seconds,
            incurred_this_round=setup_incurred,
        )
    )

    candidates: list[ClientFullRoundCandidate] = []
    client_metrics: list[ClientEncryptedRoundMetrics] = []
    client_local_processing_seconds: list[float] = []
    local_started = synchronized_perf_counter(device)
    for client_id, (loader, example_count) in enumerate(
        zip(client_loaders, sample_counts, strict=True)
    ):
        client_started = synchronized_perf_counter(device)
        local_model = copy.deepcopy(active_training_model).to(device)
        optimizer = build_optimizer(
            model=local_model,
            optimizer_name=config.optimizer,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        weighted_loss = 0.0
        trained_samples = 0
        for _ in range(config.local_epochs):
            metrics = train_one_epoch(
                model=local_model,
                loader=loader,
                optimizer=optimizer,
                device=device,
            )
            if metrics.sample_count != int(example_count):
                raise RuntimeError(
                    f"client {client_id} loader produced "
                    f"{metrics.sample_count} samples, expected {example_count}"
                )
            weighted_loss += metrics.average_loss * metrics.sample_count
            trained_samples += metrics.sample_count

        parameter_delta = np.ascontiguousarray(
            compute_layout_delta(
                local_model, active_training_model, active_layouts.parameter
            ),
            dtype=np.float32,
        )
        running_mean_delta = np.ascontiguousarray(
            compute_layout_delta(
                local_model, active_training_model, active_layouts.running_mean
            ),
            dtype=np.float32,
        )
        running_variance_delta = np.ascontiguousarray(
            compute_layout_delta(
                local_model,
                active_training_model,
                active_layouts.running_variance,
            ),
            dtype=np.float32,
        )
        candidate = ClientFullRoundCandidate(
            client_id=client_id,
            parameter_delta=parameter_delta,
            running_mean_delta=running_mean_delta,
            running_variance_delta=running_variance_delta,
        )
        client_processing_seconds = (
            synchronized_perf_counter(device) - client_started
        )
        candidates.append(candidate)
        client_local_processing_seconds.append(client_processing_seconds)
        client_metrics.append(
            ClientEncryptedRoundMetrics(
                client_id=client_id,
                example_count=int(example_count),
                average_train_loss=weighted_loss / trained_samples,
                selected_chunk_ids=(),
                rejected_chunk_ids=(),
                selected_max_abs=_maximum_absolute(parameter_delta),
                running_mean_max_abs=_maximum_absolute(running_mean_delta),
                running_variance_max_abs=_maximum_absolute(
                    running_variance_delta
                ),
                processing_seconds=client_processing_seconds,
            )
        )
        del optimizer
        del local_model

    local_training_seconds = synchronized_perf_counter(device) - local_started
    installed_setup = engine.setup_config
    if installed_setup is None:
        raise RuntimeError("crypto engine did not retain setup metadata")
    plaintext_reference = None
    plaintext_maxima = None
    if config.compute_plaintext_reference:
        plaintext_reference = _aggregate_full_reference(
            setup=installed_setup,
            candidates=candidates,
            survivor_client_ids=survivor_client_ids,
        )
        plaintext_maxima = plaintext_reference.maxima
        _validate_aggregate_maxima(
            plaintext_maxima,
            config,
            prefix="plaintext reference aggregate",
        )
    decrypted_maxima: AggregateStreamMaxima | None = None

    def validate_and_apply(round_result: OrchestratedRoundResult) -> None:
        nonlocal decrypted_maxima
        decrypted = round_result.decrypted
        decrypted_maxima = AggregateStreamMaxima(
            parameter=_maximum_absolute(decrypted.parameter_delta),
            running_mean=_maximum_absolute(decrypted.running_mean_delta),
            running_variance=_maximum_absolute(
                decrypted.running_variance_delta
            ),
        )
        _validate_aggregate_maxima(
            decrypted_maxima,
            config,
            prefix="decrypted aggregate",
        )
        apply_decrypted_result(global_model, active_layouts, decrypted)

    encrypted_started = synchronized_perf_counter(device)
    result = orchestrator.run_full_round(
        config.round_id,
        candidates,
        apply_result=validate_and_apply,
        survivor_client_ids=survivor_client_ids,
    )
    encrypted_path_seconds = (
        synchronized_perf_counter(device) - encrypted_started
    )
    if decrypted_maxima is None:
        raise RuntimeError("decrypted FULL aggregate validation did not complete")
    native_diagnostics = getattr(engine, "last_diagnostics", None)
    estimated_parallel_encrypted_path_seconds, parallel_timing = (
        _estimate_parallel_encrypted_path(
            result=result,
            native_diagnostics=native_diagnostics,
        )
    )
    if not parallel_timing["available"]:
        estimated_parallel_encrypted_path_seconds = encrypted_path_seconds
    return EncryptedSparseRoundOutcome(
        result=result,
        layouts=active_layouts,
        residuals=(),
        clients=tuple(client_metrics),
        plaintext_aggregate=plaintext_reference,
        plaintext_aggregate_maxima=plaintext_maxima,
        decrypted_aggregate_maxima=decrypted_maxima,
        learning_rate=config.learning_rate,
        setup_seconds=setup_seconds,
        estimated_parallel_setup_seconds=estimated_parallel_setup_seconds,
        setup_parallel_timing=setup_parallel_timing,
        local_training_seconds=local_training_seconds,
        encrypted_path_seconds=encrypted_path_seconds,
        client_local_processing_seconds=tuple(
            client_local_processing_seconds
        ),
        estimated_parallel_local_training_seconds=maximum_phase_seconds(
            client_local_processing_seconds
        ),
        estimated_parallel_encrypted_path_seconds=(
            estimated_parallel_encrypted_path_seconds
        ),
        parallel_timing=parallel_timing,
    )


def run_paired_encrypted_sparse_round(
    *,
    mode: str,
    reference_model: nn.Module,
    encrypted_model: nn.Module,
    encrypted_client_loaders: Sequence,
    sample_counts: Sequence[int],
    encrypted_engine: FederatedCryptoEngine,
    config: EncryptedSparseRoundConfig,
    device: torch.device,
    layouts: ModelProtocolLayouts,
    encrypted_residuals: Sequence[torch.Tensor] | None = None,
    reference_client_loaders: Sequence | None = None,
    reference_engine: FederatedCryptoEngine | None = None,
    reference_residuals: Sequence[torch.Tensor] | None = None,
    survivor_client_ids: Sequence[int] | None = None,
) -> PairedEncryptedSparseRoundOutcome:
    """Run one reference-locked or free-running paired round."""

    config = replace(config, compute_plaintext_reference=True)

    if mode == "reference-locked":
        if config.protocol_mode == "FULL":
            encrypted = run_encrypted_full_round(
                global_model=encrypted_model,
                client_loaders=encrypted_client_loaders,
                sample_counts=sample_counts,
                engine=encrypted_engine,
                config=config,
                device=device,
                layouts=layouts,
                training_model=reference_model,
            )
        else:
            encrypted = run_encrypted_sparse_round(
                global_model=encrypted_model,
                client_loaders=encrypted_client_loaders,
                sample_counts=sample_counts,
                engine=encrypted_engine,
                config=config,
                device=device,
                layouts=layouts,
                residuals=encrypted_residuals,
                training_model=reference_model,
                survivor_client_ids=survivor_client_ids,
            )
        apply_aggregate_reference(
            reference_model,
            layouts,
            encrypted.plaintext_aggregate,
        )
        return PairedEncryptedSparseRoundOutcome(
            reference=encrypted,
            encrypted=encrypted,
            reference_residuals=encrypted.residuals,
            encrypted_residuals=encrypted.residuals,
        )

    if mode != "free-running":
        raise ValueError("paired mode must be reference-locked or free-running")
    if reference_client_loaders is None or reference_engine is None:
        raise ValueError(
            "free-running mode requires reference loaders and a reference engine"
        )

    paired_rng_state = _capture_rng_state()
    runner = (
        run_encrypted_full_round
        if config.protocol_mode == "FULL"
        else run_encrypted_sparse_round
    )
    reference_kwargs = dict(
        global_model=reference_model,
        client_loaders=reference_client_loaders,
        sample_counts=sample_counts,
        engine=reference_engine,
        config=config,
        device=device,
        layouts=layouts,
    )
    if config.protocol_mode != "FULL":
        reference_kwargs["residuals"] = reference_residuals
        reference_kwargs["survivor_client_ids"] = survivor_client_ids
    reference = runner(**reference_kwargs)
    _restore_rng_state(paired_rng_state)
    encrypted_kwargs = dict(
        global_model=encrypted_model,
        client_loaders=encrypted_client_loaders,
        sample_counts=sample_counts,
        engine=encrypted_engine,
        config=config,
        device=device,
        layouts=layouts,
    )
    if config.protocol_mode != "FULL":
        encrypted_kwargs["residuals"] = encrypted_residuals
        encrypted_kwargs["survivor_client_ids"] = survivor_client_ids
    encrypted = runner(**encrypted_kwargs)
    return PairedEncryptedSparseRoundOutcome(
        reference=reference,
        encrypted=encrypted,
        reference_residuals=reference.residuals,
        encrypted_residuals=encrypted.residuals,
    )


def run_encrypted_sparse_rounds(
    *,
    global_model: nn.Module,
    client_loaders: Sequence,
    sample_counts: Sequence[int],
    engine: FederatedCryptoEngine,
    config: EncryptedSparseRoundConfig,
    rounds: int,
    device: torch.device,
    layouts: ModelProtocolLayouts | None = None,
    residuals: Sequence[torch.Tensor] | None = None,
    on_round_committed: Callable[[EncryptedSparseRoundOutcome], None] | None = None,
) -> EncryptedSparseTrainingOutcome:
    """Run increasing round IDs while carrying model and error feedback.

    Setup and its cryptographic keys are installed only for the first round.
    Every following round starts from the previously committed global model
    and receives the previous round's per-client residual tensors.
    """

    if rounds <= 0:
        raise ValueError("rounds must be positive")

    active_layouts = layouts or build_model_protocol_layouts(global_model)
    working_residuals = residuals
    completed_round_ids: list[int] = []
    final_outcome: EncryptedSparseRoundOutcome | None = None
    for offset in range(rounds):
        round_learning_rate = learning_rate_at_step(
            initial_learning_rate=config.learning_rate,
            minimum_learning_rate=config.minimum_learning_rate,
            step=offset,
            total_steps=rounds,
            schedule=config.schedule,
        )
        active_config = replace(
            config,
            learning_rate=round_learning_rate,
            round_id=config.round_id + offset,
        )
        if config.protocol_mode == "FULL":
            outcome = run_encrypted_full_round(
                global_model=global_model,
                client_loaders=client_loaders,
                sample_counts=sample_counts,
                engine=engine,
                config=active_config,
                device=device,
                layouts=active_layouts,
            )
        else:
            outcome = run_encrypted_sparse_round(
                global_model=global_model,
                client_loaders=client_loaders,
                sample_counts=sample_counts,
                engine=engine,
                config=active_config,
                device=device,
                layouts=active_layouts,
                residuals=working_residuals,
            )
        completed_round_ids.append(outcome.result.decrypted.round_id)
        working_residuals = outcome.residuals
        final_outcome = outcome
        if on_round_committed is not None:
            on_round_committed(outcome)

    if final_outcome is None:
        raise RuntimeError("no encrypted sparse round was completed")
    return EncryptedSparseTrainingOutcome(
        completed_round_ids=tuple(completed_round_ids),
        final_round=final_outcome,
    )


def _layout_document(layouts: ModelProtocolLayouts) -> dict:
    def one(layout) -> dict:
        return {
            "kind": layout.kind.value,
            "model_name": layout.model_name,
            "total_numel": layout.total_numel,
            "sha256": layout.sha256(),
            "entries": [
                {
                    "name": entry.name,
                    "shape": list(entry.shape),
                    "offset": entry.offset,
                    "numel": entry.numel,
                    "dtype": entry.dtype,
                }
                for entry in layout.entries
            ],
        }

    return {
        "parameter": one(layouts.parameter),
        "running_mean": one(layouts.running_mean),
        "running_variance": one(layouts.running_variance),
    }


def _parse_args(*, default_rounds: int = 200) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypted SPARSE/FULL CIFAR-10/100 federated training"
    )
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100"),
        default="cifar10",
    )
    parser.add_argument(
        "--model",
        choices=("resnet20", "resnet32"),
        default="resnet20",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./runs/encrypted-fl")
    parser.add_argument(
        "--crypto-parameter-profile",
        help="JSON parameter profile (default: packaged ResNet-20/CIFAR-10 profile)",
    )
    parser.add_argument("--clients", type=int)
    parser.add_argument("--rounds", type=int, default=default_rounds)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--split", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--partition-file",
        default=None,
        help="Create or reuse an exact client sample-index assignment JSON",
    )
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help=(
            "Seed used only when creating a partition file; defaults to "
            "--seed. When explicitly supplied, an existing file must match it"
        ),
    )
    parser.add_argument("--rho", type=float)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--minimum-cohort-size", type=int)
    parser.add_argument("--residual-beta", type=float, default=1.0)
    parser.add_argument("--dropout-target-cap", type=int, default=128)
    parser.add_argument("--collusion-bound", type=int, default=1)
    parser.add_argument("--experimental-dynamic-cohort", action="store_true",
                        help="v17 experimental post-upload/pre-request dropout; one attempt per round")
    parser.add_argument(
        "--dropout-round-count",
        type=int,
        default=0,
        help=(
            "number of preselected post-upload dropout rounds; FULL or SPARSE, "
            "with exactly one dropped silo per selected round"
        ),
    )
    parser.add_argument(
        "--dropout-seed",
        type=int,
        default=None,
        help="independent dropout schedule seed; defaults to --seed",
    )
    parser.add_argument(
        "--post-request-dropout-round-count",
        type=int,
        default=0,
        help=(
            "v17 SPARSE only: preselect logical rounds where one final "
            "survivor drops after the decryption request on the first attempt"
        ),
    )
    parser.add_argument(
        "--post-request-dropout-seed",
        type=int,
        default=None,
        help="independent late-dropout schedule seed; defaults to --seed",
    )
    parser.add_argument(
        "--post-request-max-retries",
        type=int,
        default=0,
        help="must be zero: this implementation uses attempt cap 1 and closes an aborted logical round",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.0)
    parser.add_argument(
        "--schedule",
        choices=["constant", "cosine"],
        default="cosine",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--aggregate-update-bound",
        "--update-bound",
        dest="update_bound",
        type=float,
        help="must match the profile aggregate parameter bound",
    )
    parser.add_argument(
        "--aggregate-aux-update-bound",
        "--aux-update-bound",
        dest="aux_update_bound",
        type=float,
        help="must match the profile aggregate auxiliary bound",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--crypto-threads",
        type=int,
        default=1,
        help=(
            "bounded outer C++ ciphertext worker count used by FULL, sparse, "
            "and auxiliary stages; OpenFHE may additionally parallelize CRT "
            "towers, so this is not a total CPU-thread limit"
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--comparison-mode",
        choices=("none", "reference-locked", "free-running"),
        default="none",
        help=(
            "none runs the ordinary encrypted trajectory; reference-locked "
            "shares one plaintext-generated update path; free-running lets "
            "plaintext and encrypted models feed back their own results"
        ),
    )
    return parser.parse_args()


def _validate_cli_args(
    args: argparse.Namespace,
    profile: CryptoParameterProfile,
) -> None:
    _validate_encrypted_experiment(
        dataset_name=getattr(args, "dataset", "cifar10"),
        model_name=getattr(args, "model", "resnet20"),
        profile=profile,
    )
    expected_values = {
        "clients": profile.clients,
        "rho": profile.selection_ratio,
        "chunk_size": profile.chunk_size,
        "minimum_cohort_size": profile.minimum_cohort_size,
        "update_bound": profile.aggregate_update_bound,
        "aux_update_bound": profile.aggregate_aux_update_bound,
    }
    for name, expected in expected_values.items():
        supplied = getattr(args, name)
        if supplied is None:
            continue
        matches = (
            supplied == expected
            if isinstance(expected, int)
            else abs(float(supplied) - float(expected)) <= 1.0e-12
        )
        if not matches:
            option = name.replace("_", "-")
            raise ValueError(
                f"--{option}={supplied} conflicts with profile value {expected}"
            )

    if profile.clients < 2:
        raise ValueError("encrypted FL requires at least two clients")
    if profile.clients > 128:
        raise ValueError(
            "v17 positive integer weights with denominator 128 support "
            "at most 128 registered clients"
        )
    if not 2 <= profile.minimum_cohort_size <= profile.clients:
        raise ValueError(
            "v17 native minimum_cohort_size must be between two and client count"
        )
    if args.local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    if getattr(args, "rounds", 1) <= 0:
        raise ValueError("rounds must be positive")
    learning_rate_at_step(
        initial_learning_rate=getattr(args, "learning_rate", 0.1),
        minimum_learning_rate=getattr(args, "minimum_learning_rate", 0.0),
        step=0,
        total_steps=getattr(args, "rounds", 1),
        schedule=getattr(args, "schedule", "cosine"),
    )
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not 1 <= getattr(args, "crypto_threads", 1) <= 256:
        raise ValueError("crypto_threads must be between 1 and 256")
    if (getattr(args, "dropout_round_count", 0) and
            not getattr(args, "experimental_dynamic_cohort", False)):
        raise ValueError("survivor dropout requires --experimental-dynamic-cohort")
    dropout_round_count = getattr(args, "dropout_round_count", 0)
    if getattr(args, "dropout_target_cap", 128) < 0:
        raise ValueError("dropout-target-cap must be nonnegative")
    if getattr(args, "experimental_dynamic_cohort", False):
        if profile.minimum_cohort_size < getattr(args, "collusion_bound", 1) + 2:
            raise ValueError("dropout extension requires tau >= c + 2")
    if not 0 <= dropout_round_count <= getattr(args, "rounds", 1):
        raise ValueError(
            "dropout-round-count must be between zero and rounds"
        )
    if dropout_round_count and getattr(args, "comparison_mode", "none") != "none":
        raise ValueError("pre-request dropout currently requires comparison-mode none")
    if dropout_round_count and (
        profile.clients - 1 < profile.minimum_cohort_size
    ):
        raise ValueError(
            "one dropout leaves fewer clients than minimum_cohort_size"
        )
    late_count = getattr(args, "post_request_dropout_round_count", 0)
    if late_count:
        raise ValueError("v17 pre-request-only experiment forbids post-request dropout injection/retries")
    if not 0 <= late_count <= getattr(args, "rounds", 1):
        raise ValueError(
            "post-request-dropout-round-count must be between zero and rounds"
        )
    if late_count and (profile.protocol_version not in {"v17"} or profile.mode != "SPARSE"):
        raise ValueError("post-request dropout requires v17 SPARSE mode")
    if late_count and dropout_round_count:
        raise ValueError(
            "pre-request and post-request dropout schedules cannot be combined"
        )
    if late_count and getattr(args, "comparison_mode", "none") != "none":
        raise ValueError("post-request dropout requires comparison-mode none")
    if getattr(args, "post_request_max_retries", 0) != 0:
        raise ValueError("attempt cap 1 forbids same-logical-round retries in this implementation")
    if getattr(args, "collusion_bound", 1) < 0:
        raise ValueError("collusion-bound must be nonnegative")
    comparison_mode = getattr(args, "comparison_mode", "none")
    if comparison_mode == "free-running" and args.num_workers != 0:
        raise ValueError(
            "free-running comparison requires --num-workers 0 so paired "
            "data order and augmentation RNG can be reproduced exactly"
        )
    if comparison_mode == "free-running" and not getattr(
        args, "deterministic", False
    ):
        raise ValueError(
            "free-running comparison requires --deterministic so numerical "
            "differences are not confounded by nondeterministic training"
        )


def _stream_maxima_document(maxima: AggregateStreamMaxima) -> dict[str, float]:
    return {
        "parameter": maxima.parameter,
        "running_mean": maxima.running_mean,
        "running_variance": maxima.running_variance,
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else None
    return numerator / denominator


def _effective_precision_bits(error: float, bound: float | None) -> float | None:
    if bound is None or bound <= 0.0 or error <= 0.0:
        return None
    return -math.log2(error / bound)


def array_error_statistics(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    reference_bound: float | None = None,
    expected_precision_bits: int | None = None,
) -> dict[str, float | int | bool | None]:
    """Describe observed-reference error without losing the configured scale."""

    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed_values = np.asarray(observed, dtype=np.float64).reshape(-1)
    if reference_values.shape != observed_values.shape:
        raise ValueError("reference and observed arrays must have the same shape")
    if not np.isfinite(reference_values).all() or not np.isfinite(
        observed_values
    ).all():
        raise ValueError("comparison arrays must contain only finite values")

    error = observed_values - reference_values
    count = int(error.size)
    if count == 0:
        minimum = maximum = mean = mean_abs = max_abs = 0.0
        rmse = standard_deviation = l2 = reference_l2 = 0.0
    else:
        minimum = float(np.min(error))
        maximum = float(np.max(error))
        mean = float(np.mean(error))
        mean_abs = float(np.mean(np.abs(error)))
        max_abs = float(np.max(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        standard_deviation = float(np.std(error))
        l2 = float(np.linalg.norm(error))
        reference_l2 = float(np.linalg.norm(reference_values))

    bound = None if reference_bound is None else float(reference_bound)
    expected_sigma = (
        None
        if bound is None or expected_precision_bits is None
        else bound * math.pow(2.0, -float(expected_precision_bits))
    )
    return {
        "count": count,
        "exact": max_abs == 0.0,
        "error_min": minimum,
        "error_max": maximum,
        "mean_error": mean,
        "mean_abs_error": mean_abs,
        "max_abs_error": max_abs,
        "rmse": rmse,
        "standard_deviation": standard_deviation,
        "l2_error": l2,
        "reference_l2": reference_l2,
        "relative_l2_error": _safe_ratio(l2, reference_l2),
        "reference_bound": bound,
        "max_abs_error_over_bound": (
            None if bound is None else _safe_ratio(max_abs, bound)
        ),
        "rmse_over_bound": None if bound is None else _safe_ratio(rmse, bound),
        "std_over_bound": (
            None
            if bound is None
            else _safe_ratio(standard_deviation, bound)
        ),
        "effective_precision_bits_from_max_abs": _effective_precision_bits(
            max_abs, bound
        ),
        "effective_precision_bits_from_rmse": _effective_precision_bits(
            rmse, bound
        ),
        "effective_precision_bits_from_std": _effective_precision_bits(
            standard_deviation, bound
        ),
        "configured_precision_bits": expected_precision_bits,
        "configured_sigma": expected_sigma,
        "rmse_over_configured_sigma": (
            None
            if expected_sigma is None
            else _safe_ratio(rmse, expected_sigma)
        ),
        "std_over_configured_sigma": (
            None
            if expected_sigma is None
            else _safe_ratio(standard_deviation, expected_sigma)
        ),
    }


def aggregate_error_document(
    reference: AggregateReference,
    observed,
    *,
    parameter_bound: float,
    auxiliary_bound: float,
    parameter_precision_bits: int,
    auxiliary_precision_bits: int,
) -> dict[str, dict]:
    """Compare all protocol streams and the combined auxiliary stream."""

    auxiliary_reference = np.concatenate(
        (reference.running_mean, reference.running_variance)
    )
    auxiliary_observed = np.concatenate(
        (observed.running_mean_delta, observed.running_variance_delta)
    )
    return {
        "parameter": array_error_statistics(
            reference.parameter,
            observed.parameter_delta,
            reference_bound=parameter_bound,
            expected_precision_bits=parameter_precision_bits,
        ),
        "running_mean": array_error_statistics(
            reference.running_mean,
            observed.running_mean_delta,
            reference_bound=auxiliary_bound,
            expected_precision_bits=auxiliary_precision_bits,
        ),
        "running_variance": array_error_statistics(
            reference.running_variance,
            observed.running_variance_delta,
            reference_bound=auxiliary_bound,
            expected_precision_bits=auxiliary_precision_bits,
        ),
        "auxiliary_combined": array_error_statistics(
            auxiliary_reference,
            auxiliary_observed,
            reference_bound=auxiliary_bound,
            expected_precision_bits=auxiliary_precision_bits,
        ),
    }


def aggregate_reference_difference_document(
    reference: AggregateReference,
    observed: AggregateReference,
) -> dict[str, dict]:
    """Measure how far two plaintext input trajectories have separated."""

    return {
        "parameter": array_error_statistics(reference.parameter, observed.parameter),
        "running_mean": array_error_statistics(
            reference.running_mean, observed.running_mean
        ),
        "running_variance": array_error_statistics(
            reference.running_variance, observed.running_variance
        ),
    }


def model_state_difference_document(
    reference_model: nn.Module,
    encrypted_model: nn.Module,
    layouts: ModelProtocolLayouts,
) -> dict[str, dict]:
    return {
        "parameter": array_error_statistics(
            flatten_layout(reference_model, layouts.parameter),
            flatten_layout(encrypted_model, layouts.parameter),
        ),
        "running_mean": array_error_statistics(
            flatten_layout(reference_model, layouts.running_mean),
            flatten_layout(encrypted_model, layouts.running_mean),
        ),
        "running_variance": array_error_statistics(
            flatten_layout(reference_model, layouts.running_variance),
            flatten_layout(encrypted_model, layouts.running_variance),
        ),
    }


def chunk_path_alignment_document(
    reference: EncryptedSparseRoundOutcome,
    encrypted: EncryptedSparseRoundOutcome,
) -> dict[str, float | int | bool]:
    reference_clients = {client.client_id: client for client in reference.clients}
    encrypted_clients = {client.client_id: client for client in encrypted.clients}
    if reference_clients.keys() != encrypted_clients.keys():
        raise ValueError("comparison client IDs do not match")

    exact_clients = 0
    client_jaccards: list[float] = []
    for client_id in sorted(reference_clients):
        left = set(reference_clients[client_id].selected_chunk_ids)
        right = set(encrypted_clients[client_id].selected_chunk_ids)
        exact_clients += int(left == right)
        union = left | right
        client_jaccards.append(1.0 if not union else len(left & right) / len(union))

    left_accepted = set(reference.result.admission.accepted_actual_chunk_ids)
    right_accepted = set(encrypted.result.admission.accepted_actual_chunk_ids)
    accepted_union = left_accepted | right_accepted
    client_count = len(reference_clients)
    return {
        "all_client_selections_exact": exact_clients == client_count,
        "exact_client_selection_count": exact_clients,
        "client_count": client_count,
        "exact_client_selection_fraction": exact_clients / client_count,
        "mean_client_selection_jaccard": sum(client_jaccards) / client_count,
        "accepted_chunks_exact": left_accepted == right_accepted,
        "accepted_chunk_jaccard": (
            1.0
            if not accepted_union
            else len(left_accepted & right_accepted) / len(accepted_union)
        ),
    }


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _accuracy_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _optional_float_text(value: float | None) -> str:
    return "exact" if value is None else f"{value:.3f}"


def _accuracy_gap_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.4f}%p"


def _client_stage_maximums(
    raw_clients: Sequence[dict], field: str = "wall_seconds"
) -> dict[str, float]:
    return {
        stage: max(
            (
                float(client.get(stage, {}).get(field, 0.0))
                for client in raw_clients
            ),
            default=0.0,
        )
        for stage in _CRYPTO_STAGE_NAMES
    }


def _server_stage_wall_seconds(raw_server: dict) -> dict[str, float]:
    return {
        stage: float(raw_server.get(stage, {}).get("wall_seconds", 0.0))
        for stage in _CRYPTO_STAGE_NAMES
    }


def _make_crypto_timing_breakdown(diagnostics: dict) -> dict:
    full_clients = diagnostics.get("full_model_client_stage_timings", ())
    sparse_clients = diagnostics.get("sparse_model_client_stage_timings", ())
    if full_clients:
        model_path = "full_model"
        model_clients = full_clients
        model_server = diagnostics.get("full_model_server_stage_timings", {})
    elif sparse_clients:
        model_path = "sparse_model"
        model_clients = sparse_clients
        model_server = diagnostics.get("sparse_model_server_stage_timings", {})
    else:
        return {"available": False, "scope": "FULL-or-SPARSE"}
    auxiliary_clients = diagnostics.get(
        "auxiliary_client_stage_timings", ()
    )
    return {
        "available": True,
        "timing_model": diagnostics.get("crypto_timing_model", "legacy-wait-inclusive"),
        "execution_model": (
            "clients-sequential-on-benchmark-host; ciphertexts-parallel-"
            "within-each-client; paper-client-latency-is-per-phase-maximum"
        ),
        "network_latency_included": False,
        "crypto_threads": int(diagnostics["crypto_thread_count"]),
        "model_path": model_path,
        "combined_client_phase_maximum_seconds": {
            "pre_aggregation_encryption_plus_key_switch": float(
                diagnostics["maximum_client_pre_aggregation_seconds"]
            ),
            "partial_decryption": float(
                diagnostics["maximum_client_partial_decryption_seconds"]
            ),
            "same_client_all_client_operations": float(
                diagnostics["maximum_client_total_crypto_seconds"]
            ),
        },
        model_path: {
            "client_maximum_executor_wait_seconds_by_stage": (
                _client_stage_maximums(model_clients, "executor_wait_seconds")
            ),
            "client_maximum_wall_seconds_by_stage": (
                _client_stage_maximums(model_clients)
            ),
            "server_wall_seconds_by_stage": _server_stage_wall_seconds(
                model_server
            ),
        },
        "auxiliary": {
            "client_maximum_executor_wait_seconds_by_stage": (
                _client_stage_maximums(auxiliary_clients, "executor_wait_seconds")
            ),
            "client_maximum_wall_seconds_by_stage": (
                _client_stage_maximums(auxiliary_clients)
            ),
            "server_wall_seconds_by_stage": _server_stage_wall_seconds(
                diagnostics.get("auxiliary_server_stage_timings", {})
            ),
        },
    }


def _aggregate_crypto_timing_breakdown(
    round_summaries: Sequence[dict],
) -> dict:
    breakdowns = [
        row["crypto_timing_breakdown"]
        for row in round_summaries
        if row.get("crypto_timing_breakdown", {}).get("available", False)
    ]
    if not breakdowns:
        return {"available": False, "scope": "FULL-or-SPARSE"}

    def sum_stage(component: str, side: str) -> dict[str, float]:
        return {
            stage: sum(
                float(row[component][side][stage]) for row in breakdowns
            )
            for stage in _CRYPTO_STAGE_NAMES
        }

    phase_names = tuple(
        breakdowns[0]["combined_client_phase_maximum_seconds"]
    )
    totals = {
        name: sum(
            float(row["combined_client_phase_maximum_seconds"][name])
            for row in breakdowns
        )
        for name in phase_names
    }
    round_count = len(breakdowns)
    model_path = breakdowns[0]["model_path"]
    return {
        "available": True,
        "rounds": round_count,
        "timing_model": breakdowns[0].get("timing_model", "legacy-wait-inclusive"),
        "execution_model": breakdowns[0]["execution_model"],
        "network_latency_included": False,
        "model_path": model_path,
        "combined_client_phase_maximum_seconds": {
            "total_over_rounds": totals,
            "mean_per_round": {
                name: value / round_count for name, value in totals.items()
            },
        },
        model_path: {
            "total_client_maximum_executor_wait_seconds_by_stage": sum_stage(
                model_path, "client_maximum_executor_wait_seconds_by_stage"
            ),
            "total_client_maximum_wall_seconds_by_stage": sum_stage(
                model_path, "client_maximum_wall_seconds_by_stage"
            ),
            "total_server_wall_seconds_by_stage": sum_stage(
                model_path, "server_wall_seconds_by_stage"
            ),
        },
        "auxiliary": {
            "total_client_maximum_executor_wait_seconds_by_stage": sum_stage(
                "auxiliary", "client_maximum_executor_wait_seconds_by_stage"
            ),
            "total_client_maximum_wall_seconds_by_stage": sum_stage(
                "auxiliary", "client_maximum_wall_seconds_by_stage"
            ),
            "total_server_wall_seconds_by_stage": sum_stage(
                "auxiliary", "server_wall_seconds_by_stage"
            ),
        },
    }


def _print_round_progress(summary: dict) -> None:
    online = summary["communication"]["online"]
    print(
        (
            f"\n=== ROUND {summary['round']:03d}/{summary['requested_rounds']:03d}"
            f" | ACCURACY: {_accuracy_text(summary['test_accuracy_percent'])} ==="
        ),
        flush=True,
    )
    comparison = summary.get("comparison")
    if comparison is not None:
        crypto_error = comparison["crypto_aggregate_error"]
        parameter = crypto_error["parameter"]
        auxiliary = crypto_error["auxiliary_combined"]
        print(
            (
                f"comparison={comparison['mode']}"
                f" | reference_accuracy="
                f"{_accuracy_text(comparison['reference_test_accuracy_percent'])}"
                f" | encrypted-reference="
                f"{_accuracy_gap_text(comparison['accuracy_gap_percentage_points'])}"
            ),
            flush=True,
        )
        print(
            (
                "crypto_error(parameter): "
                f"max_abs={parameter['max_abs_error']:.6g}"
                f" | rmse={parameter['rmse']:.6g}"
                f" | effective_p_rmse="
                f"{_optional_float_text(parameter['effective_precision_bits_from_rmse'])}"
            ),
            flush=True,
        )
        print(
            (
                "crypto_error(auxiliary): "
                f"max_abs={auxiliary['max_abs_error']:.6g}"
                f" | rmse={auxiliary['rmse']:.6g}"
                f" | effective_p_rmse="
                f"{_optional_float_text(auxiliary['effective_precision_bits_from_rmse'])}"
                f" | chunk_path_exact="
                f"{comparison['chunk_path_alignment']['all_client_selections_exact']}"
            ),
            flush=True,
        )
    print(
        (
            f"train_loss={summary['weighted_train_loss']:.6f}"
            f" | lr={summary['learning_rate']:.6g}"
            f" | accepted_chunks={summary['accepted_chunk_count']}"
            f"/{summary['parameter_chunk_count']}"
        ),
        flush=True,
    )
    print(
        (
            f"rho_requested={summary['rho_requested']:.6f}"
            f" | rho_selected={summary['rho_selected']:.6f}"
            f" | rho_eff_round={summary['rho_eff_round']:.6f}"
            f" | rho_eff_running={summary['rho_eff_running']:.6f}"
            f" | cohort_retention={summary['cohort_retention_ratio']:.6f}"
        ),
        flush=True,
    )
    dropout = summary.get("dropout")
    if dropout is not None and dropout.get("applied", False):
        diagnostics = summary.get("native_diagnostics", {})
        target_rebuild_seconds = float(
            diagnostics.get("survivor_target_reconstruction_seconds", 0.0)
        )
        rekey_client_maximum_seconds = float(
            diagnostics.get(
                "maximum_survivor_collective_key_generation_seconds",
                0.0,
            )
        )
        print(
            (
                "dropout=post-upload"
                f" | dropped_client={dropout['dropped_client_ids'][0]}"
                f" | survivors={dropout['survivor_client_ids']}"
                + (f" | full_blocks={summary['accepted_chunk_count']}"
                   if summary.get('native_effective_protocol', {}).get('protocol_mode') == 'FULL'
                   else f" | upload/final_chunks={summary['upload_accepted_chunk_count']}/{summary['accepted_chunk_count']}")
                +
                f" | target_rebuild="
                f"{target_rebuild_seconds:.3f}s"
                f" | rekey_client_max="
                f"{rekey_client_maximum_seconds:.3f}s"
            ),
            flush=True,
        )
    if comparison is not None and comparison["mode"] == "free-running":
        sparsity = comparison["sparsity"]
        print(
            (
                "comparison_rho_eff(reference/encrypted)="
                f"{sparsity['reference']['rho_eff']:.6f}/"
                f"{sparsity['encrypted']['rho_eff']:.6f}"
            ),
            flush=True,
        )
    print(
        (
            f"online_comm={online['combined_total_mb']:.3f} MB all-clients total"
            f" (uplink={online['uplink']['total_mb']:.3f} MB,"
            f" downlink={online['downlink']['total_mb']:.3f} MB,"
            f" mean/client={online['combined_mean_per_client_mb']:.3f} MB)"
        ),
        flush=True,
    )
    if summary["communication"]["setup_incurred_this_round"]:
        setup = summary["communication"]["setup"]
        parallel_setup_seconds = summary.get(
            "estimated_parallel_setup_seconds",
            summary["setup_seconds"],
        )
        print(
            f"one_time_setup_comm={setup['combined_total_mb']:.3f} MB",
            flush=True,
        )
        print(
            (
                "one_time_parallel_setup="
                f"{parallel_setup_seconds:.3f}s"
                " (independent_clients, barrier_critical_path,"
                " network_excluded)"
            ),
            flush=True,
        )
    print(
        (
            f"round_time={summary['round_seconds']:.1f}s"
            f" (setup={summary['setup_seconds']:.1f}s,"
            f" local_train={summary['local_training_seconds']:.1f}s,"
            f" encrypted_path={summary['encrypted_path_seconds']:.1f}s,"
            f" evaluation={summary['evaluation_seconds']:.1f}s)"
        ),
        flush=True,
    )
    print(
        (
            "parallel_estimate="
            f"{summary['estimated_parallel_round_seconds']:.1f}s"
            " (independent_clients, barrier_critical_path, network_excluded)"
            f" | online_runtime_parallel="
            f"{summary['estimated_parallel_protocol_seconds']:.1f}s"
            " (local_train_parallel="
            f"{summary['estimated_parallel_local_training_seconds']:.1f}s,"
            " encrypted_path_parallel="
            f"{summary['estimated_parallel_encrypted_path_seconds']:.1f}s,"
            " evaluation_excluded)"
        ),
        flush=True,
    )
    breakdown = summary.get("crypto_timing_breakdown", {})
    if breakdown.get("available", False):
        combined = breakdown["combined_client_phase_maximum_seconds"]
        model_path = breakdown["model_path"]
        model_label = "full" if model_path == "full_model" else "sparse"
        model_client = breakdown[model_path][
            "client_maximum_wall_seconds_by_stage"
        ]
        model_server = breakdown[model_path][
            "server_wall_seconds_by_stage"
        ]
        auxiliary_client = breakdown["auxiliary"][
            "client_maximum_wall_seconds_by_stage"
        ]
        auxiliary_server = breakdown["auxiliary"][
            "server_wall_seconds_by_stage"
        ]
        print(
            (
                f"{model_label}_crypto_client_max: "
                f"encrypt={model_client['encryption']:.3f}s"
                f" | key_switch="
                f"{model_client['client_to_collective_key_switch']:.3f}s"
                f" | partial_decrypt="
                f"{model_client['partial_decryption']:.3f}s"
            ),
            flush=True,
        )
        print(
            (
                f"{model_label}_crypto_server: "
                f"rlwe_aggregate={model_server['rlwe_aggregation']:.3f}s"
                f" | r2m={model_server['r2m']:.3f}s"
                f" | mlwe_aggregate={model_server['mlwe_aggregation']:.3f}s"
                f" | m2r={model_server['m2r']:.3f}s"
                f" | share_combine="
                f"{model_server['partial_decryption']:.3f}s"
                f" | decode={model_server['decoding']:.3f}s"
            ),
            flush=True,
        )
        print(
            (
                "aux_crypto_client_max: "
                f"encrypt={auxiliary_client['encryption']:.3f}s"
                f" | key_switch="
                f"{auxiliary_client['client_to_collective_key_switch']:.3f}s"
                f" | partial_decrypt="
                f"{auxiliary_client['partial_decryption']:.3f}s"
                " | aux_server="
                f"{sum(auxiliary_server.values()):.3f}s"
                " | combined_client_phase_max="
                f"pre={combined['pre_aggregation_encryption_plus_key_switch']:.3f}s,"
                f" partial={combined['partial_decryption']:.3f}s"
            ),
            flush=True,
        )


        if breakdown.get("timing_model") == "client-batch-service-v3":
            model_wait = breakdown[breakdown["model_path"]][
                "client_maximum_executor_wait_seconds_by_stage"
            ]["partial_decryption"]
            aux_wait = breakdown["auxiliary"][
                "client_maximum_executor_wait_seconds_by_stage"
            ]["partial_decryption"]
            print(
                "executor_wait_client_max (excluded from crypto service times): "
                f"model_partial={model_wait:.3f}s | aux_partial={aux_wait:.3f}s",
                flush=True,
            )


def _make_communication_runtime_summary(summary: dict) -> dict:
    """Build one explicit setup/online reporting block for final outputs."""

    communication = summary["communication"]
    setup = communication["setup"]
    online_cumulative = communication["online_cumulative"]
    online_mean = communication["online_mean_per_round"]
    client_count = int(setup["client_count"])
    completed_rounds = int(communication["completed_rounds"])
    if client_count <= 0 or completed_rounds <= 0:
        raise ValueError("communication summary requires clients and rounds")

    result = {
        "client_count": client_count,
        "completed_rounds": completed_rounds,
        "communication_accounting_model": communication[
            "accounting_model"
        ],
        "network_latency_included": False,
        "setup": {
            "scope": "one-time",
            "communication_mb": {
                "all_clients_total": setup["combined_total_mb"],
                "mean_per_client_total": setup[
                    "combined_mean_per_client_mb"
                ],
                "uplink_all_clients_total": setup["uplink"]["total_mb"],
                "uplink_mean_per_client": setup["uplink"][
                    "mean_per_client_mb"
                ],
                "downlink_all_clients_total": setup["downlink"][
                    "total_mb"
                ],
                "downlink_mean_per_client": setup["downlink"][
                    "mean_per_client_mb"
                ],
            },
            "estimated_parallel_runtime_seconds": summary[
                "total_estimated_parallel_setup_seconds"
            ],
        },
        "online_per_round": {
            "scope": "mean-per-completed-round",
            "communication_mb": {
                "all_clients_total": online_mean["combined_mb"],
                "mean_per_client_total": (
                    online_mean["combined_mb"] / client_count
                ),
                "uplink_all_clients_total": online_mean["uplink_mb"],
                "uplink_mean_per_client": (
                    online_mean["uplink_mb"] / client_count
                ),
                "downlink_all_clients_total": online_mean["downlink_mb"],
                "downlink_mean_per_client": (
                    online_mean["downlink_mb"] / client_count
                ),
            },
            "estimated_parallel_runtime_seconds": summary[
                "average_estimated_parallel_protocol_seconds_per_round"
            ],
            "estimated_parallel_local_training_seconds": summary[
                "average_estimated_parallel_local_training_seconds_per_round"
            ],
            "estimated_parallel_encrypted_path_seconds": summary[
                "average_estimated_parallel_encrypted_path_seconds_per_round"
            ],
        },
        "online_cumulative": {
            "scope": "all-completed-rounds",
            "communication_mb": {
                "all_clients_total": online_cumulative[
                    "combined_total_mb"
                ],
                "mean_per_client_total": online_cumulative[
                    "combined_mean_per_client_mb"
                ],
                "uplink_all_clients_total": online_cumulative["uplink"][
                    "total_mb"
                ],
                "uplink_mean_per_client": online_cumulative["uplink"][
                    "mean_per_client_mb"
                ],
                "downlink_all_clients_total": online_cumulative[
                    "downlink"
                ]["total_mb"],
                "downlink_mean_per_client": online_cumulative[
                    "downlink"
                ]["mean_per_client_mb"],
            },
            "estimated_parallel_runtime_seconds": summary[
                "total_estimated_parallel_protocol_seconds"
            ],
            "estimated_parallel_local_training_seconds": summary[
                "total_estimated_parallel_local_training_seconds"
            ],
            "estimated_parallel_encrypted_path_seconds": summary[
                "total_estimated_parallel_encrypted_path_seconds"
            ],
        },
        "parallel_runtime_scope": (
            "local-training-plus-encrypted-path; independent-client "
            "barrier-critical-path estimate; setup, evaluation, and network "
            "latency excluded from online runtime"
        ),
    }
    result["v17_itemized_communication"] = communication[
        "v17_itemized_communication"
    ]
    return result


def _print_final_progress(summary: dict, output_dir: Path) -> None:
    communication = summary["communication"]
    cumulative = communication["setup_plus_online_cumulative"]
    reporting = summary.get("communication_runtime_summary")
    if reporting is None:
        reporting = _make_communication_runtime_summary(summary)
    setup_reporting = reporting["setup"]
    setup_comm = setup_reporting["communication_mb"]
    online_reporting = reporting["online_per_round"]
    online_comm = online_reporting["communication_mb"]
    online_cumulative = reporting["online_cumulative"]
    itemized_communication = reporting["v17_itemized_communication"]
    best_accuracy = _accuracy_text(summary["best_accuracy_percent"])
    best_round = summary["best_round"]
    best_text = (
        best_accuracy if best_round is None else f"{best_accuracy} at round {best_round}"
    )
    print(
        (
            f"\n=== TRAINING COMPLETE | ROUNDS: {summary['completed_rounds']}"
            f" | FINAL ACCURACY:"
            f" {_accuracy_text(summary['final_test_accuracy_percent'])} ==="
        ),
        flush=True,
    )
    print(f"best_accuracy={best_text}", flush=True)
    print(
        (
            f"rho_requested={summary['rho_requested']:.6f}"
            f" | rho_selected={summary['rho_selected']:.6f}"
            f" | rho_eff={summary['rho_eff']:.6f}"
            f" | cohort_retention={summary['cohort_retention_ratio']:.6f}"
        ),
        flush=True,
    )
    print(
        (
            f"communication={cumulative['combined_total_mb']:.3f} MB"
            " setup-plus-online all-clients cumulative"
            f" (uplink={cumulative['uplink']['total_mb']:.3f} MB,"
            f" downlink={cumulative['downlink']['total_mb']:.3f} MB)"
        ),
        flush=True,
    )
    print(
        (
            "protocol_comm="
            f"{itemized_communication['protocol_total']['total_mb']:.3f} MB"
            " | initial_comm="
            f"{itemized_communication['initial_communication']['total_mb']:.3f} MB"
            " | end_to_end_total="
            f"{itemized_communication['end_to_end_total']['total_mb']:.3f} MB"
        ),
        flush=True,
    )
    print(
        (
            f"setup_comm={setup_comm['all_clients_total']:.3f} MB"
            " all-clients one-time"
            f" (uplink={setup_comm['uplink_all_clients_total']:.3f} MB total,"
            f" {setup_comm['uplink_mean_per_client']:.3f} MB/client;"
            f" downlink={setup_comm['downlink_all_clients_total']:.3f} MB total,"
            f" {setup_comm['downlink_mean_per_client']:.3f} MB/client)"
        ),
        flush=True,
    )
    print(
        (
            f"online_comm_mean_per_round={online_comm['all_clients_total']:.3f}"
            " MB all-clients"
            f" (uplink={online_comm['uplink_all_clients_total']:.3f} MB total,"
            f" {online_comm['uplink_mean_per_client']:.3f} MB/client;"
            f" downlink={online_comm['downlink_all_clients_total']:.3f} MB total,"
            f" {online_comm['downlink_mean_per_client']:.3f} MB/client)"
        ),
        flush=True,
    )
    print(
        f"total_time={summary['total_seconds']:.1f}s | outputs={output_dir}",
        flush=True,
    )
    print(
        (
            "setup_runtime_parallel="
            f"{setup_reporting['estimated_parallel_runtime_seconds']:.3f}s"
            " one-time (independent_clients, network_excluded)"
        ),
        flush=True,
    )
    print(
        (
            "online_runtime_parallel="
            f"{online_cumulative['estimated_parallel_runtime_seconds']:.1f}s"
            " total"
            " | mean/round="
            f"{online_reporting['estimated_parallel_runtime_seconds']:.1f}s"
            " | local_total="
            f"{online_cumulative['estimated_parallel_local_training_seconds']:.1f}s"
            " | encrypted_path_total="
            f"{online_cumulative['estimated_parallel_encrypted_path_seconds']:.1f}s"
            " (independent_clients, network_excluded, committed_rounds_only)"
        ),
        flush=True,
    )
    retry = summary.get("abort_accounting")
    if retry is not None and retry["aborted_attempt_count"]:
        print(
            (
                f"post_request_aborted_attempts={retry['aborted_attempt_count']}"
                " | aborted_online_comm="
                f"{retry['aborted_online_total_mb_all_clients']:.3f} MB"
                " all-clients"
                " | aborted_local_parallel="
                f"{retry['aborted_local_training_seconds_parallel_estimate']:.1f}s"
                " | aborted_encrypted_path_wall="
                f"{retry['aborted_encrypted_path_seconds_wall']:.1f}s"
                " (no parallel estimate for failed path)"
            ),
            flush=True,
        )
    comparison = summary.get("comparison")
    if comparison is not None:
        print(
            (
                f"comparison={comparison['mode']}"
                f" | final_reference_accuracy="
                f"{_accuracy_text(comparison['final_reference_accuracy_percent'])}"
                f" | final_accuracy_gap="
                f"{_accuracy_gap_text(comparison['final_accuracy_gap_percentage_points'])}"
            ),
            flush=True,
        )
        print(
            (
                "comparison_rho_eff(reference/encrypted)="
                f"{comparison['reference_rho_eff']:.6f}/"
                f"{comparison['encrypted_rho_eff']:.6f}"
                f" | gap={comparison['rho_eff_gap']:+.6f}"
            ),
            flush=True,
        )
        print(
            (
                f"worst_crypto_parameter_rmse={comparison['maximum_parameter_rmse']:.6g}"
                f" | worst_crypto_aux_rmse={comparison['maximum_auxiliary_rmse']:.6g}"
                f" | minimum_effective_p(parameter/aux)="
                f"{_optional_float_text(comparison['minimum_parameter_effective_p_rmse'])}/"
                f"{_optional_float_text(comparison['minimum_auxiliary_effective_p_rmse'])}"
            ),
            flush=True,
        )


def _save_error_feedback_state(
    path: Path,
    *,
    round_number: int,
    round_id: int,
    setup_id: bytes,
    residuals: Sequence[torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "round": round_number,
            "round_id": round_id,
            "setup_id": setup_id,
            "residuals": [residual.detach().cpu().clone() for residual in residuals],
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _make_comparison_document(
    *,
    mode: str,
    reference_outcome: EncryptedSparseRoundOutcome,
    encrypted_outcome: EncryptedSparseRoundOutcome,
    reference_model: nn.Module,
    encrypted_model: nn.Module,
    reference_evaluation: EvaluationMetrics | None,
    encrypted_evaluation: EvaluationMetrics | None,
    layouts: ModelProtocolLayouts,
    profile: CryptoParameterProfile,
) -> dict:
    reference_accuracy = (
        None
        if reference_evaluation is None
        else reference_evaluation.accuracy_percent
    )
    encrypted_accuracy = (
        None
        if encrypted_evaluation is None
        else encrypted_evaluation.accuracy_percent
    )
    accuracy_gap = (
        None
        if reference_accuracy is None or encrypted_accuracy is None
        else encrypted_accuracy - reference_accuracy
    )
    parameter_chunk_count = math.ceil(
        layouts.parameter.total_numel / profile.chunk_size
    )
    reference_sparsity = encrypted_round_sparsity_metrics(
        reference_outcome,
        total_chunk_count=parameter_chunk_count,
    )
    encrypted_sparsity = encrypted_round_sparsity_metrics(
        encrypted_outcome,
        total_chunk_count=parameter_chunk_count,
    )
    return {
        "mode": mode,
        "path_semantics": (
            "local updates and chunk selections are generated once from the "
            "plaintext reference model and shared by both update paths"
            if mode == "reference-locked"
            else "plaintext and encrypted models generate and feed back their "
            "own local updates while paired RNG and data order are aligned"
        ),
        "reference_test_loss": (
            None
            if reference_evaluation is None
            else reference_evaluation.average_loss
        ),
        "reference_test_accuracy_percent": reference_accuracy,
        "encrypted_test_loss": (
            None
            if encrypted_evaluation is None
            else encrypted_evaluation.average_loss
        ),
        "encrypted_test_accuracy_percent": encrypted_accuracy,
        "accuracy_gap_percentage_points": accuracy_gap,
        "chunk_path_alignment": chunk_path_alignment_document(
            reference_outcome, encrypted_outcome
        ),
        "sparsity": {
            "reference": reference_sparsity,
            "encrypted": encrypted_sparsity,
            "rho_eff_gap": (
                float(encrypted_sparsity["rho_eff"])
                - float(reference_sparsity["rho_eff"])
            ),
        },
        "crypto_aggregate_error": aggregate_error_document(
            encrypted_outcome.plaintext_aggregate,
            encrypted_outcome.result.decrypted,
            parameter_bound=profile.aggregate_update_bound,
            auxiliary_bound=profile.aggregate_aux_update_bound,
            parameter_precision_bits=profile.precision_bits,
            auxiliary_precision_bits=profile.aux_precision_bits,
        ),
        "plaintext_input_trajectory_difference": (
            aggregate_reference_difference_document(
                reference_outcome.plaintext_aggregate,
                encrypted_outcome.plaintext_aggregate,
            )
        ),
        "model_state_difference": model_state_difference_document(
            reference_model,
            encrypted_model,
            layouts,
        ),
    }


def _comparison_training_summary(round_summaries: Sequence[dict]) -> dict:
    comparisons = [row["comparison"] for row in round_summaries]

    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for comparison in comparisons:
            value = comparison
            for name in path:
                value = value[name]
            if value is not None:
                result.append(float(value))
        return result

    parameter_rmse = values(("crypto_aggregate_error", "parameter", "rmse"))
    auxiliary_rmse = values(
        ("crypto_aggregate_error", "auxiliary_combined", "rmse")
    )
    parameter_p = values(
        (
            "crypto_aggregate_error",
            "parameter",
            "effective_precision_bits_from_rmse",
        )
    )
    auxiliary_p = values(
        (
            "crypto_aggregate_error",
            "auxiliary_combined",
            "effective_precision_bits_from_rmse",
        )
    )
    accuracy_gaps = values(("accuracy_gap_percentage_points",))
    reference_rho_eff = values(("sparsity", "reference", "rho_eff"))
    encrypted_rho_eff = values(("sparsity", "encrypted", "rho_eff"))
    final_comparison = comparisons[-1]
    return {
        "mode": final_comparison["mode"],
        "final_reference_accuracy_percent": final_comparison[
            "reference_test_accuracy_percent"
        ],
        "final_encrypted_accuracy_percent": final_comparison[
            "encrypted_test_accuracy_percent"
        ],
        "final_accuracy_gap_percentage_points": final_comparison[
            "accuracy_gap_percentage_points"
        ],
        "maximum_absolute_accuracy_gap_percentage_points": (
            None if not accuracy_gaps else max(abs(value) for value in accuracy_gaps)
        ),
        "maximum_parameter_rmse": max(parameter_rmse, default=0.0),
        "maximum_auxiliary_rmse": max(auxiliary_rmse, default=0.0),
        "minimum_parameter_effective_p_rmse": min(parameter_p, default=None),
        "minimum_auxiliary_effective_p_rmse": min(auxiliary_p, default=None),
        "all_rounds_client_selections_exact": all(
            comparison["chunk_path_alignment"]["all_client_selections_exact"]
            for comparison in comparisons
        ),
        "all_rounds_accepted_chunks_exact": all(
            comparison["chunk_path_alignment"]["accepted_chunks_exact"]
            for comparison in comparisons
        ),
        "reference_rho_eff": (
            sum(reference_rho_eff) / len(reference_rho_eff)
        ),
        "encrypted_rho_eff": (
            sum(encrypted_rho_eff) / len(encrypted_rho_eff)
        ),
        "rho_eff_gap": (
            sum(encrypted_rho_eff) / len(encrypted_rho_eff)
            - sum(reference_rho_eff) / len(reference_rho_eff)
        ),
        "rounds": len(comparisons),
    }


def _make_round_summary(
    *,
    round_number: int,
    requested_rounds: int,
    outcome: EncryptedSparseRoundOutcome,
    evaluation: EvaluationMetrics | None,
    engine: NativeCryptoEngine,
    profile: CryptoParameterProfile,
    sample_counts: Sequence[int],
    selected_chunk_count: int,
    local_epochs: int,
    batch_size: int,
    optimizer: str,
    initial_learning_rate: float,
    minimum_learning_rate: float,
    schedule: str,
    momentum: float,
    weight_decay: float,
    total_seconds: float,
) -> dict:
    installed_setup = engine.setup_config
    if installed_setup is None:
        raise RuntimeError("native engine did not retain setup metadata")
    diagnostics = engine.last_diagnostics
    if diagnostics is None:
        raise RuntimeError("native engine did not return round diagnostics")
    effective_protocol = engine.effective_protocol
    if effective_protocol is None:
        raise RuntimeError("native engine did not return effective protocol values")
    if (
        profile.mode == "SPARSE"
        and effective_protocol["selected_chunks_per_client"]
        != selected_chunk_count
    ):
        raise RuntimeError("Python/C++ selected chunk counts do not match")

    accepted_chunk_count = (
        installed_setup.parameter_chunk_count
        if profile.mode == "FULL"
        else len(outcome.result.admission.accepted_actual_chunk_ids)
    )
    upload_admission = outcome.result.upload_admission
    if profile.mode == "SPARSE" and upload_admission is None:
        upload_admission = outcome.result.admission
    if profile.mode == "FULL":
        upload_accepted_chunk_count = 0
        uploaded_sparse_contribution_count = 0
    else:
        if upload_admission is None:
            raise RuntimeError("SPARSE round upload admission is missing")
        upload_accepted_chunk_count = len(
            upload_admission.accepted_actual_chunk_ids
        )
        uploaded_sparse_contribution_count = sum(
            len(client.accepted_actual_chunk_ids)
            for client in upload_admission.client_results
        )
    sparsity_metrics = encrypted_round_sparsity_metrics(
        outcome,
        total_chunk_count=installed_setup.parameter_chunk_count,
    )
    setup_communication = measure_setup_communication(
        setup=installed_setup,
        profile=profile,
        effective_protocol=effective_protocol,
    )
    online_communication = (
        measure_full_online_round_communication(
            setup=installed_setup,
            effective_protocol=effective_protocol,
            diagnostics=diagnostics,
        )
        if profile.mode == "FULL"
        else measure_online_round_communication(
            setup=installed_setup,
            effective_protocol=effective_protocol,
            diagnostics=diagnostics,
            accepted_chunk_count=accepted_chunk_count,
            upload_accepted_chunk_count=upload_accepted_chunk_count,
            uploaded_sparse_contribution_count=(
                uploaded_sparse_contribution_count
            ),
        )
    )
    weighted_train_loss = sum(
        client.average_train_loss * client.example_count
        for client in outcome.clients
    ) / sum(client.example_count for client in outcome.clients)
    evaluation_seconds = (
        0.0 if evaluation is None else evaluation.duration_seconds
    )
    sequential_protocol_seconds = (
        outcome.local_training_seconds + outcome.encrypted_path_seconds
    )
    uncapped_parallel_protocol_seconds = (
        outcome.estimated_parallel_local_training_seconds
        + outcome.estimated_parallel_encrypted_path_seconds
    )
    estimated_parallel_protocol_seconds = min(
        uncapped_parallel_protocol_seconds,
        sequential_protocol_seconds,
    )
    round_seconds = (
        outcome.setup_seconds
        + sequential_protocol_seconds
        + evaluation_seconds
    )
    uncapped_parallel_round_seconds = (
        outcome.estimated_parallel_setup_seconds
        + estimated_parallel_protocol_seconds
        + evaluation_seconds
    )
    estimated_parallel_round_seconds = min(
        uncapped_parallel_round_seconds,
        round_seconds,
    )
    return {
        "mode": f"encrypted-{profile.mode.lower()}-multi-round",
        "protocol": profile.protocol_version,
        "experimental_dynamic_cohort": bool(
            effective_protocol.get("experimental_dynamic_cohort", False)
        ),
        "engine": engine.engine_name,
        "round": round_number,
        "round_id": outcome.result.decrypted.round_id,
        "requested_rounds": requested_rounds,
        "setup_id": installed_setup.setup_id.hex(),
        "setup_reused": outcome.setup_seconds == 0.0,
        "crypto_parameter_profile": {
            "name": profile.name,
            "sha256": profile.sha256,
            "source": profile.source,
        },
        "clients": profile.clients,
        "client_sample_counts": list(sample_counts),
        "integer_alphas": list(installed_setup.alphas),
        "weight_denominator": installed_setup.weight_denominator,
        "rho": profile.selection_ratio,
        "rho_requested": profile.selection_ratio,
        "rho_selected": sparsity_metrics["rho_selected"],
        "rho_eff_round": sparsity_metrics["rho_eff"],
        "cohort_retention_ratio": sparsity_metrics[
            "cohort_retention_ratio"
        ],
        "chunk_size": profile.chunk_size,
        "parameter_chunk_count": installed_setup.parameter_chunk_count,
        "selected_chunks_per_client": selected_chunk_count,
        "minimum_cohort_size": profile.minimum_cohort_size,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "learning_rate": outcome.learning_rate,
        "initial_learning_rate": initial_learning_rate,
        "minimum_learning_rate": minimum_learning_rate,
        "schedule": schedule,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "accepted_chunk_ids": list(
            outcome.result.admission.accepted_actual_chunk_ids
        ),
        "accepted_chunk_count": accepted_chunk_count,
        "upload_accepted_chunk_ids": (
            []
            if upload_admission is None
            else list(upload_admission.accepted_actual_chunk_ids)
        ),
        "upload_accepted_chunk_count": upload_accepted_chunk_count,
        "uploaded_sparse_contribution_count": (
            uploaded_sparse_contribution_count
        ),
        "survivor_client_ids": list(outcome.result.survivor_client_ids),
        "dropped_client_ids": list(outcome.result.dropped_client_ids),
        "accepted_chunk_ratio": (
            accepted_chunk_count
            / installed_setup.parameter_chunk_count
        ),
        "candidate_contribution_count": sparsity_metrics[
            "candidate_contribution_count"
        ],
        "accepted_contribution_count": sparsity_metrics[
            "accepted_contribution_count"
        ],
        "rejected_contribution_count": sparsity_metrics[
            "rejected_contribution_count"
        ],
        "weighted_train_loss": weighted_train_loss,
        "test_loss": None if evaluation is None else evaluation.average_loss,
        "test_accuracy_percent": (
            None if evaluation is None else evaluation.accuracy_percent
        ),
        "setup_seconds": outcome.setup_seconds,
        "estimated_parallel_setup_seconds": (
            outcome.estimated_parallel_setup_seconds
        ),
        "local_training_seconds": outcome.local_training_seconds,
        "encrypted_path_seconds": outcome.encrypted_path_seconds,
        "survivor_transition_seconds": (
            0.0
            if outcome.result.timing is None
            else outcome.result.timing.survivor_transition_seconds
        ),
        "sequential_protocol_seconds": sequential_protocol_seconds,
        "estimated_parallel_local_training_seconds": (
            outcome.estimated_parallel_local_training_seconds
        ),
        "estimated_parallel_encrypted_path_seconds": (
            outcome.estimated_parallel_encrypted_path_seconds
        ),
        "estimated_parallel_protocol_seconds": (
            estimated_parallel_protocol_seconds
        ),
        "evaluation_seconds": evaluation_seconds,
        "round_seconds": round_seconds,
        "estimated_parallel_round_seconds": estimated_parallel_round_seconds,
        "parallel_timing": {
            "model": outcome.parallel_timing.get(
                "model", "barrier-critical-path-independent-clients-v2"
            ),
            "client_count": len(outcome.clients),
            "network_latency_included": False,
            "setup": outcome.setup_parallel_timing,
            "client_local_processing_seconds": list(
                outcome.client_local_processing_seconds
            ),
            "maximum_client_local_processing_seconds": (
                outcome.estimated_parallel_local_training_seconds
            ),
            "encrypted_path": outcome.parallel_timing,
            "uncapped_estimated_parallel_protocol_seconds": (
                uncapped_parallel_protocol_seconds
            ),
            "uncapped_estimated_parallel_round_seconds": (
                uncapped_parallel_round_seconds
            ),
            "parallel_protocol_capped_to_sequential_wall": (
                uncapped_parallel_protocol_seconds
                > sequential_protocol_seconds
            ),
            "parallel_round_capped_to_measured_wall": (
                uncapped_parallel_round_seconds > round_seconds
            ),
        },
        "total_seconds": total_seconds,
        "native_diagnostics": diagnostics,
        "crypto_timing_breakdown": _make_crypto_timing_breakdown(
            diagnostics
        ),
        "native_effective_protocol": effective_protocol,
        "communication": {
            "accounting_model": setup_communication["accounting_model"],
            "setup_incurred_this_round": round_number == 1,
            "setup": setup_communication,
            "online": online_communication,
        },
        "plaintext_aggregate_max_abs": (
            None if outcome.plaintext_aggregate_maxima is None
            else _stream_maxima_document(outcome.plaintext_aggregate_maxima)
        ),
        "plaintext_reference_computed": outcome.plaintext_aggregate is not None,
        "decrypted_aggregate_max_abs": _stream_maxima_document(
            outcome.decrypted_aggregate_maxima
        ),
        "clients_detail": [
            {
                "client_id": client.client_id,
                "example_count": client.example_count,
                "average_train_loss": client.average_train_loss,
                "selected_chunk_ids": list(client.selected_chunk_ids),
                "rejected_chunk_ids": list(client.rejected_chunk_ids),
                "selected_max_abs": client.selected_max_abs,
                "running_mean_max_abs": client.running_mean_max_abs,
                "running_variance_max_abs": client.running_variance_max_abs,
                "processing_seconds": client.processing_seconds,
                "residual_l2": (
                    None
                    if profile.mode == "FULL"
                    else float(
                        torch.linalg.vector_norm(
                            outcome.residuals[client.client_id].to(
                                torch.float64
                            )
                        ).item()
                    )
                ),
            }
            for client in outcome.clients
        ],
    }


def _aborted_attempt_document(
    *,
    attempt: AbortedEncryptedRoundAttempt,
    engine: NativeCryptoEngine,
    logical_round_number: int,
) -> dict:
    setup = engine.setup_config
    effective = engine.effective_protocol
    if setup is None or effective is None or attempt.native_traffic is None:
        raise RuntimeError(
            "v17 aborted attempt requires native pre-response traffic counters"
        )
    communication = measure_aborted_sparse_round_communication(
        setup=setup,
        effective_protocol=effective,
        aborted_traffic=attempt.native_traffic,
    )
    return {
        "logical_round": logical_round_number,
        "logical_round_index": attempt.logical_round_index,
        "native_round_id": attempt.native_round_id,
        "reason": "post-final-request-survivor-dropout",
        "model_and_residual_committed": False,
        "setup_seconds": attempt.setup_seconds,
        "estimated_parallel_setup_seconds": (
            attempt.estimated_parallel_setup_seconds
        ),
        "setup_parallel_timing": attempt.setup_parallel_timing,
        "local_training_seconds": attempt.local_training_seconds,
        "estimated_parallel_local_training_seconds": (
            attempt.estimated_parallel_local_training_seconds
        ),
        "encrypted_path_wall_seconds": attempt.encrypted_path_wall_seconds,
        "encrypted_path_parallel_estimate_available": False,
        "communication": communication,
    }


def _load_cli_profile(path, expected_mode=None):
    if path is None and expected_mode == "FULL":
        path = Path(__file__).parent / "profiles/resnet20-cifar10-full-shared-v17.json"
    profile = load_crypto_parameter_profile(path)
    if expected_mode is not None and profile.mode != expected_mode:
        raise ValueError(f"encrypted-{expected_mode.lower()} requires a {expected_mode} profile; got {profile.mode}")
    return profile


def main(*, default_rounds: int = 200, expected_mode: str | None = None) -> None:
    args = _parse_args(default_rounds=default_rounds)
    profile = _load_cli_profile(args.crypto_parameter_profile, expected_mode)
    _validate_cli_args(args, profile)
    total_started = time.perf_counter()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dropout_seed = args.seed if args.dropout_seed is None else args.dropout_seed
    dropout_schedule = build_dropout_schedule(
        total_rounds=args.rounds,
        dropout_round_count=args.dropout_round_count,
        client_count=profile.clients,
        seed=dropout_seed,
    )
    dropout_by_round = {
        entry.round_id: entry for entry in dropout_schedule
    }
    dropout_schedule_document = {
        "schema": "post-upload-single-silo-dropout-v1",
        "seed": dropout_seed,
        "total_rounds": args.rounds,
        "dropout_round_count": len(dropout_schedule),
        "client_count": profile.clients,
        "weight_policy": "fixed-integer-alphas-no-renormalization",
        "recovery_policy": "dropped-silo-recovers-next-round",
        "events": [
            {
                "round": entry.round_id + 1,
                "round_id": entry.round_id,
                "dropped_client_id": entry.dropped_client_id,
                "survivor_client_ids": list(entry.survivor_client_ids),
            }
            for entry in dropout_schedule
        ],
    }
    dropout_schedule_sha256 = hashlib.sha256(
        json.dumps(
            dropout_schedule_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    late_dropout_seed = (
        args.seed
        if args.post_request_dropout_seed is None
        else args.post_request_dropout_seed
    )
    late_dropout_schedule = build_dropout_schedule(
        total_rounds=args.rounds,
        dropout_round_count=args.post_request_dropout_round_count,
        client_count=profile.clients,
        seed=late_dropout_seed,
    )
    late_dropout_by_round = {
        entry.round_id: entry for entry in late_dropout_schedule
    }
    late_dropout_schedule_document = {
        "schema": "v17-post-final-request-single-silo-dropout-v1",
        "seed": late_dropout_seed,
        "total_rounds": args.rounds,
        "dropout_round_count": len(late_dropout_schedule),
        "max_retries_per_logical_round": args.post_request_max_retries,
        "events": [
            {
                "logical_round": entry.round_id + 1,
                "logical_round_index": entry.round_id,
                "dropped_client_id": entry.dropped_client_id,
            }
            for entry in late_dropout_schedule
        ],
    }
    late_dropout_schedule_sha256 = hashlib.sha256(
        json.dumps(
            late_dropout_schedule_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    profile_model_name = _validate_encrypted_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        profile=profile,
    )
    train_set, test_set = _load_encrypted_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        download=not args.no_download,
    )
    partition_seed = (
        args.seed
        if args.partition_seed is None
        else args.partition_seed
    )
    partitions, partition_info = load_or_create_partitions(
        targets=train_set.targets,
        dataset_name=args.dataset,
        num_clients=profile.clients,
        partition=args.split,
        dirichlet_alpha=args.alpha,
        partition_seed=partition_seed,
        partition_file=args.partition_file,
        validate_partition_seed=(
            args.partition_seed is not None
        ),
    )
    client_datasets = build_client_datasets(train_set, partitions)
    client_loaders = build_client_loaders(
        client_datasets=client_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    reference_client_loaders = (
        build_client_loaders(
            client_datasets=client_datasets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        )
        if args.comparison_mode == "free-running"
        else None
    )
    sample_counts = client_sample_counts(client_datasets)
    test_loader = build_test_loader(
        test_set=test_set,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    global_model = _build_encrypted_model(args.model).to(device)
    reference_model = (
        copy.deepcopy(global_model).to(device)
        if args.comparison_mode != "none"
        else None
    )
    layouts = build_model_protocol_layouts(
        global_model,
        model_name=profile_model_name,
    )
    profile.validate_layouts(layouts)
    selected_chunk_count = profile.selected_chunk_count(
        layouts.parameter.total_numel
    )
    master_seed = hashlib.sha256(
        f"fl-v17-encrypted-{profile.mode.lower()}:{args.seed}".encode("utf-8")
    ).digest()
    engine = NativeCryptoEngine(
        replace(
            profile.native_options(master_seed=master_seed),
            crypto_threads=args.crypto_threads,
            experimental_dynamic_cohort=getattr(args, "experimental_dynamic_cohort", False),
            dropout_target_cap=args.dropout_target_cap,
            collusion_bound=args.collusion_bound,
        )
    )
    reference_engine = (
        PlaintextCryptoEngine(experimental_dynamic_cohort=getattr(args, "experimental_dynamic_cohort", False))
        if args.comparison_mode == "free-running"
        else None
    )
    round_config = EncryptedSparseRoundConfig(
        chunk_size=profile.chunk_size,
        selected_chunk_count=selected_chunk_count,
        minimum_cohort_size=profile.minimum_cohort_size,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        schedule=args.schedule,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        residual_beta=args.residual_beta,
        aggregate_update_bound=profile.aggregate_update_bound,
        aggregate_auxiliary_update_bound=profile.aggregate_aux_update_bound,
        protocol_mode=profile.mode,
        compute_plaintext_reference=args.comparison_mode != "none",
    )

    print(
        (
            f"\n=== ENCRYPTED {profile.mode} FL RUN ===\n"
            f"dataset={args.dataset} | model={args.model}\n"
            f"rounds={args.rounds} | clients={profile.clients}"
            f" | device={device} | profile={profile.name}\n"
            f"optimizer={args.optimizer} | lr={args.learning_rate:g}"
            f" | schedule={args.schedule} | batch_size={args.batch_size}\n"
            f"comparison_mode={args.comparison_mode}"
            f" | protocol_version={profile.protocol_version}"
            f" | configured_p={profile.precision_bits}"
            f" | configured_p_aux={profile.aux_precision_bits}\n"
            f"crypto_threads={args.crypto_threads}"
            f" | outer_ciphertext_parallelism=full-sparse-aux-enabled\n"
            f"dropout_round_count={len(dropout_schedule)}"
            f" | dropout_seed={dropout_seed}"
            f" | dropout_schedule_sha256={dropout_schedule_sha256}\n"
            f"post_request_dropout_round_count={len(late_dropout_schedule)}"
            f" | post_request_seed={late_dropout_seed}"
            f" | post_request_schedule_sha256={late_dropout_schedule_sha256}\n"
            f"split={args.split} | alpha="
            f"{args.alpha if args.split == 'noniid' else 'N/A'}"
            f" | partition_source={partition_info['source']}"
            f" | partition_sha256={partition_info['partition_sha256']}"
        ),
        flush=True,
    )

    write_json(output_dir / "crypto_parameter_profile.json", profile.document)
    write_json(output_dir / "protocol_layouts.json", _layout_document(layouts))
    write_json(
        output_dir / "dropout_schedule.json",
        {
            **dropout_schedule_document,
            "sha256": dropout_schedule_sha256,
        },
    )
    write_json(
        output_dir / "post_request_dropout_schedule.json",
        {
            **late_dropout_schedule_document,
            "sha256": late_dropout_schedule_sha256,
        },
    )
    partition_report = partition_document(
        targets=train_set.targets,
        partitions=partitions,
    )
    partition_report["assignment"] = partition_info
    write_json(output_dir / "partition.json", partition_report)
    residuals: tuple[torch.Tensor, ...] | None = None
    reference_residuals: tuple[torch.Tensor, ...] | None = None
    round_summaries: list[dict] = []
    aborted_attempts: list[dict] = []
    next_native_round_id = 0
    best_accuracy: float | None = None
    best_round: int | None = None
    best_reference_accuracy: float | None = None
    best_reference_round: int | None = None

    for round_index in range(args.rounds):
        round_number = round_index + 1
        dropout_event = dropout_by_round.get(round_index)
        late_dropout_event = late_dropout_by_round.get(round_index)
        aborted_this_round: list[dict] = []
        survivor_client_ids = (
            None
            if dropout_event is None
            else dropout_event.survivor_client_ids
        )
        round_learning_rate = learning_rate_at_step(
            initial_learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            step=round_index,
            total_steps=args.rounds,
            schedule=args.schedule,
        )
        active_round_config = replace(
            round_config,
            learning_rate=round_learning_rate,
            round_id=next_native_round_id,
        )
        reference_outcome: EncryptedSparseRoundOutcome | None = None
        if args.comparison_mode == "none":
            if profile.mode == "FULL":
                outcome = run_encrypted_full_round(
                    global_model=global_model,
                    client_loaders=client_loaders,
                    sample_counts=sample_counts,
                    engine=engine,
                    config=active_round_config,
                    device=device,
                    layouts=layouts,
                    survivor_client_ids=survivor_client_ids,
                )
                next_native_round_id += 1
            else:
                next_native_round_id += 1
                try:
                    outcome = run_encrypted_sparse_round(
                        global_model=global_model,
                        client_loaders=client_loaders,
                        sample_counts=sample_counts,
                        engine=engine,
                        config=active_round_config,
                        device=device,
                        layouts=layouts,
                        residuals=residuals,
                        survivor_client_ids=survivor_client_ids,
                        logical_round_index=round_index,
                    )
                except EncryptedRoundAborted as error:
                    document = _aborted_attempt_document(
                        attempt=error.attempt, engine=engine,
                        logical_round_number=round_number,
                    )
                    document["status"] = "AbortPost-no-update"
                    aborted_attempts.append(document)
                    write_json(output_dir / "aborted_attempts.json",
                               {"attempts": aborted_attempts})
                    torch.save({"logical_round": round_number,
                                "status": "AbortPost-no-update",
                                "model_state_dict": global_model.state_dict(),
                                "residuals": residuals},
                               output_dir / "no_update_checkpoint.pt")
                    print(f"ROUND {round_number}: AbortPost; committed model/residual retained; no retry")
                    continue
            residuals = outcome.residuals
        else:
            if reference_model is None:
                raise RuntimeError("paired comparison model was not initialized")
            paired = run_paired_encrypted_sparse_round(
                mode=args.comparison_mode,
                reference_model=reference_model,
                encrypted_model=global_model,
                encrypted_client_loaders=client_loaders,
                sample_counts=sample_counts,
                encrypted_engine=engine,
                config=active_round_config,
                device=device,
                layouts=layouts,
                encrypted_residuals=residuals,
                reference_client_loaders=reference_client_loaders,
                reference_engine=reference_engine,
                reference_residuals=reference_residuals,
                survivor_client_ids=survivor_client_ids,
            )
            reference_outcome = paired.reference
            outcome = paired.encrypted
            reference_residuals = paired.reference_residuals
            residuals = paired.encrypted_residuals
            next_native_round_id += 1

        evaluation: EvaluationMetrics | None = None
        reference_evaluation: EvaluationMetrics | None = None
        if not args.skip_evaluation:
            evaluation = evaluate(global_model, test_loader, device)
            if reference_model is not None:
                reference_evaluation = evaluate(reference_model, test_loader, device)
        round_summary = _make_round_summary(
            round_number=round_number,
            requested_rounds=args.rounds,
            outcome=outcome,
            evaluation=evaluation,
            engine=engine,
            profile=profile,
            sample_counts=sample_counts,
            selected_chunk_count=selected_chunk_count,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            initial_learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            schedule=args.schedule,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            total_seconds=time.perf_counter() - total_started,
        )
        round_summary["split"] = args.split
        round_summary["dataset"] = args.dataset
        round_summary["model"] = args.model
        round_summary["alpha"] = (
            args.alpha if args.split == "noniid" else None
        )
        round_summary["partition"] = partition_info
        round_summary["dropout"] = {
            "applied": dropout_event is not None,
            "schedule_seed": dropout_seed,
            "schedule_sha256": dropout_schedule_sha256,
            "dropped_client_ids": (
                []
                if dropout_event is None
                else [dropout_event.dropped_client_id]
            ),
            "survivor_client_ids": list(
                range(profile.clients)
                if dropout_event is None
                else dropout_event.survivor_client_ids
            ),
            "weight_policy": "fixed-integer-alphas-no-renormalization",
            "recovers_next_round": dropout_event is not None,
            "injection_phase": "after-complete-upload-before-decryption-request",
            "rejoin_policy": "sync-committed-global-state; Sparse residual reset after missed successful Commit",
            "attempt_cap": 1,
            "target_cap": args.dropout_target_cap,
        }
        round_summary["post_request_dropout"] = {
            "scheduled": late_dropout_event is not None,
            "dropped_client_id": (
                None if late_dropout_event is None
                else late_dropout_event.dropped_client_id
            ),
            "aborted_attempts_before_commit": aborted_this_round,
            "successful_native_round_id": outcome.result.decrypted.round_id,
        }
        completed_with_current = [*round_summaries, round_summary]
        total_candidate_contributions = sum(
            int(row["candidate_contribution_count"])
            for row in completed_with_current
        )
        total_accepted_contributions = sum(
            int(row["accepted_contribution_count"])
            for row in completed_with_current
        )
        round_summary["rho_eff_running"] = (
            sum(float(row["rho_eff_round"]) for row in completed_with_current)
            / len(completed_with_current)
        )
        round_summary["cohort_retention_running"] = (
            total_accepted_contributions
            / total_candidate_contributions
        )
        if args.comparison_mode != "none":
            if reference_model is None or reference_outcome is None:
                raise RuntimeError("paired comparison outcome is missing")
            round_summary["comparison"] = _make_comparison_document(
                mode=args.comparison_mode,
                reference_outcome=reference_outcome,
                encrypted_outcome=outcome,
                reference_model=reference_model,
                encrypted_model=global_model,
                reference_evaluation=reference_evaluation,
                encrypted_evaluation=evaluation,
                layouts=layouts,
                profile=profile,
            )
        completed_online_communication = [
            row["communication"]["online"] for row in round_summaries
        ] + [round_summary["communication"]["online"]]
        round_summary["communication_cumulative"] = cumulative_communication(
            setup=round_summary["communication"]["setup"],
            online_rounds=completed_online_communication,
        )
        round_summaries.append(round_summary)

        round_stem = f"round_{round_number:04d}"
        write_json(output_dir / "rounds" / f"{round_stem}.json", round_summary)
        write_json(
            output_dir / "rounds_summary.json",
            {
                "mode": f"encrypted-{profile.mode.lower()}-multi-round",
                "rounds": round_summaries,
            },
        )
        save_checkpoint(
            output_dir / "checkpoints" / f"model_{round_stem}.pt",
            global_model,
            metadata=round_summary,
        )
        if reference_model is not None:
            save_checkpoint(
                output_dir / "checkpoints" / f"reference_model_{round_stem}.pt",
                reference_model,
                metadata=round_summary,
            )
        if profile.mode == "SPARSE":
            _save_error_feedback_state(
                output_dir / "checkpoints" / f"error_feedback_{round_stem}.pt",
                round_number=round_number,
                round_id=outcome.result.decrypted.round_id,
                setup_id=engine.setup_config.setup_id,
                residuals=residuals,
            )

        if evaluation is not None and (
            best_accuracy is None or evaluation.accuracy_percent > best_accuracy
        ):
            best_accuracy = evaluation.accuracy_percent
            best_round = round_number
            save_checkpoint(
                output_dir / "best_model.pt",
                global_model,
                metadata=round_summary,
            )
        if reference_evaluation is not None and (
            best_reference_accuracy is None
            or reference_evaluation.accuracy_percent > best_reference_accuracy
        ):
            best_reference_accuracy = reference_evaluation.accuracy_percent
            best_reference_round = round_number
            save_checkpoint(
                output_dir / "best_reference_model.pt",
                reference_model,
                metadata=round_summary,
            )
        _print_round_progress(round_summary)

    if not round_summaries:
        write_json(output_dir / "summary.json", {
            "completed_rounds": 0, "closed_without_update": len(aborted_attempts),
            "requested_rounds": args.rounds,
        })
        return
    final_round = round_summaries[-1]
    total_candidate_contributions = sum(
        int(row["candidate_contribution_count"])
        for row in round_summaries
    )
    total_accepted_contributions = sum(
        int(row["accepted_contribution_count"])
        for row in round_summaries
    )
    final_summary = {
        "mode": f"encrypted-{profile.mode.lower()}-multi-round",
        "protocol": profile.protocol_version,
        "experimental_dynamic_cohort": getattr(args, "experimental_dynamic_cohort", False),
        "engine": engine.engine_name,
        "rounds": args.rounds,
        "completed_rounds": len(round_summaries),
        "setup_id": final_round["setup_id"],
        "crypto_parameter_profile": final_round["crypto_parameter_profile"],
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "initial_learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "schedule": args.schedule,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "comparison_mode": args.comparison_mode,
        "crypto_threads": args.crypto_threads,
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "alpha": args.alpha if args.split == "noniid" else None,
        "partition_seed": partition_info["partition_seed"],
        "partition_file": partition_info["partition_file"],
        "partition_source": partition_info["source"],
        "partition_sha256": partition_info["partition_sha256"],
        "dropout_round_count": len(dropout_schedule),
        "dropout_seed": dropout_seed,
        "dropout_schedule_sha256": dropout_schedule_sha256,
        "dropout_schedule": dropout_schedule_document["events"],
        "completed_dropout_rounds": sum(
            1 for row in round_summaries if row["dropout"]["applied"]
        ),
        "rho_requested": final_round["rho_requested"],
        "rho_selected": final_round["rho_selected"],
        "rho_eff": (
            sum(float(row["rho_eff_round"]) for row in round_summaries)
            / len(round_summaries)
        ),
        "cohort_retention_ratio": (
            total_accepted_contributions
            / total_candidate_contributions
        ),
        "total_candidate_contribution_count": (
            total_candidate_contributions
        ),
        "total_accepted_contribution_count": (
            total_accepted_contributions
        ),
        "total_rejected_contribution_count": (
            total_candidate_contributions
            - total_accepted_contributions
        ),
        "best_round": best_round,
        "best_accuracy_percent": best_accuracy,
        "best_reference_round": best_reference_round,
        "best_reference_accuracy_percent": best_reference_accuracy,
        "final_test_loss": final_round["test_loss"],
        "final_test_accuracy_percent": final_round["test_accuracy_percent"],
        "total_setup_seconds": (
            sum(row["setup_seconds"] for row in round_summaries)
            + sum(row["setup_seconds"] for row in aborted_attempts)
        ),
        "setup_parallel_timing": (
            next(
                (
                    row["setup_parallel_timing"]
                    for row in aborted_attempts
                    if row["setup_seconds"] > 0
                ),
                round_summaries[0]["parallel_timing"]["setup"],
            )
        ),
        "total_estimated_parallel_setup_seconds": sum(
            row["estimated_parallel_setup_seconds"]
            for row in round_summaries
        ) + sum(
            row["estimated_parallel_setup_seconds"]
            for row in aborted_attempts
        ),
        "total_local_training_seconds": sum(
            row["local_training_seconds"] for row in round_summaries
        ),
        "total_estimated_parallel_local_training_seconds": sum(
            row["estimated_parallel_local_training_seconds"]
            for row in round_summaries
        ),
        "average_estimated_parallel_local_training_seconds_per_round": (
            sum(
                row["estimated_parallel_local_training_seconds"]
                for row in round_summaries
            )
            / len(round_summaries)
        ),
        "total_encrypted_path_seconds": sum(
            row["encrypted_path_seconds"] for row in round_summaries
        ),
        "total_estimated_parallel_encrypted_path_seconds": sum(
            row["estimated_parallel_encrypted_path_seconds"]
            for row in round_summaries
        ),
        "average_estimated_parallel_encrypted_path_seconds_per_round": (
            sum(
                row["estimated_parallel_encrypted_path_seconds"]
                for row in round_summaries
            )
            / len(round_summaries)
        ),
        "total_sequential_protocol_seconds": sum(
            row["sequential_protocol_seconds"] for row in round_summaries
        ),
        "total_estimated_parallel_protocol_seconds": sum(
            row["estimated_parallel_protocol_seconds"]
            for row in round_summaries
        ),
        "total_estimated_parallel_round_seconds": sum(
            row["estimated_parallel_round_seconds"]
            for row in round_summaries
        ),
        "average_estimated_parallel_protocol_seconds_per_round": (
            sum(
                row["estimated_parallel_protocol_seconds"]
                for row in round_summaries
            )
            / len(round_summaries)
        ),
        "total_evaluation_seconds": sum(
            row["evaluation_seconds"] for row in round_summaries
        ),
        "total_seconds": time.perf_counter() - total_started,
        "communication": final_round["communication_cumulative"],
        "post_request_dropout_schedule": {
            **late_dropout_schedule_document,
            "sha256": late_dropout_schedule_sha256,
        },
        "aborted_attempts": aborted_attempts,
        "final_round": final_round,
    }
    aborted_uplink_bits = sum(
        int(row["communication"]["uplink"]["total_bits"])
        for row in aborted_attempts
    )
    aborted_downlink_bits = sum(
        int(row["communication"]["downlink"]["total_bits"])
        for row in aborted_attempts
    )
    successful_online = final_summary["communication"]["online_cumulative"]
    final_summary["abort_accounting"] = {
        "aborted_attempt_count": len(aborted_attempts),
        "aborted_native_round_ids": [
            row["native_round_id"] for row in aborted_attempts
        ],
        "aborted_online_uplink_mb_all_clients": (
            aborted_uplink_bits / 8_000_000
        ),
        "aborted_online_downlink_mb_all_clients": (
            aborted_downlink_bits / 8_000_000
        ),
        "aborted_online_total_mb_all_clients": (
            (aborted_uplink_bits + aborted_downlink_bits) / 8_000_000
        ),
        "online_total_mb_including_aborts_all_clients": (
            int(successful_online["combined_total_bits"])
            + aborted_uplink_bits + aborted_downlink_bits
        ) / 8_000_000,
        "aborted_local_training_seconds_wall": sum(
            row["local_training_seconds"] for row in aborted_attempts
        ),
        "aborted_local_training_seconds_parallel_estimate": sum(
            row["estimated_parallel_local_training_seconds"]
            for row in aborted_attempts
        ),
        "aborted_encrypted_path_seconds_wall": sum(
            row["encrypted_path_wall_seconds"] for row in aborted_attempts
        ),
        "aborted_encrypted_path_parallel_estimate_available": False,
        "abort_control_metadata_included": False,
    }
    final_summary["crypto_timing_breakdown"] = (
        _aggregate_crypto_timing_breakdown(round_summaries)
    )
    if args.comparison_mode != "none":
        final_summary["comparison"] = _comparison_training_summary(
            round_summaries
        )
        write_json(
            output_dir / "comparison_summary.json",
            {
                "summary": final_summary["comparison"],
                "rounds": [row["comparison"] for row in round_summaries],
            },
        )
    # Keep this final so the compact setup/online report is the last block in
    # summary.json and is easy to find without traversing round diagnostics.
    final_summary["communication_runtime_summary"] = (
        _make_communication_runtime_summary(final_summary)
    )
    write_json(output_dir / "summary.json", final_summary)
    write_json(output_dir / "round_summary.json", final_round)
    save_checkpoint(output_dir / "final_model.pt", global_model, final_summary)
    if reference_model is not None:
        save_checkpoint(
            output_dir / "final_reference_model.pt",
            reference_model,
            final_summary,
        )
    save_checkpoint(
        output_dir / "model_after_encrypted_round.pt",
        global_model,
        final_summary,
    )
    _save_error_feedback_state(
        output_dir / "error_feedback_state.pt",
        round_number=args.rounds,
        round_id=int(final_round["round_id"]),
        setup_id=engine.setup_config.setup_id,
        residuals=residuals,
    )
    if (
        args.comparison_mode == "free-running"
        and reference_engine is not None
        and reference_engine.setup_config is not None
        and reference_residuals is not None
    ):
        _save_error_feedback_state(
            output_dir / "reference_error_feedback_state.pt",
            round_number=args.rounds,
            round_id=int(final_round["round_id"]),
            setup_id=reference_engine.setup_config.setup_id,
            residuals=reference_residuals,
        )
    _print_final_progress(final_summary, output_dir)


def main_one_round() -> None:
    """Compatibility entry point whose omitted --rounds value remains one."""

    main(default_rounds=1)


if __name__ == "__main__":
    main()
