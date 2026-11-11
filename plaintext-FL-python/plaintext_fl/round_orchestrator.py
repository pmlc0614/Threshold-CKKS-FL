"""Python-owned v17 round orchestration independent of the engine backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import time

import numpy as np

from .crypto_engine import (
    AdmissionPlan,
    CryptoEngineSetup,
    DecryptedRoundResult,
    FederatedCryptoEngine,
)
from .protocol_contract import RoundPhase


@dataclass(frozen=True)
class ClientRoundCandidate:
    """Local-training output held by Python until cohort admission."""

    client_id: int
    selected_actual_chunk_ids: tuple[int, ...]
    selected_values: np.ndarray
    running_mean_delta: np.ndarray
    running_variance_delta: np.ndarray

    def validate(self, setup: CryptoEngineSetup) -> None:
        if self.client_id not in setup.client_ids:
            raise ValueError(f"client {self.client_id} is not registered")
        if not self.selected_actual_chunk_ids:
            raise ValueError("selected chunk IDs must not be empty")
        if len(set(self.selected_actual_chunk_ids)) != len(
            self.selected_actual_chunk_ids
        ):
            raise ValueError("selected chunk IDs must be unique")
        if any(
            chunk_id < 0 or chunk_id >= setup.parameter_chunk_count
            for chunk_id in self.selected_actual_chunk_ids
        ):
            raise ValueError("selected chunk ID is outside the parameter layout")
        if not isinstance(self.selected_values, np.ndarray):
            raise TypeError("selected_values must be a numpy array")
        if self.selected_values.dtype != np.float32:
            raise TypeError("selected_values must use float32")
        if self.selected_values.shape != (
            len(self.selected_actual_chunk_ids),
            setup.chunk_size,
        ):
            raise ValueError("selected_values have an invalid shape")
        if not self.selected_values.flags.c_contiguous:
            raise ValueError("selected_values must be C-contiguous")
        if not np.isfinite(self.selected_values).all():
            raise ValueError("selected_values contain NaN or infinity")

        auxiliary = (
            (
                "running_mean_delta",
                self.running_mean_delta,
                setup.running_mean_layout.total_numel,
            ),
            (
                "running_variance_delta",
                self.running_variance_delta,
                setup.running_variance_layout.total_numel,
            ),
        )
        for name, values, expected_size in auxiliary:
            if not isinstance(values, np.ndarray):
                raise TypeError(f"{name} must be a numpy array")
            if values.dtype != np.float32:
                raise TypeError(f"{name} must use float32")
            if values.shape != (expected_size,):
                raise ValueError(f"{name} has an invalid shape")
            if not values.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or infinity")


@dataclass(frozen=True)
class ClientFullRoundCandidate:
    """Complete local model and BatchNorm deltas for a FULL round."""

    client_id: int
    parameter_delta: np.ndarray
    running_mean_delta: np.ndarray
    running_variance_delta: np.ndarray

    def validate(self, setup: CryptoEngineSetup) -> None:
        if setup.protocol_mode != "FULL":
            raise ValueError("full candidates require a FULL setup")
        if self.client_id not in setup.client_ids:
            raise ValueError(f"client {self.client_id} is not registered")
        for name, values, expected_size in (
            (
                "parameter_delta",
                self.parameter_delta,
                setup.parameter_layout.total_numel,
            ),
            (
                "running_mean_delta",
                self.running_mean_delta,
                setup.running_mean_layout.total_numel,
            ),
            (
                "running_variance_delta",
                self.running_variance_delta,
                setup.running_variance_layout.total_numel,
            ),
        ):
            if not isinstance(values, np.ndarray):
                raise TypeError(f"{name} must be a numpy array")
            if values.dtype != np.float32 or values.shape != (expected_size,):
                raise ValueError(f"{name} must be a one-dimensional float32 array")
            if not values.flags.c_contiguous or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite and C-contiguous")


@dataclass(frozen=True)
class OrchestrationTiming:
    """Sequential measurements around barrier-separated round operations."""

    client_selection_submission_seconds: tuple[float, ...]
    client_value_submission_seconds: tuple[float, ...]
    survivor_transition_seconds: float
    finalize_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class OrchestratedRoundResult:
    admission: AdmissionPlan
    decrypted: DecryptedRoundResult
    rejected_actual_chunk_ids_by_client: tuple[tuple[int, tuple[int, ...]], ...]
    timing: OrchestrationTiming | None = None
    upload_admission: AdmissionPlan | None = None
    survivor_client_ids: tuple[int, ...] = ()
    dropped_client_ids: tuple[int, ...] = ()

    def rejected_for_client(self, client_id: int) -> tuple[int, ...]:
        for candidate_client_id, rejected_ids in self.rejected_actual_chunk_ids_by_client:
            if candidate_client_id == client_id:
                return rejected_ids
        raise KeyError(f"no rejection result for client {client_id}")


ApplyRoundResult = Callable[[OrchestratedRoundResult], None]
BeforeValueSubmission = Callable[[AdmissionPlan], None]
AfterSurvivorTransition = Callable[[AdmissionPlan], None]


class FederatedRoundOrchestrator:
    """Own round order and transactions without knowing engine internals."""

    def __init__(self, engine: FederatedCryptoEngine) -> None:
        self._engine = engine

    @property
    def engine(self) -> FederatedCryptoEngine:
        return self._engine

    def setup(self, config: CryptoEngineSetup) -> None:
        self._engine.setup(config)

    def run_round(
        self,
        round_id: int,
        candidates: Sequence[ClientRoundCandidate],
        apply_result: ApplyRoundResult | None = None,
        before_value_submission: BeforeValueSubmission | None = None,
        survivor_client_ids: Sequence[int] | None = None,
        after_survivor_transition: AfterSurvivorTransition | None = None,
        logical_round_index: int | None = None,
        post_request_dropouts: Sequence[int] = (),
    ) -> OrchestratedRoundResult:
        """Execute two-phase submission and commit only after model apply.

        Candidate values remain in Python memory while only IDs are submitted.
        After ``finalize_admission`` the approved rows are copied into compact
        float32 arrays and sent to the engine.  Any failure aborts the engine
        round, so residual/model updates can be treated transactionally.
        """

        setup = self._engine.setup_config
        if setup is None:
            raise RuntimeError("orchestrator setup has not been installed")
        if len(candidates) != len(setup.clients):
            raise ValueError("one candidate is required for every client")

        candidate_by_client: dict[int, ClientRoundCandidate] = {}
        for candidate in candidates:
            candidate.validate(setup)
            if candidate.client_id in candidate_by_client:
                raise ValueError(f"duplicate candidate for client {candidate.client_id}")
            candidate_by_client[candidate.client_id] = candidate
        if set(candidate_by_client) != set(setup.client_ids):
            raise ValueError("candidate client set does not match setup")

        orchestrator_started = time.perf_counter()
        client_selection_seconds: list[float] = []
        client_value_seconds: list[float] = []
        if logical_round_index is None:
            self._engine.begin_round(round_id)
        else:
            self._engine.begin_round(
                round_id, logical_round_index=logical_round_index
            )
        try:
            # Phase 1: labels only.  No candidate value buffer crosses here.
            for client_id in setup.client_ids:
                client_started = time.perf_counter()
                candidate = candidate_by_client[client_id]
                self._engine.submit_selection(
                    client_id,
                    candidate.selected_actual_chunk_ids,
                )
                client_selection_seconds.append(
                    time.perf_counter() - client_started
                )

            admission = self._engine.finalize_admission()
            admission.validate(setup)
            if admission.round_id != round_id:
                raise ValueError("engine returned admission for the wrong round")

            expected_cohort_sizes = [0] * setup.parameter_chunk_count
            for candidate in candidates:
                for chunk_id in candidate.selected_actual_chunk_ids:
                    expected_cohort_sizes[chunk_id] += 1
            if admission.cohort_sizes != tuple(expected_cohort_sizes):
                raise ValueError("engine returned incorrect cohort sizes")
            if before_value_submission is not None:
                before_value_submission(admission)

            # Phase 2: compact and submit only approved rows.
            for client_id in setup.client_ids:
                client_started = time.perf_counter()
                candidate = candidate_by_client[client_id]
                accepted_ids = admission.for_client(
                    client_id
                ).accepted_actual_chunk_ids
                expected_accepted_ids = tuple(
                    chunk_id
                    for chunk_id in candidate.selected_actual_chunk_ids
                    if chunk_id in set(admission.accepted_actual_chunk_ids)
                )
                if accepted_ids != expected_accepted_ids:
                    raise ValueError(
                        "engine returned an invalid per-client admission result"
                    )
                row_by_chunk_id = {
                    chunk_id: row
                    for row, chunk_id in enumerate(
                        candidate.selected_actual_chunk_ids
                    )
                }
                accepted_values = np.ascontiguousarray(
                    np.stack(
                        [
                            candidate.selected_values[row_by_chunk_id[chunk_id]]
                            for chunk_id in accepted_ids
                        ],
                        axis=0,
                    )
                    if accepted_ids
                    else np.empty((0, setup.chunk_size), dtype=np.float32),
                    dtype=np.float32,
                )
                self._engine.submit_accepted_values(
                    client_id,
                    accepted_ids,
                    accepted_values,
                )
                self._engine.submit_auxiliary_deltas(
                    client_id,
                    candidate.running_mean_delta,
                    candidate.running_variance_delta,
                )
                client_value_seconds.append(
                    time.perf_counter() - client_started
                )

            active_admission = admission
            transition_seconds = 0.0
            active_survivors = setup.client_ids
            dropped_client_ids: tuple[int, ...] = ()
            if survivor_client_ids is not None:
                active_survivors = tuple(
                    int(value) for value in survivor_client_ids
                )
                if len(set(active_survivors)) != len(active_survivors):
                    raise ValueError("survivor client IDs must be unique")
                if not set(active_survivors).issubset(set(setup.client_ids)):
                    raise ValueError("survivor client IDs must be registered")
                active_survivor_set = set(active_survivors)
                dropped_client_ids = tuple(
                    client_id
                    for client_id in setup.client_ids
                    if client_id not in active_survivor_set
                )
                transition_started = time.perf_counter()
                active_admission = self._engine.set_round_survivors(
                    active_survivors
                )
                transition_seconds = time.perf_counter() - transition_started
                active_admission.validate(setup)
                if active_admission.round_id != round_id:
                    raise ValueError(
                        "engine returned survivor admission for the wrong round"
                    )
                if after_survivor_transition is not None:
                    after_survivor_transition(active_admission)

            rejected = tuple(
                (
                    client_id,
                    tuple(
                        chunk_id
                        for chunk_id in candidate_by_client[
                            client_id
                        ].selected_actual_chunk_ids
                        if chunk_id
                        not in set(
                            active_admission.for_client(
                                client_id
                            ).accepted_actual_chunk_ids
                        )
                    ),
                )
                for client_id in setup.client_ids
            )

            finalize_started = time.perf_counter()
            if post_request_dropouts:
                decrypted = self._engine.finalize_round(
                    post_request_dropouts=post_request_dropouts
                )
            else:
                decrypted = self._engine.finalize_round()
            finalize_seconds = time.perf_counter() - finalize_started
            decrypted.validate(setup)
            if decrypted.round_id != round_id:
                raise ValueError("engine returned a result for the wrong round")
            if (
                decrypted.accepted_actual_chunk_ids
                != active_admission.accepted_actual_chunk_ids
                or decrypted.cohort_sizes != active_admission.cohort_sizes
            ):
                raise ValueError("decrypted result does not match admission")
            result = OrchestratedRoundResult(
                admission=active_admission,
                decrypted=decrypted,
                rejected_actual_chunk_ids_by_client=rejected,
                upload_admission=admission,
                survivor_client_ids=tuple(active_survivors),
                dropped_client_ids=dropped_client_ids,
            )
            if apply_result is not None:
                apply_result(result)
            self._engine.commit_round()
            return OrchestratedRoundResult(
                admission=result.admission,
                decrypted=result.decrypted,
                rejected_actual_chunk_ids_by_client=(
                    result.rejected_actual_chunk_ids_by_client
                ),
                timing=OrchestrationTiming(
                    client_selection_submission_seconds=tuple(
                        client_selection_seconds
                    ),
                    client_value_submission_seconds=tuple(
                        client_value_seconds
                    ),
                    survivor_transition_seconds=transition_seconds,
                    finalize_seconds=finalize_seconds,
                    total_seconds=time.perf_counter() - orchestrator_started,
                ),
                upload_admission=result.upload_admission,
                survivor_client_ids=result.survivor_client_ids,
                dropped_client_ids=result.dropped_client_ids,
            )
        except Exception:
            if self._engine.phase not in {
                None,
                RoundPhase.COMMITTED,
                RoundPhase.ABORTED,
            }:
                self._engine.abort_round()
            raise

    def run_full_round(
        self,
        round_id: int,
        candidates: Sequence[ClientFullRoundCandidate],
        apply_result: ApplyRoundResult | None = None,
        before_value_submission: Callable[[], None] | None = None,
        survivor_client_ids: Sequence[int] | None = None,
    ) -> OrchestratedRoundResult:
        """Submit complete deltas without selection, admission, R2M, or M2R."""

        setup = self._engine.setup_config
        if setup is None:
            raise RuntimeError("orchestrator setup has not been installed")
        if setup.protocol_mode != "FULL":
            raise ValueError("run_full_round requires a FULL setup")
        if len(candidates) != len(setup.clients):
            raise ValueError("one candidate is required for every client")
        candidate_by_client: dict[int, ClientFullRoundCandidate] = {}
        for candidate in candidates:
            candidate.validate(setup)
            if candidate.client_id in candidate_by_client:
                raise ValueError(f"duplicate candidate for client {candidate.client_id}")
            candidate_by_client[candidate.client_id] = candidate
        if set(candidate_by_client) != set(setup.client_ids):
            raise ValueError("candidate client set does not match setup")

        survivors = setup.client_ids if survivor_client_ids is None else tuple(survivor_client_ids)
        if (len(set(survivors)) != len(survivors) or not set(survivors).issubset(setup.client_ids)
                or len(survivors) < setup.minimum_cohort_size):
            raise ValueError("invalid FULL survivor set")

        empty_admission = AdmissionPlan(
            setup_id=setup.setup_id,
            round_id=round_id,
            client_results=(),
            accepted_actual_chunk_ids=(),
            cohort_sizes=(),
        )
        empty_admission.validate(setup)
        orchestrator_started = time.perf_counter()
        client_value_seconds: list[float] = []
        self._engine.begin_round(round_id)
        try:
            if before_value_submission is not None:
                before_value_submission()
            for client_id in setup.client_ids:
                client_started = time.perf_counter()
                candidate = candidate_by_client[client_id]
                self._engine.submit_full_parameter_delta(
                    client_id,
                    candidate.parameter_delta,
                )
                self._engine.submit_auxiliary_deltas(
                    client_id,
                    candidate.running_mean_delta,
                    candidate.running_variance_delta,
                )
                client_value_seconds.append(time.perf_counter() - client_started)

            transition_seconds = 0.0
            if survivor_client_ids is not None:
                transition_started = time.perf_counter()
                self._engine.set_round_survivors(survivors)
                transition_seconds = time.perf_counter() - transition_started
            finalize_started = time.perf_counter()
            decrypted = self._engine.finalize_round()
            finalize_seconds = time.perf_counter() - finalize_started
            decrypted.validate(setup)
            if decrypted.round_id != round_id:
                raise ValueError("engine returned a result for the wrong round")
            result = OrchestratedRoundResult(
                admission=empty_admission,
                decrypted=decrypted,
                rejected_actual_chunk_ids_by_client=tuple(
                    (client_id, ()) for client_id in setup.client_ids
                ),
                survivor_client_ids=survivors,
                dropped_client_ids=tuple(i for i in setup.client_ids if i not in survivors),
            )
            if apply_result is not None:
                apply_result(result)
            self._engine.commit_round()
            return OrchestratedRoundResult(
                admission=result.admission,
                decrypted=result.decrypted,
                rejected_actual_chunk_ids_by_client=(
                    result.rejected_actual_chunk_ids_by_client
                ),
                timing=OrchestrationTiming(
                    client_selection_submission_seconds=(),
                    client_value_submission_seconds=tuple(client_value_seconds),
                    survivor_transition_seconds=transition_seconds,
                    finalize_seconds=finalize_seconds,
                    total_seconds=time.perf_counter() - orchestrator_started,
                ),
                upload_admission=result.upload_admission,
                survivor_client_ids=result.survivor_client_ids,
                dropped_client_ids=result.dropped_client_ids,
            )
        except Exception:
            if self._engine.phase not in {
                None,
                RoundPhase.COMMITTED,
                RoundPhase.ABORTED,
            }:
                self._engine.abort_round()
            raise
