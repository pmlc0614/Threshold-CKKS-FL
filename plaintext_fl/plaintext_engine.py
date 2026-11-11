"""Plaintext reference backend for the engine-neutral v17 interface."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .crypto_engine import (
    AdmissionPlan,
    CryptoEngineSetup,
    DecryptedRoundResult,
    FederatedCryptoEngine,
)
from .protocol_contract import (
    ClientAdmissionResult,
    RoundPhase,
    is_allowed_round_transition,
)


_TERMINAL_PHASES = {RoundPhase.COMMITTED, RoundPhase.ABORTED}


def _float32_vector(values: np.ndarray, expected_size: int, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if values.dtype != np.float32:
        raise TypeError(f"{name} must use float32")
    if values.ndim != 1 or values.size != expected_size:
        raise ValueError(f"{name} has an invalid shape")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return values.copy()


class PlaintextCryptoEngine(FederatedCryptoEngine):
    """Deterministic reference engine with encrypted-engine semantics.

    This implementation intentionally follows the same two-phase API as the
    C++ backend.  It is not a security mechanism; it is the executable oracle
    used to test orchestration and future pybind11 parity.
    """

    def __init__(self, *, experimental_dynamic_cohort: bool = False) -> None:
        self.experimental_dynamic_cohort = experimental_dynamic_cohort
        self._setup: CryptoEngineSetup | None = None
        self._phase: RoundPhase | None = None
        self._round_id: int | None = None
        self._last_round_id: int | None = None
        self._logical_round_index: int | None = None
        self._last_closed_logical_round: int | None = None
        self._selections: dict[int, tuple[int, ...]] = {}
        self._admission: AdmissionPlan | None = None
        self._upload_admission: AdmissionPlan | None = None
        self._survivor_client_ids: tuple[int, ...] | None = None
        self._values: dict[int, np.ndarray] = {}
        self._full_values: dict[int, np.ndarray] = {}
        self._running_means: dict[int, np.ndarray] = {}
        self._running_variances: dict[int, np.ndarray] = {}

    @property
    def engine_name(self) -> str:
        return "plaintext-reference-v17"

    @property
    def setup_config(self) -> CryptoEngineSetup | None:
        return self._setup

    @property
    def phase(self) -> RoundPhase | None:
        return self._phase

    def setup(self, config: CryptoEngineSetup) -> None:
        config.validate()
        if self._phase is not None and self._phase not in _TERMINAL_PHASES:
            raise RuntimeError("cannot replace setup during an active round")
        self._setup = config
        self._phase = None
        self._round_id = None
        self._last_round_id = None
        self._logical_round_index = None
        self._last_closed_logical_round = None
        self._clear_round_payloads()

    def begin_round(self, round_id: int, *, logical_round_index: int | None = None) -> None:
        self._require_setup()
        if self._phase is RoundPhase.ABORTED and not self.experimental_dynamic_cohort:
            raise RuntimeError("fixed-cohort execution ended: retries are forbidden")
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self._phase is not None and self._phase not in _TERMINAL_PHASES:
            raise RuntimeError("another round is already active")
        if self._last_round_id is not None and round_id <= self._last_round_id:
            raise ValueError("round_id must increase monotonically")

        logical = round_id if logical_round_index is None else logical_round_index
        if logical < 0 or (self._last_closed_logical_round is not None and
                           logical != self._last_closed_logical_round + 1):
            raise ValueError("closed logical round cannot be retried; start the next round")
        self._logical_round_index = logical

        self._clear_round_payloads()
        self._round_id = int(round_id)
        self._phase = RoundPhase.CREATED
        self._transition(
            RoundPhase.SELECTION_COLLECTING
            if self._require_setup().protocol_mode == "SPARSE"
            else RoundPhase.VALUES_COLLECTING
        )

    def submit_selection(
        self,
        client_id: int,
        selected_actual_chunk_ids: Sequence[int],
    ) -> None:
        config = self._require_setup()
        if config.protocol_mode != "SPARSE":
            raise RuntimeError("FULL rounds do not accept sparse selections")
        self._require_phase(RoundPhase.SELECTION_COLLECTING)
        self._require_registered_client(client_id)
        if client_id in self._selections:
            raise ValueError(f"duplicate selection for client {client_id}")

        chunk_ids = tuple(int(chunk_id) for chunk_id in selected_actual_chunk_ids)
        if not chunk_ids:
            raise ValueError("selected chunk IDs must not be empty")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("selected chunk IDs must be unique")
        if any(
            chunk_id < 0 or chunk_id >= config.parameter_chunk_count
            for chunk_id in chunk_ids
        ):
            raise ValueError("selected chunk ID is outside the parameter layout")
        self._selections[client_id] = chunk_ids

    def finalize_admission(self) -> AdmissionPlan:
        config = self._require_setup()
        if config.protocol_mode != "SPARSE":
            raise RuntimeError("FULL rounds do not perform sparse admission")
        self._require_phase(RoundPhase.SELECTION_COLLECTING)
        if set(self._selections) != set(config.client_ids):
            if not self.experimental_dynamic_cohort:
                self.abort_round()
            raise RuntimeError("selection is required from every registered client")

        cohort_sizes = [0] * config.parameter_chunk_count
        for chunk_ids in self._selections.values():
            for chunk_id in chunk_ids:
                cohort_sizes[chunk_id] += 1

        accepted = tuple(
            chunk_id
            for chunk_id, cohort_size in enumerate(cohort_sizes)
            if cohort_size >= config.minimum_cohort_size
        )
        accepted_set = set(accepted)
        results = tuple(
            ClientAdmissionResult(
                setup_id=config.setup_id,
                round_id=self._require_round_id(),
                client_id=client_id,
                accepted_actual_chunk_ids=tuple(
                    chunk_id
                    for chunk_id in self._selections[client_id]
                    if chunk_id in accepted_set
                ),
            )
            for client_id in config.client_ids
        )
        for result in results:
            result.validate()

        self._admission = AdmissionPlan(
            setup_id=config.setup_id,
            round_id=self._require_round_id(),
            client_results=results,
            accepted_actual_chunk_ids=accepted,
            cohort_sizes=tuple(cohort_sizes),
        )
        self._admission.validate(config)
        self._transition(RoundPhase.ADMISSION_FINALIZED)
        return self._admission

    def submit_accepted_values(
        self,
        client_id: int,
        accepted_actual_chunk_ids: Sequence[int],
        values: np.ndarray,
    ) -> None:
        config = self._require_setup()
        if config.protocol_mode != "SPARSE":
            raise RuntimeError("FULL rounds require a dense parameter delta")
        self._enter_values_phase()
        self._require_registered_client(client_id)
        if client_id in self._values:
            raise ValueError(f"duplicate accepted values for client {client_id}")

        expected_ids = self._require_admission().for_client(
            client_id
        ).accepted_actual_chunk_ids
        actual_ids = tuple(int(chunk_id) for chunk_id in accepted_actual_chunk_ids)
        if actual_ids != expected_ids:
            raise ValueError("submitted chunk IDs do not match the admission result")
        if not isinstance(values, np.ndarray):
            raise TypeError("accepted values must be a numpy array")
        if values.dtype != np.float32:
            raise TypeError("accepted values must use float32")
        if values.shape != (len(expected_ids), config.chunk_size):
            raise ValueError("accepted values have an invalid shape")
        if not values.flags.c_contiguous:
            raise ValueError("accepted values must be C-contiguous")
        if not np.isfinite(values).all():
            raise ValueError("accepted values contain NaN or infinity")
        self._values[client_id] = values.copy()

    def submit_full_parameter_delta(
        self,
        client_id: int,
        parameter_delta: np.ndarray,
    ) -> None:
        config = self._require_setup()
        if config.protocol_mode != "FULL":
            raise RuntimeError("SPARSE rounds do not accept FULL parameter deltas")
        self._require_phase(RoundPhase.VALUES_COLLECTING)
        self._require_registered_client(client_id)
        if client_id in self._full_values:
            raise ValueError(f"duplicate FULL values for client {client_id}")
        self._full_values[client_id] = _float32_vector(
            parameter_delta,
            config.parameter_layout.total_numel,
            "parameter_delta",
        )

    def submit_auxiliary_deltas(
        self,
        client_id: int,
        running_mean_delta: np.ndarray,
        running_variance_delta: np.ndarray,
    ) -> None:
        config = self._require_setup()
        self._enter_values_phase()
        self._require_registered_client(client_id)
        if client_id in self._running_means:
            raise ValueError(f"duplicate auxiliary deltas for client {client_id}")

        validated_mean = _float32_vector(
            running_mean_delta,
            config.running_mean_layout.total_numel,
            "running_mean_delta",
        )
        validated_variance = _float32_vector(
            running_variance_delta,
            config.running_variance_layout.total_numel,
            "running_variance_delta",
        )
        self._running_means[client_id] = validated_mean
        self._running_variances[client_id] = validated_variance

    def set_round_survivors(
        self,
        survivor_client_ids: Sequence[int],
    ) -> AdmissionPlan:
        config = self._require_setup()
        if not self.experimental_dynamic_cohort:
            if self._phase is not None and self._phase not in _TERMINAL_PHASES:
                self.abort_round()
            raise RuntimeError("fixed-cohort execution forbids survivor reconfiguration")
        if config.protocol_mode != "SPARSE":
            raise RuntimeError("post-upload dropout requires SPARSE mode")
        self._require_phase(RoundPhase.VALUES_COLLECTING)
        expected_clients = set(config.client_ids)
        if (
            set(self._values) != expected_clients
            or set(self._running_means) != expected_clients
            or set(self._running_variances) != expected_clients
        ):
            raise RuntimeError(
                "survivors can be fixed only after every model/aux upload"
            )
        survivors = tuple(int(value) for value in survivor_client_ids)
        if len(set(survivors)) != len(survivors):
            raise ValueError("survivor client IDs must be unique")
        if not set(survivors).issubset(expected_clients):
            raise ValueError("survivor client IDs must be registered")
        if len(survivors) < config.minimum_cohort_size:
            raise ValueError("survivor count is below minimum cohort size")
        if self._survivor_client_ids is not None:
            raise RuntimeError("round survivors have already been fixed")

        survivor_set = set(survivors)
        cohort_sizes = [0] * config.parameter_chunk_count
        for client_id in config.client_ids:
            if client_id not in survivor_set:
                continue
            for chunk_id in self._selections[client_id]:
                cohort_sizes[chunk_id] += 1
        accepted = tuple(
            chunk_id
            for chunk_id, cohort_size in enumerate(cohort_sizes)
            if cohort_size >= config.minimum_cohort_size
        )
        accepted_set = set(accepted)
        results = tuple(
            ClientAdmissionResult(
                setup_id=config.setup_id,
                round_id=self._require_round_id(),
                client_id=client_id,
                accepted_actual_chunk_ids=(
                    tuple(
                        chunk_id
                        for chunk_id in self._selections[client_id]
                        if chunk_id in accepted_set
                    )
                    if client_id in survivor_set
                    else ()
                ),
            )
            for client_id in config.client_ids
        )
        self._upload_admission = self._require_admission()
        self._admission = AdmissionPlan(
            setup_id=config.setup_id,
            round_id=self._require_round_id(),
            client_results=results,
            accepted_actual_chunk_ids=accepted,
            cohort_sizes=tuple(cohort_sizes),
        )
        self._admission.validate(config)
        self._survivor_client_ids = tuple(
            client_id
            for client_id in config.client_ids
            if client_id in survivor_set
        )
        return self._admission

    def finalize_round(self) -> DecryptedRoundResult:
        config = self._require_setup()
        self._enter_values_phase()
        expected_clients = set(config.client_ids)
        model_values = (
            self._values
            if config.protocol_mode == "SPARSE"
            else self._full_values
        )
        if set(model_values) != expected_clients:
            if not self.experimental_dynamic_cohort:
                self.abort_round()
            raise RuntimeError("model values are required from every client")
        if set(self._running_means) != expected_clients:
            if not self.experimental_dynamic_cohort:
                self.abort_round()
            raise RuntimeError("auxiliary deltas are required from every client")

        self._transition(RoundPhase.ENCRYPTED)
        parameter_chunks = np.zeros(
            (config.parameter_chunk_count, config.chunk_size),
            dtype=np.float64,
        )
        running_mean = np.zeros(
            config.running_mean_layout.total_numel,
            dtype=np.float64,
        )
        running_variance = np.zeros(
            config.running_variance_layout.total_numel,
            dtype=np.float64,
        )

        admission = (
            self._require_admission()
            if config.protocol_mode == "SPARSE"
            else AdmissionPlan(
                setup_id=config.setup_id,
                round_id=self._require_round_id(),
                client_results=(),
                accepted_actual_chunk_ids=(),
                cohort_sizes=(),
            )
        )
        admission.validate(config)
        full_parameter = np.zeros(
            config.parameter_layout.total_numel,
            dtype=np.float64,
        )
        survivor_set = set(
            config.client_ids
            if self._survivor_client_ids is None
            else self._survivor_client_ids
        )
        upload_admission = self._upload_admission or admission
        for client, alpha in zip(config.clients, config.alphas):
            client_id = client.client_id
            if client_id not in survivor_set:
                continue
            if config.protocol_mode == "SPARSE":
                accepted_ids = admission.for_client(
                    client_id
                ).accepted_actual_chunk_ids
                uploaded_ids = upload_admission.for_client(
                    client_id
                ).accepted_actual_chunk_ids
                uploaded_row = {
                    chunk_id: row for row, chunk_id in enumerate(uploaded_ids)
                }
                for chunk_id in accepted_ids:
                    parameter_chunks[chunk_id] += (
                        alpha * self._values[client_id][uploaded_row[chunk_id]]
                    )
            else:
                full_parameter += alpha * self._full_values[client_id]
            running_mean += alpha * self._running_means[client_id]
            running_variance += alpha * self._running_variances[client_id]

        denominator = float(config.weight_denominator)
        parameter_delta = (
            (
                parameter_chunks.reshape(-1)[
                    : config.parameter_layout.total_numel
                ]
                if config.protocol_mode == "SPARSE"
                else full_parameter
            )
            / denominator
        ).astype(np.float32, copy=False)
        running_mean_delta = (running_mean / denominator).astype(
            np.float32,
            copy=False,
        )
        running_variance_delta = (running_variance / denominator).astype(
            np.float32,
            copy=False,
        )

        self._transition(RoundPhase.DECRYPTING)
        result = DecryptedRoundResult(
            setup_id=config.setup_id,
            round_id=self._require_round_id(),
            accepted_actual_chunk_ids=admission.accepted_actual_chunk_ids,
            cohort_sizes=admission.cohort_sizes,
            parameter_delta=np.ascontiguousarray(parameter_delta),
            running_mean_delta=np.ascontiguousarray(running_mean_delta),
            running_variance_delta=np.ascontiguousarray(running_variance_delta),
        )
        result.validate(config)
        return result

    def commit_round(self) -> None:
        self._require_phase(RoundPhase.DECRYPTING)
        self._transition(RoundPhase.COMMITTED)
        self._last_round_id = self._require_round_id()
        self._last_closed_logical_round = self._logical_round_index

    def abort_round(self) -> None:
        if self._phase is None:
            raise RuntimeError("no round is active")
        if self._phase in _TERMINAL_PHASES:
            raise RuntimeError("terminal rounds cannot be aborted")
        self._transition(RoundPhase.ABORTED)
        self._last_round_id = self._require_round_id()
        self._last_closed_logical_round = self._logical_round_index

    def _require_setup(self) -> CryptoEngineSetup:
        if self._setup is None:
            raise RuntimeError("crypto engine setup has not been installed")
        return self._setup

    def _require_round_id(self) -> int:
        if self._round_id is None:
            raise RuntimeError("no round is active")
        return self._round_id

    def _require_admission(self) -> AdmissionPlan:
        if self._admission is None:
            raise RuntimeError("admission has not been finalized")
        return self._admission

    def _require_phase(self, expected: RoundPhase) -> None:
        if self._phase is not expected:
            actual = self._phase.value if self._phase is not None else "none"
            raise RuntimeError(
                f"operation requires phase {expected.value}, got {actual}"
            )

    def _require_registered_client(self, client_id: int) -> None:
        if client_id not in self._require_setup().client_ids:
            raise ValueError(f"client {client_id} is not registered")

    def _enter_values_phase(self) -> None:
        if self._phase is RoundPhase.ADMISSION_FINALIZED:
            self._transition(RoundPhase.VALUES_COLLECTING)
        else:
            self._require_phase(RoundPhase.VALUES_COLLECTING)

    def _transition(self, following: RoundPhase) -> None:
        if self._phase is None or not is_allowed_round_transition(
            self._phase,
            following,
        ):
            current = self._phase.value if self._phase is not None else "none"
            raise RuntimeError(f"invalid round transition: {current} -> {following.value}")
        self._phase = following

    def _clear_round_payloads(self) -> None:
        self._selections = {}
        self._admission = None
        self._upload_admission = None
        self._survivor_client_ids = None
        self._values = {}
        self._full_values = {}
        self._running_means = {}
        self._running_variances = {}
