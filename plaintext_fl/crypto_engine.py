"""Engine-neutral boundary for the v17 federated cryptographic protocol.

The Python federated-learning layer depends only on the abstractions in this
module.  A plaintext reference implementation lives in ``plaintext_engine``;
the future pybind11 adapter must implement the same interface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .protocol_contract import (
    ClientAdmissionResult,
    LayoutKind,
    RegisteredClientWeight,
    RoundPhase,
    TensorLayout,
    WEIGHT_DENOMINATOR,
    quantize_registered_client_weights,
)


@dataclass(frozen=True)
class CryptoEngineSetup:
    """Immutable setup shared by every engine implementation."""

    setup_id: bytes
    parameter_layout: TensorLayout
    running_mean_layout: TensorLayout
    running_variance_layout: TensorLayout
    clients: tuple[RegisteredClientWeight, ...]
    chunk_size: int
    minimum_cohort_size: int
    weight_denominator: int = WEIGHT_DENOMINATOR
    protocol_mode: str = "SPARSE"

    def validate(self) -> None:
        if len(self.setup_id) != 16:
            raise ValueError("setup_id must contain 16 bytes")

        expected_layout_kinds = (
            (self.parameter_layout, LayoutKind.PARAMETER),
            (self.running_mean_layout, LayoutKind.RUNNING_MEAN),
            (self.running_variance_layout, LayoutKind.RUNNING_VARIANCE),
        )
        model_names: set[str] = set()
        for layout, expected_kind in expected_layout_kinds:
            layout.validate()
            if layout.kind is not expected_kind:
                raise ValueError(
                    f"expected {expected_kind.value} layout, "
                    f"got {layout.kind.value}"
                )
            model_names.add(layout.model_name)

        if len(model_names) != 1:
            raise ValueError("all layouts must describe the same model")
        if not self.clients:
            raise ValueError("at least one registered client is required")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 1 <= self.minimum_cohort_size <= len(self.clients):
            raise ValueError(
                "minimum_cohort_size must be between one and client count"
            )
        if self.weight_denominator != WEIGHT_DENOMINATOR:
            raise ValueError(
                f"v17 requires weight_denominator={WEIGHT_DENOMINATOR}"
            )
        if self.protocol_mode not in {"SPARSE", "FULL"}:
            raise ValueError("protocol_mode must be SPARSE or FULL")

        client_ids: set[int] = set()
        for client in self.clients:
            client.validate()
            if client.client_id in client_ids:
                raise ValueError(f"duplicate client_id: {client.client_id}")
            client_ids.add(client.client_id)

        # This also verifies the v17 alpha_i >= 1 requirement.
        quantize_registered_client_weights(
            self.clients,
            denominator=self.weight_denominator,
        )

    @property
    def client_ids(self) -> tuple[int, ...]:
        return tuple(client.client_id for client in self.clients)

    @property
    def alphas(self) -> tuple[int, ...]:
        self.validate()
        return quantize_registered_client_weights(
            self.clients,
            denominator=self.weight_denominator,
        )

    @property
    def parameter_chunk_count(self) -> int:
        self.validate()
        return math.ceil(self.parameter_layout.total_numel / self.chunk_size)


@dataclass(frozen=True)
class AdmissionPlan:
    """Minimum-cohort decision produced before any values are submitted."""

    setup_id: bytes
    round_id: int
    client_results: tuple[ClientAdmissionResult, ...]
    accepted_actual_chunk_ids: tuple[int, ...]
    cohort_sizes: tuple[int, ...]

    def validate(self, setup: CryptoEngineSetup) -> None:
        setup.validate()
        if self.setup_id != setup.setup_id:
            raise ValueError("admission setup_id mismatch")
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if setup.protocol_mode == "FULL":
            if self.client_results or self.accepted_actual_chunk_ids or self.cohort_sizes:
                raise ValueError("FULL admission metadata must be empty")
            return
        if len(self.cohort_sizes) != setup.parameter_chunk_count:
            raise ValueError("cohort size count does not match parameter layout")
        if any(
            not isinstance(size, int) or not 0 <= size <= len(setup.clients)
            for size in self.cohort_sizes
        ):
            raise ValueError("cohort sizes must be valid client counts")
        expected_accepted = tuple(
            chunk_id
            for chunk_id, size in enumerate(self.cohort_sizes)
            if size >= setup.minimum_cohort_size
        )
        if self.accepted_actual_chunk_ids != expected_accepted:
            raise ValueError("accepted chunk IDs do not match cohort sizes")

        result_client_ids: list[int] = []
        for result in self.client_results:
            result.validate()
            if result.setup_id != setup.setup_id or result.round_id != self.round_id:
                raise ValueError("client admission metadata mismatch")
            result_client_ids.append(result.client_id)
        if tuple(result_client_ids) != setup.client_ids:
            raise ValueError("client admission order does not match setup")

    def for_client(self, client_id: int) -> ClientAdmissionResult:
        matches = tuple(
            result
            for result in self.client_results
            if result.client_id == client_id
        )
        if len(matches) != 1:
            raise KeyError(f"no unique admission result for client {client_id}")
        return matches[0]


@dataclass(frozen=True)
class DecryptedRoundResult:
    """Engine output after encryption, aggregation, and joint decryption.

    Arrays cross the runtime boundary as contiguous CPU float32 buffers.  The
    plaintext engine computes with float64 internally and returns the same
    public representation expected from the future pybind11 engine.
    """

    setup_id: bytes
    round_id: int
    accepted_actual_chunk_ids: tuple[int, ...]
    cohort_sizes: tuple[int, ...]
    parameter_delta: np.ndarray
    running_mean_delta: np.ndarray
    running_variance_delta: np.ndarray
    phase_audit: dict | None = None

    def validate(self, setup: CryptoEngineSetup) -> None:
        setup.validate()
        if self.setup_id != setup.setup_id:
            raise ValueError("round result setup_id mismatch")
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if setup.protocol_mode == "FULL":
            if self.accepted_actual_chunk_ids or self.cohort_sizes:
                raise ValueError("FULL decrypted result must not contain cohorts")
        elif len(self.cohort_sizes) != setup.parameter_chunk_count:
            raise ValueError("cohort size count does not match parameter layout")
        if tuple(sorted(set(self.accepted_actual_chunk_ids))) != (
            self.accepted_actual_chunk_ids
        ):
            raise ValueError("accepted chunk IDs must be sorted and unique")

        expected_sizes = (
            ("parameter_delta", self.parameter_delta, setup.parameter_layout.total_numel),
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
        for name, values, expected_size in expected_sizes:
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


class PostRequestDropoutError(RuntimeError):
    """The final decryption request was sent but a survivor disappeared."""


class FederatedCryptoEngine(ABC):
    """Stateful engine port owned by the Python round orchestrator."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable backend name for logs and checkpoints."""

    @property
    @abstractmethod
    def setup_config(self) -> CryptoEngineSetup | None:
        """Return the installed setup, or ``None`` before setup."""

    @property
    @abstractmethod
    def phase(self) -> RoundPhase | None:
        """Return the current round phase."""

    @abstractmethod
    def setup(self, config: CryptoEngineSetup) -> None:
        """Install immutable setup material and registered client weights."""

    @abstractmethod
    def begin_round(
        self, round_id: int, *, logical_round_index: int | None = None
    ) -> None:
        """Start selection collection for one round."""

    @abstractmethod
    def submit_selection(
        self,
        client_id: int,
        selected_actual_chunk_ids: Sequence[int],
    ) -> None:
        """Submit only chunk IDs; update values must not be present yet."""

    @abstractmethod
    def finalize_admission(self) -> AdmissionPlan:
        """Apply minimum cohort and return per-client accepted IDs."""

    @abstractmethod
    def submit_accepted_values(
        self,
        client_id: int,
        accepted_actual_chunk_ids: Sequence[int],
        values: np.ndarray,
    ) -> None:
        """Submit float32 values for exactly the approved IDs."""

    def submit_full_parameter_delta(
        self,
        client_id: int,
        parameter_delta: np.ndarray,
    ) -> None:
        """Submit one complete float32 parameter delta in FULL mode."""

        raise NotImplementedError("this engine does not implement FULL mode")

    @abstractmethod
    def submit_auxiliary_deltas(
        self,
        client_id: int,
        running_mean_delta: np.ndarray,
        running_variance_delta: np.ndarray,
    ) -> None:
        """Submit the two dense BatchNorm delta streams."""

    def set_round_survivors(
        self,
        survivor_client_ids: Sequence[int],
    ) -> AdmissionPlan:
        """Rebuild an SPARSE round target after every upload has arrived.

        Implementations must keep the registered integer weights and fixed
        denominator unchanged.  Backends that do not support post-upload
        dropout may retain this default rejection.
        """

        raise NotImplementedError(
            "this engine does not implement post-upload dropout"
        )

    @abstractmethod
    def finalize_round(
        self, *, post_request_dropouts: Sequence[int] = ()
    ) -> DecryptedRoundResult:
        """Run the backend pipeline and return the jointly decrypted deltas."""

    @abstractmethod
    def commit_round(self) -> None:
        """Commit after Python successfully applies the decrypted result."""

    @abstractmethod
    def abort_round(self) -> None:
        """Abort the active non-terminal round."""
