"""pybind11-backed implementation of the protocol-v17 crypto engine port."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any, Sequence

import numpy as np

from .crypto_engine import (
    AdmissionPlan,
    CryptoEngineSetup,
    DecryptedRoundResult,
    FederatedCryptoEngine,
    PostRequestDropoutError,
)
from .wire_contract import canonical_parameter_descriptor, serialize_public_setup, parse_public_setup

from .protocol_contract import (
    BINDING_ABI_VERSION,
    PROTOCOL_VERSION,
    ClientAdmissionResult,
    RoundPhase,
    TensorLayout,
)


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


def _parse_stage_timings(raw_timings: Any) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    raw_timings = raw_timings or {}
    for name in _CRYPTO_STAGE_NAMES:
        raw = raw_timings.get(name, {})
        wall_seconds = float(raw.get("wall_seconds", 0.0))
        work_seconds = float(raw.get("work_seconds", 0.0))
        executor_wait = float(raw.get("executor_wait_seconds", 0.0))
        task_count = int(raw.get("task_count", 0))
        if (
            not math.isfinite(wall_seconds)
            or wall_seconds < 0.0
            or not math.isfinite(work_seconds)
            or work_seconds < 0.0
            or not math.isfinite(executor_wait)
            or executor_wait < 0.0
            or task_count < 0
        ):
            raise RuntimeError("native crypto stage timing is invalid")
        result[name] = {
            "wall_seconds": wall_seconds,
            "work_seconds": work_seconds,
            "executor_wait_seconds": executor_wait,
            "elapsed_including_executor_wait_seconds": wall_seconds + executor_wait,
            "task_count": task_count,
        }
    return result


def _parse_client_stage_timings(raw_timings: Any) -> tuple[dict, ...]:
    return tuple(_parse_stage_timings(raw) for raw in (raw_timings or ()))


class NativeBindingUnavailableError(ImportError):
    """Raised when the optional C++ extension cannot be loaded."""


@dataclass(frozen=True)
class NativeCryptoEngineOptions:
    """C++ protocol/profile overrides applied before cryptographic setup.

    The production default is the CIFAR-10/ResNet-20 sparse profile.  The
    ``smoke`` profile and its overrides exist for fast boundary tests.
    """

    profile: str = "resnet20-cifar10"
    protocol_version: str = "v17"
    experimental_dynamic_cohort: bool = False
    dropout_target_cap: int = 128
    collusion_bound: int = 0
    protocol_mode: str = "SPARSE"
    ring_dimension: int | None = None
    mlwe_rank: int | None = None
    q_modulus_bits: tuple[int, ...] | None = None
    p_modulus_bits: tuple[int, ...] | None = None
    hybrid_large_digits: int | None = None
    maximum_qp_bits: int | None = None
    model_scaling_factor: int | None = None
    aux_scaling_factor: int | None = None
    precision_bits: int | None = None
    aux_precision_bits: int | None = None
    selection_ratio: float | None = None
    selected_chunks_per_client: int | None = None
    update_bound: float | None = None
    aux_update_bound: float | None = None
    compression_c1: int | None = None
    compression_ca: int | None = None
    compression_c2: int | None = None
    compression_cd: int | None = None
    aux_compression_c1: int | None = None
    aux_compression_ca: int | None = None
    aux_compression_c2: int | None = None
    aux_compression_cd: int | None = None
    smudging_sigma: float | None = None
    aux_smudging_sigma: float | None = None
    model_delivery_scale: float | None = None
    aux_delivery_scale: float | None = None
    use_sparse_compression: bool = True
    use_auxiliary_compression: bool = True
    measure_pre_smudging_noise: bool = False
    crypto_threads: int = 1
    master_seed: bytes | None = None

    def validate(self) -> None:
        if self.profile not in {"resnet20-cifar10", "smoke"}:
            raise ValueError("unsupported native crypto profile")
        if not isinstance(self.experimental_dynamic_cohort, bool):
            raise ValueError("experimental_dynamic_cohort must be boolean")
        if self.protocol_version not in {"v17"}:
            raise ValueError("protocol_version must be v17")
        if self.protocol_mode not in {"SPARSE", "FULL"}:
            raise ValueError("protocol_mode must be SPARSE or FULL")
        if (
            self.selection_ratio is not None
            and self.selected_chunks_per_client is not None
        ):
            raise ValueError(
                "set only selection_ratio or selected_chunks_per_client"
            )
        for scale in (self.model_scaling_factor, self.aux_scaling_factor):
            if scale is not None and (isinstance(scale, bool) or not isinstance(scale, int)
                                      or not 0 < scale < 2**256):
                raise ValueError("scaling factors must be integers in [1, 2^256)")
        positive_integers = (
            ("ring_dimension", self.ring_dimension),
            ("mlwe_rank", self.mlwe_rank),
            ("hybrid_large_digits", self.hybrid_large_digits),
            ("maximum_qp_bits", self.maximum_qp_bits),
            ("model_scaling_factor", self.model_scaling_factor),
            ("aux_scaling_factor", self.aux_scaling_factor),
            ("precision_bits", self.precision_bits),
            ("aux_precision_bits", self.aux_precision_bits),
            ("selected_chunks_per_client", self.selected_chunks_per_client),
            ("compression_c1", self.compression_c1),
            ("compression_ca", self.compression_ca),
            ("compression_c2", self.compression_c2),
            ("compression_cd", self.compression_cd),
            ("aux_compression_c1", self.aux_compression_c1),
            ("aux_compression_ca", self.aux_compression_ca),
            ("aux_compression_c2", self.aux_compression_c2),
            ("aux_compression_cd", self.aux_compression_cd),
            ("crypto_threads", self.crypto_threads),
        )
        for name, value in positive_integers:
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.selection_ratio is not None and not (
            math.isfinite(self.selection_ratio)
            and 0.0 < self.selection_ratio <= 1.0
        ):
            raise ValueError("selection_ratio must be in (0, 1]")
        for name, values in (
            ("q_modulus_bits", self.q_modulus_bits),
            ("p_modulus_bits", self.p_modulus_bits),
        ):
            if values is not None and (not values or any(value <= 0 for value in values)):
                raise ValueError(f"{name} must contain positive values")
        for name, value in (
            ("update_bound", self.update_bound),
            ("aux_update_bound", self.aux_update_bound),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("smudging_sigma", self.smudging_sigma),
            ("aux_smudging_sigma", self.aux_smudging_sigma),
        ):
            if value is not None and (
                not math.isfinite(value) or value < 0.0
            ):
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("model_delivery_scale", self.model_delivery_scale),
            ("aux_delivery_scale", self.aux_delivery_scale),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")
        for name, value in positive_integers:
            if "compression_" in name and value is not None and value > 63:
                raise ValueError(f"{name} must be at most 63")
        if self.master_seed is not None and len(self.master_seed) != 32:
            raise ValueError("master_seed must contain 32 bytes")
        for name in ("dropout_target_cap", "collusion_bound"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.crypto_threads > 256:
            raise ValueError("crypto_threads must be at most 256")

    def to_binding_dict(self) -> dict[str, Any]:
        self.validate()
        result: dict[str, Any] = {
            "profile": self.profile,
            "protocol_version": self.protocol_version,
            "experimental_dynamic_cohort": self.experimental_dynamic_cohort,
            "dropout_target_cap": self.dropout_target_cap,
            "collusion_bound": self.collusion_bound,
            "protocol_mode": self.protocol_mode,
            "use_sparse_compression": self.use_sparse_compression,
            "use_auxiliary_compression": self.use_auxiliary_compression,
            "measure_pre_smudging_noise": self.measure_pre_smudging_noise,
            "crypto_threads": self.crypto_threads,
        }
        for name in (
            "ring_dimension",
            "mlwe_rank",
            "q_modulus_bits",
            "p_modulus_bits",
            "hybrid_large_digits",
            "maximum_qp_bits",
            "model_scaling_factor",
            "aux_scaling_factor",
            "precision_bits",
            "aux_precision_bits",
            "selection_ratio",
            "selected_chunks_per_client",
            "update_bound",
            "aux_update_bound",
            "compression_c1",
            "compression_ca",
            "compression_c2",
            "compression_cd",
            "aux_compression_c1",
            "aux_compression_ca",
            "aux_compression_c2",
            "aux_compression_cd",
            "smudging_sigma",
            "aux_smudging_sigma",
            "model_delivery_scale",
            "aux_delivery_scale",
            "master_seed",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    def validate_effective_protocol(self, effective: dict[str, Any]) -> None:
        """Verify that C++ applied every explicitly supplied numeric option."""

        if effective.get("protocol_mode") != self.protocol_mode:
            raise RuntimeError("native setup effective protocol mismatch: protocol_mode")
        if effective.get("experimental_dynamic_cohort") is not self.experimental_dynamic_cohort:
            raise RuntimeError("native execution policy mismatch")
        if effective.get("protocol_version") != self.protocol_version:
            raise RuntimeError("native setup effective protocol mismatch: protocol_version")
        if self.protocol_version == "v17":
            expected_codec = (
                "CHUNK-BITMAP-MSB0-V1"
                if self.protocol_mode == "SPARSE"
                else "NONE"
            )
            if effective.get("support_codec") != expected_codec:
                raise RuntimeError(
                    "native setup effective protocol mismatch: support_codec"
                )
            expected_code = 1 if self.protocol_mode == "SPARSE" else 0
            if effective.get("support_codec_code") != expected_code:
                raise RuntimeError(
                    "native setup effective protocol mismatch: support_codec_code"
                )
        for name, expected in (
            ("experimental_dynamic_cohort", self.experimental_dynamic_cohort),
            ("dropout_attempt_cap", 1),
            ("dropout_target_cap", self.dropout_target_cap),
            ("collusion_bound", self.collusion_bound),
        ):
            if effective.get(name) != expected:
                raise RuntimeError(f"native setup effective protocol mismatch: {name}")
        sequence_names = {"q_modulus_bits", "p_modulus_bits"}
        float_names = {
            "selection_ratio",
            "update_bound",
            "aux_update_bound",
            "smudging_sigma",
            "aux_smudging_sigma",
            "model_delivery_scale",
            "aux_delivery_scale",
        }
        for name in (
            "ring_dimension",
            "mlwe_rank",
            "q_modulus_bits",
            "p_modulus_bits",
            "hybrid_large_digits",
            "maximum_qp_bits",
            "model_scaling_factor",
            "aux_scaling_factor",
            "precision_bits",
            "aux_precision_bits",
            "selection_ratio",
            "update_bound",
            "aux_update_bound",
            "compression_c1",
            "compression_ca",
            "compression_c2",
            "compression_cd",
            "aux_compression_c1",
            "aux_compression_ca",
            "aux_compression_c2",
            "aux_compression_cd",
            "smudging_sigma",
            "aux_smudging_sigma",
            "model_delivery_scale",
            "aux_delivery_scale",
        ):
            expected = getattr(self, name)
            if expected is None:
                continue
            if name not in effective:
                raise RuntimeError(
                    f"native setup omitted effective protocol field: {name}"
                )
            actual = effective[name]
            if name in sequence_names:
                matches = tuple(int(value) for value in actual) == tuple(expected)
            elif name in float_names:
                matches = math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            else:
                matches = int(actual) == int(expected)
            if not matches:
                raise RuntimeError(
                    f"native setup effective protocol mismatch: {name}"
                )


def _load_native_module() -> ModuleType:
    try:
        module = importlib.import_module("ciphertext_fl_cpp")
    except (ImportError, OSError) as error:
        raise NativeBindingUnavailableError(
            "ciphertext_fl_cpp is unavailable; build ciphertext-FL-cpp with "
            "-DBUILD_FEDERATED_PYTHON_BINDINGS=ON and add build/python "
            "to PYTHONPATH"
        ) from error

    if getattr(module, "BINDING_ABI_VERSION", None) != BINDING_ABI_VERSION:
        raise NativeBindingUnavailableError("native binding ABI version mismatch")
    if getattr(module, "PROTOCOL_VERSION", None) != PROTOCOL_VERSION:
        raise NativeBindingUnavailableError("native protocol version mismatch")
    return module


def native_binding_available() -> bool:
    """Return whether the ABI-compatible C++ extension can be imported."""

    try:
        _load_native_module()
    except NativeBindingUnavailableError:
        return False
    return True


def _layout_payload(layout: TensorLayout) -> dict[str, Any]:
    layout.validate()
    return {
        "kind": layout.kind.value,
        "model_name": layout.model_name,
        "entries": [
            {
                "name": entry.name,
                "shape": entry.shape,
                "offset": entry.offset,
                "numel": entry.numel,
                "dtype": entry.dtype,
            }
            for entry in layout.entries
        ],
    }


def _require_float32_array(
    values: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if values.dtype != np.float32:
        raise TypeError(f"{name} must use float32")
    if values.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")


class NativeCryptoEngine(FederatedCryptoEngine):
    """Python adapter for ``ciphertext_fl_cpp.NativeCryptoEngine``."""

    def __init__(
        self,
        options: NativeCryptoEngineOptions | None = None,
        *,
        native_module: ModuleType | None = None,
    ) -> None:
        self._options = options or NativeCryptoEngineOptions()
        self._options.validate()
        module = native_module or _load_native_module()
        if getattr(module, "BINDING_ABI_VERSION", None) != BINDING_ABI_VERSION:
            raise NativeBindingUnavailableError("native binding ABI version mismatch")
        if getattr(module, "PROTOCOL_VERSION", None) != PROTOCOL_VERSION:
            raise NativeBindingUnavailableError("native protocol version mismatch")
        self._backend = module.NativeCryptoEngine(
            self._options.to_binding_dict()
        )
        self._setup: CryptoEngineSetup | None = None
        self._setup_diagnostics: dict[str, Any] | None = None
        self._last_diagnostics: dict[str, Any] | None = None
        self._effective_protocol: dict[str, Any] | None = None
        self._public_setup_binding: bytes | None = None

    @property
    def engine_name(self) -> str:
        return "openfhe-pybind11-v17"

    @property
    def setup_config(self) -> CryptoEngineSetup | None:
        return self._setup

    @property
    def phase(self) -> RoundPhase | None:
        if self._setup is None:
            return None
        value = self._backend.phase
        return None if value is None else RoundPhase(value)

    @property
    def last_diagnostics(self) -> dict[str, Any] | None:
        """Ciphertext counters and timings from the latest decrypted round."""

        return None if self._last_diagnostics is None else dict(self._last_diagnostics)

    @property
    def last_aborted_round_traffic(self) -> dict[str, Any] | None:
        """Pre-response wire counters for the latest v17 late abort."""

        raw = self._backend.last_aborted_round_traffic
        return None if raw is None else dict(raw)

    @property
    def setup_diagnostics(self) -> dict[str, Any] | None:
        """Sequential and idealized parallel timing for one-time setup."""

        return (
            None
            if self._setup_diagnostics is None
            else dict(self._setup_diagnostics)
        )

    @property
    def public_setup_binding(self) -> bytes | None:
        """Canonical public metadata; contains no secret keys or Kidx."""
        return self._public_setup_binding

    @property
    def effective_protocol(self) -> dict[str, Any] | None:
        return (
            None
            if self._effective_protocol is None
            else dict(self._effective_protocol)
        )

    def setup(self, config: CryptoEngineSetup) -> None:
        config.validate()
        native = self._backend.setup(
            _layout_payload(config.parameter_layout),
            _layout_payload(config.running_mean_layout),
            _layout_payload(config.running_variance_layout),
            tuple(
                (client.client_id, client.example_count)
                for client in config.clients
            ),
            config.chunk_size,
            config.minimum_cohort_size,
        )

        expected = {
            "parameter_layout_hash": bytes.fromhex(config.parameter_layout.sha256()),
            "running_mean_layout_hash": bytes.fromhex(
                config.running_mean_layout.sha256()
            ),
            "running_variance_layout_hash": bytes.fromhex(
                config.running_variance_layout.sha256()
            ),
            "parameter_numel": config.parameter_layout.total_numel,
            "running_mean_numel": config.running_mean_layout.total_numel,
            "running_variance_numel": config.running_variance_layout.total_numel,
            "chunk_size": config.chunk_size,
            "chunk_count": config.parameter_chunk_count,
            "client_ids": list(config.client_ids),
            "alphas": list(config.alphas),
            "protocol_mode": config.protocol_mode,
        }
        for key, value in expected.items():
            if native[key] != value:
                raise RuntimeError(f"native setup manifest mismatch: {key}")

        effective_protocol = dict(native["effective_protocol"])
        self._options.validate_effective_protocol(effective_protocol)
        if (
            self._options.protocol_version == "v17"
            and self._options.protocol_mode == "SPARSE"
            and effective_protocol.get("support_bitmap_bytes")
            != (config.parameter_chunk_count + 7) // 8
        ):
            raise RuntimeError(
                "native setup effective protocol mismatch: support_bitmap_bytes"
            )

        descriptor = canonical_parameter_descriptor(effective_protocol)
        if (bytes(native["setup_descriptor"]) != descriptor
                or len(descriptor) != effective_protocol["setup_descriptor_bytes"]):
            raise RuntimeError("native canonical setup descriptor mismatch")
        expected_binding = serialize_public_setup(
            bytes(native["setup_id"]), bytes(native["crs_seed"]), descriptor,
            config.client_ids, (bytes.fromhex(config.parameter_layout.sha256()),
                               bytes.fromhex(config.running_mean_layout.sha256()),
                               bytes.fromhex(config.running_variance_layout.sha256())))
        binding = bytes(native["public_setup_binding"])
        parse_public_setup(binding)
        if binding != expected_binding:
            raise RuntimeError("native public setup binding mismatch")

        setup_timing = dict(native["setup_timing"])
        client_initial = tuple(
            float(value)
            for value in setup_timing["client_initial_setup_seconds"]
        )
        client_collective = tuple(
            float(value)
            for value in setup_timing[
                "client_collective_key_generation_seconds"
            ]
        )
        if len(client_initial) != len(config.clients) or len(
            client_collective
        ) != len(config.clients):
            raise RuntimeError("native setup timing client count mismatch")
        scalar_names = (
            "sequential_seconds",
            "server_seconds",
            "maximum_initial_client_seconds",
            "maximum_collective_key_generation_seconds",
            "estimated_parallel_seconds",
        )
        scalar_values = {
            name: float(setup_timing[name]) for name in scalar_names
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                *client_initial,
                *client_collective,
                *scalar_values.values(),
            )
        ):
            raise RuntimeError(
                "native setup timing values must be finite and non-negative"
            )
        self._setup_diagnostics = {
            "model": str(setup_timing["model"]),
            "network_latency_included": bool(
                setup_timing["network_latency_included"]
            ),
            **scalar_values,
            "client_initial_setup_seconds": list(client_initial),
            "client_collective_key_generation_seconds": list(
                client_collective
            ),
        }

        setup_id = bytes(native["setup_id"])
        if len(setup_id) != 16:
            raise RuntimeError("native setup_id must contain 16 bytes")
        self._setup = replace(config, setup_id=setup_id)
        self._setup.validate()
        self._last_diagnostics = None
        self._effective_protocol = effective_protocol
        self._public_setup_binding = binding

    def begin_round(
        self, round_id: int, *, logical_round_index: int | None = None
    ) -> None:
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        if logical_round_index is not None and logical_round_index < 0:
            raise ValueError("logical_round_index must be non-negative")
        self._require_setup()
        if logical_round_index is None:
            self._backend.begin_round(round_id)
        else:
            self._backend.begin_round(round_id, logical_round_index)

    def submit_selection(
        self,
        client_id: int,
        selected_actual_chunk_ids: Sequence[int],
    ) -> None:
        self._require_setup()
        self._backend.submit_selection(
            client_id,
            tuple(int(chunk_id) for chunk_id in selected_actual_chunk_ids),
        )

    def finalize_admission(self) -> AdmissionPlan:
        setup = self._require_setup()
        native = self._backend.finalize_admission()
        return self._admission_from_native(native, setup)

    @staticmethod
    def _admission_from_native(
        native: Any,
        setup: CryptoEngineSetup,
    ) -> AdmissionPlan:
        clients = tuple(
            ClientAdmissionResult(
                setup_id=bytes(item["setup_id"]),
                round_id=int(item["round_id"]),
                client_id=int(item["client_id"]),
                accepted_actual_chunk_ids=tuple(
                    int(value) for value in item["accepted_actual_chunk_ids"]
                ),
            )
            for item in native["client_results"]
        )
        result = AdmissionPlan(
            setup_id=bytes(native["setup_id"]),
            round_id=int(native["round_id"]),
            client_results=clients,
            accepted_actual_chunk_ids=tuple(
                int(value) for value in native["accepted_actual_chunk_ids"]
            ),
            cohort_sizes=tuple(int(value) for value in native["cohort_sizes"]),
        )
        result.validate(setup)
        return result

    def submit_accepted_values(
        self,
        client_id: int,
        accepted_actual_chunk_ids: Sequence[int],
        values: np.ndarray,
    ) -> None:
        setup = self._require_setup()
        chunk_ids = tuple(int(value) for value in accepted_actual_chunk_ids)
        _require_float32_array(
            values,
            (len(chunk_ids), setup.chunk_size),
            "accepted values",
        )
        self._backend.submit_accepted_values(client_id, chunk_ids, values)

    def submit_full_parameter_delta(
        self,
        client_id: int,
        parameter_delta: np.ndarray,
    ) -> None:
        setup = self._require_setup()
        if setup.protocol_mode != "FULL":
            raise ValueError("full parameter deltas require FULL mode")
        _require_float32_array(
            parameter_delta,
            (setup.parameter_layout.total_numel,),
            "parameter_delta",
        )
        self._backend.submit_full_parameter_delta(client_id, parameter_delta)

    def submit_auxiliary_deltas(
        self,
        client_id: int,
        running_mean_delta: np.ndarray,
        running_variance_delta: np.ndarray,
    ) -> None:
        setup = self._require_setup()
        _require_float32_array(
            running_mean_delta,
            (setup.running_mean_layout.total_numel,),
            "running_mean_delta",
        )
        _require_float32_array(
            running_variance_delta,
            (setup.running_variance_layout.total_numel,),
            "running_variance_delta",
        )
        self._backend.submit_auxiliary_deltas(
            client_id,
            running_mean_delta,
            running_variance_delta,
        )

    def set_round_survivors(
        self,
        survivor_client_ids: Sequence[int],
    ) -> AdmissionPlan | None:
        setup = self._require_setup()
        if setup.protocol_mode not in {"SPARSE", "FULL"}:
            raise ValueError("post-upload dropout requires SPARSE or FULL mode")
        survivors = tuple(int(value) for value in survivor_client_ids)
        if len(set(survivors)) != len(survivors):
            raise ValueError("survivor client IDs must be unique")
        if not set(survivors).issubset(set(setup.client_ids)):
            raise ValueError("survivor client IDs must be registered")
        native = self._backend.set_round_survivors(survivors)
        admission = (
            self._admission_from_native(native, setup)
            if setup.protocol_mode == "SPARSE" else None
        )
        returned_survivors = tuple(
            int(value) for value in native["survivor_client_ids"]
        )
        dropped = tuple(int(value) for value in native["dropped_client_ids"])
        if set(returned_survivors) != set(survivors):
            raise RuntimeError("native survivor client set mismatch")
        if set(returned_survivors).intersection(dropped) or (
            set(returned_survivors).union(dropped) != set(setup.client_ids)
        ):
            raise RuntimeError("native survivor/dropout partition is invalid")
        timings = tuple(
            float(value)
            for value in native["client_collective_key_generation_seconds"]
        )
        timing_scalars = (
            float(native["server_target_reconstruction_seconds"]),
            float(native["maximum_client_collective_key_generation_seconds"]),
            *timings,
        )
        if len(timings) != len(setup.clients) or any(
            not math.isfinite(value) or value < 0.0
            for value in timing_scalars
        ):
            raise RuntimeError("native survivor transition timing is invalid")
        return admission

    def finalize_round(
        self, *, post_request_dropouts: Sequence[int] = ()
    ) -> DecryptedRoundResult:
        setup = self._require_setup()
        dropouts = tuple(int(value) for value in post_request_dropouts)
        if self._options.experimental_dynamic_cohort:
            # Single-process experiment: each surviving client acknowledges the
            # exact setup/t/rid/revision/target/support/group context. Native
            # enforcement also protects callers that bypass this adapter.
            context = self._backend.prepare_finalization()
            count_position = 2 + len(setup.setup_id) + 3
            count = context[count_position]
            for client_id in context[count_position + 1:count_position + 1 + count]:
                self._backend.acknowledge_finalization(client_id, context)
        try:
            native = (
                self._backend.finalize_round(dropouts)
                if dropouts else self._backend.finalize_round()
            )
        except RuntimeError as error:
            if "v17 final survivor dropped after decryption request" in str(error):
                raise PostRequestDropoutError(str(error)) from error
            raise
        result = DecryptedRoundResult(
            setup_id=bytes(native["setup_id"]),
            round_id=int(native["round_id"]),
            accepted_actual_chunk_ids=tuple(
                int(value) for value in native["accepted_actual_chunk_ids"]
            ),
            cohort_sizes=tuple(int(value) for value in native["cohort_sizes"]),
            parameter_delta=np.asarray(native["parameter_delta"]),
            running_mean_delta=np.asarray(native["running_mean_delta"]),
            running_variance_delta=np.asarray(native["running_variance_delta"]),
            phase_audit=native.get("phase_audit"),
        )
        result.validate(setup)
        diagnostic_names = (
            "sparse_plan_hash",
            "sparse_client_ciphertext_count",
            "sparse_survivor_client_ciphertext_count",
            "sparse_discarded_client_ciphertext_count",
            "accepted_sparse_contribution_count",
            "aggregated_mlwe_ciphertext_count",
            "sparse_server_ciphertext_count",
            "selection_bitmap_count",
            "admission_bitmap_recipient_count",
            "upload_support_bitmap_count",
            "result_recipient_count",
            "full_model_client_ciphertext_count",
            "full_model_aggregate_ciphertext_count",
            "r2m_ciphertext_count",
            "m2r_ciphertext_count",
            "auxiliary_client_ciphertext_count",
            "auxiliary_survivor_client_ciphertext_count",
            "auxiliary_discarded_client_ciphertext_count",
            "auxiliary_aggregate_ciphertext_count",
        )
        self._last_diagnostics = {
            name: int(native[name]) for name in diagnostic_names
        }
        if "codec_work_timings" in native:
            self._last_diagnostics["codec_work_timings"] = native["codec_work_timings"]

        client_encryption_seconds = tuple(
            float(value) for value in native["client_encryption_seconds"]
        )
        client_key_switch_seconds = tuple(
            float(value)
            for value in native.get(
                "client_key_switch_seconds",
                (0.0,) * len(setup.clients),
            )
        )
        client_partial_decryption_seconds = tuple(
            float(value)
            for value in native["client_partial_decryption_seconds"]
        )
        if (
            len(client_encryption_seconds) != len(setup.clients)
            or len(client_key_switch_seconds) != len(setup.clients)
            or len(client_partial_decryption_seconds) != len(setup.clients)
        ):
            raise RuntimeError("native timing client count mismatch")
        client_total_crypto_seconds = tuple(
            float(value)
            for value in native.get(
                "client_total_crypto_seconds",
                tuple(
                    encryption + key_switch + partial
                    for encryption, key_switch, partial in zip(
                        client_encryption_seconds,
                        client_key_switch_seconds,
                        client_partial_decryption_seconds,
                        strict=True,
                    )
                ),
            )
        )
        if len(client_total_crypto_seconds) != len(setup.clients):
            raise RuntimeError("native timing client count mismatch")
        timing_values = (
            *client_encryption_seconds,
            *client_key_switch_seconds,
            *client_partial_decryption_seconds,
            *client_total_crypto_seconds,
            float(native["crypto_sequential_seconds"]),
            float(native["crypto_server_seconds"]),
            float(native["estimated_parallel_crypto_seconds"]),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in timing_values):
            raise RuntimeError("native timings must be finite and non-negative")
        crypto_thread_count = int(native.get("crypto_thread_count", 1))
        if not 1 <= crypto_thread_count <= 256:
            raise RuntimeError("native crypto thread count is invalid")
        crypto_stage_timings = _parse_stage_timings(
            native.get("crypto_stage_timings", {})
        )
        full_model_client_stage_timings = _parse_client_stage_timings(
            native.get("full_model_client_stage_timings", ())
        )
        sparse_model_client_stage_timings = _parse_client_stage_timings(
            native.get("sparse_model_client_stage_timings", ())
        )
        auxiliary_client_stage_timings = _parse_client_stage_timings(
            native.get("auxiliary_client_stage_timings", ())
        )
        if (
            full_model_client_stage_timings
            and len(full_model_client_stage_timings) != len(setup.clients)
        ) or (
            sparse_model_client_stage_timings
            and len(sparse_model_client_stage_timings) != len(setup.clients)
        ) or (
            auxiliary_client_stage_timings
            and len(auxiliary_client_stage_timings) != len(setup.clients)
        ):
            raise RuntimeError("native stage timing client count mismatch")
        full_model_server_stage_timings = _parse_stage_timings(
            native.get("full_model_server_stage_timings", {})
        )
        sparse_model_server_stage_timings = _parse_stage_timings(
            native.get("sparse_model_server_stage_timings", {})
        )
        auxiliary_server_stage_timings = _parse_stage_timings(
            native.get("auxiliary_server_stage_timings", {})
        )
        maximum_client_pre_aggregation_seconds = float(
            native.get(
                "maximum_client_pre_aggregation_seconds",
                max(
                    (
                        encryption + key_switch
                        for encryption, key_switch in zip(
                            client_encryption_seconds,
                            client_key_switch_seconds,
                            strict=True,
                        )
                    ),
                    default=0.0,
                ),
            )
        )
        maximum_client_partial_decryption_seconds = float(
            native.get(
                "maximum_client_partial_decryption_seconds",
                max(client_partial_decryption_seconds, default=0.0),
            )
        )
        maximum_client_total_crypto_seconds = float(
            native.get(
                "maximum_client_total_crypto_seconds",
                max(client_total_crypto_seconds, default=0.0),
            )
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                maximum_client_pre_aggregation_seconds,
                maximum_client_partial_decryption_seconds,
                maximum_client_total_crypto_seconds,
            )
        ):
            raise RuntimeError("native client maximum timing is invalid")
        survivor_reconfiguration_applied = bool(
            native.get("survivor_reconfiguration_applied", False)
        )
        survivor_client_ids = tuple(
            int(value) for value in native.get("survivor_client_ids", ())
        )
        dropped_client_ids = tuple(
            int(value) for value in native.get("dropped_client_ids", ())
        )
        survivor_collective_key_generation_seconds = tuple(
            float(value)
            for value in native.get(
                "survivor_collective_key_generation_seconds",
                (0.0,) * len(setup.clients),
            )
        )
        survivor_timing_values = (
            float(native.get("survivor_target_reconstruction_seconds", 0.0)),
            float(
                native.get(
                    "maximum_survivor_collective_key_generation_seconds",
                    0.0,
                )
            ),
            float(
                native.get(
                    "survivor_reconfiguration_sequential_seconds",
                    0.0,
                )
            ),
            *survivor_collective_key_generation_seconds,
        )
        if (
            len(survivor_collective_key_generation_seconds)
            != len(setup.clients)
            or len(set(survivor_client_ids)) != len(survivor_client_ids)
            or len(set(dropped_client_ids)) != len(dropped_client_ids)
            or set(survivor_client_ids).intersection(dropped_client_ids)
            or set(survivor_client_ids).union(dropped_client_ids)
            != set(setup.client_ids)
            or survivor_reconfiguration_applied != bool(dropped_client_ids)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in survivor_timing_values
            )
        ):
            raise RuntimeError("native survivor diagnostics are invalid")
        if (
            self._last_diagnostics["sparse_client_ciphertext_count"]
            != self._last_diagnostics[
                "sparse_survivor_client_ciphertext_count"
            ]
            + self._last_diagnostics[
                "sparse_discarded_client_ciphertext_count"
            ]
            or self._last_diagnostics["auxiliary_client_ciphertext_count"]
            != self._last_diagnostics[
                "auxiliary_survivor_client_ciphertext_count"
            ]
            + self._last_diagnostics[
                "auxiliary_discarded_client_ciphertext_count"
            ]
        ):
            raise RuntimeError("native survivor ciphertext counts are invalid")
        self._last_diagnostics.update(
            {
                "crypto_timing_model": native.get(
                    "crypto_timing_model", "legacy-wait-inclusive"
                ),
                "crypto_sequential_seconds": float(
                    native["crypto_sequential_seconds"]
                ),
                "crypto_server_seconds": float(native["crypto_server_seconds"]),
                "estimated_parallel_crypto_seconds": float(
                    native["estimated_parallel_crypto_seconds"]
                ),
                "client_encryption_seconds": client_encryption_seconds,
                "client_key_switch_seconds": client_key_switch_seconds,
                "client_partial_decryption_seconds": (
                    client_partial_decryption_seconds
                ),
                "client_total_crypto_seconds": client_total_crypto_seconds,
                "maximum_client_pre_aggregation_seconds": (
                    maximum_client_pre_aggregation_seconds
                ),
                "maximum_client_partial_decryption_seconds": (
                    maximum_client_partial_decryption_seconds
                ),
                "maximum_client_total_crypto_seconds": (
                    maximum_client_total_crypto_seconds
                ),
                "crypto_thread_count": crypto_thread_count,
                "crypto_stage_timings": crypto_stage_timings,
                "full_model_client_stage_timings": (
                    full_model_client_stage_timings
                ),
                "sparse_model_client_stage_timings": (
                    sparse_model_client_stage_timings
                ),
                "auxiliary_client_stage_timings": (
                    auxiliary_client_stage_timings
                ),
                "full_model_server_stage_timings": (
                    full_model_server_stage_timings
                ),
                "sparse_model_server_stage_timings": (
                    sparse_model_server_stage_timings
                ),
                "auxiliary_server_stage_timings": (
                    auxiliary_server_stage_timings
                ),
                "survivor_reconfiguration_applied": (
                    survivor_reconfiguration_applied
                ),
                "survivor_target_key_cache_hit": bool(
                    native.get("survivor_target_key_cache_hit", False)
                ),
                "survivor_new_target_sizes": tuple(
                    int(value)
                    for value in native.get("survivor_new_target_sizes", ())
                ),
                "survivor_client_ids": survivor_client_ids,
                "dropped_client_ids": dropped_client_ids,
                "survivor_target_reconstruction_seconds": float(
                    native.get("survivor_target_reconstruction_seconds", 0.0)
                ),
                "survivor_collective_key_generation_seconds": (
                    survivor_collective_key_generation_seconds
                ),
                "maximum_survivor_collective_key_generation_seconds": float(
                    native.get(
                        "maximum_survivor_collective_key_generation_seconds",
                        0.0,
                    )
                ),
                "survivor_reconfiguration_sequential_seconds": float(
                    native.get(
                        "survivor_reconfiguration_sequential_seconds",
                        0.0,
                    )
                ),
            }
        )
        return result

    def commit_round(self) -> None:
        self._require_setup()
        self._backend.commit_round()

    def abort_round(self) -> None:
        self._require_setup()
        self._backend.abort_round()

    def _require_setup(self) -> CryptoEngineSetup:
        if self._setup is None:
            raise RuntimeError("native crypto engine setup has not been installed")
        return self._setup
