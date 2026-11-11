"""User-supplied, immutable numeric profile for the native v17 engine."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_adapter import ModelProtocolLayouts
from .native_engine import NativeCryptoEngineOptions
from .chunk_layout import selected_chunk_count_from_ratio


PROFILE_SCHEMA = "fl-v17-crypto-parameter-profile-v3"
DEFAULT_PROFILE_RESOURCE = (
    "profiles/resnet20-cifar10-sp-k64-rho020-shared-v17.json"
)


def _require_exact_keys(
    document: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{context} fields mismatch; missing={missing}, unknown={unknown}"
        )


def _positive_scale(value: Any, name: str) -> int:
    # Decimal strings make large JSON integers interoperable with native readers.
    if isinstance(value, str):
        if not value or len(value) > 78 or value[0] == "0" or not value.isascii() or not value.isdigit():
            raise ValueError(f"{name} must be a canonical decimal integer")
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**256:
        raise ValueError(f"{name} must be an integer in [1, 2^256)")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _nonnegative_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


@dataclass(frozen=True)
class CryptoParameterProfile:
    name: str
    protocol_version: str
    model_name: str
    mode: str
    quantization_enabled: bool
    clients: int
    minimum_cohort_size: int
    selection_ratio: float
    safety_factor: float
    parameter_numel: int
    running_mean_numel: int
    running_variance_numel: int
    parameter_layout_sha256: str
    running_mean_layout_sha256: str
    running_variance_layout_sha256: str
    ring_dimension: int
    mlwe_rank: int
    q_modulus_bits: tuple[int, ...]
    p_modulus_bits: tuple[int, ...]
    hybrid_large_digits: int
    maximum_qp_bits: int
    model_scaling_factor: int
    aux_scaling_factor: int
    precision_bits: int
    aux_precision_bits: int
    aggregate_update_bound: float
    aggregate_aux_update_bound: float
    compression_c1: int
    compression_ca: int
    compression_c2: int
    compression_cd: int
    aux_compression_c1: int
    aux_compression_ca: int
    aux_compression_c2: int
    aux_compression_cd: int
    smudging_sigma: float
    aux_smudging_sigma: float
    model_delivery_scale: float
    aux_delivery_scale: float
    document: dict[str, Any]
    sha256: str
    source: str

    @property
    def chunk_size(self) -> int:
        return (
            self.ring_dimension
            if self.mode == "FULL"
            else self.ring_dimension // self.mlwe_rank
        )

    @property
    def auxiliary_numel(self) -> int:
        return self.running_mean_numel + self.running_variance_numel

    def selected_chunk_count(self, parameter_numel: int | None = None) -> int:
        dimension = (
            self.parameter_numel
            if parameter_numel is None
            else parameter_numel
        )
        return selected_chunk_count_from_ratio(
            total_numel=dimension,
            chunk_size=self.chunk_size,
            rho=self.selection_ratio,
        )

    def validate_layouts(self, layouts: ModelProtocolLayouts) -> None:
        layouts.validate()
        expected = (
            (
                layouts.parameter,
                self.parameter_numel,
                self.parameter_layout_sha256,
                "parameter",
            ),
            (
                layouts.running_mean,
                self.running_mean_numel,
                self.running_mean_layout_sha256,
                "running_mean",
            ),
            (
                layouts.running_variance,
                self.running_variance_numel,
                self.running_variance_layout_sha256,
                "running_variance",
            ),
        )
        for layout, numel, digest, name in expected:
            if layout.model_name != self.model_name:
                raise ValueError(f"{name} layout model name does not match profile")
            if layout.total_numel != numel:
                raise ValueError(f"{name} layout dimension does not match profile")
            if layout.sha256() != digest:
                raise ValueError(f"{name} layout hash does not match profile")

    def native_options(self, *, master_seed: bytes) -> NativeCryptoEngineOptions:
        return NativeCryptoEngineOptions(
            profile="resnet20-cifar10",
            protocol_version=self.protocol_version,
            protocol_mode=self.mode,
            ring_dimension=self.ring_dimension,
            mlwe_rank=self.mlwe_rank,
            q_modulus_bits=self.q_modulus_bits,
            p_modulus_bits=self.p_modulus_bits,
            hybrid_large_digits=self.hybrid_large_digits,
            maximum_qp_bits=self.maximum_qp_bits,
            model_scaling_factor=self.model_scaling_factor,
            aux_scaling_factor=self.aux_scaling_factor,
            precision_bits=self.precision_bits,
            aux_precision_bits=self.aux_precision_bits,
            selection_ratio=self.selection_ratio,
            update_bound=self.aggregate_update_bound,
            aux_update_bound=self.aggregate_aux_update_bound,
            compression_c1=self.compression_c1,
            compression_ca=self.compression_ca,
            compression_c2=self.compression_c2,
            compression_cd=self.compression_cd,
            aux_compression_c1=self.aux_compression_c1,
            aux_compression_ca=self.aux_compression_ca,
            aux_compression_c2=self.aux_compression_c2,
            aux_compression_cd=self.aux_compression_cd,
            smudging_sigma=self.smudging_sigma,
            aux_smudging_sigma=self.aux_smudging_sigma,
            model_delivery_scale=self.model_delivery_scale,
            aux_delivery_scale=self.aux_delivery_scale,
            master_seed=master_seed,
        )


def _parse_profile(document: dict[str, Any], source: str) -> CryptoParameterProfile:
    _require_exact_keys(
        document,
        {
            "schema",
            "name",
            "protocol_version",
            "model_name",
            "experiment",
            "dimensions",
            "ring",
            "scaling",
            "precision",
            "aggregate_bounds",
            "compression",
            "smudging",
            "delivery",
        },
        "profile",
    )
    if document["schema"] != PROFILE_SCHEMA:
        raise ValueError("unsupported crypto parameter profile schema")
    if document["protocol_version"] not in {"v17"}:
        raise ValueError(
            "crypto parameter profile must use protocol v17"
        )
    for name in ("name", "model_name"):
        if not isinstance(document[name], str) or not document[name]:
            raise ValueError(f"{name} must be a non-empty string")

    experiment = document["experiment"]
    dimensions = document["dimensions"]
    ring = document["ring"]
    scaling = document["scaling"]
    precision = document["precision"]
    aggregate_bounds = document["aggregate_bounds"]
    compression = document["compression"]
    smudging = document["smudging"]
    delivery = document["delivery"]
    for name, section in (
        ("experiment", experiment),
        ("dimensions", dimensions),
        ("ring", ring),
        ("scaling", scaling),
        ("precision", precision),
        ("aggregate_bounds", aggregate_bounds),
        ("compression", compression),
        ("smudging", smudging),
        ("delivery", delivery),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be an object")

    _require_exact_keys(
        experiment,
        {
            "mode",
            "clients",
            "minimum_cohort_size",
            "selection_ratio",
            "safety_factor",
            "quantization_enabled",
        },
        "experiment",
    )
    _require_exact_keys(
        dimensions,
        {
            "parameter_numel",
            "running_mean_numel",
            "running_variance_numel",
            "auxiliary_numel",
            "parameter_layout_sha256",
            "running_mean_layout_sha256",
            "running_variance_layout_sha256",
        },
        "dimensions",
    )
    _require_exact_keys(
        ring,
        {
            "ring_dimension",
            "mlwe_rank",
            "q_modulus_bits",
            "p_modulus_bits",
            "hybrid_large_digits",
            "maximum_qp_bits",
        },
        "ring",
    )
    _require_exact_keys(
        scaling,
        {"model_factor", "aux_factor"},
        "scaling",
    )
    _require_exact_keys(
        precision,
        {"parameter_bits", "auxiliary_bits"},
        "precision",
    )
    _require_exact_keys(
        aggregate_bounds,
        {"parameter", "auxiliary"},
        "aggregate_bounds",
    )
    compression_names = {
        "c1",
        "ca",
        "c2",
        "cd",
        "aux_c1",
        "aux_ca",
        "aux_c2",
        "aux_cd",
    }
    _require_exact_keys(compression, compression_names, "compression")
    _require_exact_keys(
        smudging,
        {"parameter_sigma", "auxiliary_sigma"},
        "smudging",
    )
    _require_exact_keys(
        delivery,
        {"parameter_scale", "auxiliary_scale"},
        "delivery",
    )

    clients = _positive_integer(experiment["clients"], "clients")
    minimum_cohort_size = _positive_integer(
        experiment["minimum_cohort_size"], "minimum_cohort_size"
    )
    selection_ratio = _positive_number(
        experiment["selection_ratio"], "selection_ratio"
    )
    if selection_ratio > 1.0:
        raise ValueError("selection_ratio must be at most one")
    safety_factor = _positive_number(experiment["safety_factor"], "safety_factor")
    if not 2 <= minimum_cohort_size <= clients:
        raise ValueError("minimum_cohort_size must be between two and clients")
    if experiment["mode"] not in {"SPARSE", "FULL"}:
        raise ValueError("encrypted profiles must use SPARSE or FULL mode")
    if experiment["mode"] == "FULL" and selection_ratio != 1.0:
        raise ValueError("FULL profiles require selection_ratio=1")
    if experiment["quantization_enabled"] is not True:
        raise ValueError("the supplied profile requires quantization")

    if not isinstance(ring["q_modulus_bits"], list):
        raise ValueError("q_modulus_bits must be an array")
    if not isinstance(ring["p_modulus_bits"], list):
        raise ValueError("p_modulus_bits must be an array")
    q_modulus_bits = tuple(
        _positive_integer(value, "q_modulus_bits")
        for value in ring["q_modulus_bits"]
    )
    p_modulus_bits = tuple(
        _positive_integer(value, "p_modulus_bits")
        for value in ring["p_modulus_bits"]
    )
    if not q_modulus_bits or not p_modulus_bits:
        raise ValueError("Q and P modulus bit lists must not be empty")
    if len(p_modulus_bits) != 1:
        raise ValueError("exactly one P modulus tower is required")
    hybrid_large_digits = _positive_integer(
        ring["hybrid_large_digits"], "hybrid_large_digits"
    )
    if hybrid_large_digits != len(q_modulus_bits):
        raise ValueError("hybrid_large_digits must equal the Q tower count")
    if (
        document["protocol_version"] in {"v17"}
        and len(q_modulus_bits) != 2
    ):
        raise ValueError("v17 require exactly two Q primes and m=2")
    if len(q_modulus_bits) > 2 and any(
        bits != q_modulus_bits[1] for bits in q_modulus_bits[2:]
    ):
        raise ValueError(
            "all Q towers after the first must use the same bit size"
        )

    values = {
        name: _positive_integer(compression[name], name)
        for name in compression_names
    }
    if any(value > 63 for value in values.values()):
        raise ValueError("compression bit counts must be at most 63")
    aggregate_update_bound = _positive_number(
        aggregate_bounds["parameter"], "aggregate parameter bound"
    )
    aggregate_aux_update_bound = _positive_number(
        aggregate_bounds["auxiliary"], "aggregate auxiliary bound"
    )
    model_delivery_scale = _positive_number(
        delivery["parameter_scale"], "parameter_scale"
    )
    aux_delivery_scale = _positive_number(
        delivery["auxiliary_scale"], "auxiliary_scale"
    )
    if (
        model_delivery_scale * aggregate_update_bound
        > 2 ** (values["cd"] - 1) - 1
    ):
        raise ValueError("parameter delivery range exceeds cd")
    if (
        aux_delivery_scale * aggregate_aux_update_bound
        > 2 ** (values["aux_cd"] - 1) - 1
    ):
        raise ValueError("auxiliary delivery range exceeds aux_cd")

    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    profile = CryptoParameterProfile(
        name=document["name"],
        protocol_version=document["protocol_version"],
        model_name=document["model_name"],
        mode=experiment["mode"],
        quantization_enabled=experiment["quantization_enabled"],
        clients=clients,
        minimum_cohort_size=minimum_cohort_size,
        selection_ratio=selection_ratio,
        safety_factor=safety_factor,
        parameter_numel=_positive_integer(
            dimensions["parameter_numel"], "parameter_numel"
        ),
        running_mean_numel=_positive_integer(
            dimensions["running_mean_numel"], "running_mean_numel"
        ),
        running_variance_numel=_positive_integer(
            dimensions["running_variance_numel"], "running_variance_numel"
        ),
        parameter_layout_sha256=_sha256(
            dimensions["parameter_layout_sha256"], "parameter_layout_sha256"
        ),
        running_mean_layout_sha256=_sha256(
            dimensions["running_mean_layout_sha256"],
            "running_mean_layout_sha256",
        ),
        running_variance_layout_sha256=_sha256(
            dimensions["running_variance_layout_sha256"],
            "running_variance_layout_sha256",
        ),
        ring_dimension=_positive_integer(ring["ring_dimension"], "ring_dimension"),
        mlwe_rank=_positive_integer(ring["mlwe_rank"], "mlwe_rank"),
        q_modulus_bits=q_modulus_bits,
        p_modulus_bits=p_modulus_bits,
        hybrid_large_digits=hybrid_large_digits,
        maximum_qp_bits=_positive_integer(
            ring["maximum_qp_bits"], "maximum_qp_bits"
        ),
        model_scaling_factor=_positive_scale(
            scaling["model_factor"], "model_factor"
        ),
        aux_scaling_factor=_positive_scale(
            scaling["aux_factor"], "aux_factor"
        ),
        precision_bits=_positive_integer(
            precision["parameter_bits"], "parameter_bits"
        ),
        aux_precision_bits=_positive_integer(
            precision["auxiliary_bits"], "auxiliary_bits"
        ),
        aggregate_update_bound=aggregate_update_bound,
        aggregate_aux_update_bound=aggregate_aux_update_bound,
        compression_c1=values["c1"],
        compression_ca=values["ca"],
        compression_c2=values["c2"],
        compression_cd=values["cd"],
        aux_compression_c1=values["aux_c1"],
        aux_compression_ca=values["aux_ca"],
        aux_compression_c2=values["aux_c2"],
        aux_compression_cd=values["aux_cd"],
        smudging_sigma=_nonnegative_number(
            smudging["parameter_sigma"], "parameter_sigma"
        ),
        aux_smudging_sigma=_nonnegative_number(
            smudging["auxiliary_sigma"], "auxiliary_sigma"
        ),
        model_delivery_scale=model_delivery_scale,
        aux_delivery_scale=aux_delivery_scale,
        document=document,
        sha256=hashlib.sha256(canonical).hexdigest(),
        source=source,
    )
    if profile.ring_dimension & (profile.ring_dimension - 1):
        raise ValueError("ring_dimension must be a power of two")
    if profile.ring_dimension % profile.mlwe_rank:
        raise ValueError("mlwe_rank must divide ring_dimension")
    if (
        sum(profile.q_modulus_bits) + sum(profile.p_modulus_bits)
        > profile.maximum_qp_bits
    ):
        raise ValueError("Q/P bit total exceeds maximum_qp_bits")
    if dimensions["auxiliary_numel"] != profile.auxiliary_numel:
        raise ValueError("auxiliary_numel does not match the two auxiliary layouts")
    return profile


def load_crypto_parameter_profile(
    path: str | Path | None = None,
) -> CryptoParameterProfile:
    if path is None:
        resource = files("plaintext_fl").joinpath(DEFAULT_PROFILE_RESOURCE)
        source = f"package:{DEFAULT_PROFILE_RESOURCE}"
        text = resource.read_text(encoding="utf-8")
    else:
        source_path = Path(path).expanduser().resolve()
        source = str(source_path)
        text = source_path.read_text(encoding="utf-8")
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("crypto parameter profile root must be an object")
    return _parse_profile(document, source)
