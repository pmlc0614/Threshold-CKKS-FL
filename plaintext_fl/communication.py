"""Logical all-unicast communication accounting for encrypted FL.

The pybind11 calls run inside one process, so their Python object sizes are not
network traffic.  This module instead models the distributed v17 protocol:
coefficient and final-delivery values are bit-packed at their configured
``c`` widths, common downlink payloads are charged once per recipient, and
server-internal M2R/R2M work is excluded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .crypto_engine import CryptoEngineSetup

if TYPE_CHECKING:
    from .crypto_parameter_profile import CryptoParameterProfile


ACCOUNTING_MODEL = "logical-all-unicast-bit-packed-v1"
V17_ACCOUNTING_MODEL = "v17-logical-all-unicast-bit-packed-bitmap-v1"
BYTES_PER_MB = 1_000_000


def _bits_to_mb(bits: int | float) -> float:
    return bits / 8.0 / BYTES_PER_MB


def _integer(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(mapping: Mapping[str, Any], name: str) -> int:
    value = _integer(mapping, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_integer_sequence(
    mapping: Mapping[str, Any],
    name: str,
) -> tuple[int, ...]:
    values = mapping.get(name)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must contain positive integers")
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in result
    ):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _optional_non_negative_integer(
    mapping: Mapping[str, Any],
    name: str,
    default: int,
) -> int:
    if name not in mapping:
        return default
    return _integer(mapping, name)


def _direction(
    breakdown_bits: Mapping[str, int],
    *,
    client_count: int,
) -> dict[str, Any]:
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    normalized: dict[str, int] = {}
    for name, value in breakdown_bits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} bits must be a non-negative integer")
        normalized[name] = value
    total_bits = sum(normalized.values())
    return {
        "total_bits": total_bits,
        "total_bytes": (total_bits + 7) // 8,
        "total_mb": _bits_to_mb(total_bits),
        "mean_per_client_bits": total_bits / client_count,
        "mean_per_client_bytes": total_bits / 8.0 / client_count,
        "mean_per_client_mb": _bits_to_mb(total_bits) / client_count,
        "breakdown_bits": normalized,
        "breakdown_bytes": {
            name: (value + 7) // 8 for name, value in normalized.items()
        },
    }


def _communication_document(
    *,
    client_count: int,
    uplink_bits: Mapping[str, int],
    downlink_bits: Mapping[str, int],
) -> dict[str, Any]:
    uplink = _direction(uplink_bits, client_count=client_count)
    downlink = _direction(downlink_bits, client_count=client_count)
    combined_bits = uplink["total_bits"] + downlink["total_bits"]
    return {
        "accounting_model": ACCOUNTING_MODEL,
        "client_count": client_count,
        "transport_headers_included": False,
        "payload_scope": (
            "bit-packed cryptographic/model values plus sparse labels and seeds; "
            "TCP/TLS/IP and replay/debug framing excluded"
        ),
        "uplink": uplink,
        "downlink": downlink,
        "combined_total_bits": combined_bits,
        "combined_total_bytes": (combined_bits + 7) // 8,
        "combined_total_mb": _bits_to_mb(combined_bits),
        "combined_mean_per_client_bits": combined_bits / client_count,
        "combined_mean_per_client_bytes": combined_bits / 8.0 / client_count,
        "combined_mean_per_client_mb": (
            _bits_to_mb(combined_bits) / client_count
        ),
    }


def _bit_quantity(
    bits: int | float,
    *,
    client_count: int,
) -> dict[str, int | float]:
    """Describe a logical bit count without assuming byte-aligned fields."""

    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)) or bits < 0:
        raise ValueError("bits must be non-negative")
    total_bytes: int | float = (
        (bits + 7) // 8 if isinstance(bits, int) else bits / 8.0
    )
    return {
        "total_bits": bits,
        "total_bytes": total_bytes,
        "total_mb": _bits_to_mb(bits),
        "mean_per_registered_client_bits": bits / client_count,
        "mean_per_registered_client_bytes": bits / 8.0 / client_count,
        "mean_per_registered_client_mb": _bits_to_mb(bits) / client_count,
    }


def _communication_item(
    *,
    direction: str,
    profile_scope: str,
    bits_per_occurrence: int,
    occurrence_count: int,
    client_count: int,
    applicable: bool = True,
    included_in_total: bool = True,
    condition: str,
    formula: str,
) -> dict[str, Any]:
    if bits_per_occurrence < 0 or occurrence_count < 0:
        raise ValueError("communication item counts must be non-negative")
    total_bits = bits_per_occurrence * occurrence_count
    result: dict[str, Any] = {
        "direction": direction,
        "profile_scope": profile_scope,
        "applicable": applicable,
        "included_in_total": included_in_total,
        "condition": condition,
        "formula": formula,
        "bits_per_occurrence": bits_per_occurrence,
        "occurrence_count": occurrence_count,
    }
    result.update(_bit_quantity(total_bits, client_count=client_count))
    return result


def _itemized_section(
    items: Mapping[str, Mapping[str, Any]],
    *,
    client_count: int,
) -> dict[str, Any]:
    included_bits = sum(
        int(item["total_bits"])
        for item in items.values()
        if bool(item["included_in_total"])
    )
    direction_bits: dict[str, int] = {}
    for item in items.values():
        if not bool(item["included_in_total"]):
            continue
        direction = str(item["direction"])
        direction_bits[direction] = (
            direction_bits.get(direction, 0) + int(item["total_bits"])
        )
    return {
        "items": {name: dict(item) for name, item in items.items()},
        "direction_totals": {
            direction: _bit_quantity(bits, client_count=client_count)
            for direction, bits in direction_bits.items()
        },
        "total": _bit_quantity(included_bits, client_count=client_count),
    }


def _sum_itemized_rounds(
    online_rounds: list[Mapping[str, Any]],
    *,
    client_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cumulative: dict[str, dict[str, Any]] = {}
    for round_document in online_rounds:
        itemization = round_document["v17_itemization"]
        items = itemization["items"]
        for name, item in items.items():
            if name not in cumulative:
                cumulative[name] = dict(item)
                cumulative[name]["occurrence_count"] = 0
                cumulative[name]["total_bits"] = 0
            cumulative[name]["occurrence_count"] += int(
                item["occurrence_count"]
            )
            cumulative[name]["total_bits"] += int(item["total_bits"])

    for item in cumulative.values():
        quantity = _bit_quantity(
            int(item["total_bits"]), client_count=client_count
        )
        item.update(quantity)

    cumulative_section = _itemized_section(
        cumulative, client_count=client_count
    )
    round_count = len(online_rounds)
    mean_items: dict[str, dict[str, Any]] = {}
    for name, item in cumulative.items():
        mean_item = dict(item)
        mean_item["occurrence_count"] = item["occurrence_count"] / round_count
        mean_item.update(
            _bit_quantity(
                item["total_bits"] / round_count,
                client_count=client_count,
            )
        )
        mean_items[name] = mean_item

    included_mean_bits = sum(
        float(item["total_bits"])
        for item in mean_items.values()
        if bool(item["included_in_total"])
    )
    direction_mean_bits: dict[str, float] = {}
    for item in mean_items.values():
        if not bool(item["included_in_total"]):
            continue
        direction = str(item["direction"])
        direction_mean_bits[direction] = (
            direction_mean_bits.get(direction, 0.0)
            + float(item["total_bits"])
        )
    mean_section = {
        "items": mean_items,
        "direction_totals": {
            direction: _bit_quantity(bits, client_count=client_count)
            for direction, bits in direction_mean_bits.items()
        },
        "total": _bit_quantity(
            included_mean_bits, client_count=client_count
        ),
    }
    return cumulative_section, mean_section


def measure_setup_communication(
    *,
    setup: CryptoEngineSetup,
    profile: CryptoParameterProfile,
    effective_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Model one fixed-survivor SPARSE or FULL setup as all-unicast."""

    if profile.protocol_version != "v17":
        raise ValueError("communication accounting requires protocol v17")
    setup.validate()
    client_count = len(setup.clients)
    if client_count != profile.clients:
        raise ValueError("setup/profile client counts do not match")

    # A QP polynomial is modeled at the configured fixed-width tower budget.
    # Uniform p1 and M2R common-a polynomials are regenerated from the CRS seed.
    qp_bits_per_coefficient = sum(profile.q_modulus_bits) + sum(
        profile.p_modulus_bits
    )
    qp_polynomial_bits = profile.ring_dimension * qp_bits_per_coefficient
    digit_count = profile.hybrid_large_digits

    layout_bytes = sum(
        len(layout.canonical_bytes())
        for layout in (
            setup.parameter_layout,
            setup.running_mean_layout,
            setup.running_variance_layout,
        )
    )
    profile_bytes = len(
        json.dumps(
            profile.document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    setup_descriptor_bytes = profile_bytes
    q_primes: tuple[int, ...] = ()
    p_primes: tuple[int, ...] = ()
    if profile.protocol_version in {"v17"} and effective_protocol is not None:
        setup_descriptor_bytes = _positive_integer(
            effective_protocol, "setup_descriptor_bytes"
        )
        q_primes = _positive_integer_sequence(effective_protocol, "q_primes")
        p_primes = _positive_integer_sequence(effective_protocol, "p_primes")
        if tuple(value.bit_length() for value in q_primes) != tuple(
            profile.q_modulus_bits
        ):
            raise ValueError("native Q prime sequence differs from the profile")
        if tuple(value.bit_length() for value in p_primes) != tuple(
            profile.p_modulus_bits
        ):
            raise ValueError("native P prime sequence differs from the profile")
    public_descriptor_and_layout_bytes = setup_descriptor_bytes + layout_bytes
    uplink_bits = {
        "client_registration": client_count * (64 + 64),
        "collective_public_key_p0_shares": client_count * qp_polynomial_bits,
        "m2r_key_b_shares": (
            0
            if profile.mode == "FULL" or profile.mlwe_rank == 1
            else client_count
                * profile.mlwe_rank
                * digit_count
                * qp_polynomial_bits
        ),
        "client_to_collective_key_pairs": (
            client_count * 2 * digit_count * qp_polynomial_bits
        ),
    }
    initial_model_value_count = (
        setup.parameter_layout.total_numel
        + setup.running_mean_layout.total_numel
        + setup.running_variance_layout.total_numel
    )
    downlink_bits = {
        "protocol_profile_and_layouts": (
            client_count * public_descriptor_and_layout_bytes * 8
        ),
        # Layout hashes already live in the transmitted profile document.
        "setup_identifier": client_count * 128,
        # Each client receives only its own logical ID and integer alpha_i.
        # A full cohort table is neither required nor charged.
        "client_id_and_weight_assignment": client_count * (64 + 32),
        "crs_and_group_index_seeds": (
            client_count * (1 if profile.mode == "FULL" else 2) * 128
        ),
        "collective_public_key_p0": client_count * qp_polynomial_bits,
        "initial_global_model_float32": (
            client_count * initial_model_value_count * 32
        ),
    }
    result = _communication_document(
        client_count=client_count,
        uplink_bits=uplink_bits,
        downlink_bits=downlink_bits,
    )
    result["parameters"] = {
        "ring_dimension": profile.ring_dimension,
        "qp_bits_per_coefficient": qp_bits_per_coefficient,
        "qp_polynomial_bits": qp_polynomial_bits,
        "mlwe_rank": profile.mlwe_rank,
        "hybrid_large_digits": digit_count,
        "initial_model_value_count": initial_model_value_count,
        "setup_descriptor_bytes": setup_descriptor_bytes,
        "q_primes": list(q_primes),
        "p_primes": list(p_primes),
    }
    sparse_profile = profile.mode == "SPARSE"
    uses_m2r = sparse_profile and profile.mlwe_rank > 1
    descriptor_bits_per_client = (
        public_descriptor_and_layout_bytes * 8
        + 128  # setup_id
        + 128  # public seed_CRS
    )
    protocol_items = {
        "crs_setup_descriptor": _communication_item(
            direction="server_to_registered_clients",
            profile_scope="COMMON",
            bits_per_occurrence=descriptor_bits_per_client,
            occurrence_count=client_count,
            client_count=client_count,
            condition="one-time setup",
            formula=(
                "n * (native canonical setup descriptor/layout bytes + "
                "128-bit setup_id + 128-bit seed_CRS)"
            ),
        ),
        "collective_pk_share": _communication_item(
            direction="client_to_server",
            profile_scope="COMMON",
            bits_per_occurrence=qp_polynomial_bits,
            occurrence_count=client_count,
            client_count=client_count,
            condition="one share from each registered client",
            formula="n * N * log2(QP)",
        ),
        "initial_collective_target_descriptor": _communication_item(
            direction="server_to_registered_clients",
            profile_scope="COMMON",
            bits_per_occurrence=qp_polynomial_bits + client_count,
            occurrence_count=client_count,
            client_count=client_count,
            condition="initial target is the registered cohort C",
            formula="n * (N * log2(QP) + n-bit survivor bitmap)",
        ),
        "initial_client_to_collective_ksk": _communication_item(
            direction="client_to_server",
            profile_scope="COMMON",
            bits_per_occurrence=(
                2 * digit_count * qp_polynomial_bits
            ),
            occurrence_count=client_count,
            client_count=client_count,
            condition="generated once for the initial collective target C",
            formula="n * 2m * N * log2(QP)",
        ),
        "m2r_swk_share": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                profile.mlwe_rank * digit_count * qp_polynomial_bits
                if uses_m2r
                else 0
            ),
            occurrence_count=client_count if uses_m2r else 0,
            client_count=client_count,
            applicable=uses_m2r,
            included_in_total=uses_m2r,
            condition=(
                "sparse k > 1"
                if uses_m2r
                else "not used by FULL or sparse k=1"
            ),
            formula="n * k * m * N * log2(QP)",
        ),
        "index_key_setup": _communication_item(
            direction="designated_client_to_other_sparse_clients",
            profile_scope="SPARSE",
            bits_per_occurrence=128 if sparse_profile else 0,
            occurrence_count=client_count - 1 if sparse_profile else 0,
            client_count=client_count,
            applicable=sparse_profile,
            included_in_total=sparse_profile,
            condition=(
                "designated client i0 distributes Kidx; server excluded"
                if sparse_profile
                else "not used by FULL"
            ),
            formula="128 * (n - 1)",
        ),
        "other_setup_control_metadata": _communication_item(
            direction="setup_participants",
            profile_scope="COMMON",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=True,
            included_in_total=False,
            condition=(
                "registration, acknowledgement, client identifiers, and "
                "server-only alpha_i are excluded"
            ),
            formula="excluded control-plane payload",
        ),
    }
    initial_items = {
        "initial_model_parameters_float32": _communication_item(
            direction="server_to_registered_clients",
            profile_scope="APPLICATION_INITIAL",
            bits_per_occurrence=setup.parameter_layout.total_numel * 32,
            occurrence_count=client_count,
            client_count=client_count,
            condition="one initial global trainable-model delivery",
            formula="n * 32d",
        ),
        "initial_auxiliary_state_float32": _communication_item(
            direction="server_to_registered_clients",
            profile_scope="APPLICATION_INITIAL",
            bits_per_occurrence=(
                setup.running_mean_layout.total_numel
                + setup.running_variance_layout.total_numel
            )
            * 32,
            occurrence_count=client_count,
            client_count=client_count,
            condition="one initial global auxiliary-state delivery",
            formula="n * 32d_aux",
        ),
    }
    protocol_setup = _itemized_section(
        protocol_items, client_count=client_count
    )
    initial_communication = _itemized_section(
        initial_items, client_count=client_count
    )
    setup_itemization = {
        "accounting_model": V17_ACCOUNTING_MODEL,
        "transport_headers_included": False,
        "control_metadata_included": False,
        "protocol_setup": protocol_setup,
        "initial_communication": initial_communication,
        "end_to_end_total": _bit_quantity(
            int(protocol_setup["total"]["total_bits"])
            + int(initial_communication["total"]["total_bits"]),
            client_count=client_count,
        ),
    }
    result["v17_itemization"] = setup_itemization
    return result


def measure_full_online_round_communication(
    *,
    setup: CryptoEngineSetup,
    effective_protocol: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure a direct-RLWE FULL round with no sparse routing metadata."""

    if effective_protocol.get("protocol_version", "v17") != "v17":
        raise ValueError("communication accounting requires protocol v17")
    setup.validate()
    if setup.protocol_mode != "FULL":
        raise ValueError("FULL communication requires a FULL setup")
    client_count = len(setup.clients)
    ring_dimension = _positive_integer(effective_protocol, "ring_dimension")
    c1 = _positive_integer(effective_protocol, "compression_c1")
    ca = _positive_integer(effective_protocol, "compression_ca")
    c2 = _positive_integer(effective_protocol, "compression_c2")
    cd = _positive_integer(effective_protocol, "compression_cd")
    aux_c1 = _positive_integer(effective_protocol, "aux_compression_c1")
    aux_ca = _positive_integer(effective_protocol, "aux_compression_ca")
    aux_c2 = _positive_integer(effective_protocol, "aux_compression_c2")
    aux_cd = _positive_integer(effective_protocol, "aux_compression_cd")
    model_client_ciphertexts = _positive_integer(
        diagnostics, "full_model_client_ciphertext_count"
    )
    model_aggregate_ciphertexts = _positive_integer(
        diagnostics, "full_model_aggregate_ciphertext_count"
    )
    auxiliary_client_ciphertexts = _positive_integer(
        diagnostics, "auxiliary_client_ciphertext_count"
    )
    auxiliary_aggregate_ciphertexts = _positive_integer(
        diagnostics, "auxiliary_aggregate_ciphertext_count"
    )
    if _integer(diagnostics, "r2m_ciphertext_count") != 0:
        raise ValueError("FULL path unexpectedly reported R2M ciphertexts")
    if _integer(diagnostics, "m2r_ciphertext_count") != 0:
        raise ValueError("FULL path unexpectedly reported M2R ciphertexts")
    survivor_reconfiguration = bool(
        diagnostics.get("survivor_reconfiguration_applied", False)
    )
    survivor_target_cache_hit = bool(
        diagnostics.get("survivor_target_key_cache_hit", False)
    )
    survivor_ids = tuple(
        int(value)
        for value in diagnostics.get("survivor_client_ids", ())
    )
    dropped_ids = tuple(
        int(value)
        for value in diagnostics.get("dropped_client_ids", ())
    )
    if survivor_reconfiguration:
        if (
            not survivor_ids
            or len(set(survivor_ids)) != len(survivor_ids)
            or len(set(dropped_ids)) != len(dropped_ids)
            or set(survivor_ids).intersection(dropped_ids)
            or len(survivor_ids) + len(dropped_ids) != client_count
        ):
            raise ValueError("invalid survivor/dropout diagnostics")
        survivor_count = len(survivor_ids)
    else:
        survivor_count = client_count
        survivor_ids = setup.client_ids
        dropped_ids = ()
    new_target_sizes_raw = diagnostics.get("survivor_new_target_sizes")
    if new_target_sizes_raw is None:
        new_target_client_count = (
            survivor_count
            if survivor_reconfiguration and not survivor_target_cache_hit
            else 0
        )
    else:
        new_target_sizes = tuple(int(value) for value in new_target_sizes_raw)
        if any(value < 1 or value > client_count for value in new_target_sizes):
            raise ValueError("invalid survivor target size")
        new_target_client_count = sum(new_target_sizes)
    q_tower_bits = _positive_integer_sequence(
        effective_protocol, "q_modulus_bits"
    )
    p_tower_bits = _positive_integer_sequence(
        effective_protocol, "p_modulus_bits"
    )
    hybrid_large_digits = _positive_integer(
        effective_protocol, "hybrid_large_digits"
    )
    qp_polynomial_bits = ring_dimension * (
        sum(q_tower_bits) + sum(p_tower_bits)
    )
    updated_target_bits = (
        new_target_client_count * (qp_polynomial_bits + client_count)
    )
    updated_ksk_bits = (
        new_target_client_count
        * 2
        * hybrid_large_digits
        * qp_polynomial_bits
    )
    parameter_numel = setup.parameter_layout.total_numel
    auxiliary_numel = (
        setup.running_mean_layout.total_numel
        + setup.running_variance_layout.total_numel
    )
    expected_model_blocks = (
        parameter_numel + ring_dimension - 1
    ) // ring_dimension
    if model_client_ciphertexts != client_count * expected_model_blocks:
        raise ValueError("FULL client ciphertext count is inconsistent")
    if model_aggregate_ciphertexts != expected_model_blocks:
        raise ValueError("FULL aggregate ciphertext count is inconsistent")

    uplink_bits = {
        "updated_client_to_collective_ksk": updated_ksk_bits,
        "model_a_seeds": model_client_ciphertexts * 128,
        "model_client_b_c1": (
            model_client_ciphertexts * ring_dimension * c1
        ),
        "auxiliary_a_seeds": auxiliary_client_ciphertexts * 128,
        "auxiliary_client_b_c1": (
            auxiliary_client_ciphertexts * ring_dimension * aux_c1
        ),
        "model_partial_decryption_c2": (
            survivor_count
            * model_aggregate_ciphertexts
            * ring_dimension
            * c2
        ),
        "auxiliary_partial_decryption_c2": (
            survivor_count
            * auxiliary_aggregate_ciphertexts
            * ring_dimension
            * aux_c2
        ),
    }
    dd_seed_bits = 128
    downlink_bits = {
        "updated_collective_target_descriptor": updated_target_bits,
        "dd_public_seed": survivor_count * dd_seed_bits,
        "model_aggregate_a_ca": (
            survivor_count
            * model_aggregate_ciphertexts
            * ring_dimension
            * ca
        ),
        "auxiliary_aggregate_a_ca": (
            survivor_count
            * auxiliary_aggregate_ciphertexts
            * ring_dimension
            * aux_ca
        ),
        "final_model_values_cd": client_count * parameter_numel * cd,
        "final_auxiliary_values_cd": client_count * auxiliary_numel * aux_cd,
    }
    result = _communication_document(
        client_count=client_count,
        uplink_bits=uplink_bits,
        downlink_bits=downlink_bits,
    )
    result["parameters"] = {
        "protocol_mode": "FULL",
        "survivor_client_ids": list(survivor_ids),
        "dropped_client_ids": list(dropped_ids),
        "survivor_new_target_sizes": list(new_target_sizes_raw or (() if not new_target_client_count else (survivor_count,))),
        "ring_dimension": ring_dimension,
        "parameter_value_count": parameter_numel,
        "parameter_block_count": expected_model_blocks,
        "model_client_ciphertext_count": model_client_ciphertexts,
        "model_aggregate_ciphertext_count": model_aggregate_ciphertexts,
        "auxiliary_value_count": auxiliary_numel,
        "auxiliary_client_ciphertext_count": auxiliary_client_ciphertexts,
        "auxiliary_aggregate_ciphertext_count": (
            auxiliary_aggregate_ciphertexts
        ),
        "r2m_ciphertext_count": 0,
        "m2r_ciphertext_count": 0,
        "c1": c1,
        "ca": ca,
        "c2": c2,
        "cd": cd,
        "aux_c1": aux_c1,
        "aux_ca": aux_ca,
        "aux_c2": aux_c2,
        "aux_cd": aux_cd,
    }
    items = {
        "updated_collective_target_descriptor": _communication_item(
            direction="server_to_current_survivors",
            profile_scope="COMMON",
            bits_per_occurrence=(
                qp_polynomial_bits + client_count
                if new_target_client_count
                else 0
            ),
            occurrence_count=(
                new_target_client_count
            ),
            client_count=client_count,
            condition=(
                "post-upload survivor target S*"
                if new_target_client_count
                else "cached collective target reused"
            ),
            formula="|S*| * (N*log2(QP) + n-bit survivor bitmap)",
        ),
        "updated_client_to_collective_ksk": _communication_item(
            direction="client_to_server",
            profile_scope="COMMON",
            bits_per_occurrence=(
                2 * hybrid_large_digits * qp_polynomial_bits
                if new_target_client_count
                else 0
            ),
            occurrence_count=(
                new_target_client_count
            ),
            client_count=client_count,
            condition=(
                "fresh KSK_i_to_S* from each survivor"
                if new_target_client_count
                else "cached client-to-collective KSK reused"
            ),
            formula="|S*| * 2m * N * log2(QP)",
        ),
        "sparse_selection_bitmap": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "sparse_admission_bitmap": _communication_item(
            direction="server_to_current_survivors",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "sparse_upload_support_bitmap": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "sparse_final_context_bitmap": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "sparse_result_bitmap": _communication_item(
            direction="server_to_clients",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "sparse_model_ciphertext_uplink": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "full_model_ciphertext_uplink": _communication_item(
            direction="client_to_server",
            profile_scope="FULL",
            bits_per_occurrence=128 + ring_dimension * c1,
            occurrence_count=model_client_ciphertexts,
            client_count=client_count,
            condition="FULL model path",
            formula="n * R_model * (128 + N*c1)",
        ),
        "auxiliary_ciphertext_uplink": _communication_item(
            direction="client_to_server",
            profile_scope="AUXILIARY",
            bits_per_occurrence=128 + ring_dimension * aux_c1,
            occurrence_count=auxiliary_client_ciphertexts,
            client_count=client_count,
            condition="all profiles",
            formula="n * R_aux * (128 + N*c1_aux)",
        ),
        "sparse_model_aggregate_a_downlink": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "full_model_aggregate_a_downlink": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="FULL",
            bits_per_occurrence=(
                model_aggregate_ciphertexts * ring_dimension * ca
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="one transmission per final survivor",
            formula="|S*| * R_model * N * ca",
        ),
        "auxiliary_aggregate_a_downlink": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="AUXILIARY",
            bits_per_occurrence=(
                auxiliary_aggregate_ciphertexts * ring_dimension * aux_ca
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="one transmission per final survivor",
            formula="|S*| * R_aux * N * ca_aux",
        ),
        "sparse_model_partial_decryption": _communication_item(
            direction="final_survivor_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "full_model_partial_decryption": _communication_item(
            direction="final_survivor_to_server",
            profile_scope="FULL",
            bits_per_occurrence=(
                model_aggregate_ciphertexts * ring_dimension * c2
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="one transmission per final survivor",
            formula="|S*| * R_model * N * c2",
        ),
        "auxiliary_partial_decryption": _communication_item(
            direction="final_survivor_to_server",
            profile_scope="AUXILIARY",
            bits_per_occurrence=(
                auxiliary_aggregate_ciphertexts * ring_dimension * aux_c2
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="one transmission per final survivor",
            formula="|S*| * R_aux * N * c2_aux",
        ),
        "sparse_model_result": _communication_item(
            direction="server_to_clients",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by FULL",
            formula="0",
        ),
        "full_model_result": _communication_item(
            direction="server_to_clients",
            profile_scope="FULL",
            bits_per_occurrence=parameter_numel * cd,
            occurrence_count=client_count,
            client_count=client_count,
            condition="fixed ordering; no coordinate labels",
            formula="n * d * cd",
        ),
        "auxiliary_result": _communication_item(
            direction="server_to_clients",
            profile_scope="AUXILIARY",
            bits_per_occurrence=auxiliary_numel * aux_cd,
            occurrence_count=client_count,
            client_count=client_count,
            condition="fixed-order dense auxiliary update",
            formula="n * d_aux * cd_aux",
        ),
        "dd_public_seed": _communication_item(
            direction="server_to_final_survivors", profile_scope="COMMON",
            bits_per_occurrence=dd_seed_bits, occurrence_count=survivor_count,
            client_count=client_count, applicable=bool(dd_seed_bits),
            condition="one shared late seed per finalized v17 attempt",
            formula="128 * |S*|",
        ),
        "round_control_metadata": _communication_item(
            direction="round_participants",
            profile_scope="COMMON",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            included_in_total=False,
            condition=(
                "round id, survivor bitmap, and finalize/abort flags excluded"
            ),
            formula="excluded control-plane payload",
        ),
    }
    section = _itemized_section(items, client_count=client_count)
    full_itemization = {
        "accounting_model": V17_ACCOUNTING_MODEL,
        "protocol_mode": "FULL",
        "survivor_client_ids": list(survivor_ids),
        "dropped_client_ids": list(dropped_ids),
        "survivor_new_target_sizes": list(new_target_sizes_raw or (() if not new_target_client_count else (survivor_count,))),
        "final_survivor_count": survivor_count,
        **section,
    }
    result["v17_itemization"] = full_itemization
    return result


def measure_online_round_communication(
    *,
    setup: CryptoEngineSetup,
    effective_protocol: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    accepted_chunk_count: int,
    upload_accepted_chunk_count: int | None = None,
    uploaded_sparse_contribution_count: int | None = None,
) -> dict[str, Any]:
    """Measure one completed online round from its actual native counts."""

    if effective_protocol.get("protocol_version", "v17") != "v17":
        raise ValueError("communication accounting requires protocol v17")
    setup.validate()
    if accepted_chunk_count < 0:
        raise ValueError("accepted_chunk_count must be non-negative")
    if upload_accepted_chunk_count is None:
        upload_accepted_chunk_count = accepted_chunk_count
    if upload_accepted_chunk_count < accepted_chunk_count:
        raise ValueError(
            "upload accepted chunk count cannot be below final accepted count"
        )
    client_count = len(setup.clients)
    ring_dimension = _positive_integer(effective_protocol, "ring_dimension")
    mlwe_rank = _positive_integer(effective_protocol, "mlwe_rank")
    selected_chunks = _positive_integer(
        effective_protocol, "selected_chunks_per_client"
    )
    c1 = _positive_integer(effective_protocol, "compression_c1")
    ca = _positive_integer(effective_protocol, "compression_ca")
    c2 = _positive_integer(effective_protocol, "compression_c2")
    cd = _positive_integer(effective_protocol, "compression_cd")
    aux_c1 = _positive_integer(effective_protocol, "aux_compression_c1")
    aux_ca = _positive_integer(effective_protocol, "aux_compression_ca")
    aux_c2 = _positive_integer(effective_protocol, "aux_compression_c2")
    aux_cd = _positive_integer(effective_protocol, "aux_compression_cd")

    sparse_client_ciphertexts = _integer(
        diagnostics, "sparse_client_ciphertext_count"
    )
    accepted_contributions = _integer(
        diagnostics, "accepted_sparse_contribution_count"
    )
    if uploaded_sparse_contribution_count is None:
        uploaded_sparse_contribution_count = accepted_contributions
    if uploaded_sparse_contribution_count < accepted_contributions:
        raise ValueError(
            "uploaded sparse contributions cannot be below final contributions"
        )
    if upload_accepted_chunk_count > setup.parameter_chunk_count:
        raise ValueError("upload accepted chunk count exceeds parameter chunks")
    if uploaded_sparse_contribution_count > client_count * selected_chunks:
        raise ValueError("uploaded sparse contributions exceed selections")
    aggregated_mlwe_ciphertexts = _integer(
        diagnostics, "aggregated_mlwe_ciphertext_count"
    )
    sparse_server_ciphertexts = _integer(
        diagnostics, "sparse_server_ciphertext_count"
    )
    auxiliary_client_ciphertexts = _integer(
        diagnostics, "auxiliary_client_ciphertext_count"
    )
    auxiliary_aggregate_ciphertexts = _integer(
        diagnostics, "auxiliary_aggregate_ciphertext_count"
    )
    survivor_reconfiguration = bool(
        diagnostics.get("survivor_reconfiguration_applied", False)
    )
    survivor_target_cache_hit = bool(
        diagnostics.get("survivor_target_key_cache_hit", False)
    )
    survivor_ids = tuple(
        int(value)
        for value in diagnostics.get("survivor_client_ids", ())
    )
    dropped_ids = tuple(
        int(value)
        for value in diagnostics.get("dropped_client_ids", ())
    )
    if survivor_reconfiguration:
        if (
            not survivor_ids
            or len(set(survivor_ids)) != len(survivor_ids)
            or len(set(dropped_ids)) != len(dropped_ids)
            or set(survivor_ids).intersection(dropped_ids)
            or len(survivor_ids) + len(dropped_ids) != client_count
        ):
            raise ValueError("invalid survivor/dropout diagnostics")
        survivor_count = len(survivor_ids)
    else:
        survivor_count = client_count
        survivor_ids = setup.client_ids
        dropped_ids = ()
    new_target_sizes_raw = diagnostics.get("survivor_new_target_sizes")
    if new_target_sizes_raw is None:
        new_target_client_count = (
            survivor_count
            if survivor_reconfiguration and not survivor_target_cache_hit
            else 0
        )
    else:
        new_target_sizes = tuple(int(value) for value in new_target_sizes_raw)
        if any(value < 1 or value > client_count for value in new_target_sizes):
            raise ValueError("invalid survivor target size")
        new_target_client_count = sum(new_target_sizes)
    q_tower_bits = _positive_integer_sequence(
        effective_protocol, "q_modulus_bits"
    )
    p_tower_bits = _positive_integer_sequence(
        effective_protocol, "p_modulus_bits"
    )
    hybrid_large_digits = _positive_integer(
        effective_protocol, "hybrid_large_digits"
    )
    qp_polynomial_bits = ring_dimension * (
        sum(q_tower_bits) + sum(p_tower_bits)
    )
    updated_target_bits = (
        new_target_client_count * (qp_polynomial_bits + client_count)
    )
    updated_ksk_bits = (
        new_target_client_count
        * 2
        * hybrid_large_digits
        * qp_polynomial_bits
    )
    if accepted_contributions > client_count * selected_chunks:
        raise ValueError("accepted sparse contributions exceed selections")
    # Rank-one sparse execution stays in RLWE throughout and therefore has no
    # intermediate MLWE ciphertexts.  Higher-rank profiles still aggregate one
    # MLWE ciphertext per accepted chunk before the server-internal M2R step.
    expected_aggregated_mlwe_ciphertexts = (
        0 if mlwe_rank == 1 else accepted_chunk_count
    )
    if aggregated_mlwe_ciphertexts != expected_aggregated_mlwe_ciphertexts:
        raise ValueError("accepted chunk and aggregated MLWE counts differ")

    chunk_size = setup.chunk_size
    chunk_count = setup.parameter_chunk_count
    label_bits = max(1, (chunk_count - 1).bit_length())
    support_bitmap_bytes = (chunk_count + 7) // 8
    support_bitmap_bits = support_bitmap_bytes * 8
    protocol_version = str(effective_protocol.get("protocol_version", "v17"))
    uses_v17_bitmap = protocol_version == "v17"
    selection_bitmap_count = _optional_non_negative_integer(
        diagnostics, "selection_bitmap_count", client_count
    )
    admission_bitmap_recipient_count = _optional_non_negative_integer(
        diagnostics, "admission_bitmap_recipient_count", client_count
    )
    upload_support_bitmap_count = _optional_non_negative_integer(
        diagnostics, "upload_support_bitmap_count", client_count
    )
    result_recipient_count = _optional_non_negative_integer(
        diagnostics, "result_recipient_count", client_count
    )
    auxiliary_numel = (
        setup.running_mean_layout.total_numel
        + setup.running_variance_layout.total_numel
    )

    uplink_bits = {
        (
            "selection_support_bitmap"
            if uses_v17_bitmap
            else "selection_labels"
        ): (
            selection_bitmap_count * support_bitmap_bits
            if uses_v17_bitmap
            else client_count * selected_chunks * label_bits
        ),
        (
            "upload_support_bitmap"
            if uses_v17_bitmap
            else "sparse_packet_labels"
        ): (
            upload_support_bitmap_count * support_bitmap_bits
            if uses_v17_bitmap
            else uploaded_sparse_contribution_count * label_bits
        ),
        "sparse_a_seeds": sparse_client_ciphertexts * 128,
        "sparse_client_b_c1": (
            uploaded_sparse_contribution_count * chunk_size * c1
        ),
        "updated_client_to_collective_ksk": updated_ksk_bits,
        "auxiliary_a_seeds": auxiliary_client_ciphertexts * 128,
        "auxiliary_client_b_c1": (
            auxiliary_client_ciphertexts * ring_dimension * aux_c1
        ),
        "sparse_partial_decryption_c2": (
            survivor_count * accepted_chunk_count * chunk_size * c2
        ),
        "auxiliary_partial_decryption_c2": (
            survivor_count
            * auxiliary_aggregate_ciphertexts
            * ring_dimension
            * aux_c2
        ),
    }
    dd_seed_bits = 128
    downlink_bits = {
        "dd_public_seed": survivor_count * dd_seed_bits,
        "updated_collective_target_descriptor": updated_target_bits,
        (
            "admission_support_bitmap"
            if uses_v17_bitmap
            else "admission_accepted_labels"
        ): (
            admission_bitmap_recipient_count * support_bitmap_bits
            if uses_v17_bitmap
            else client_count * upload_accepted_chunk_count * label_bits
        ),
        "final_context_support_bitmap": (
            survivor_count * support_bitmap_bits
            if uses_v17_bitmap
            else 0
        ),
        "sparse_aggregate_a_ca": (
            survivor_count * sparse_server_ciphertexts * ring_dimension * ca
        ),
        "auxiliary_aggregate_a_ca": (
            survivor_count
            * auxiliary_aggregate_ciphertexts
            * ring_dimension
            * aux_ca
        ),
        "result_support_bitmap": (
            result_recipient_count * support_bitmap_bits
            if uses_v17_bitmap
            else 0
        ),
        "final_sparse_values_cd": (
            result_recipient_count * accepted_chunk_count * chunk_size * cd
        ),
        "final_auxiliary_values_cd": (
            result_recipient_count * auxiliary_numel * aux_cd
        ),
    }
    result = _communication_document(
        client_count=client_count,
        uplink_bits=uplink_bits,
        downlink_bits=downlink_bits,
    )
    result["parameters"] = {
        "ring_dimension": ring_dimension,
        "mlwe_rank": mlwe_rank,
        "chunk_size": chunk_size,
        "parameter_chunk_count": chunk_count,
        "label_bits": label_bits,
        "support_codec": (
            "CHUNK-BITMAP-MSB0-V1" if uses_v17_bitmap else "LEGACY-LABEL-LIST"
        ),
        "support_bitmap_bytes": support_bitmap_bytes,
        "support_bitmap_bits": support_bitmap_bits,
        "selection_bitmap_count": selection_bitmap_count,
        "admission_bitmap_recipient_count": admission_bitmap_recipient_count,
        "upload_support_bitmap_count": upload_support_bitmap_count,
        "result_recipient_count": result_recipient_count,
        "selected_chunks_per_client": selected_chunks,
        "accepted_chunk_count": accepted_chunk_count,
        "upload_accepted_chunk_count": upload_accepted_chunk_count,
        "accepted_sparse_contribution_count": accepted_contributions,
        "uploaded_sparse_contribution_count": (
            uploaded_sparse_contribution_count
        ),
        "survivor_reconfiguration_applied": survivor_reconfiguration,
        "survivor_target_key_cache_hit": survivor_target_cache_hit,
        "survivor_client_ids": list(survivor_ids),
        "dropped_client_ids": list(dropped_ids),
        "final_survivor_count": survivor_count,
        "aggregated_mlwe_ciphertext_count": aggregated_mlwe_ciphertexts,
        "sparse_client_ciphertext_count": sparse_client_ciphertexts,
        "sparse_server_ciphertext_count": sparse_server_ciphertexts,
        "auxiliary_client_ciphertext_count": auxiliary_client_ciphertexts,
        "auxiliary_aggregate_ciphertext_count": (
            auxiliary_aggregate_ciphertexts
        ),
        "auxiliary_value_count": auxiliary_numel,
        "c1": c1,
        "ca": ca,
        "c2": c2,
        "cd": cd,
        "aux_c1": aux_c1,
        "aux_ca": aux_ca,
        "aux_c2": aux_c2,
        "aux_cd": aux_cd,
    }
    items = {
        "updated_collective_target_descriptor": _communication_item(
            direction="server_to_current_survivors",
            profile_scope="COMMON",
            bits_per_occurrence=(
                qp_polynomial_bits + client_count
                if new_target_client_count
                else 0
            ),
            occurrence_count=(
                new_target_client_count
            ),
            client_count=client_count,
            condition=(
                "post-upload survivor target S*"
                if new_target_client_count
                else "cached collective target reused"
            ),
            formula="|S*| * (N*log2(QP) + n-bit survivor bitmap)",
        ),
        "updated_client_to_collective_ksk": _communication_item(
            direction="client_to_server",
            profile_scope="COMMON",
            bits_per_occurrence=(
                2 * hybrid_large_digits * qp_polynomial_bits
                if new_target_client_count
                else 0
            ),
            occurrence_count=(
                new_target_client_count
            ),
            client_count=client_count,
            condition=(
                "fresh KSK_i_to_S* from each survivor"
                if new_target_client_count
                else "cached client-to-collective KSK reused"
            ),
            formula="|S*| * 2m * N * log2(QP)",
        ),
        "sparse_label_index_metadata": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                0 if uses_v17_bitmap else selected_chunks * label_bits
            ),
            occurrence_count=0 if uses_v17_bitmap else client_count,
            client_count=client_count,
            applicable=not uses_v17_bitmap,
            included_in_total=not uses_v17_bitmap,
            condition=(
                "legacy selection labels"
                if not uses_v17_bitmap
                else "replaced by CHUNK-BITMAP-MSB0-V1 in v17"
            ),
            formula="n * K_rho * ceil(log2(M))",
        ),
        "sparse_selection_bitmap": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                support_bitmap_bits if uses_v17_bitmap else 0
            ),
            occurrence_count=(
                selection_bitmap_count if uses_v17_bitmap else 0
            ),
            client_count=client_count,
            applicable=uses_v17_bitmap,
            included_in_total=uses_v17_bitmap,
            condition="one v17 selection bitmap per received announcement",
            formula="8*b_J*a_t",
        ),
        "sparse_admission_bitmap": _communication_item(
            direction="server_to_current_survivors",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                support_bitmap_bits if uses_v17_bitmap else 0
            ),
            occurrence_count=(
                admission_bitmap_recipient_count if uses_v17_bitmap else 0
            ),
            client_count=client_count,
            applicable=uses_v17_bitmap,
            included_in_total=uses_v17_bitmap,
            condition="one copy per actual admission revision recipient",
            formula="8*b_J*sum_v(s_t,v)",
        ),
        "sparse_upload_support_bitmap": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                support_bitmap_bits if uses_v17_bitmap else 0
            ),
            occurrence_count=(
                upload_support_bitmap_count if uses_v17_bitmap else 0
            ),
            client_count=client_count,
            applicable=uses_v17_bitmap,
            included_in_total=uses_v17_bitmap,
            condition="one immutable upload-support header per local collection",
            formula="8*b_J*u_t",
        ),
        "sparse_final_context_bitmap": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                support_bitmap_bits if uses_v17_bitmap else 0
            ),
            occurrence_count=survivor_count if uses_v17_bitmap else 0,
            client_count=client_count,
            applicable=uses_v17_bitmap,
            included_in_total=uses_v17_bitmap,
            condition="one finalized support copy per decryptor",
            formula="8*b_J*n_t",
        ),
        "sparse_result_bitmap": _communication_item(
            direction="server_to_clients",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                support_bitmap_bits if uses_v17_bitmap else 0
            ),
            occurrence_count=(
                result_recipient_count if uses_v17_bitmap else 0
            ),
            client_count=client_count,
            applicable=uses_v17_bitmap,
            included_in_total=uses_v17_bitmap,
            condition="one finalized support copy per result recipient",
            formula="8*b_J*d_t_recv",
        ),
        "sparse_model_ciphertext_uplink": _communication_item(
            direction="client_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=0,
            occurrence_count=1,
            client_count=client_count,
            condition="actual accepted packet and stream counts",
            formula="R_i seeds plus c1-compressed accepted chunk streams",
        ),
        "full_model_ciphertext_uplink": _communication_item(
            direction="client_to_server",
            profile_scope="FULL",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by SPARSE",
            formula="0",
        ),
        "auxiliary_ciphertext_uplink": _communication_item(
            direction="client_to_server",
            profile_scope="AUXILIARY",
            bits_per_occurrence=128 + ring_dimension * aux_c1,
            occurrence_count=auxiliary_client_ciphertexts,
            client_count=client_count,
            condition="all profiles",
            formula="n * R_aux * (128 + N*c1_aux)",
        ),
        "sparse_model_aggregate_a_downlink": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                sparse_server_ciphertexts * ring_dimension * ca
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="sent only to final survivors",
            formula="|S*| * R_t * N * ca",
        ),
        "full_model_aggregate_a_downlink": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="FULL",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by SPARSE",
            formula="0",
        ),
        "auxiliary_aggregate_a_downlink": _communication_item(
            direction="server_to_final_survivors",
            profile_scope="AUXILIARY",
            bits_per_occurrence=(
                auxiliary_aggregate_ciphertexts * ring_dimension * aux_ca
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="sent only to final survivors",
            formula="|S*| * R_aux * N * ca_aux",
        ),
        "sparse_model_partial_decryption": _communication_item(
            direction="final_survivor_to_server",
            profile_scope="SPARSE",
            bits_per_occurrence=accepted_chunk_count * chunk_size * c2,
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="one share from each final survivor",
            formula="|S*| * sum_r(c2*q_r*B)",
        ),
        "full_model_partial_decryption": _communication_item(
            direction="final_survivor_to_server",
            profile_scope="FULL",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by SPARSE",
            formula="0",
        ),
        "auxiliary_partial_decryption": _communication_item(
            direction="final_survivor_to_server",
            profile_scope="AUXILIARY",
            bits_per_occurrence=(
                auxiliary_aggregate_ciphertexts * ring_dimension * aux_c2
            ),
            occurrence_count=survivor_count,
            client_count=client_count,
            condition="one share from each final survivor",
            formula="|S*| * R_aux * N * c2_aux",
        ),
        "sparse_model_result": _communication_item(
            direction="server_to_clients",
            profile_scope="SPARSE",
            bits_per_occurrence=(
                accepted_chunk_count * chunk_size * cd
            ),
            occurrence_count=result_recipient_count,
            client_count=client_count,
            condition=(
                "v17 result values follow the separately counted result bitmap"
                if uses_v17_bitmap
                else "legacy result values use the cached accepted-label order"
            ),
            formula="d_t_recv * |Y*| * B*cd",
        ),
        "full_model_result": _communication_item(
            direction="server_to_clients",
            profile_scope="FULL",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            applicable=False,
            included_in_total=False,
            condition="not used by SPARSE",
            formula="0",
        ),
        "auxiliary_result": _communication_item(
            direction="server_to_clients",
            profile_scope="AUXILIARY",
            bits_per_occurrence=auxiliary_numel * aux_cd,
            occurrence_count=result_recipient_count,
            client_count=client_count,
            condition="fixed-order dense auxiliary update",
            formula="n * d_aux * cd_aux",
        ),
        "dd_public_seed": _communication_item(
            direction="server_to_final_survivors", profile_scope="COMMON",
            bits_per_occurrence=dd_seed_bits, occurrence_count=survivor_count,
            client_count=client_count, applicable=bool(dd_seed_bits),
            condition="one shared late seed per finalized v17 attempt",
            formula="128 * |S*|",
        ),
        "round_control_metadata": _communication_item(
            direction="round_participants",
            profile_scope="COMMON",
            bits_per_occurrence=0,
            occurrence_count=0,
            client_count=client_count,
            included_in_total=False,
            condition=(
                "admission/finalize/abort flags and survivor bitmap excluded"
            ),
            formula="excluded control-plane payload",
        ),
    }
    sparse_uplink = items["sparse_model_ciphertext_uplink"]
    sparse_uplink_bits = (
        sparse_client_ciphertexts * 128
        + uploaded_sparse_contribution_count * chunk_size * c1
    )
    sparse_uplink["bits_per_occurrence"] = sparse_uplink_bits
    sparse_uplink.update(
        _bit_quantity(sparse_uplink_bits, client_count=client_count)
    )
    section = _itemized_section(items, client_count=client_count)
    sparse_itemization = {
        "accounting_model": V17_ACCOUNTING_MODEL,
        "protocol_mode": "SPARSE",
        "final_survivor_count": survivor_count,
        **section,
    }
    result["v17_itemization"] = sparse_itemization
    return result


def measure_aborted_sparse_round_communication(
    *,
    setup: CryptoEngineSetup,
    effective_protocol: Mapping[str, Any],
    aborted_traffic: Mapping[str, Any],
) -> dict[str, Any]:
    """Count v17 traffic sent before a final-request dropout abort.

    Uploads, survivor rekeys, and aggregate-A downlinks were sent; no partial
    decryption response or decoded result was released. Control framing is
    excluded by the same convention as successful-round accounting.
    """

    if effective_protocol.get("protocol_version", "v17") != "v17":
        raise ValueError("communication accounting requires protocol v17")
    accepted_count = int(aborted_traffic["accepted_chunk_count"])
    upload_count = int(aborted_traffic["upload_accepted_chunk_count"])
    survivor_ids = tuple(int(value) for value in aborted_traffic["survivor_client_ids"])
    dropped_ids = tuple(int(value) for value in aborted_traffic["dropped_client_ids"])
    mlwe_rank = int(effective_protocol["mlwe_rank"])
    full_equivalent = measure_online_round_communication(
        setup=setup,
        effective_protocol=effective_protocol,
        diagnostics={
            "sparse_client_ciphertext_count": int(
                aborted_traffic["sparse_client_ciphertext_count"]
            ),
            "accepted_sparse_contribution_count": int(
                aborted_traffic["accepted_sparse_contribution_count"]
            ),
            "aggregated_mlwe_ciphertext_count": (
                accepted_count if mlwe_rank > 1 else 0
            ),
            "sparse_server_ciphertext_count": int(
                aborted_traffic["sparse_server_ciphertext_count"]
            ),
            "auxiliary_client_ciphertext_count": int(
                aborted_traffic["auxiliary_client_ciphertext_count"]
            ),
            "auxiliary_aggregate_ciphertext_count": int(
                aborted_traffic["auxiliary_aggregate_ciphertext_count"]
            ),
            "survivor_reconfiguration_applied": bool(dropped_ids),
            "survivor_client_ids": survivor_ids,
            "dropped_client_ids": dropped_ids,
            "survivor_new_target_sizes": tuple(
                int(value)
                for value in aborted_traffic["survivor_new_target_sizes"]
            ),
            "selection_bitmap_count": int(
                aborted_traffic.get(
                    "selection_bitmap_count", len(setup.clients)
                )
            ),
            "admission_bitmap_recipient_count": int(
                aborted_traffic.get(
                    "admission_bitmap_recipient_count", len(setup.clients)
                )
            ),
            "upload_support_bitmap_count": int(
                aborted_traffic.get(
                    "upload_support_bitmap_count", len(setup.clients)
                )
            ),
            "result_recipient_count": 0,
        },
        accepted_chunk_count=accepted_count,
        upload_accepted_chunk_count=upload_count,
        uploaded_sparse_contribution_count=int(
            aborted_traffic["uploaded_sparse_contribution_count"]
        ),
    )
    uplink = dict(full_equivalent["uplink"]["breakdown_bits"])
    downlink = dict(full_equivalent["downlink"]["breakdown_bits"])
    uplink["sparse_partial_decryption_c2"] = 0
    uplink["auxiliary_partial_decryption_c2"] = 0
    downlink["final_sparse_values_cd"] = 0
    downlink["final_auxiliary_values_cd"] = 0
    downlink["result_support_bitmap"] = 0
    report = _communication_document(
        client_count=len(setup.clients),
        uplink_bits=uplink,
        downlink_bits=downlink,
    )
    report["abort_boundary"] = "final-request-sent-before-partial-responses"
    report["native_round_id"] = int(aborted_traffic["round_id"])
    report["control_framing_included"] = False
    return report


def cumulative_communication(
    *,
    setup: Mapping[str, Any],
    online_rounds: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine one setup with completed online rounds."""

    if not online_rounds:
        raise ValueError("online_rounds must not be empty")
    client_count = int(setup["client_count"])
    online_uplink_bits = sum(
        int(round_document["uplink"]["total_bits"])
        for round_document in online_rounds
    )
    online_downlink_bits = sum(
        int(round_document["downlink"]["total_bits"])
        for round_document in online_rounds
    )
    online = _communication_document(
        client_count=client_count,
        uplink_bits={"completed_rounds": online_uplink_bits},
        downlink_bits={"completed_rounds": online_downlink_bits},
    )
    setup_uplink_bits = int(setup["uplink"]["total_bits"])
    setup_downlink_bits = int(setup["downlink"]["total_bits"])
    combined = _communication_document(
        client_count=client_count,
        uplink_bits={
            "setup": setup_uplink_bits,
            "online": online_uplink_bits,
        },
        downlink_bits={
            "setup": setup_downlink_bits,
            "online": online_downlink_bits,
        },
    )
    online_itemized_cumulative, online_itemized_mean = (
        _sum_itemized_rounds(
            online_rounds,
            client_count=client_count,
        )
    )
    setup_itemization = setup["v17_itemization"]
    setup_protocol_bits = int(
        setup_itemization["protocol_setup"]["total"]["total_bits"]
    )
    initial_bits = int(
        setup_itemization["initial_communication"]["total"]["total_bits"]
    )
    online_protocol_bits = int(
        online_itemized_cumulative["total"]["total_bits"]
    )
    protocol_bits = setup_protocol_bits + online_protocol_bits
    end_to_end_bits = protocol_bits + initial_bits
    itemized = {
        "accounting_model": V17_ACCOUNTING_MODEL,
        "transport_headers_included": False,
        "control_metadata_included": False,
        "completed_rounds": len(online_rounds),
        "setup": dict(setup_itemization),
        "online_mean_per_round": online_itemized_mean,
        "online_cumulative": online_itemized_cumulative,
        "v17_protocol_total": _bit_quantity(
            protocol_bits, client_count=client_count
        ),
        "protocol_total": _bit_quantity(
            protocol_bits, client_count=client_count
        ),
        "initial_communication": _bit_quantity(
            initial_bits, client_count=client_count
        ),
        "end_to_end_total": _bit_quantity(
            end_to_end_bits, client_count=client_count
        ),
        "total_identity": (
            "end_to_end_total = protocol_total + initial_communication"
        ),
    }
    cumulative_result = {
        "accounting_model": ACCOUNTING_MODEL,
        "completed_rounds": len(online_rounds),
        "setup": dict(setup),
        "online_cumulative": online,
        "setup_plus_online_cumulative": combined,
        "v17_itemized_communication": itemized,
        "online_mean_per_round": {
            "uplink_bits": online_uplink_bits / len(online_rounds),
            "uplink_bytes": online_uplink_bits / 8.0 / len(online_rounds),
            "uplink_mb": (
                _bits_to_mb(online_uplink_bits) / len(online_rounds)
            ),
            "downlink_bits": online_downlink_bits / len(online_rounds),
            "downlink_bytes": online_downlink_bits / 8.0 / len(online_rounds),
            "downlink_mb": (
                _bits_to_mb(online_downlink_bits) / len(online_rounds)
            ),
            "combined_bits": (
                online_uplink_bits + online_downlink_bits
            ) / len(online_rounds),
            "combined_bytes": (
                online_uplink_bits + online_downlink_bits
            ) / 8.0 / len(online_rounds),
            "combined_mb": _bits_to_mb(
                online_uplink_bits + online_downlink_bits
            ) / len(online_rounds),
        },
    }
    return cumulative_result
