"""FP32 communication accounting for plaintext federated learning.

Sparse v17 uses the same chunk universe and B-padding as the encrypted path,
with ``CHUNK-BITMAP-MSB0-V1`` support bodies over actual chunk indices.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .support_bitmap import SUPPORT_CODEC_NAME, bitmap_byte_count


PLAINTEXT_VALUE_BYTES = 4


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def measure_plaintext_full_round_communication(
    *,
    client_count: int,
    parameter_numel: int,
    auxiliary_numel: int,
) -> dict[str, Any]:
    """Measure a fixed-order dense FP32 round with all-unicast downlink."""

    client_count = _positive_integer("client_count", client_count)
    parameter_numel = _positive_integer("parameter_numel", parameter_numel)
    auxiliary_numel = _non_negative_integer("auxiliary_numel", auxiliary_numel)
    values_per_client = parameter_numel + auxiliary_numel
    logical_bytes = values_per_client * PLAINTEXT_VALUE_BYTES
    all_client_bytes = client_count * logical_bytes
    return {
        "profile": "FULL",
        "padding_included": False,
        "transmitted_chunk_index_bits": 0,
        "candidate_index_bits": 0,
        "candidate_index_bytes_per_client": 0,
        "candidate_index_bytes": 0,
        "accepted_label_bytes": 0,
        "accepted_wire_value_numel": client_count * parameter_numel,
        "rejected_wire_value_numel": 0,
        "accepted_aggregate_value_numel": parameter_numel,
        "accepted_model_fp32_uplink_bytes": (
            client_count * parameter_numel * PLAINTEXT_VALUE_BYTES
        ),
        "dense_aux_fp32_uplink_bytes": (
            client_count * auxiliary_numel * PLAINTEXT_VALUE_BYTES
        ),
        "accepted_only_fp32_uplink_bytes": all_client_bytes,
        "accepted_only_logical_downlink_bytes": logical_bytes,
        "accepted_only_unicast_downlink_bytes": all_client_bytes,
    }


def measure_plaintext_round_communication(
    *,
    profile: str,
    cohort_sizes: Sequence[int],
    minimum_cohort_size: int,
    client_count: int,
    selected_chunk_count: int,
    parameter_numel: int,
    chunk_size: int,
    auxiliary_numel: int,
) -> dict[str, Any]:
    """Measure one plaintext FL round under the v17 support convention.

    SPARSE sends selection/admission/upload-support/result bitmaps and pads
    every accepted model chunk to ``chunk_size``. FULL uses fixed dense
    ordering and therefore carries neither bitmap support nor chunk padding.
    """

    normalized_profile = profile.upper()
    if normalized_profile not in {"SPARSE", "FULL"}:
        raise ValueError("profile must be SPARSE or FULL")
    client_count = _positive_integer("client_count", client_count)
    selected_chunk_count = _positive_integer(
        "selected_chunk_count", selected_chunk_count
    )
    parameter_numel = _positive_integer("parameter_numel", parameter_numel)
    chunk_size = _positive_integer("chunk_size", chunk_size)
    auxiliary_numel = _non_negative_integer("auxiliary_numel", auxiliary_numel)
    minimum_cohort_size = _positive_integer(
        "minimum_cohort_size", minimum_cohort_size
    )

    total_chunk_count = math.ceil(parameter_numel / chunk_size)
    if len(cohort_sizes) != total_chunk_count:
        raise ValueError("cohort size count does not match model chunk count")
    normalized_cohorts = tuple(
        _non_negative_integer("cohort_size", int(value))
        for value in cohort_sizes
    )
    if any(value > client_count for value in normalized_cohorts):
        raise ValueError("cohort size cannot exceed client count")
    if sum(normalized_cohorts) != client_count * selected_chunk_count:
        raise ValueError("cohort sizes do not match client selections")

    canonical_chunk_index_bits = max(1, (total_chunk_count - 1).bit_length())
    dense_auxiliary_uplink_bytes = (
        client_count * auxiliary_numel * PLAINTEXT_VALUE_BYTES
    )

    if normalized_profile == "FULL":
        if selected_chunk_count != total_chunk_count or any(
            cohort != client_count for cohort in normalized_cohorts
        ):
            raise ValueError("FULL communication requires every client coordinate")
        result = measure_plaintext_full_round_communication(
            client_count=client_count,
            parameter_numel=parameter_numel,
            auxiliary_numel=auxiliary_numel,
        )
        result["canonical_chunk_index_bits"] = canonical_chunk_index_bits
        return result
    else:
        accepted = tuple(
            cohort >= minimum_cohort_size for cohort in normalized_cohorts
        )
        candidate_contribution_count = sum(normalized_cohorts)
        support_bitmap_bytes = bitmap_byte_count(total_chunk_count)
        support_bitmap_bits = support_bitmap_bytes * 8
        selection_bitmap_bytes = client_count * support_bitmap_bytes
        upload_support_bitmap_bytes = client_count * support_bitmap_bytes
        admission_bitmap_bytes = client_count * support_bitmap_bytes
        result_bitmap_bytes = client_count * support_bitmap_bytes
        candidate_index_bits = selection_bitmap_bytes * 8
        candidate_index_bytes_per_client = support_bitmap_bytes
        candidate_index_bytes = selection_bitmap_bytes
        accepted_chunk_count = sum(accepted)
        accepted_label_bytes = support_bitmap_bytes
        accepted_wire_value_numel = (
            sum(
                cohort
                for cohort, is_accepted in zip(
                    normalized_cohorts, accepted, strict=True
                )
                if is_accepted
            )
            * chunk_size
        )
        rejected_wire_value_numel = (
            sum(
                cohort
                for cohort, is_accepted in zip(
                    normalized_cohorts, accepted, strict=True
                )
                if not is_accepted
            )
            * chunk_size
        )
        accepted_aggregate_value_numel = accepted_chunk_count * chunk_size

    accepted_model_uplink_bytes = (
        accepted_wire_value_numel * PLAINTEXT_VALUE_BYTES
    )
    uplink_bytes = (
        selection_bitmap_bytes
        + upload_support_bitmap_bytes
        + accepted_model_uplink_bytes
        + dense_auxiliary_uplink_bytes
    )
    logical_downlink_bytes = (
        support_bitmap_bytes
        + support_bitmap_bytes
        + (accepted_aggregate_value_numel + auxiliary_numel)
        * PLAINTEXT_VALUE_BYTES
    )
    unicast_downlink_bytes = logical_downlink_bytes * client_count

    return {
        "profile": normalized_profile,
        "padding_included": True,
        "support_codec": SUPPORT_CODEC_NAME,
        "support_bitmap_bytes": support_bitmap_bytes,
        "support_bitmap_bits": support_bitmap_bits,
        "selection_bitmap_bytes": selection_bitmap_bytes,
        "admission_bitmap_bytes": admission_bitmap_bytes,
        "upload_support_bitmap_bytes": upload_support_bitmap_bytes,
        "result_bitmap_bytes": result_bitmap_bytes,
        "support_bitmap_uplink_bytes": (
            selection_bitmap_bytes + upload_support_bitmap_bytes
        ),
        "support_bitmap_downlink_bytes": (
            admission_bitmap_bytes + result_bitmap_bytes
        ),
        "canonical_chunk_index_bits": canonical_chunk_index_bits,
        "transmitted_chunk_index_bits": (
            0 if normalized_profile == "FULL" else 1
        ),
        "candidate_index_bits": candidate_index_bits,
        "candidate_index_bytes_per_client": candidate_index_bytes_per_client,
        "candidate_index_bytes": candidate_index_bytes,
        "accepted_label_bytes": accepted_label_bytes,
        "accepted_wire_value_numel": accepted_wire_value_numel,
        "rejected_wire_value_numel": rejected_wire_value_numel,
        "accepted_aggregate_value_numel": accepted_aggregate_value_numel,
        "accepted_model_fp32_uplink_bytes": accepted_model_uplink_bytes,
        "dense_aux_fp32_uplink_bytes": dense_auxiliary_uplink_bytes,
        "accepted_only_fp32_uplink_bytes": uplink_bytes,
        "accepted_only_logical_downlink_bytes": logical_downlink_bytes,
        "accepted_only_unicast_downlink_bytes": unicast_downlink_bytes,
    }
