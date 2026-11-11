import unittest

from plaintext_fl.communication import (
    measure_aborted_sparse_round_communication,
    measure_online_round_communication,
)
from plaintext_fl.crypto_engine import CryptoEngineSetup
from plaintext_fl.protocol_contract import (
    LayoutEntry,
    LayoutKind,
    RegisteredClientWeight,
    TensorLayout,
)


def make_layout(kind, name, numel):
    return TensorLayout(
        kind=kind,
        model_name="dropout-communication-test",
        entries=(LayoutEntry(name, (numel,), 0, numel),),
    )


class DropoutCommunicationTest(unittest.TestCase):
    def test_v17_counts_each_bitmap_phase_with_actual_recipients(self):
        setup = CryptoEngineSetup(
            setup_id=bytes(16),
            parameter_layout=make_layout(LayoutKind.PARAMETER, "weight", 6),
            running_mean_layout=make_layout(LayoutKind.RUNNING_MEAN, "mean", 2),
            running_variance_layout=make_layout(
                LayoutKind.RUNNING_VARIANCE, "variance", 2
            ),
            clients=tuple(
                RegisteredClientWeight(client_id, 1)
                for client_id in range(4)
            ),
            chunk_size=2,
            minimum_cohort_size=3,
        )
        effective = {
            "protocol_version": "v17",
            "ring_dimension": 8,
            "mlwe_rank": 4,
            "selected_chunks_per_client": 2,
            "compression_c1": 5,
            "compression_ca": 6,
            "compression_c2": 4,
            "compression_cd": 3,
            "aux_compression_c1": 7,
            "aux_compression_ca": 8,
            "aux_compression_c2": 6,
            "aux_compression_cd": 5,
            "q_modulus_bits": [3, 3],
            "p_modulus_bits": [4],
            "hybrid_large_digits": 2,
        }
        diagnostics = {
            "sparse_client_ciphertext_count": 4,
            "accepted_sparse_contribution_count": 3,
            "aggregated_mlwe_ciphertext_count": 1,
            "sparse_server_ciphertext_count": 1,
            "auxiliary_client_ciphertext_count": 4,
            "auxiliary_aggregate_ciphertext_count": 1,
            "survivor_reconfiguration_applied": True,
            "survivor_client_ids": (0, 1, 2),
            "dropped_client_ids": (3,),
            "selection_bitmap_count": 4,
            "admission_bitmap_recipient_count": 7,
            "upload_support_bitmap_count": 4,
            "result_recipient_count": 4,
        }
        report = measure_online_round_communication(
            setup=setup,
            effective_protocol=effective,
            diagnostics=diagnostics,
            accepted_chunk_count=1,
            upload_accepted_chunk_count=1,
            uploaded_sparse_contribution_count=4,
        )

        self.assertEqual(report["parameters"]["support_bitmap_bytes"], 1)
        self.assertEqual(
            report["uplink"]["breakdown_bits"]["selection_support_bitmap"],
            4 * 8,
        )
        self.assertEqual(
            report["uplink"]["breakdown_bits"]["upload_support_bitmap"],
            4 * 8,
        )
        self.assertEqual(
            report["downlink"]["breakdown_bits"]["admission_support_bitmap"],
            7 * 8,
        )
        self.assertEqual(
            report["downlink"]["breakdown_bits"]["final_context_support_bitmap"],
            3 * 8,
        )
        self.assertEqual(
            report["downlink"]["breakdown_bits"]["result_support_bitmap"],
            4 * 8,
        )
        self.assertNotIn(
            "sparse_packet_labels", report["uplink"]["breakdown_bits"]
        )
        items = report["v17_itemization"]["items"]
        self.assertEqual(items["sparse_selection_bitmap"]["total_bits"], 32)
        self.assertEqual(items["sparse_admission_bitmap"]["total_bits"], 56)
        self.assertEqual(items["sparse_upload_support_bitmap"]["total_bits"], 32)
        self.assertEqual(items["sparse_final_context_bitmap"]["total_bits"], 24)
        self.assertEqual(items["sparse_result_bitmap"]["total_bits"], 32)
        self.assertEqual(
            report["v17_itemization"]["total"]["total_bits"],
            report["combined_total_bits"],
        )

    def test_upload_survivor_and_rekey_traffic_use_distinct_counts(self):
        setup = CryptoEngineSetup(
            setup_id=bytes(16),
            parameter_layout=make_layout(LayoutKind.PARAMETER, "weight", 6),
            running_mean_layout=make_layout(
                LayoutKind.RUNNING_MEAN, "mean", 2
            ),
            running_variance_layout=make_layout(
                LayoutKind.RUNNING_VARIANCE, "variance", 2
            ),
            clients=tuple(
                RegisteredClientWeight(client_id, 1)
                for client_id in range(4)
            ),
            chunk_size=2,
            minimum_cohort_size=3,
        )
        effective = {
            "protocol_version": "v17",
            "ring_dimension": 8,
            "mlwe_rank": 4,
            "selected_chunks_per_client": 2,
            "compression_c1": 5,
            "compression_ca": 6,
            "compression_c2": 4,
            "compression_cd": 3,
            "aux_compression_c1": 7,
            "aux_compression_ca": 8,
            "aux_compression_c2": 6,
            "aux_compression_cd": 5,
            "q_modulus_bits": [3, 3],
            "p_modulus_bits": [4],
            "hybrid_large_digits": 2,
        }
        diagnostics = {
            "sparse_client_ciphertext_count": 4,
            "accepted_sparse_contribution_count": 3,
            "aggregated_mlwe_ciphertext_count": 1,
            "sparse_server_ciphertext_count": 1,
            "auxiliary_client_ciphertext_count": 4,
            "auxiliary_aggregate_ciphertext_count": 1,
            "survivor_reconfiguration_applied": True,
            "survivor_client_ids": (0, 1, 2),
            "dropped_client_ids": (3,),
        }
        report = measure_online_round_communication(
            setup=setup,
            effective_protocol=effective,
            diagnostics=diagnostics,
            accepted_chunk_count=1,
            upload_accepted_chunk_count=1,
            uploaded_sparse_contribution_count=4,
        )

        self.assertEqual(report["downlink"]["breakdown_bits"]["dd_public_seed"], 3 * 128)
        qp_polynomial_bits = 8 * (3 + 3 + 4)
        uplink = report["uplink"]["breakdown_bits"]
        downlink = report["downlink"]["breakdown_bits"]
        self.assertEqual(uplink["sparse_client_b_c1"], 4 * 2 * 5)
        self.assertEqual(uplink["sparse_partial_decryption_c2"], 3 * 2 * 4)
        self.assertEqual(
            uplink["updated_client_to_collective_ksk"],
            3 * 2 * 2 * qp_polynomial_bits,
        )
        self.assertEqual(
            downlink["updated_collective_target_descriptor"],
            3 * (qp_polynomial_bits + 4),
        )
        self.assertEqual(downlink["sparse_aggregate_a_ca"], 3 * 8 * 6)
        items = report["v17_itemization"]["items"]
        self.assertEqual(
            items["sparse_selection_bitmap"]["occurrence_count"], 4
        )
        self.assertEqual(
            items["sparse_model_aggregate_a_downlink"]["occurrence_count"],
            3,
        )

        cached = measure_online_round_communication(
            setup=setup,
            effective_protocol=effective,
            diagnostics={**diagnostics, "survivor_target_key_cache_hit": True},
            accepted_chunk_count=1,
            upload_accepted_chunk_count=1,
            uploaded_sparse_contribution_count=4,
        )
        self.assertEqual(
            cached["uplink"]["breakdown_bits"]["updated_client_to_collective_ksk"],
            0,
        )
        self.assertEqual(
            cached["downlink"]["breakdown_bits"]["updated_collective_target_descriptor"],
            0,
        )
        repeated = measure_online_round_communication(
            setup=setup,
            effective_protocol=effective,
            diagnostics={**diagnostics, "survivor_new_target_sizes": (4, 3)},
            accepted_chunk_count=1,
            upload_accepted_chunk_count=1,
            uploaded_sparse_contribution_count=4,
        )
        self.assertEqual(
            repeated["uplink"]["breakdown_bits"]["updated_client_to_collective_ksk"],
            7 * 2 * 2 * qp_polynomial_bits,
        )
        aborted = measure_aborted_sparse_round_communication(
            setup=setup,
            effective_protocol=effective,
            aborted_traffic={
                "round_id": 7,
                "sparse_client_ciphertext_count": 4,
                "auxiliary_client_ciphertext_count": 4,
                "auxiliary_aggregate_ciphertext_count": 1,
                "sparse_server_ciphertext_count": 1,
                "upload_accepted_chunk_count": 1,
                "accepted_chunk_count": 1,
                "uploaded_sparse_contribution_count": 4,
                "accepted_sparse_contribution_count": 3,
                "survivor_client_ids": (0, 1, 2),
                "dropped_client_ids": (3,),
                "survivor_new_target_sizes": (3,),
            },
        )
        self.assertEqual(aborted["downlink"]["breakdown_bits"]["dd_public_seed"], 3 * 128)
        self.assertGreater(aborted["combined_total_bits"], 0)
        self.assertEqual(
            aborted["uplink"]["breakdown_bits"]["sparse_partial_decryption_c2"],
            0,
        )
        self.assertEqual(
            aborted["downlink"]["breakdown_bits"]["final_sparse_values_cd"],
            0,
        )
        self.assertLess(aborted["combined_total_bits"], report["combined_total_bits"])


if __name__ == "__main__":
    unittest.main()
