from tests.profile_fixtures import communication_profile
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import unittest

from plaintext_fl.communication import (
    cumulative_communication,
    measure_full_online_round_communication,
    measure_online_round_communication,
    measure_setup_communication,
)
from plaintext_fl.crypto_engine import CryptoEngineSetup
from plaintext_fl.crypto_parameter_profile import load_crypto_parameter_profile
from plaintext_fl.encrypted_training import (
    _make_communication_runtime_summary,
    _print_final_progress,
    _print_round_progress,
)
from plaintext_fl.model import resnet20
from plaintext_fl.model_adapter import build_model_protocol_layouts
from plaintext_fl.protocol_contract import RegisteredClientWeight


class CommunicationAccountingTest(unittest.TestCase):
    def test_retired_version_cannot_be_reported_as_v17(self):
        for version in ("v23", "v27", "v28"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "requires protocol v17"):
                    measure_online_round_communication(setup=self.setup,
                        effective_protocol={"protocol_version": version},
                        diagnostics={}, accepted_chunk_count=0)

    def test_full_mode_has_no_m2r_or_sparse_label_traffic(self):
        repository = Path(__file__).parents[1]
        profile = load_crypto_parameter_profile(
            repository
            / "plaintext_fl"
            / "profiles"
            / "resnet32-cifar100-full-shared-v17.json"
        )
        from plaintext_fl.model import resnet32

        layouts = build_model_protocol_layouts(
            resnet32(), model_name="resnet32-cifar100"
        )
        setup = CryptoEngineSetup(
            setup_id=bytes(16),
            parameter_layout=layouts.parameter,
            running_mean_layout=layouts.running_mean,
            running_variance_layout=layouts.running_variance,
            clients=tuple(
                RegisteredClientWeight(client_id, 6250)
                for client_id in range(8)
            ),
            chunk_size=8192,
            minimum_cohort_size=3,
            protocol_mode="FULL",
        )
        setup_report = measure_setup_communication(setup=setup, profile=profile)
        self.assertEqual(
            setup_report["uplink"]["breakdown_bits"]["m2r_key_b_shares"], 0
        )
        effective = profile.native_options(master_seed=bytes(32)).to_binding_dict()
        diagnostics = {
            "full_model_client_ciphertext_count": 8 * 58,
            "full_model_aggregate_ciphertext_count": 58,
            "r2m_ciphertext_count": 0,
            "m2r_ciphertext_count": 0,
            "auxiliary_client_ciphertext_count": 8,
            "auxiliary_aggregate_ciphertext_count": 1,
        }
        report = measure_full_online_round_communication(
            setup=setup,
            effective_protocol=effective,
            diagnostics=diagnostics,
        )
        v17 = measure_full_online_round_communication(
            setup=setup, effective_protocol={**effective, "protocol_version": "v17"},
            diagnostics=diagnostics,
        )
        self.assertEqual(v17["combined_total_bits"], report["combined_total_bits"])
        self.assertEqual(v17["v17_itemization"]["items"]["dd_public_seed"]["total_bits"], 8 * 128)
        self.assertNotIn("selection_labels", report["uplink"]["breakdown_bits"])
        self.assertEqual(report["parameters"]["parameter_block_count"], 58)
        self.assertEqual(report["parameters"]["m2r_ciphertext_count"], 0)
        items = report["v17_itemization"]["items"]
        self.assertFalse(items["sparse_model_ciphertext_uplink"]["applicable"])
        self.assertGreater(
            items["full_model_ciphertext_uplink"]["total_bits"], 0
        )
        self.assertEqual(
            report["v17_itemization"]["total"]["total_bits"],
            sum(
                item["total_bits"]
                for item in items.values()
                if item["included_in_total"]
            ),
        )

    @classmethod
    def setUpClass(cls):
        cls.profile = communication_profile()
        layouts = build_model_protocol_layouts(resnet20())
        cls.setup = CryptoEngineSetup(
            setup_id=bytes(16),
            parameter_layout=layouts.parameter,
            running_mean_layout=layouts.running_mean,
            running_variance_layout=layouts.running_variance,
            clients=tuple(
                RegisteredClientWeight(client_id, 6250)
                for client_id in range(cls.profile.clients)
            ),
            chunk_size=cls.profile.chunk_size,
            minimum_cohort_size=cls.profile.minimum_cohort_size,
        )

    def effective_protocol(self):
        options = self.profile.native_options(master_seed=bytes(32))
        payload = options.to_binding_dict()
        effective = {
            name: value
            for name, value in payload.items()
            if name
            not in {
                "profile",
                "master_seed",
                "use_sparse_compression",
                "use_auxiliary_compression",
                "measure_pre_smudging_noise",
                "crypto_threads",
            }
        }
        # The profile supplies rho; C++ derives and returns this effective
        # value during setup. Mirror that post-setup payload in this unit test.
        effective["selected_chunks_per_client"] = (
            self.profile.selected_chunk_count()
        )
        return effective

    def test_setup_counts_qp_key_material_and_initial_model(self):
        report = measure_setup_communication(
            setup=self.setup,
            profile=self.profile,
        )
        polynomial_bits = 8192 * (60 + 60)
        self.assertEqual(
            report["uplink"]["breakdown_bits"][
                "collective_public_key_p0_shares"
            ],
            8 * polynomial_bits,
        )
        self.assertEqual(
            report["uplink"]["breakdown_bits"]["m2r_key_b_shares"],
            8 * 4 * 2 * polynomial_bits,
        )
        self.assertEqual(
            report["uplink"]["breakdown_bits"][
                "client_to_collective_key_pairs"
            ],
            8 * 2 * 2 * polynomial_bits,
        )
        self.assertEqual(
            report["downlink"]["breakdown_bits"][
                "initial_global_model_float32"
            ],
            8 * (272474 + 1568) * 32,
        )
        v16 = report["v17_itemization"]
        setup_items = v16["protocol_setup"]["items"]
        initial_items = v16["initial_communication"]["items"]
        self.assertEqual(
            setup_items["collective_pk_share"]["total_bits"],
            8 * polynomial_bits,
        )
        self.assertEqual(
            setup_items["initial_client_to_collective_ksk"]["total_bits"],
            8 * 2 * 2 * polynomial_bits,
        )
        self.assertEqual(
            setup_items["index_key_setup"]["total_bits"],
            128 * 7,
        )
        self.assertEqual(
            initial_items["initial_model_parameters_float32"]["total_bits"],
            8 * 272474 * 32,
        )
        self.assertEqual(
            initial_items["initial_auxiliary_state_float32"]["total_bits"],
            8 * 1568 * 32,
        )
        self.assertEqual(
            v16["end_to_end_total"]["total_bits"],
            v16["protocol_setup"]["total"]["total_bits"]
            + v16["initial_communication"]["total"]["total_bits"],
        )

    def test_rank_one_sparse_setup_omits_m2r_key_traffic(self):
        rank_one_profile = replace(self.profile, mlwe_rank=1)
        report = measure_setup_communication(
            setup=self.setup,
            profile=rank_one_profile,
        )
        self.assertEqual(
            report["uplink"]["breakdown_bits"]["m2r_key_b_shares"], 0
        )
        item = report["v17_itemization"]["protocol_setup"]["items"][
            "m2r_swk_share"
        ]
        self.assertFalse(item["applicable"])
        self.assertEqual(item["total_bits"], 0)

    def test_v17_setup_uses_native_canonical_descriptor_size(self):
        repository = Path(__file__).parents[1]
        profile = load_crypto_parameter_profile(
            repository / "plaintext_fl" / "profiles" / "resnet32-cifar100-sp-shared-v17.json"
        )
        effective = profile.native_options(master_seed=bytes(32)).to_binding_dict()
        effective.update(
            setup_descriptor_bytes=320,
            q_primes=(281474976694273, 140737488486401),
            p_primes=(1152921504606830593,),
        )
        report = measure_setup_communication(
            setup=replace(
                self.setup,
                chunk_size=profile.chunk_size,
                minimum_cohort_size=profile.minimum_cohort_size,
            ),
            profile=profile,
            effective_protocol=effective,
        )
        self.assertEqual(report["parameters"]["setup_descriptor_bytes"], 320)
        item = report["v17_itemization"]["protocol_setup"]["items"][
            "crs_setup_descriptor"
        ]
        layout_bytes = sum(
            len(layout.canonical_bytes())
            for layout in (
                self.setup.parameter_layout,
                self.setup.running_mean_layout,
                self.setup.running_variance_layout,
            )
        )
        self.assertEqual(
            item["bits_per_occurrence"],
            (320 + layout_bytes) * 8 + 128 + 128,
        )

    def test_online_uses_actual_counts_and_all_unicast_downlink(self):
        diagnostics = {
            "sparse_client_ciphertext_count": 66,
            "accepted_sparse_contribution_count": 251,
            "aggregated_mlwe_ciphertext_count": 52,
            "sparse_server_ciphertext_count": 13,
            "auxiliary_client_ciphertext_count": 8,
            "auxiliary_aggregate_ciphertext_count": 1,
        }
        report = measure_online_round_communication(
            setup=self.setup,
            effective_protocol=self.effective_protocol(),
            diagnostics=diagnostics,
            accepted_chunk_count=52,
        )
        uplink = report["uplink"]["breakdown_bits"]
        downlink = report["downlink"]["breakdown_bits"]
        self.assertEqual(uplink["selection_support_bitmap"], 8 * 17 * 8)
        self.assertEqual(uplink["sparse_client_b_c1"], 251 * 2048 * 22)
        self.assertEqual(
            uplink["sparse_partial_decryption_c2"],
            8 * 52 * 2048 * 18,
        )
        self.assertEqual(
            downlink["sparse_aggregate_a_ca"],
            8 * 13 * 8192 * 40,
        )
        self.assertEqual(
            downlink["final_sparse_values_cd"],
            8 * 52 * 2048 * 15,
        )
        self.assertEqual(
            downlink["final_auxiliary_values_cd"],
            8 * 1568 * 15,
        )
        self.assertEqual(
            report["uplink"]["mean_per_client_bits"],
            report["uplink"]["total_bits"] / 8,
        )
        self.assertAlmostEqual(
            report["combined_total_mb"],
            report["combined_total_bytes"] / 1_000_000,
        )
        self.assertAlmostEqual(
            report["uplink"]["mean_per_client_mb"],
            report["uplink"]["total_bytes"] / 8 / 1_000_000,
        )
        self.assertNotIn("combined_total_mib", report)
        self.assertNotIn("total_mib", report["uplink"])
        items = report["v17_itemization"]["items"]
        self.assertEqual(
            items["sparse_selection_bitmap"]["total_bits"],
            8 * 17 * 8,
        )
        self.assertEqual(
            items["sparse_model_ciphertext_uplink"]["total_bits"],
            66 * 128 + 251 * 2048 * 22,
        )
        self.assertEqual(
            items["sparse_model_result"]["total_bits"],
            8 * 52 * 2048 * 15,
        )
        self.assertNotIn(
            "sparse_packet_labels",
            report["v17_itemization"]["items"],
        )

    def test_rank_one_online_accepts_direct_rlwe_diagnostics(self):
        rank_one_setup = replace(self.setup, chunk_size=8192)
        effective_protocol = self.effective_protocol()
        effective_protocol["mlwe_rank"] = 1
        diagnostics = {
            "sparse_client_ciphertext_count": 120,
            "accepted_sparse_contribution_count": 120,
            "aggregated_mlwe_ciphertext_count": 0,
            "sparse_server_ciphertext_count": 30,
            "auxiliary_client_ciphertext_count": 8,
            "auxiliary_aggregate_ciphertext_count": 1,
        }

        report = measure_online_round_communication(
            setup=rank_one_setup,
            effective_protocol=effective_protocol,
            diagnostics=diagnostics,
            accepted_chunk_count=30,
        )

        self.assertEqual(report["parameters"]["mlwe_rank"], 1)
        self.assertEqual(
            report["parameters"]["aggregated_mlwe_ciphertext_count"],
            0,
        )
        self.assertGreater(report["combined_total_bits"], 0)

    def test_post_upload_dropout_counts_uploads_survivors_and_rekey(self):
        diagnostics = {
            "sparse_client_ciphertext_count": 66,
            "accepted_sparse_contribution_count": 190,
            "aggregated_mlwe_ciphertext_count": 40,
            "sparse_server_ciphertext_count": 10,
            "auxiliary_client_ciphertext_count": 8,
            "auxiliary_aggregate_ciphertext_count": 1,
            "survivor_reconfiguration_applied": True,
            "survivor_client_ids": (0, 1, 2, 3, 4, 5, 6),
            "dropped_client_ids": (7,),
        }
        effective = self.effective_protocol()
        report = measure_online_round_communication(
            setup=self.setup,
            effective_protocol=effective,
            diagnostics=diagnostics,
            accepted_chunk_count=40,
            upload_accepted_chunk_count=52,
            uploaded_sparse_contribution_count=251,
        )

        uplink = report["uplink"]["breakdown_bits"]
        downlink = report["downlink"]["breakdown_bits"]
        qp_polynomial_bits = 8192 * (
            sum(effective["q_modulus_bits"])
            + sum(effective["p_modulus_bits"])
        )
        self.assertEqual(uplink["sparse_client_b_c1"], 251 * 2048 * 22)
        self.assertEqual(
            uplink["sparse_partial_decryption_c2"],
            7 * 40 * 2048 * 18,
        )
        self.assertEqual(
            uplink["updated_client_to_collective_ksk"],
            7 * 2 * effective["hybrid_large_digits"] * qp_polynomial_bits,
        )
        self.assertEqual(
            downlink["updated_collective_target_descriptor"],
            7 * (qp_polynomial_bits + 8),
        )
        self.assertEqual(
            downlink["sparse_aggregate_a_ca"],
            7 * 10 * 8192 * 40,
        )
        items = report["v17_itemization"]["items"]
        self.assertEqual(
            items["sparse_selection_bitmap"]["occurrence_count"], 8
        )
        self.assertEqual(
            items["sparse_model_aggregate_a_downlink"]["occurrence_count"],
            7,
        )

    def test_cumulative_charges_setup_once_and_sums_rounds(self):
        setup = measure_setup_communication(
            setup=self.setup,
            profile=self.profile,
        )
        diagnostics = {
            "sparse_client_ciphertext_count": 40,
            "accepted_sparse_contribution_count": 120,
            "aggregated_mlwe_ciphertext_count": 30,
            "sparse_server_ciphertext_count": 8,
            "auxiliary_client_ciphertext_count": 8,
            "auxiliary_aggregate_ciphertext_count": 1,
        }
        online = measure_online_round_communication(
            setup=self.setup,
            effective_protocol=self.effective_protocol(),
            diagnostics=diagnostics,
            accepted_chunk_count=30,
        )
        cumulative = cumulative_communication(
            setup=setup,
            online_rounds=[online, online],
        )
        self.assertEqual(cumulative["completed_rounds"], 2)
        self.assertEqual(
            cumulative["online_cumulative"]["uplink"]["total_bits"],
            2 * online["uplink"]["total_bits"],
        )
        self.assertEqual(
            cumulative["setup_plus_online_cumulative"]["uplink"][
                "total_bits"
            ],
            setup["uplink"]["total_bits"]
            + 2 * online["uplink"]["total_bits"],
        )
        self.assertAlmostEqual(
            cumulative["online_mean_per_round"]["combined_mb"],
            online["combined_total_bytes"] / 1_000_000,
        )
        v16 = cumulative["v17_itemized_communication"]
        self.assertEqual(
            v16["online_cumulative"]["items"][
                "sparse_model_ciphertext_uplink"
            ]["total_bits"],
            2
            * online["v17_itemization"]["items"][
                "sparse_model_ciphertext_uplink"
            ]["total_bits"],
        )
        self.assertEqual(
            v16["end_to_end_total"]["total_bits"],
            v16["v17_protocol_total"]["total_bits"]
            + v16["initial_communication"]["total_bits"],
        )

    def test_terminal_progress_is_compact_and_highlights_accuracy(self):
        setup = measure_setup_communication(
            setup=self.setup,
            profile=self.profile,
        )
        diagnostics = {
            "sparse_client_ciphertext_count": 40,
            "accepted_sparse_contribution_count": 120,
            "aggregated_mlwe_ciphertext_count": 30,
            "sparse_server_ciphertext_count": 8,
            "auxiliary_client_ciphertext_count": 8,
            "auxiliary_aggregate_ciphertext_count": 1,
        }
        online = measure_online_round_communication(
            setup=self.setup,
            effective_protocol=self.effective_protocol(),
            diagnostics=diagnostics,
            accepted_chunk_count=30,
        )
        cumulative = cumulative_communication(
            setup=setup,
            online_rounds=[online],
        )
        round_summary = {
            "round": 1,
            "requested_rounds": 5,
            "test_accuracy_percent": 72.5,
            "weighted_train_loss": 1.25,
            "learning_rate": 0.1,
            "rho_requested": 0.3,
            "rho_selected": 0.3,
            "rho_eff_round": 0.28,
            "rho_eff_running": 0.28,
            "cohort_retention_ratio": 0.933333,
            "accepted_chunk_count": 30,
            "parameter_chunk_count": 134,
            "communication": {
                "online": online,
                "setup": setup,
                "setup_incurred_this_round": True,
            },
            "round_seconds": 12.0,
            "setup_seconds": 2.0,
            "estimated_parallel_setup_seconds": 1.0,
            "local_training_seconds": 5.0,
            "encrypted_path_seconds": 4.0,
            "evaluation_seconds": 1.0,
            "estimated_parallel_round_seconds": 7.0,
            "estimated_parallel_local_training_seconds": 2.5,
            "estimated_parallel_encrypted_path_seconds": 1.5,
            "estimated_parallel_protocol_seconds": 4.0,
        }
        final_summary = {
            "communication": cumulative,
            "best_accuracy_percent": 72.5,
            "best_round": 1,
            "completed_rounds": 1,
            "final_test_accuracy_percent": 72.5,
            "rho_requested": 0.3,
            "rho_selected": 0.3,
            "rho_eff": 0.28,
            "cohort_retention_ratio": 0.933333,
            "total_estimated_parallel_protocol_seconds": 4.0,
            "average_estimated_parallel_protocol_seconds_per_round": 4.0,
            "total_estimated_parallel_setup_seconds": 1.0,
            "total_estimated_parallel_local_training_seconds": 2.5,
            "average_estimated_parallel_local_training_seconds_per_round": 2.5,
            "total_estimated_parallel_encrypted_path_seconds": 1.5,
            "average_estimated_parallel_encrypted_path_seconds_per_round": 1.5,
            "total_seconds": 12.0,
        }
        reporting = _make_communication_runtime_summary(final_summary)
        final_summary["communication_runtime_summary"] = reporting
        self.assertEqual(reporting["client_count"], 8)
        self.assertEqual(
            reporting["setup"]["communication_mb"][
                "uplink_mean_per_client"
            ],
            setup["uplink"]["mean_per_client_mb"],
        )
        self.assertAlmostEqual(
            reporting["online_per_round"]["communication_mb"][
                "downlink_mean_per_client"
            ],
            online["downlink"]["mean_per_client_mb"],
        )
        self.assertEqual(
            reporting["setup"]["estimated_parallel_runtime_seconds"],
            1.0,
        )
        self.assertEqual(
            reporting["online_cumulative"][
                "estimated_parallel_runtime_seconds"
            ],
            4.0,
        )
        v16 = reporting["v17_itemized_communication"]
        self.assertEqual(
            v16["end_to_end_total"]["total_bits"],
            v16["v17_protocol_total"]["total_bits"]
            + v16["initial_communication"]["total_bits"],
        )
        output = StringIO()
        with redirect_stdout(output):
            _print_round_progress(round_summary)
            _print_final_progress(final_summary, Path("runs/example"))
        terminal = output.getvalue()
        self.assertIn("ROUND 001/005 | ACCURACY: 72.50%", terminal)
        self.assertIn("online_comm=", terminal)
        self.assertIn("rho_eff_round=0.280000", terminal)
        self.assertIn("rho_eff=0.280000", terminal)
        self.assertIn("MB", terminal)
        self.assertIn("TRAINING COMPLETE", terminal)
        self.assertIn("parallel_estimate=7.0s", terminal)
        self.assertIn("online_runtime_parallel=4.0s", terminal)
        self.assertIn("local_train_parallel=2.5s", terminal)
        self.assertIn("encrypted_path_parallel=1.5s", terminal)
        self.assertIn("evaluation_excluded", terminal)
        self.assertIn("setup_comm=", terminal)
        self.assertIn("protocol_comm=", terminal)
        self.assertIn("initial_comm=", terminal)
        self.assertIn("end_to_end_total=", terminal)
        self.assertIn("online_comm_mean_per_round=", terminal)
        self.assertIn("MB/client", terminal)
        self.assertIn("setup_runtime_parallel=1.000s", terminal)
        self.assertIn("local_total=2.5s", terminal)
        self.assertNotIn('"native_diagnostics"', terminal)


if __name__ == "__main__":
    unittest.main()
