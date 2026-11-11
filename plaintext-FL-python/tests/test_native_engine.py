from dataclasses import replace
import unittest
import os
import json

import numpy as np

from plaintext_fl.communication import measure_aborted_sparse_round_communication
from plaintext_fl.crypto_engine import CryptoEngineSetup, PostRequestDropoutError
from plaintext_fl.native_engine import (
    NativeCryptoEngine,
    NativeCryptoEngineOptions,
    native_binding_available,
)
from plaintext_fl.protocol_contract import (
    LayoutEntry,
    LayoutKind,
    RegisteredClientWeight,
    RoundPhase,
    TensorLayout,
)
from plaintext_fl.round_orchestrator import (
    ClientFullRoundCandidate,
    ClientRoundCandidate,
    FederatedRoundOrchestrator,
)


def make_layout(kind, name, numel):
    return TensorLayout(
        kind=kind,
        model_name="pybind-tiny-model",
        entries=(
            LayoutEntry(
                name=name,
                shape=(numel,),
                offset=0,
                numel=numel,
            ),
        ),
    )


def make_setup():
    return CryptoEngineSetup(
        # The native setup replaces this transport placeholder with the
        # setup ID derived from the generated OpenFHE setup material.
        setup_id=bytes(16),
        parameter_layout=make_layout(LayoutKind.PARAMETER, "weight", 600),
        running_mean_layout=make_layout(
            LayoutKind.RUNNING_MEAN,
            "bn.running_mean",
            2,
        ),
        running_variance_layout=make_layout(
            LayoutKind.RUNNING_VARIANCE,
            "bn.running_var",
            2,
        ),
        clients=tuple(
            RegisteredClientWeight(client_id=client_id, example_count=1)
            for client_id in (30, 10, 20, 40)
        ),
        chunk_size=256,
        minimum_cohort_size=3,
    )


def make_options():
    return NativeCryptoEngineOptions(
        profile="smoke",
        ring_dimension=1024,
        mlwe_rank=4,
        q_modulus_bits=(28, 28),
        p_modulus_bits=(55,),
        hybrid_large_digits=2,
        maximum_qp_bits=160,
        model_scaling_factor=2**47,
        aux_scaling_factor=2**43,
        precision_bits=16,
        aux_precision_bits=16,
        selected_chunks_per_client=2,
        update_bound=0.49401719,
        aux_update_bound=11.8428,
        compression_c1=23,
        compression_ca=36,
        compression_c2=19,
        compression_cd=16,
        aux_compression_c1=23,
        aux_compression_ca=37,
        aux_compression_c2=19,
        aux_compression_cd=16,
        smudging_sigma=0.0,
        aux_smudging_sigma=0.0,
        model_delivery_scale=65536,
        aux_delivery_scale=2760,
        use_sparse_compression=False,
        use_auxiliary_compression=False,
    )


def make_candidates():
    selections = ((0, 1), (0, 1), (1, 2), (1, 2))
    result = []
    for position, (client_id, selection) in enumerate(
        zip((30, 10, 20, 40), selections, strict=True),
        start=1,
    ):
        value = np.float32(position / 10.0)
        result.append(
            ClientRoundCandidate(
                client_id=client_id,
                selected_actual_chunk_ids=selection,
                selected_values=np.full((2, 256), value, dtype=np.float32),
                running_mean_delta=np.array(
                    [value, 2.0 * value],
                    dtype=np.float32,
                ),
                running_variance_delta=np.array(
                    [value, 2.0 * value],
                    dtype=np.float32,
                ),
            )
        )
    return tuple(result)


def make_full_setup():
    return replace(
        make_setup(),
        chunk_size=1024,
        protocol_mode="FULL",
    )


def make_full_candidates():
    values = (2.0, 2.0, -1.5, -1.5)
    return tuple(
        ClientFullRoundCandidate(
            client_id=client_id,
            parameter_delta=np.full(600, value, dtype=np.float32),
            running_mean_delta=np.array([value, value], dtype=np.float32),
            running_variance_delta=np.array([value, value], dtype=np.float32),
        )
        for client_id, value in zip(
            (30, 10, 20, 40), values, strict=True
        )
    )


@unittest.skipUnless(
    native_binding_available(),
    "ciphertext_fl_cpp was not built or is not on PYTHONPATH",
)
class NativeCryptoEngineIntegrationTest(unittest.TestCase):
    def test_native_boundary_rejects_retired_versions(self):
        import ciphertext_fl_cpp as native
        for version in ("v23", "v27", "v28"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "must be 'v17'"):
                    native.NativeCryptoEngine({"profile": "smoke", "protocol_version": version}).setup(
                        {}, {}, {}, (), 256, 3)

    def test_default_v17_setup_is_fresh_even_with_same_master_seed(self):
        options = replace(make_options(), master_seed=bytes(32))
        first, second = NativeCryptoEngine(options), NativeCryptoEngine(options)
        first.setup(make_setup())
        second.setup(make_setup())
        self.assertNotEqual(first.setup_config.setup_id, second.setup_config.setup_id)
        for engine in (first, second):
            self.assertEqual(engine.effective_protocol["protocol_version"], "v17")
            self.assertEqual(engine.effective_protocol["support_codec"], "CHUNK-BITMAP-MSB0-V1")
            self.assertFalse(engine.effective_protocol["experimental_dynamic_cohort"])

    def test_default_v17_compressed_full_and_sparse_rounds(self):
        for mode in ("FULL", "SPARSE"):
            with self.subTest(mode=mode):
                options = replace(make_options(), protocol_mode=mode,
                    use_sparse_compression=True, use_auxiliary_compression=True)
                if mode == "FULL":
                    options = replace(options, mlwe_rank=1, selection_ratio=1.0,
                                      selected_chunks_per_client=None)
                engine = NativeCryptoEngine(options)
                runner = FederatedRoundOrchestrator(engine)
                runner.setup(make_full_setup() if mode == "FULL" else make_setup())
                result = (runner.run_full_round(0, make_full_candidates()) if mode == "FULL"
                          else runner.run_round(0, make_candidates()))
                self.assertEqual(engine.last_diagnostics["crypto_timing_model"],
                                 "client-batch-service-v3")
                # Default fixed-cohort uploads are prepared inside finalize.
                # Their time must not be added to finalize elapsed a second time.
                self.assertLessEqual(engine.last_diagnostics["crypto_sequential_seconds"],
                                     result.timing.finalize_seconds)
                expected = np.zeros(600, dtype=np.float32)
                expected[:] = 0.25 if mode == "FULL" else 0.0
                if mode == "SPARSE": expected[256:512] = 0.25
                np.testing.assert_allclose(result.decrypted.parameter_delta, expected, atol=2e-4, rtol=0)
                self.assertEqual(engine.phase, RoundPhase.COMMITTED)

    def test_codec_timings_are_optional_serializable_and_per_round(self):
        options = replace(make_options(), protocol_version="v17",
                          smudging_sigma=1.0, aux_smudging_sigma=1.0,
                          use_sparse_compression=True, use_auxiliary_compression=True,
                          crypto_threads=8)
        engine = NativeCryptoEngine(options)
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())
        snapshots = []
        for round_id in (0, 1):
            orchestrator.run_round(round_id, make_candidates())
            timings = engine.last_diagnostics.get("codec_work_timings")
            if os.environ.get("FL_CODEC_TIMING") != "1":
                self.assertIsNone(timings)
                continue
            self.assertEqual(set(timings), {
                "public_polynomial", "gaussian", "format_and_crt", "quantize",
                "pack", "unpack", "code_sum", "reconstruct"})
            json.dumps(timings, allow_nan=False)
            for row in timings.values():
                self.assertGreater(row["calls"], 0)
                self.assertGreaterEqual(row["work_seconds"], 0)
            snapshots.append(timings)
        if snapshots:
            self.assertEqual(
                {k: v["calls"] for k, v in snapshots[0].items()},
                {k: v["calls"] for k, v in snapshots[1].items()})

    def test_v17_exposes_exact_phase_audit_through_real_binding(self):
        options = replace(make_options(), protocol_version="v17",
                          measure_pre_smudging_noise=True,
                          smudging_sigma=1.0, aux_smudging_sigma=1.0)
        engine = NativeCryptoEngine(options)
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())
        result = orchestrator.run_round(0, make_candidates()).decrypted
        audit = result.phase_audit
        self.assertIsNotNone(audit)
        self.assertEqual(len(audit["model"]["phase_residues"]), 600)
        self.assertEqual(len(audit["aux"]["phase_residues"]), 4)
        active = audit["model"]["integer_references"][256:512]
        expected = sum(32 * round(float(np.float32(i / 10)) * 2**47) for i in range(1, 5))
        self.assertTrue(all(int(value) == expected for value in active))
        self.assertTrue(all(value == "" for value in audit["model"]["phase_residues"][:256]))

    def test_v17_late_abort_closes_round_and_allows_next_logical_round(self):
        options = replace(
            make_options(),
            experimental_dynamic_cohort=True,
            protocol_version="v17",
            model_scaling_factor=2**35,
            aux_scaling_factor=2**35,
            update_bound=1.0,
            aux_update_bound=1.0,
            compression_c1=25,
            compression_ca=52,
            compression_c2=24,
            aux_compression_c1=25,
            aux_compression_ca=52,
            aux_compression_c2=24,
            model_delivery_scale=16000,
            use_sparse_compression=True,
            use_auxiliary_compression=True,
        )
        engine = NativeCryptoEngine(options)
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())
        applied = []

        with self.assertRaises(PostRequestDropoutError):
            orchestrator.run_round(
                0,
                make_candidates(),
                logical_round_index=0,
                post_request_dropouts=(30,),
                apply_result=lambda result: applied.append(result),
            )
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        self.assertEqual(applied, [])
        traffic = engine.last_aborted_round_traffic
        self.assertIsNotNone(traffic)
        self.assertEqual(traffic["round_id"], 0)
        self.assertGreater(traffic["sparse_client_ciphertext_count"], 0)
        self.assertGreater(traffic["auxiliary_client_ciphertext_count"], 0)
        communication = measure_aborted_sparse_round_communication(
            setup=engine.setup_config,
            effective_protocol=engine.effective_protocol,
            aborted_traffic=traffic,
        )
        self.assertGreater(communication["combined_total_bits"], 0)
        self.assertEqual(
            communication["uplink"]["breakdown_bits"][
                "sparse_partial_decryption_c2"
            ],
            0,
        )

        with self.assertRaisesRegex(ValueError, "aborted logical round is closed"):
            engine.begin_round(1, logical_round_index=0)
        result = orchestrator.run_round(
            1,
            make_candidates(),
            logical_round_index=1,
            apply_result=lambda value: applied.append(value),
        )
        self.assertEqual(result.decrypted.round_id, 1)
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertEqual(len(applied), 1)

    def test_fixed_cohort_missing_selection_aborts_and_forbids_retry(self):
        engine = NativeCryptoEngine(make_options())
        engine.setup(make_setup())
        engine.begin_round(0)
        with self.assertRaises(RuntimeError):
            engine.finalize_admission()
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        with self.assertRaisesRegex(RuntimeError, "retries"):
            engine.begin_round(1)

    def test_direct_full_finalization_error_is_terminal(self):
        engine = NativeCryptoEngine(replace(make_options(), protocol_mode="FULL"))
        engine.setup(make_full_setup())
        engine.begin_round(0)
        for client_id in engine.setup_config.client_ids:
            engine.submit_full_parameter_delta(client_id, np.zeros(600, dtype=np.float32))
            engine.submit_auxiliary_deltas(client_id, np.zeros(2, dtype=np.float32),
                                          np.zeros(2, dtype=np.float32))
        # Unsupported late-dropout request fails before the crypto tasks start.
        # Call the native adapter directly so no orchestrator can hide the bug.
        with self.assertRaises(RuntimeError):
            engine.finalize_round(post_request_dropouts=(30,))
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        with self.assertRaisesRegex(RuntimeError, "retries"):
            engine.begin_round(1)

    def test_fixed_cohort_rejects_survivors_without_applying_result(self):
        engine = NativeCryptoEngine(make_options())
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())
        applied = []
        with self.assertRaisesRegex(RuntimeError, "survivor"):
            orchestrator.run_round(0, make_candidates(),
                                   survivor_client_ids=(10, 20, 40),
                                   apply_result=applied.append)
        self.assertEqual(applied, [])
        self.assertEqual(engine.phase, RoundPhase.ABORTED)

    def test_post_upload_dropout_crosses_pybind_and_rekeys_survivors(self):
        engine = NativeCryptoEngine(replace(make_options(), experimental_dynamic_cohort=True))
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())

        result = orchestrator.run_round(
            0,
            make_candidates(),
            survivor_client_ids=(10, 20, 40),
        )

        self.assertEqual(result.survivor_client_ids, (10, 20, 40))
        self.assertEqual(result.dropped_client_ids, (30,))
        self.assertEqual(result.admission.accepted_actual_chunk_ids, (1,))
        expected_parameter = np.zeros(600, dtype=np.float32)
        expected_parameter[256:512] = 0.225
        np.testing.assert_allclose(
            result.decrypted.parameter_delta,
            expected_parameter,
            rtol=0.0,
            atol=2.0e-4,
        )
        diagnostics = engine.last_diagnostics
        self.assertTrue(diagnostics["survivor_reconfiguration_applied"])
        self.assertEqual(
            diagnostics["survivor_client_ids"], (10, 20, 40)
        )
        self.assertEqual(diagnostics["dropped_client_ids"], (30,))
        self.assertGreater(
            diagnostics["sparse_discarded_client_ciphertext_count"], 0
        )
        self.assertGreater(
            diagnostics["auxiliary_discarded_client_ciphertext_count"], 0
        )

    def test_setup_rejects_factor_that_exceeds_actual_q(self):
        options = replace(make_options(), model_scaling_factor=2**49)
        engine = NativeCryptoEngine(options)
        with self.assertRaisesRegex(ValueError, "correctness.plaintext"):
            engine.setup(make_setup())

    def test_python_orchestrator_crosses_pybind_and_real_crypto_path(self):
        engine = NativeCryptoEngine(make_options())
        orchestrator = FederatedRoundOrchestrator(engine)
        placeholder_setup = make_setup()
        orchestrator.setup(placeholder_setup)

        setup_timing = engine.setup_diagnostics
        self.assertIsNotNone(setup_timing)
        self.assertEqual(
            len(setup_timing["client_initial_setup_seconds"]),
            len(placeholder_setup.clients),
        )
        self.assertEqual(
            len(setup_timing["client_collective_key_generation_seconds"]),
            len(placeholder_setup.clients),
        )
        self.assertGreater(setup_timing["estimated_parallel_seconds"], 0.0)
        self.assertLessEqual(
            setup_timing["estimated_parallel_seconds"],
            setup_timing["sequential_seconds"],
        )

        self.assertEqual(
            engine.effective_protocol["q_modulus_bits"],
            [28, 28],
        )
        self.assertEqual(
            engine.effective_protocol["hybrid_large_digits"],
            2,
        )

        self.assertNotEqual(
            engine.setup_config.setup_id,
            placeholder_setup.setup_id,
        )
        result = orchestrator.run_round(0, make_candidates())

        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertEqual(result.admission.accepted_actual_chunk_ids, (1,))
        expected_parameter = np.zeros(600, dtype=np.float32)
        expected_parameter[256:512] = 0.25
        np.testing.assert_allclose(
            result.decrypted.parameter_delta,
            expected_parameter,
            rtol=0.0,
            atol=2.0e-4,
        )
        np.testing.assert_allclose(
            result.decrypted.running_mean_delta,
            np.array([0.25, 0.5], dtype=np.float32),
            rtol=0.0,
            atol=2.0e-4,
        )
        np.testing.assert_allclose(
            result.decrypted.running_variance_delta,
            np.array([0.25, 0.5], dtype=np.float32),
            rtol=0.0,
            atol=2.0e-4,
        )
        self.assertGreater(
            engine.last_diagnostics["sparse_client_ciphertext_count"],
            0,
        )
        self.assertEqual(
            len(engine.last_diagnostics["client_encryption_seconds"]),
            len(placeholder_setup.clients),
        )
        self.assertGreater(
            engine.last_diagnostics["estimated_parallel_crypto_seconds"],
            0.0,
        )
        self.assertEqual(engine.last_diagnostics["crypto_thread_count"], 1)
        self.assertEqual(
            set(engine.last_diagnostics["crypto_stage_timings"]),
            {
                "encryption",
                "client_to_collective_key_switch",
                "rlwe_aggregation",
                "r2m",
                "mlwe_aggregation",
                "m2r",
                "partial_decryption",
                "decoding",
            },
        )
        sparse_clients = engine.last_diagnostics[
            "sparse_model_client_stage_timings"
        ]
        self.assertEqual(len(sparse_clients), len(placeholder_setup.clients))
        for client_timing in sparse_clients:
            self.assertGreater(client_timing["encryption"]["task_count"], 0)
            self.assertGreater(
                client_timing["client_to_collective_key_switch"]["task_count"],
                0,
            )
            self.assertGreater(
                client_timing["partial_decryption"]["task_count"],
                0,
            )
            self.assertEqual(client_timing["r2m"]["task_count"], 0)
            self.assertEqual(client_timing["m2r"]["task_count"], 0)
        sparse_server = engine.last_diagnostics[
            "sparse_model_server_stage_timings"
        ]
        for stage in (
            "r2m",
            "mlwe_aggregation",
            "m2r",
            "partial_decryption",
            "decoding",
        ):
            self.assertGreater(sparse_server[stage]["task_count"], 0)
        self.assertIsNotNone(result.timing)

    def test_full_round_crosses_pybind_without_sparse_conversion(self):
        options = replace(
            make_options(),
            protocol_mode="FULL",
            mlwe_rank=1,
            selection_ratio=1.0,
            selected_chunks_per_client=None,
            crypto_threads=3,
        )
        engine = NativeCryptoEngine(options)
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_full_setup())

        result = orchestrator.run_full_round(0, make_full_candidates())

        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertEqual(engine.last_diagnostics["crypto_thread_count"], 3)
        self.assertEqual(result.admission.accepted_actual_chunk_ids, ())
        np.testing.assert_allclose(
            result.decrypted.parameter_delta,
            np.full(600, 0.25, dtype=np.float32),
            rtol=0.0,
            atol=2.0e-4,
        )
        diagnostics = engine.last_diagnostics
        self.assertEqual(diagnostics["full_model_client_ciphertext_count"], 4)
        self.assertEqual(diagnostics["full_model_aggregate_ciphertext_count"], 1)
        self.assertEqual(diagnostics["sparse_client_ciphertext_count"], 0)
        self.assertEqual(diagnostics["r2m_ciphertext_count"], 0)
        self.assertEqual(diagnostics["m2r_ciphertext_count"], 0)
        self.assertEqual(len(diagnostics["client_key_switch_seconds"]), 4)
        self.assertEqual(len(diagnostics["client_total_crypto_seconds"]), 4)
        self.assertEqual(
            len(diagnostics["full_model_client_stage_timings"]),
            4,
        )
        self.assertEqual(
            len(diagnostics["auxiliary_client_stage_timings"]),
            4,
        )
        for client_timing in diagnostics["full_model_client_stage_timings"]:
            self.assertEqual(client_timing["encryption"]["task_count"], 1)
            self.assertEqual(
                client_timing["client_to_collective_key_switch"]["task_count"],
                1,
            )
            self.assertEqual(
                client_timing["partial_decryption"]["task_count"],
                1,
            )
            self.assertEqual(client_timing["r2m"]["task_count"], 0)
            self.assertEqual(client_timing["m2r"]["task_count"], 0)
        self.assertEqual(
            diagnostics["full_model_server_stage_timings"]["rlwe_aggregation"][
                "task_count"
            ],
            1,
        )
        self.assertAlmostEqual(
            diagnostics["maximum_client_pre_aggregation_seconds"],
            max(
                encryption + key_switch
                for encryption, key_switch in zip(
                    diagnostics["client_encryption_seconds"],
                    diagnostics["client_key_switch_seconds"],
                    strict=True,
                )
            ),
        )
        for stage in (
            "encryption",
            "client_to_collective_key_switch",
            "rlwe_aggregation",
            "partial_decryption",
            "decoding",
        ):
            self.assertGreater(
                diagnostics["crypto_stage_timings"][stage]["task_count"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
