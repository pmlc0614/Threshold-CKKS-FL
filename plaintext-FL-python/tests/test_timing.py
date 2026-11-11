import unittest

from plaintext_fl.encrypted_training import _estimate_parallel_encrypted_path
from plaintext_fl.native_engine import _parse_stage_timings
from plaintext_fl.round_orchestrator import (
    OrchestratedRoundResult,
    OrchestrationTiming,
)
from plaintext_fl.timing import (
    estimated_barrier_parallel_seconds,
    maximum_phase_seconds,
)


class ParallelTimingTest(unittest.TestCase):
    def test_wait_diagnostics_survive_native_adapter(self):
        raw = {"partial_decryption": {
            "wall_seconds": 0.1, "work_seconds": 0.2,
            "executor_wait_seconds": 1.3, "task_count": 2,
        }}
        row = _parse_stage_timings(raw)["partial_decryption"]
        self.assertEqual(row["executor_wait_seconds"], 1.3)
        self.assertAlmostEqual(row["elapsed_including_executor_wait_seconds"], 1.4)
        raw["partial_decryption"]["executor_wait_seconds"] = float('nan')
        with self.assertRaises(RuntimeError):
            _parse_stage_timings(raw)

    def test_service_estimate_ignores_executor_wait_and_uses_phase_maxima(self):
        result = OrchestratedRoundResult(
            admission=None, decrypted=None, rejected_actual_chunk_ids_by_client=(),
            timing=OrchestrationTiming(
                client_selection_submission_seconds=(),
                client_value_submission_seconds=(0.0, 0.0),
                survivor_transition_seconds=0.0, finalize_seconds=12.0,
                total_seconds=12.0,
            ),
        )
        native = {
            "crypto_timing_model": "client-batch-service-v3",
            "client_encryption_seconds": (1.0, 2.0),
            "client_key_switch_seconds": (2.0, 1.0),
            "client_partial_decryption_seconds": (0.5, 1.0),
            "crypto_sequential_seconds": 12.0,
            "crypto_server_seconds": 4.5,
            "auxiliary_client_stage_timings": [
                {"partial_decryption": {"executor_wait_seconds": 100.0}}
            ],
        }
        estimated, document = _estimate_parallel_encrypted_path(
            result=result, native_diagnostics=native,
        )
        self.assertEqual(estimated, 8.5)  # max(1+2,2+1) + max(.5,1) + 4.5
        self.assertTrue(document["executor_wait_excluded_from_client_times"])
        self.assertEqual(document["model"], "barrier-critical-path-independent-clients-v3")

    def test_maximum_phase_uses_slowest_client(self):
        self.assertEqual(maximum_phase_seconds([1.0, 2.5, 2.0]), 2.5)

    def test_barrier_estimate_sums_phase_maxima_and_server(self):
        self.assertEqual(
            estimated_barrier_parallel_seconds(
                client_phases=[[1.0, 2.0], [4.0, 3.0]],
                server_seconds=5.0,
            ),
            11.0,
        )

    def test_negative_time_is_rejected(self):
        with self.assertRaises(ValueError):
            maximum_phase_seconds([1.0, -1.0])

    def test_encrypted_parallel_estimate_never_exceeds_measured_path(self):
        result = OrchestratedRoundResult(
            admission=None,
            decrypted=None,
            rejected_actual_chunk_ids_by_client=(),
            timing=OrchestrationTiming(
                client_selection_submission_seconds=(),
                client_value_submission_seconds=(1.0, 1.0),
                survivor_transition_seconds=0.0,
                finalize_seconds=2.0,
                total_seconds=4.0,
            ),
        )
        estimated, diagnostics = _estimate_parallel_encrypted_path(
            result=result,
            native_diagnostics={
                "client_encryption_seconds": (10.0, 10.0),
                "client_partial_decryption_seconds": (10.0, 10.0),
                "crypto_sequential_seconds": 2.0,
                "crypto_server_seconds": 0.0,
            },
        )
        self.assertEqual(estimated, 4.0)
        self.assertTrue(diagnostics["capped_to_sequential_encrypted_path"])
        self.assertGreater(
            diagnostics["uncapped_barrier_estimate_seconds"],
            diagnostics["sequential_encrypted_path_upper_bound_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
