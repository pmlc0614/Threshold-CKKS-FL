"""Dropout control tests run against actual OpenFHE, not a mocked aggregator."""
from dataclasses import replace
import unittest
import numpy as np
from plaintext_fl.native_engine import NativeCryptoEngine, native_binding_available
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator
from plaintext_fl.protocol_contract import RoundPhase
from plaintext_fl.communication import measure_full_online_round_communication
from tests.test_native_engine import (make_options, make_setup, make_candidates,
                                      make_full_setup, make_full_candidates)


@unittest.skipUnless(native_binding_available(), "ABI 22 native binding required")
class DropoutV17Test(unittest.TestCase):
    def engine(self, full=False, cap=1):
        engine = NativeCryptoEngine(replace(make_options(),
            protocol_mode="FULL" if full else "SPARSE",
            experimental_dynamic_cohort=True, dropout_target_cap=cap,
            collusion_bound=1))
        runner = FederatedRoundOrchestrator(engine)
        runner.setup(make_full_setup() if full else make_setup())
        return engine, runner

    def test_full_survivors_weights_cache_cap_and_rejoin(self):
        engine, runner = self.engine(full=True)
        candidates = tuple(replace(c, parameter_delta=c.parameter_delta / 10,
                                  running_mean_delta=c.running_mean_delta / 10,
                                  running_variance_delta=c.running_variance_delta / 10)
                           for c in make_full_candidates())
        for rid in (0, 1):
            result = runner.run_full_round(rid, candidates, survivor_client_ids=(10,20,40))
            np.testing.assert_allclose(result.decrypted.parameter_delta, -0.025, atol=2e-4)
            np.testing.assert_allclose(result.decrypted.running_mean_delta, -0.025, atol=2e-4)
            self.assertEqual(engine.last_diagnostics['survivor_target_key_cache_hit'], rid == 1)
            report = measure_full_online_round_communication(setup=engine.setup_config,
                effective_protocol=engine.effective_protocol, diagnostics=engine.last_diagnostics)
            self.assertEqual(report['v17_itemization']['final_survivor_count'], 3)
            self.assertEqual(report['combined_total_bits'], report['v17_itemization']['total']['total_bits'])
            rekey = report['uplink']['breakdown_bits']['updated_client_to_collective_ksk']
            self.assertEqual(rekey == 0, rid == 1)
        with self.assertRaisesRegex(RuntimeError, 'target cap exhausted'):
            runner.run_full_round(2, candidates, survivor_client_ids=(30,20,40))
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        with self.assertRaisesRegex(ValueError, 'aborted logical round is closed'):
            engine.begin_round(3, logical_round_index=2)
        result = runner.run_full_round(3, candidates)
        np.testing.assert_allclose(result.decrypted.parameter_delta, 0.025, atol=2e-4)

    def upload(self, engine):
        engine.begin_round(0)
        candidates = make_candidates()
        for c in candidates: engine.submit_selection(c.client_id, c.selected_actual_chunk_ids)
        admission = engine.finalize_admission()
        for c in candidates:
            chunks = admission.for_client(c.client_id).accepted_actual_chunk_ids
            rows = [c.selected_actual_chunk_ids.index(i) for i in chunks]
            engine.submit_accepted_values(c.client_id, chunks, c.selected_values[rows])
            engine.submit_auxiliary_deltas(c.client_id, c.running_mean_delta, c.running_variance_delta)

    def test_native_finalization_rejects_missing_and_stale_acknowledgements(self):
        engine, _ = self.engine()
        self.upload(engine)
        old = engine._backend.prepare_finalization()
        for cid in engine.setup_config.client_ids:
            engine._backend.acknowledge_finalization(cid, old)
        engine.set_round_survivors((10,20,40))
        current = engine._backend.prepare_finalization()
        self.assertNotEqual(old, current)
        with self.assertRaisesRegex(ValueError, 'stale or mismatched'):
            engine._backend.acknowledge_finalization(10, old)
        with self.assertRaisesRegex(ValueError, 'not from a survivor'):
            engine._backend.acknowledge_finalization(30, current)
        for cid in (10,20): engine._backend.acknowledge_finalization(cid, current)
        with self.assertRaisesRegex(RuntimeError, 'all final survivors'):
            engine._backend.finalize_round()
        self.assertEqual(engine.phase, RoundPhase.ABORTED)

    def test_sparse_empty_final_support_still_commits_auxiliary(self):
        engine, runner = self.engine()
        candidates = list(make_candidates())
        candidates[2] = replace(candidates[2], selected_actual_chunk_ids=(0,2))
        result = runner.run_round(0, candidates, survivor_client_ids=(10,20,40))
        self.assertEqual(result.admission.accepted_actual_chunk_ids, ())
        np.testing.assert_array_equal(result.decrypted.parameter_delta, np.zeros(600))
        np.testing.assert_allclose(result.decrypted.running_mean_delta, [0.225, 0.45], atol=2e-4)
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        result = runner.run_round(1, make_candidates())
        self.assertEqual(result.survivor_client_ids, engine.setup_config.client_ids)


if __name__ == '__main__': unittest.main()
