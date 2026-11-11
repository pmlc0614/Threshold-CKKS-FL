import unittest
from dataclasses import replace
from plaintext_fl.plaintext_engine import PlaintextCryptoEngine
from plaintext_fl.protocol_contract import RoundPhase
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator
from tests.test_crypto_engine import make_setup, make_candidates, make_dropout_setup, make_dropout_candidates

class FixedCohortTests(unittest.TestCase):
    def test_missing_selection_ends_execution(self):
        engine = PlaintextCryptoEngine()
        engine.setup(make_setup())
        engine.begin_round(0)
        engine.submit_selection(10, (0,))
        with self.assertRaises(RuntimeError): engine.finalize_admission()
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        with self.assertRaisesRegex(RuntimeError, "retries"): engine.begin_round(1)

    def test_survivor_change_never_applies_model(self):
        engine = PlaintextCryptoEngine()
        runner = FederatedRoundOrchestrator(engine)
        runner.setup(make_dropout_setup())
        applied = []
        with self.assertRaisesRegex(RuntimeError, "survivor"):
            runner.run_round(0, make_dropout_candidates(), survivor_client_ids=(10,20,40),
                             apply_result=applied.append)
        self.assertEqual(applied, [])
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        with self.assertRaises(RuntimeError): engine.begin_round(1)

    def test_missing_values_aborts_full_and_sparse(self):
        for mode in ("FULL", "SPARSE"):
            engine = PlaintextCryptoEngine()
            engine.setup(replace(make_setup(), protocol_mode=mode))
            engine.begin_round(0)
            if mode == "SPARSE":
                engine.submit_selection(10, (0,))
                engine.submit_selection(20, (0,))
                engine.finalize_admission()
            with self.assertRaises(RuntimeError): engine.finalize_round()
            self.assertEqual(engine.phase, RoundPhase.ABORTED)
            with self.assertRaises(RuntimeError): engine.begin_round(1)
