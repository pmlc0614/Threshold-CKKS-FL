import unittest

import numpy as np

from plaintext_fl.crypto_engine import CryptoEngineSetup, PostRequestDropoutError
from plaintext_fl.plaintext_engine import PlaintextCryptoEngine
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
        model_name="tiny-model",
        entries=(
            LayoutEntry(
                name=name,
                shape=(numel,),
                offset=0,
                numel=numel,
            ),
        ),
    )


def make_setup(minimum_cohort_size=2):
    return CryptoEngineSetup(
        setup_id=bytes(range(16)),
        parameter_layout=make_layout(LayoutKind.PARAMETER, "weight", 10),
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
        clients=(
            RegisteredClientWeight(client_id=10, example_count=1),
            RegisteredClientWeight(client_id=20, example_count=3),
        ),
        chunk_size=2,
        minimum_cohort_size=minimum_cohort_size,
    )


def make_candidates():
    return (
        ClientRoundCandidate(
            client_id=10,
            selected_actual_chunk_ids=(0, 1),
            selected_values=np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            running_mean_delta=np.array([1.0, 2.0], dtype=np.float32),
            running_variance_delta=np.array([2.0, 4.0], dtype=np.float32),
        ),
        ClientRoundCandidate(
            client_id=20,
            selected_actual_chunk_ids=(1, 2),
            selected_values=np.array(
                [[5.0, 6.0], [7.0, 8.0]],
                dtype=np.float32,
            ),
            running_mean_delta=np.array([5.0, 6.0], dtype=np.float32),
            running_variance_delta=np.array([6.0, 8.0], dtype=np.float32),
        ),
    )


def make_dropout_setup():
    return CryptoEngineSetup(
        setup_id=bytes(range(16)),
        parameter_layout=make_layout(LayoutKind.PARAMETER, "weight", 6),
        running_mean_layout=make_layout(
            LayoutKind.RUNNING_MEAN, "bn.running_mean", 2
        ),
        running_variance_layout=make_layout(
            LayoutKind.RUNNING_VARIANCE, "bn.running_var", 2
        ),
        clients=tuple(
            RegisteredClientWeight(client_id=client_id, example_count=1)
            for client_id in (30, 10, 20, 40)
        ),
        chunk_size=2,
        minimum_cohort_size=3,
    )


def make_dropout_candidates():
    selections = ((0, 1), (0, 1), (1, 2), (1, 2))
    return tuple(
        ClientRoundCandidate(
            client_id=client_id,
            selected_actual_chunk_ids=selection,
            selected_values=np.full(
                (2, 2), position / 10.0, dtype=np.float32
            ),
            running_mean_delta=np.full(
                2, position / 10.0, dtype=np.float32
            ),
            running_variance_delta=np.full(
                2, position / 10.0, dtype=np.float32
            ),
        )
        for position, (client_id, selection) in enumerate(
            zip((30, 10, 20, 40), selections, strict=True),
            start=1,
        )
    )


class RecordingPlaintextEngine(PlaintextCryptoEngine):
    def __init__(self):
        super().__init__()
        self.events = []
        self.admission_finalized = False

    def submit_selection(self, client_id, selected_actual_chunk_ids):
        self.events.append(("selection", client_id))
        super().submit_selection(client_id, selected_actual_chunk_ids)

    def finalize_admission(self):
        self.events.append(("admission", None))
        result = super().finalize_admission()
        self.admission_finalized = True
        return result

    def submit_accepted_values(
        self,
        client_id,
        accepted_actual_chunk_ids,
        values,
    ):
        if not self.admission_finalized:
            raise AssertionError("values crossed the boundary before admission")
        self.events.append(("values", client_id))
        super().submit_accepted_values(
            client_id,
            accepted_actual_chunk_ids,
            values,
        )


class CryptoEngineTest(unittest.TestCase):
    def test_post_upload_dropout_keeps_fixed_weights_and_excludes_silo(self):
        engine = PlaintextCryptoEngine(experimental_dynamic_cohort=True)
        orchestrator = FederatedRoundOrchestrator(engine)
        setup = make_dropout_setup()
        orchestrator.setup(setup)

        transitions = []
        result = orchestrator.run_round(
            0,
            make_dropout_candidates(),
            survivor_client_ids=(10, 20, 40),
            after_survivor_transition=lambda admission: transitions.append(
                admission.accepted_actual_chunk_ids
            ),
        )

        self.assertEqual(
            result.upload_admission.accepted_actual_chunk_ids, (1,)
        )
        self.assertEqual(result.admission.accepted_actual_chunk_ids, (1,))
        self.assertEqual(result.survivor_client_ids, (10, 20, 40))
        self.assertEqual(result.dropped_client_ids, (30,))
        self.assertEqual(result.rejected_for_client(30), (0, 1))
        self.assertEqual(transitions, [(1,)])
        expected = np.zeros(6, dtype=np.float32)
        expected[2:4] = 0.225
        np.testing.assert_allclose(result.decrypted.parameter_delta, expected)
        np.testing.assert_allclose(
            result.decrypted.running_mean_delta,
            np.full(2, 0.225, dtype=np.float32),
        )
        self.assertGreaterEqual(result.timing.survivor_transition_seconds, 0.0)

    def test_full_round_skips_selection_and_matches_weighted_dense_result(self):
        setup = CryptoEngineSetup(
            setup_id=bytes(range(16)),
            parameter_layout=make_layout(LayoutKind.PARAMETER, "weight", 5),
            running_mean_layout=make_layout(
                LayoutKind.RUNNING_MEAN, "bn.running_mean", 2
            ),
            running_variance_layout=make_layout(
                LayoutKind.RUNNING_VARIANCE, "bn.running_var", 2
            ),
            clients=(
                RegisteredClientWeight(client_id=10, example_count=1),
                RegisteredClientWeight(client_id=20, example_count=3),
            ),
            chunk_size=4,
            minimum_cohort_size=2,
            protocol_mode="FULL",
        )
        candidates = (
            ClientFullRoundCandidate(
                client_id=10,
                parameter_delta=np.arange(5, dtype=np.float32),
                running_mean_delta=np.array([1, 2], dtype=np.float32),
                running_variance_delta=np.array([2, 3], dtype=np.float32),
            ),
            ClientFullRoundCandidate(
                client_id=20,
                parameter_delta=np.arange(5, dtype=np.float32) + 4,
                running_mean_delta=np.array([5, 6], dtype=np.float32),
                running_variance_delta=np.array([6, 7], dtype=np.float32),
            ),
        )
        engine = PlaintextCryptoEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(setup)
        result = orchestrator.run_full_round(0, candidates)

        np.testing.assert_allclose(
            result.decrypted.parameter_delta,
            (candidates[0].parameter_delta + 3 * candidates[1].parameter_delta)
            / 4,
        )
        self.assertEqual(result.admission.accepted_actual_chunk_ids, ())
        self.assertEqual(result.decrypted.cohort_sizes, ())
        self.assertEqual(result.timing.client_selection_submission_seconds, ())
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)

    def test_setup_freezes_v17_weights_and_dimensions(self):
        setup = make_setup()
        setup.validate()

        self.assertEqual(setup.alphas, (32, 96))
        self.assertEqual(setup.parameter_chunk_count, 5)

    def test_orchestrator_submits_values_only_after_admission(self):
        engine = RecordingPlaintextEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        setup = make_setup()
        orchestrator.setup(setup)

        result = orchestrator.run_round(0, make_candidates())

        self.assertIsNotNone(result.timing)
        self.assertEqual(
            len(result.timing.client_selection_submission_seconds),
            len(setup.clients),
        )
        self.assertEqual(
            len(result.timing.client_value_submission_seconds),
            len(setup.clients),
        )
        self.assertGreaterEqual(result.timing.finalize_seconds, 0.0)
        self.assertGreaterEqual(result.timing.total_seconds, 0.0)

        self.assertEqual(
            engine.events,
            [
                ("selection", 10),
                ("selection", 20),
                ("admission", None),
                ("values", 10),
                ("values", 20),
            ],
        )
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertEqual(result.admission.accepted_actual_chunk_ids, (1,))
        self.assertEqual(result.rejected_for_client(10), (0,))
        self.assertEqual(result.rejected_for_client(20), (2,))

    def test_pre_value_validation_failure_aborts_before_values_cross(self):
        engine = RecordingPlaintextEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())

        def reject_aggregate(_admission):
            raise ValueError("aggregate bound exceeded")

        with self.assertRaisesRegex(ValueError, "aggregate bound exceeded"):
            orchestrator.run_round(
                0,
                make_candidates(),
                before_value_submission=reject_aggregate,
            )

        self.assertEqual(
            engine.events,
            [("selection", 10), ("selection", 20), ("admission", None)],
        )
        self.assertEqual(engine.phase, RoundPhase.ABORTED)

    def test_plaintext_engine_matches_weighted_sparse_and_auxiliary_result(self):
        engine = PlaintextCryptoEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())

        result = orchestrator.run_round(0, make_candidates()).decrypted

        np.testing.assert_array_equal(
            result.parameter_delta,
            np.array(
                [0.0, 0.0, 4.5, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            result.running_mean_delta,
            np.array([4.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.running_variance_delta,
            np.array([5.0, 7.0], dtype=np.float32),
        )

    def test_unapproved_values_are_rejected(self):
        engine = PlaintextCryptoEngine()
        engine.setup(make_setup())
        engine.begin_round(0)
        engine.submit_selection(10, (0, 1))
        engine.submit_selection(20, (1, 2))
        engine.finalize_admission()

        with self.assertRaises(ValueError):
            engine.submit_accepted_values(
                10,
                (0, 1),
                np.zeros((2, 2), dtype=np.float32),
            )

    def test_model_apply_failure_aborts_instead_of_committing(self):
        engine = PlaintextCryptoEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())

        def fail_apply(_result):
            raise RuntimeError("model update failed")

        with self.assertRaisesRegex(RuntimeError, "model update failed"):
            orchestrator.run_round(
                0,
                make_candidates(),
                apply_result=fail_apply,
            )

        self.assertEqual(engine.phase, RoundPhase.ABORTED)

    def test_late_dropout_closes_logical_round(self):
        class RetryEngine(PlaintextCryptoEngine):
            def __init__(self):
                super().__init__(experimental_dynamic_cohort=True)
                self.started = []

            def begin_round(self, round_id, *, logical_round_index=None):
                self.started.append((round_id, logical_round_index))
                return super().begin_round(round_id, logical_round_index=logical_round_index)

            def finalize_round(self, *, post_request_dropouts=()):
                if post_request_dropouts:
                    self.abort_round()
                    raise PostRequestDropoutError("simulated late dropout")
                return super().finalize_round()

        engine = RetryEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())
        applied = []
        with self.assertRaises(PostRequestDropoutError):
            orchestrator.run_round(
                0,
                make_candidates(),
                logical_round_index=0,
                post_request_dropouts=(10,),
                apply_result=lambda result: applied.append(result),
            )
        self.assertEqual(engine.phase, RoundPhase.ABORTED)
        self.assertEqual(applied, [])
        with self.assertRaisesRegex(ValueError, "cannot be retried"):
            engine.begin_round(1, logical_round_index=0)
        result = orchestrator.run_round(
            1,
            make_candidates(),
            logical_round_index=1,
            apply_result=lambda value: applied.append(value),
        )
        self.assertEqual(engine.started, [(0, 0), (1, 0), (1, 1)])
        self.assertEqual(result.decrypted.round_id, 1)
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertEqual(len(applied), 1)

    def test_all_rejected_round_still_completes_with_zero_parameter_delta(self):
        setup = make_setup(minimum_cohort_size=2)
        candidates = list(make_candidates())
        candidates[1] = ClientRoundCandidate(
            client_id=20,
            selected_actual_chunk_ids=(3, 4),
            selected_values=np.array(
                [[5.0, 6.0], [7.0, 8.0]],
                dtype=np.float32,
            ),
            running_mean_delta=np.array([5.0, 6.0], dtype=np.float32),
            running_variance_delta=np.array([6.0, 8.0], dtype=np.float32),
        )
        orchestrator = FederatedRoundOrchestrator(PlaintextCryptoEngine())
        orchestrator.setup(setup)

        result = orchestrator.run_round(0, candidates).decrypted

        self.assertEqual(result.accepted_actual_chunk_ids, ())
        np.testing.assert_array_equal(
            result.parameter_delta,
            np.zeros(10, dtype=np.float32),
        )

    def test_round_ids_must_increase_after_commit(self):
        engine = PlaintextCryptoEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(make_setup())
        orchestrator.run_round(0, make_candidates())

        with self.assertRaises(ValueError):
            orchestrator.run_round(0, make_candidates())


if __name__ == "__main__":
    unittest.main()
