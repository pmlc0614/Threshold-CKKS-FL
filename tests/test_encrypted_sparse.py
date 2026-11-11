import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from plaintext_fl.encrypted_training import (
    EncryptedSparseRoundConfig,
    _build_encrypted_model,
    _load_encrypted_dataset,
    _make_comparison_document,
    _parse_args,
    array_error_statistics,
    chunk_path_alignment_document,
    encrypted_round_sparsity_metrics,
    model_state_difference_document,
    run_encrypted_full_round,
    run_encrypted_sparse_round,
    run_encrypted_sparse_rounds,
    run_paired_encrypted_sparse_round,
)
from plaintext_fl.model_adapter import build_model_protocol_layouts, flatten_layout
from plaintext_fl.native_engine import (
    NativeCryptoEngine,
    NativeCryptoEngineOptions,
    native_binding_available,
)
from plaintext_fl.plaintext_engine import PlaintextCryptoEngine
from plaintext_fl.protocol_contract import RoundPhase
from plaintext_fl.training import learning_rate_at_step


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 2)

    def forward(self, values):
        return self.linear2(self.relu(self.bn(self.linear1(values))))


def make_loaders():
    first = TensorDataset(
        torch.tensor(
            [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 0, 1, 1]),
    )
    second = TensorDataset(
        torch.tensor(
            [[1.0, 1.0], [2.0, 1.0], [-1.0, -1.0], [-2.0, -1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 0, 1, 1]),
    )
    return (
        DataLoader(first, batch_size=2, shuffle=False),
        DataLoader(second, batch_size=2, shuffle=False),
    )


class CountingPlaintextCryptoEngine(PlaintextCryptoEngine):
    def __init__(self):
        super().__init__()
        self.setup_calls = 0

    def setup(self, config):
        self.setup_calls += 1
        super().setup(config)


class EncryptedSparseRoundTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyClassifier()
        self.layouts = build_model_protocol_layouts(
            self.model,
            model_name="tiny-encrypted-round",
        )

    def run_round(
        self,
        engine,
        *,
        chunk_size,
        selected_chunk_count,
        aggregate_auxiliary_bound=10.0,
    ):
        before = flatten_layout(
            self.model,
            self.layouts.parameter,
        ).copy()
        outcome = run_encrypted_sparse_round(
            global_model=self.model,
            client_loaders=make_loaders(),
            sample_counts=(4, 4),
            engine=engine,
            config=EncryptedSparseRoundConfig(
                chunk_size=chunk_size,
                selected_chunk_count=selected_chunk_count,
                minimum_cohort_size=2,
                local_epochs=1,
                learning_rate=0.05,
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=10.0,
                aggregate_auxiliary_update_bound=aggregate_auxiliary_bound,
                compute_plaintext_reference=True,
            ),
            device=torch.device("cpu"),
            layouts=self.layouts,
        )
        after = flatten_layout(self.model, self.layouts.parameter)
        return before, after, outcome

    def test_local_training_to_plaintext_reference_commit(self):
        total_numel = self.layouts.parameter.total_numel
        before, after, outcome = self.run_round(
            PlaintextCryptoEngine(),
            chunk_size=4,
            selected_chunk_count=(total_numel + 3) // 4,
        )

        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(
            outcome.result.admission.accepted_actual_chunk_ids,
            (0, 1, 2, 3),
        )
        self.assertEqual(len(outcome.residuals), 2)
        self.assertTrue(
            all(
                torch.count_nonzero(value) == 0
                for value in outcome.residuals
            )
        )
        self.assertEqual(len(outcome.clients), 2)
        sparsity = encrypted_round_sparsity_metrics(
            outcome,
            total_chunk_count=4,
        )
        self.assertEqual(sparsity["accepted_contribution_count"], 8)
        self.assertEqual(sparsity["rho_selected"], 1.0)
        self.assertEqual(sparsity["rho_eff"], 1.0)
        self.assertEqual(sparsity["cohort_retention_ratio"], 1.0)
        self.assertEqual(
            outcome.plaintext_aggregate.parameter.shape,
            (self.layouts.parameter.total_numel,),
        )

    def test_full_round_uses_no_sparse_admission_or_residual(self):
        before = flatten_layout(self.model, self.layouts.parameter).copy()
        outcome = run_encrypted_full_round(
            global_model=self.model,
            client_loaders=make_loaders(),
            sample_counts=(4, 4),
            engine=PlaintextCryptoEngine(),
            config=EncryptedSparseRoundConfig(
                chunk_size=8,
                selected_chunk_count=2,
                minimum_cohort_size=2,
                local_epochs=1,
                learning_rate=0.05,
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=10.0,
                aggregate_auxiliary_update_bound=10.0,
                protocol_mode="FULL",
            ),
            device=torch.device("cpu"),
            layouts=self.layouts,
        )
        after = flatten_layout(self.model, self.layouts.parameter)
        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(outcome.result.admission.accepted_actual_chunk_ids, ())
        self.assertEqual(outcome.result.decrypted.cohort_sizes, ())
        self.assertTrue(
            all(not client.selected_chunk_ids for client in outcome.clients)
        )
        self.assertEqual(outcome.residuals, ())

    def test_error_statistics_report_bound_relative_effective_p(self):
        statistics = array_error_statistics(
            np.zeros(4, dtype=np.float64),
            np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64),
            reference_bound=8.0,
            expected_precision_bits=3,
        )

        self.assertAlmostEqual(statistics["rmse"], 1.0)
        self.assertAlmostEqual(
            statistics["effective_precision_bits_from_rmse"], 3.0
        )
        self.assertAlmostEqual(statistics["rmse_over_configured_sigma"], 1.0)

    def test_reference_locked_uses_one_chunk_path_and_updates_two_models(self):
        reference_model = copy.deepcopy(self.model)
        encrypted_model = copy.deepcopy(self.model)
        paired = run_paired_encrypted_sparse_round(
            mode="reference-locked",
            reference_model=reference_model,
            encrypted_model=encrypted_model,
            encrypted_client_loaders=make_loaders(),
            sample_counts=(4, 4),
            encrypted_engine=PlaintextCryptoEngine(),
            config=EncryptedSparseRoundConfig(
                chunk_size=4,
                selected_chunk_count=1,
                minimum_cohort_size=2,
                learning_rate=0.05,
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=10.0,
                aggregate_auxiliary_update_bound=10.0,
            ),
            device=torch.device("cpu"),
            layouts=self.layouts,
        )

        alignment = chunk_path_alignment_document(
            paired.reference, paired.encrypted
        )
        difference = model_state_difference_document(
            reference_model, encrypted_model, self.layouts
        )
        self.assertTrue(alignment["all_client_selections_exact"])
        self.assertEqual(difference["parameter"]["max_abs_error"], 0.0)
        self.assertEqual(difference["running_mean"]["max_abs_error"], 0.0)
        self.assertEqual(difference["running_variance"]["max_abs_error"], 0.0)
        document = _make_comparison_document(
            mode="reference-locked",
            reference_outcome=paired.reference,
            encrypted_outcome=paired.encrypted,
            reference_model=reference_model,
            encrypted_model=encrypted_model,
            reference_evaluation=None,
            encrypted_evaluation=None,
            layouts=self.layouts,
            profile=SimpleNamespace(
                aggregate_update_bound=10.0,
                aggregate_aux_update_bound=10.0,
                precision_bits=14,
                aux_precision_bits=14,
                chunk_size=4,
            ),
        )
        json.dumps(document, allow_nan=False)
        self.assertEqual(
            document["crypto_aggregate_error"]["parameter"][
                "configured_precision_bits"
            ],
            14,
        )
        self.assertEqual(
            document["sparsity"]["reference"]["rho_eff"],
            document["sparsity"]["encrypted"]["rho_eff"],
        )
        self.assertEqual(document["sparsity"]["rho_eff_gap"], 0.0)

    def test_free_running_keeps_independent_plaintext_trajectories(self):
        reference_model = copy.deepcopy(self.model)
        encrypted_model = copy.deepcopy(self.model)
        paired = run_paired_encrypted_sparse_round(
            mode="free-running",
            reference_model=reference_model,
            encrypted_model=encrypted_model,
            reference_client_loaders=make_loaders(),
            encrypted_client_loaders=make_loaders(),
            sample_counts=(4, 4),
            reference_engine=PlaintextCryptoEngine(),
            encrypted_engine=PlaintextCryptoEngine(),
            config=EncryptedSparseRoundConfig(
                chunk_size=4,
                selected_chunk_count=1,
                minimum_cohort_size=2,
                learning_rate=0.05,
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=10.0,
                aggregate_auxiliary_update_bound=10.0,
            ),
            device=torch.device("cpu"),
            layouts=self.layouts,
        )

        self.assertIsNot(paired.reference, paired.encrypted)
        self.assertTrue(
            chunk_path_alignment_document(
                paired.reference, paired.encrypted
            )["all_client_selections_exact"]
        )
        self.assertEqual(
            model_state_difference_document(
                reference_model, encrypted_model, self.layouts
            )["parameter"]["max_abs_error"],
            0.0,
        )

    def test_local_value_may_exceed_an_in_range_aggregate_bound(self):
        total_numel = self.layouts.parameter.total_numel
        _, _, outcome = self.run_round(
            PlaintextCryptoEngine(),
            chunk_size=4,
            selected_chunk_count=(total_numel + 3) // 4,
            aggregate_auxiliary_bound=0.1894,
        )

        self.assertGreater(
            max(client.running_variance_max_abs for client in outcome.clients),
            0.1894,
        )
        self.assertLessEqual(
            outcome.plaintext_aggregate_maxima.running_variance,
            0.1894,
        )
        self.assertLessEqual(
            outcome.decrypted_aggregate_maxima.running_variance,
            0.1894,
        )

    def test_multiple_rounds_reuse_setup_and_advance_model_and_residuals(self):
        engine = CountingPlaintextCryptoEngine()
        before = flatten_layout(self.model, self.layouts.parameter).copy()
        observed_rounds = []

        def observe(outcome):
            observed_rounds.append(
                (
                    outcome.result.decrypted.round_id,
                    outcome.setup_seconds,
                    outcome.learning_rate,
                    tuple(value.clone() for value in outcome.residuals),
                )
            )

        training = run_encrypted_sparse_rounds(
            global_model=self.model,
            client_loaders=make_loaders(),
            sample_counts=(4, 4),
            engine=engine,
            config=EncryptedSparseRoundConfig(
                chunk_size=4,
                selected_chunk_count=1,
                minimum_cohort_size=2,
                learning_rate=0.05,
                minimum_learning_rate=0.005,
                schedule="cosine",
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=10.0,
                aggregate_auxiliary_update_bound=10.0,
                round_id=7,
            ),
            rounds=3,
            device=torch.device("cpu"),
            layouts=self.layouts,
            on_round_committed=observe,
        )

        self.assertEqual(engine.setup_calls, 1)
        self.assertEqual(training.completed_round_ids, (7, 8, 9))
        self.assertEqual(
            [setup_seconds for _, setup_seconds, _, _ in observed_rounds[1:]],
            [0.0, 0.0],
        )
        expected_learning_rates = [
            learning_rate_at_step(
                initial_learning_rate=0.05,
                minimum_learning_rate=0.005,
                step=step,
                total_steps=3,
                schedule="cosine",
            )
            for step in range(3)
        ]
        for observed, expected in zip(
            (item[2] for item in observed_rounds),
            expected_learning_rates,
            strict=True,
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertFalse(
            np.array_equal(before, flatten_layout(self.model, self.layouts.parameter))
        )
        self.assertTrue(
            any(torch.count_nonzero(value) > 0 for value in training.residuals)
        )
        self.assertTrue(
            any(
                not torch.equal(first, last)
                for first, last in zip(
                    observed_rounds[0][3],
                    training.residuals,
                    strict=True,
                )
            )
        )
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)

    def test_encrypted_training_cli_defaults_match_plaintext_experiment(self):
        with patch("sys.argv", ["encrypted-sparse"]):
            args = _parse_args()

        self.assertEqual(args.rounds, 200)
        self.assertEqual(args.local_epochs, 1)
        self.assertEqual(args.optimizer, "sgd")
        self.assertEqual(args.learning_rate, 0.1)
        self.assertEqual(args.minimum_learning_rate, 0.0)
        self.assertEqual(args.schedule, "cosine")
        self.assertEqual(args.momentum, 0.9)
        self.assertEqual(args.weight_decay, 5e-4)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.comparison_mode, "none")
        self.assertEqual(args.dataset, "cifar10")
        self.assertEqual(args.model, "resnet20")
        self.assertIsNone(args.partition_file)
        self.assertIsNone(args.partition_seed)

    def test_cifar100_resnet32_cli_and_dispatch(self):
        with patch(
            "sys.argv",
            [
                "encrypted-sparse",
                "--dataset",
                "cifar100",
                "--model",
                "resnet32",
            ],
        ):
            args = _parse_args()

        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(args.model, "resnet32")
        self.assertEqual(_build_encrypted_model(args.model).fc.out_features, 100)

        with patch(
            "plaintext_fl.encrypted_training.load_cifar100",
            return_value=("train", "test"),
        ) as load_cifar100:
            result = _load_encrypted_dataset(
                dataset_name=args.dataset,
                data_dir="./data",
                download=False,
            )

        self.assertEqual(result, ("train", "test"))
        load_cifar100.assert_called_once_with(
            data_dir="./data",
            download=False,
        )

    def test_legacy_one_round_entry_point_keeps_one_round_default(self):
        with patch("sys.argv", ["encrypted-sparse-one-round"]):
            args = _parse_args(default_rounds=1)

        self.assertEqual(args.rounds, 1)

    def test_fixed_partition_cli(self):
        with patch(
            "sys.argv",
            [
                "encrypted-sparse",
                "--partition-file",
                "partitions/cifar10-fixed.json",
                "--partition-seed",
                "42",
            ],
        ):
            args = _parse_args()

        self.assertEqual(
            args.partition_file,
            "partitions/cifar10-fixed.json",
        )
        self.assertEqual(args.partition_seed, 42)

    def test_reused_setup_rejects_changed_round_contract(self):
        engine = PlaintextCryptoEngine()
        self.run_round(
            engine,
            chunk_size=4,
            selected_chunk_count=4,
        )
        with self.assertRaisesRegex(ValueError, "installed setup: chunk_size"):
            run_encrypted_sparse_round(
                global_model=self.model,
                client_loaders=make_loaders(),
                sample_counts=(4, 4),
                engine=engine,
                config=EncryptedSparseRoundConfig(
                    chunk_size=8,
                    selected_chunk_count=2,
                    minimum_cohort_size=2,
                    learning_rate=0.05,
                    momentum=0.0,
                    weight_decay=0.0,
                    aggregate_update_bound=10.0,
                    aggregate_auxiliary_update_bound=10.0,
                    round_id=1,
                ),
                device=torch.device("cpu"),
                layouts=self.layouts,
            )

@unittest.skipUnless(
    native_binding_available(),
    "ciphertext_fl_cpp was not built or is not on PYTHONPATH",
)
class NativeEncryptedSparseRoundIntegrationTest(unittest.TestCase):
    def test_local_training_crosses_real_encrypted_path_and_updates_model(self):
        torch.manual_seed(19)
        model = TinyClassifier()
        layouts = build_model_protocol_layouts(
            model,
            model_name="tiny-native-trained-round",
        )
        before = flatten_layout(model, layouts.parameter).copy()
        engine = NativeCryptoEngine(
            NativeCryptoEngineOptions(
                profile="smoke",
                ring_dimension=1024,
                mlwe_rank=4,
                q_modulus_bits=(28, 28),
                p_modulus_bits=(55,),
                hybrid_large_digits=2,
                maximum_qp_bits=160,
                selected_chunks_per_client=1,
                update_bound=1.0,
                aux_update_bound=1.0,
                smudging_sigma=0.0,
                aux_smudging_sigma=0.0,
                use_sparse_compression=False,
                use_auxiliary_compression=False,
            )
        )

        outcome = run_encrypted_sparse_round(
            global_model=model,
            client_loaders=make_loaders(),
            sample_counts=(4, 4),
            engine=engine,
            config=EncryptedSparseRoundConfig(
                chunk_size=256,
                selected_chunk_count=1,
                minimum_cohort_size=2,
                learning_rate=0.05,
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=1.0,
                aggregate_auxiliary_update_bound=1.0,
            ),
            device=torch.device("cpu"),
            layouts=layouts,
        )

        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertFalse(
            np.array_equal(before, flatten_layout(model, layouts.parameter))
        )
        self.assertEqual(
            outcome.result.admission.accepted_actual_chunk_ids,
            (0,),
        )
        self.assertGreater(
            engine.last_diagnostics["sparse_client_ciphertext_count"],
            0,
        )

    def test_two_real_encrypted_rounds_reuse_one_native_setup(self):
        torch.manual_seed(23)
        model = TinyClassifier()
        layouts = build_model_protocol_layouts(
            model,
            model_name="tiny-native-multi-round",
        )
        engine = NativeCryptoEngine(
            NativeCryptoEngineOptions(
                profile="smoke",
                ring_dimension=1024,
                mlwe_rank=4,
                q_modulus_bits=(28, 28),
                p_modulus_bits=(55,),
                hybrid_large_digits=2,
                maximum_qp_bits=160,
                selected_chunks_per_client=1,
                update_bound=1.0,
                aux_update_bound=1.0,
                smudging_sigma=0.0,
                aux_smudging_sigma=0.0,
                use_sparse_compression=False,
                use_auxiliary_compression=False,
            )
        )
        training = run_encrypted_sparse_rounds(
            global_model=model,
            client_loaders=make_loaders(),
            sample_counts=(4, 4),
            engine=engine,
            config=EncryptedSparseRoundConfig(
                chunk_size=256,
                selected_chunk_count=1,
                minimum_cohort_size=2,
                learning_rate=0.05,
                momentum=0.0,
                weight_decay=0.0,
                aggregate_update_bound=1.0,
                aggregate_auxiliary_update_bound=1.0,
                round_id=11,
            ),
            rounds=2,
            device=torch.device("cpu"),
            layouts=layouts,
        )

        self.assertEqual(training.completed_round_ids, (11, 12))
        self.assertEqual(training.final_round.setup_seconds, 0.0)
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)
        self.assertGreater(
            engine.last_diagnostics["sparse_client_ciphertext_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
