import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from plaintext_fl.aggregation import (
    ClientSparseUpdate,
    aggregate_sparse_fedavg,
    sparse_communication_counts,
)
from plaintext_fl.federated_sparse import (
    _print_final_summary,
    _print_round_progress,
    build_sparse_model,
    load_sparse_dataset,
    parse_args,
    recommended_safety_scaled_bound,
    ROUND_METRIC_CONTEXT_FIELDS,
    round_metric_context,
    validate_sparse_chunk_selection,
    validate_sparse_experiment,
    validate_sparse_rho,
)
from plaintext_fl.model import ResNet20, ResNet32
from plaintext_fl.plaintext_communication import (
    measure_plaintext_round_communication,
)
from plaintext_fl.sparsify import (
    compress_with_error_feedback,
    restore_rejected_chunks_to_residual,
    select_topk_chunks,
    selected_chunk_count_from_ratio,
)


class SparseFederatedTest(unittest.TestCase):
    def test_round_metric_csv_schema_includes_dataset_and_model(self):
        row = round_metric_context(
            round_number=1,
            dataset_name="cifar100",
            model_name="resnet32",
        )
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=ROUND_METRIC_CONTEXT_FIELDS,
        )

        writer.writeheader()
        writer.writerow(row)

        self.assertEqual(
            output.getvalue().splitlines(),
            ["round,dataset,model", "1,cifar100,resnet32"],
        )

    def test_supported_dataset_model_pairs(self):
        validate_sparse_experiment("cifar10", "resnet20")
        validate_sparse_experiment("cifar100", "resnet32")

        with self.assertRaisesRegex(
            ValueError,
            "unsupported sparse dataset/model pair",
        ):
            validate_sparse_experiment("cifar100", "resnet20")

    def test_cifar100_resnet32_cli(self):
        with patch(
            "sys.argv",
            [
                "federated_sparse",
                "--dataset",
                "cifar100",
                "--model",
                "resnet32",
                "--u-max",
                "0.75",
                "--u-max-aux",
                "12.5",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(args.model, "resnet32")
        self.assertEqual(args.parameter_update_bound, 0.75)
        self.assertEqual(args.running_update_bound, 12.5)

    def test_sparse_model_factory_supports_both_experiments(self):
        self.assertIsInstance(build_sparse_model("resnet20"), ResNet20)
        self.assertIsInstance(build_sparse_model("resnet32"), ResNet32)

    @patch("plaintext_fl.federated_sparse.load_cifar100")
    def test_sparse_dataset_loader_dispatches_cifar100(self, load_cifar100):
        load_cifar100.return_value = ("train", "test")

        result = load_sparse_dataset(
            dataset_name="cifar100",
            data_dir="data",
            download=False,
        )

        self.assertEqual(result, ("train", "test"))
        load_cifar100.assert_called_once_with(
            data_dir="data",
            download=False,
        )

    def test_compact_terminal_output_highlights_accuracy_and_ranges(self):
        row = {
            "round": 7,
            "accuracy_percent": 81.25,
            "weighted_train_loss": 0.75,
            "test_loss": 0.625,
            "learning_rate": 0.08,
            "rho_requested": 0.3,
            "rho_selected": 0.300375,
            "rho_eff_round": 0.28125,
            "rho_eff_running": 0.279,
            "cohort_retention_ratio": 0.93633,
            "accepted_chunk_count": 55,
            "total_chunk_count": 267,
            "parameter_update_min": -0.2,
            "parameter_update_max": 0.3,
            "parameter_update_max_abs": 0.3,
            "observed_parameter_update_max_abs_so_far": 0.4,
            "running_mean_update_min": -1.0,
            "running_mean_update_max": 1.5,
            "running_mean_update_max_abs": 1.5,
            "running_var_update_min": -2.0,
            "running_var_update_max": 3.0,
            "running_var_update_max_abs": 3.0,
            "observed_running_update_max_abs_so_far": 4.0,
            "round_seconds": 12.5,
            "estimated_parallel_round_seconds": 5.5,
            "estimated_parallel_protocol_seconds": 4.5,
        }
        summary = {
            "rounds": 200,
            "final_accuracy_percent": 90.0,
            "best_accuracy_percent": 90.4,
            "best_round": 196,
            "rho_requested": 0.3,
            "rho_selected": 0.300375,
            "rho_eff": 0.281,
            "cohort_retention_ratio": 0.9355,
            "observed_parameter_update_max_abs": 0.4,
            "recommended_parameter_update_bound": 0.6,
            "observed_running_update_max_abs": 4.0,
            "observed_running_mean_update_max_abs": 1.5,
            "observed_running_var_update_max_abs": 4.0,
            "recommended_running_update_bound": 6.0,
            "total_estimated_parallel_protocol_seconds": 900.0,
            "average_estimated_parallel_protocol_seconds_per_round": 4.5,
            "total_seconds": 100.0,
        }
        output = StringIO()
        with redirect_stdout(output):
            _print_round_progress(row, 200)
            _print_final_summary(summary, Path("runs/example"))
        terminal = output.getvalue()
        self.assertIn("ROUND 007/200 | TEST ACCURACY: 81.25%", terminal)
        self.assertIn("PARAMETER UPDATE (clean aggregate)", terminal)
        self.assertIn("U_max_so_far=0.4", terminal)
        self.assertIn("U_max_aug_so_far=4", terminal)
        self.assertIn("rho_eff_round=0.281250", terminal)
        self.assertIn("rho_eff=0.281000", terminal)
        self.assertIn("parallel_estimate=5.5s", terminal)
        self.assertIn("estimated_parallel_protocol_total=900.0s", terminal)
        self.assertIn("FINAL TEST ACCURACY: 90.00%", terminal)
        self.assertNotIn('"wire_update_count"', terminal)

    def test_accepted_only_communication_counts(self):
        counts = sparse_communication_counts(
            cohort_sizes=torch.tensor(
                [0, 1, 2, 3, 5],
                dtype=torch.int64,
            ),
            minimum_cohort_size=3,
        )

        self.assertEqual(counts.union_chunk_count, 4)
        self.assertEqual(counts.accepted_chunk_count, 2)
        self.assertEqual(counts.rejected_chunk_count, 2)
        self.assertEqual(
            counts.candidate_contribution_count,
            11,
        )
        self.assertEqual(
            counts.accepted_contribution_count,
            8,
        )
        self.assertEqual(
            counts.rejected_contribution_count,
            3,
        )

    def test_sparse_plaintext_communication_uses_v17_bitmaps_and_padding(self):
        communication = measure_plaintext_round_communication(
            profile="SPARSE",
            cohort_sizes=(3, 0, 3),
            minimum_cohort_size=3,
            client_count=3,
            selected_chunk_count=2,
            parameter_numel=10,
            chunk_size=4,
            auxiliary_numel=2,
        )

        # v17 keeps both accepted chunks at B=4 on wire and sends one-byte
        # selection/upload/admission/result bitmaps over J=3 positions.
        self.assertEqual(communication["accepted_wire_value_numel"], 24)
        self.assertEqual(communication["candidate_index_bytes"], 3)
        self.assertEqual(communication["selection_bitmap_bytes"], 3)
        self.assertEqual(communication["upload_support_bitmap_bytes"], 3)
        self.assertEqual(communication["admission_bitmap_bytes"], 3)
        self.assertEqual(communication["result_bitmap_bytes"], 3)
        self.assertEqual(
            communication["accepted_only_fp32_uplink_bytes"],
            126,
        )
        self.assertEqual(
            communication["accepted_only_logical_downlink_bytes"],
            42,
        )
        self.assertEqual(
            communication["accepted_only_unicast_downlink_bytes"],
            126,
        )
        self.assertTrue(communication["padding_included"])

    def test_full_plaintext_communication_has_no_indices_or_padding(self):
        communication = measure_plaintext_round_communication(
            profile="FULL",
            cohort_sizes=(3, 3, 3),
            minimum_cohort_size=3,
            client_count=3,
            selected_chunk_count=3,
            parameter_numel=10,
            chunk_size=4,
            auxiliary_numel=2,
        )

        self.assertEqual(communication["candidate_index_bits"], 0)
        self.assertEqual(communication["candidate_index_bytes"], 0)
        self.assertEqual(communication["accepted_label_bytes"], 0)
        self.assertEqual(communication["accepted_wire_value_numel"], 30)
        self.assertEqual(
            communication["accepted_only_fp32_uplink_bytes"],
            144,
        )
        self.assertEqual(
            communication["accepted_only_logical_downlink_bytes"],
            48,
        )
        self.assertEqual(
            communication["accepted_only_unicast_downlink_bytes"],
            144,
        )
        self.assertFalse(communication["padding_included"])

    def test_resnet32_full_plaintext_payload_uses_exact_dimensions(self):
        communication = measure_plaintext_round_communication(
            profile="FULL",
            cohort_sizes=(8,) * 58,
            minimum_cohort_size=3,
            client_count=8,
            selected_chunk_count=58,
            parameter_numel=472_756,
            chunk_size=8_192,
            auxiliary_numel=2_464,
        )

        expected_all_client_bytes = 8 * (472_756 + 2_464) * 4
        self.assertEqual(
            communication["accepted_only_fp32_uplink_bytes"],
            expected_all_client_bytes,
        )
        self.assertEqual(
            communication["accepted_only_unicast_downlink_bytes"],
            expected_all_client_bytes,
        )
        self.assertEqual(
            communication["accepted_only_logical_downlink_bytes"],
            (472_756 + 2_464) * 4,
        )

    def test_noise_exponents_use_p_and_p_aux_cli_names(self):
        with patch(
            "sys.argv",
            [
                "federated_sparse",
                "--noise-p",
                "12",
                "--aux-noise-p",
                "14",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.noise_p, 12.0)
        self.assertEqual(args.aux_noise_p, 14.0)

    def test_minimum_cohort_size_cli(self):
        with patch(
            "sys.argv",
            [
                "federated_sparse",
                "--minimum-cohort-size",
                "3",
            ],
        ):
            args = parse_args()

        self.assertEqual(
            args.minimum_cohort_size,
            3,
        )

    def test_fixed_partition_cli(self):
        with patch(
            "sys.argv",
            [
                "federated_sparse",
                "--partition-file",
                "partitions/cifar10-fixed.json",
                "--partition-seed",
                "123",
            ],
        ):
            args = parse_args()

        self.assertEqual(
            args.partition_file,
            "partitions/cifar10-fixed.json",
        )
        self.assertEqual(args.partition_seed, 123)

    def test_legacy_noise_exponent_cli_names_still_work(self):
        with patch(
            "sys.argv",
            [
                "federated_sparse",
                "--parameter-noise-p",
                "11",
                "--running-noise-p",
                "13",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.noise_p, 11.0)
        self.assertEqual(args.aux_noise_p, 13.0)

    def test_recommended_bounds_use_multiplicative_margin(self):
        self.assertAlmostEqual(
            recommended_safety_scaled_bound(0.32935),
            0.494025,
        )

        self.assertAlmostEqual(
            recommended_safety_scaled_bound(8.7765),
            13.16475,
        )

        self.assertEqual(
            recommended_safety_scaled_bound(0.0),
            0.0,
        )

    def test_rho_is_converted_to_chunk_count(self):
        selected = (
            selected_chunk_count_from_ratio(
                total_numel=272_474,
                chunk_size=1024,
                rho=0.3,
            )
        )

        # ceil(0.3 * 272474 / 1024) = 80. This intentionally differs
        # from ceil(0.3 * ceil(272474 / 1024)) = 81.
        self.assertEqual(selected, 80)

    def test_rho_is_applied_before_partial_chunk_ceiling(self):
        self.assertEqual(
            selected_chunk_count_from_ratio(
                total_numel=9,
                chunk_size=4,
                rho=0.34,
            ),
            1,
        )

    def test_invalid_rho_is_rejected(self):
        for rho in [0.0, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                selected_chunk_count_from_ratio(
                    total_numel=100,
                    chunk_size=10,
                    rho=rho,
                )

    def test_sparse_mode_rejects_full_rho_and_effective_full_selection(self):
        with self.assertRaisesRegex(ValueError, "federated_full"):
            validate_sparse_rho(1.0)
        with self.assertRaisesRegex(ValueError, "fewer than all"):
            validate_sparse_chunk_selection(
                selected_chunk_count=58,
                total_chunk_count=58,
            )

    def test_exact_sparse_fedavg(self):
        first_selection = select_topk_chunks(
            update=torch.tensor(
                [4.0, 0.0, 0.0, 0.0]
            ),
            chunk_size=2,
            selected_chunk_count=1,
        )

        second_selection = select_topk_chunks(
            update=torch.tensor(
                [0.0, 0.0, 8.0, 0.0]
            ),
            chunk_size=2,
            selected_chunk_count=1,
        )

        aggregate = aggregate_sparse_fedavg(
            updates=[
                ClientSparseUpdate(
                    client_id=0,
                    round_id=0,
                    num_examples=1,
                    selection=first_selection,
                ),
                ClientSparseUpdate(
                    client_id=1,
                    round_id=0,
                    num_examples=3,
                    selection=second_selection,
                ),
            ]
        )

        # Client 0: 1/4 * [4, 0, 0, 0]
        # Client 1: 3/4 * [0, 0, 8, 0]
        expected = torch.tensor(
            [1.0, 0.0, 6.0, 0.0],
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            aggregate,
            expected,
        )

    def test_error_feedback_sends_omitted_chunk_later(self):
        first_selection, residual, _ = (
            compress_with_error_feedback(
                update=torch.tensor(
                    [10.0, 0.0, 1.0, 0.0]
                ),
                previous_residual=None,
                chunk_size=2,
                selected_chunk_count=1,
                residual_beta=1.0,
            )
        )

        self.assertEqual(
            first_selection.chunk_ids.tolist(),
            [0],
        )

        torch.testing.assert_close(
            residual,
            torch.tensor(
                [0.0, 0.0, 1.0, 0.0]
            ),
        )

        second_selection, new_residual, _ = (
            compress_with_error_feedback(
                update=torch.zeros(4),
                previous_residual=residual,
                chunk_size=2,
                selected_chunk_count=1,
                residual_beta=1.0,
            )
        )

        self.assertEqual(
            second_selection.chunk_ids.tolist(),
            [1],
        )

        torch.testing.assert_close(
            second_selection.to_dense(),
            torch.tensor(
                [0.0, 0.0, 1.0, 0.0]
            ),
        )

        torch.testing.assert_close(
            new_residual,
            torch.zeros(4),
        )

    def test_cohort_rejected_chunk_returns_to_residual(self):
        selection = select_topk_chunks(
            update=torch.tensor(
                [4.0, 0.0, 1.0, 0.0]
            ),
            chunk_size=2,
            selected_chunk_count=2,
        )

        restored = (
            restore_rejected_chunks_to_residual(
                residual=torch.zeros(4),
                selection=selection,
                accepted_chunk_ids={0},
            )
        )

        torch.testing.assert_close(
            restored,
            torch.tensor(
                [0.0, 0.0, 1.0, 0.0]
            ),
        )


if __name__ == "__main__":
    unittest.main()
