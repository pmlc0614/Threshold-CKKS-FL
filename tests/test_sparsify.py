import unittest

import torch

from plaintext_fl.sparsify import (
    SparseSelection,
    add_gaussian_noise,
    chunk_count,
    compress_with_error_feedback,
    compression_residual,
    compute_chunk_scores,
    effective_sparsity_metrics,
    restore_rejected_chunks_to_residual,
    select_topk_chunks,
)


class SparsificationTest(unittest.TestCase):
    def test_effective_sparsity_uses_accepted_client_contributions(self):
        metrics = effective_sparsity_metrics(
            accepted_contribution_count=4,
            client_count=2,
            selected_chunk_count=3,
            total_chunk_count=10,
        )

        self.assertEqual(metrics["candidate_contribution_count"], 6)
        self.assertAlmostEqual(metrics["rho_selected"], 0.3)
        self.assertAlmostEqual(metrics["rho_eff"], 0.2)
        self.assertAlmostEqual(metrics["cohort_retention_ratio"], 2 / 3)

    def test_effective_sparsity_rejects_impossible_contribution_count(self):
        with self.assertRaisesRegex(ValueError, "candidate contribution"):
            effective_sparsity_metrics(
                accepted_contribution_count=7,
                client_count=2,
                selected_chunk_count=3,
                total_chunk_count=10,
            )

    def test_chunk_count(self):
        self.assertEqual(
            chunk_count(
                total_numel=8,
                chunk_size=4,
            ),
            2,
        )

        self.assertEqual(
            chunk_count(
                total_numel=9,
                chunk_size=4,
            ),
            3,
        )

    def test_chunk_scores(self):
        update = torch.tensor(
            [
                3.0, 4.0, 0.0, 0.0,
                0.0, 5.0, 0.0, 0.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        chunks, scores = compute_chunk_scores(
            update=update,
            chunk_size=4,
        )

        self.assertEqual(
            tuple(chunks.shape),
            (3, 4),
        )

        torch.testing.assert_close(
            scores,
            torch.tensor(
                [25.0, 25.0, 1.0],
                dtype=torch.float64,
            ),
        )

    def test_topk_tie_uses_smaller_chunk_id(self):
        update = torch.tensor(
            [
                3.0, 4.0, 0.0, 0.0,
                0.0, 5.0, 0.0, 0.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        selection = select_topk_chunks(
            update=update,
            chunk_size=4,
            selected_chunk_count=2,
        )

        torch.testing.assert_close(
            selection.chunk_ids,
            torch.tensor(
                [0, 1],
                dtype=torch.int64,
            ),
        )

        torch.testing.assert_close(
            selection.values[0],
            torch.tensor(
                [3.0, 4.0, 0.0, 0.0],
            ),
        )

        torch.testing.assert_close(
            selection.values[1],
            torch.tensor(
                [0.0, 5.0, 0.0, 0.0],
            ),
        )

    def test_last_chunk_is_zero_padded(self):
        update = torch.tensor(
            [
                1.0, 1.0, 1.0, 1.0,
                100.0,
            ],
            dtype=torch.float32,
        )

        selection = select_topk_chunks(
            update=update,
            chunk_size=4,
            selected_chunk_count=1,
        )

        self.assertEqual(
            selection.chunk_ids.tolist(),
            [1],
        )

        torch.testing.assert_close(
            selection.values[0],
            torch.tensor(
                [100.0, 0.0, 0.0, 0.0],
            ),
        )

        self.assertEqual(
            selection.selected_valid_numel,
            1,
        )

        self.assertEqual(
            selection.wire_value_numel,
            4,
        )

    def test_selecting_all_chunks_restores_update(self):
        update = torch.arange(
            1,
            10,
            dtype=torch.float32,
        )

        selection = select_topk_chunks(
            update=update,
            chunk_size=4,
            selected_chunk_count=3,
        )

        restored = selection.to_dense()

        torch.testing.assert_close(
            restored,
            update,
            rtol=0,
            atol=0,
        )

    def test_compression_residual(self):
        update = torch.tensor(
            [
                1.0, 2.0,
                10.0, 20.0,
            ],
        )

        selection = select_topk_chunks(
            update=update,
            chunk_size=2,
            selected_chunk_count=1,
        )

        residual = compression_residual(
            original_update=update,
            selection=selection,
        )

        torch.testing.assert_close(
            residual,
            torch.tensor(
                [1.0, 2.0, 0.0, 0.0],
            ),
        )

    def test_error_feedback(self):
        update = torch.tensor(
            [
                1.0, 2.0,
                10.0, 20.0,
            ],
        )

        previous_residual = torch.tensor(
            [
                3.0, 4.0,
                0.0, 0.0,
            ],
        )

        (
            selection,
            new_residual,
            compression_input,
        ) = compress_with_error_feedback(
            update=update,
            previous_residual=previous_residual,
            chunk_size=2,
            selected_chunk_count=1,
            residual_beta=1.0,
        )

        torch.testing.assert_close(
            compression_input,
            torch.tensor(
                [
                    4.0, 6.0,
                    10.0, 20.0,
                ]
            ),
        )

        self.assertEqual(
            selection.chunk_ids.tolist(),
            [1],
        )

        torch.testing.assert_close(
            new_residual,
            torch.tensor(
                [
                    4.0, 6.0,
                    0.0, 0.0,
                ]
            ),
        )

    def test_vectorized_residual_preserves_partial_chunk_semantics(self):
        selection = SparseSelection(
            total_numel=5,
            chunk_size=3,
            total_chunk_count=2,
            chunk_ids=torch.tensor([1], dtype=torch.int64),
            values=torch.tensor([[10.0, 20.0, 99.0]]),
            scores=torch.zeros(2, dtype=torch.float64),
        )

        residual = compression_residual(
            original_update=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            selection=selection,
        )

        torch.testing.assert_close(
            residual,
            torch.tensor([1.0, 2.0, 3.0, -6.0, -15.0]),
            rtol=0,
            atol=0,
        )

    def test_vectorized_rejected_restore_matches_chunk_reference(self):
        selection = SparseSelection(
            total_numel=10,
            chunk_size=4,
            total_chunk_count=3,
            chunk_ids=torch.tensor([2, 0, 1], dtype=torch.int64),
            values=torch.tensor(
                [
                    [9.0, 10.0, 91.0, 92.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
            scores=torch.zeros(3, dtype=torch.float64),
        )
        initial = torch.arange(10, dtype=torch.float32)
        expected = initial.clone()
        for row_position, chunk_id in enumerate(
            selection.chunk_ids.tolist()
        ):
            if int(chunk_id) == 0:
                continue
            begin = int(chunk_id) * selection.chunk_size
            end = min(begin + selection.chunk_size, selection.total_numel)
            expected[begin:end] += selection.values[
                row_position,
                : end - begin,
            ]

        actual = restore_rejected_chunks_to_residual(
            residual=initial,
            selection=selection,
            accepted_chunk_ids={0},
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_error_feedback_reports_nonnegative_phase_timings(self):
        timing = {"stale": -1.0}

        compress_with_error_feedback(
            update=torch.arange(1, 17, dtype=torch.float32),
            previous_residual=torch.zeros(16, dtype=torch.float32),
            chunk_size=4,
            selected_chunk_count=2,
            residual_beta=1.0,
            timing=timing,
        )

        expected_fields = {
            "compression_prepare_seconds",
            "selection_input_prepare_seconds",
            "score_input_prepare_seconds",
            "chunk_score_seconds",
            "topk_seconds",
            "selection_validation_seconds",
            "residual_seconds",
            "total_seconds",
        }
        self.assertEqual(set(timing), expected_fields)
        self.assertTrue(all(value >= 0.0 for value in timing.values()))
        self.assertGreaterEqual(
            timing["total_seconds"],
            timing["compression_prepare_seconds"],
        )
        self.assertGreaterEqual(
            timing["total_seconds"],
            timing["chunk_score_seconds"],
        )
        self.assertGreaterEqual(
            timing["total_seconds"],
            timing["residual_seconds"],
        )

    def test_gaussian_noise_is_reproducible(self):
        reference_update = torch.linspace(
            -1.0,
            1.0,
            10_000,
            dtype=torch.float32,
        )

        selection = select_topk_chunks(
            update=reference_update,
            chunk_size=10_000,
            selected_chunk_count=1,
        )

        first, first_sigma = add_gaussian_noise(
            selection=selection,
            reference_update=reference_update,
            std_ratio=0.5,
            seed=1234,
        )

        second, second_sigma = add_gaussian_noise(
            selection=selection,
            reference_update=reference_update,
            std_ratio=0.5,
            seed=1234,
        )

        self.assertEqual(
            first_sigma,
            second_sigma,
        )

        torch.testing.assert_close(
            first.values,
            second.values,
            rtol=0,
            atol=0,
        )

        noise = (
            first.values.to(torch.float64)
            - selection.values.to(torch.float64)
        )

        self.assertAlmostEqual(
            float(
                noise.std(
                    unbiased=False
                ).item()
            ),
            first_sigma,
            delta=first_sigma * 0.03,
        )

    def test_zero_noise_does_not_change_values(self):
        update = torch.tensor(
            [1.0, 2.0, 3.0, 4.0],
        )

        selection = select_topk_chunks(
            update=update,
            chunk_size=2,
            selected_chunk_count=1,
        )

        noisy, sigma = add_gaussian_noise(
            selection=selection,
            reference_update=update,
            std_ratio=0.0,
            seed=42,
        )

        self.assertEqual(
            sigma,
            0.0,
        )

        torch.testing.assert_close(
            noisy.values,
            selection.values,
            rtol=0,
            atol=0,
        )

    def test_nonfinite_update_is_rejected(self):
        update = torch.tensor(
            [1.0, float("nan")],
        )

        with self.assertRaises(ValueError):
            select_topk_chunks(
                update=update,
                chunk_size=2,
                selected_chunk_count=1,
            )


if __name__ == "__main__":
    unittest.main()
