import unittest

import torch

from plaintext_fl.aggregation import (
    aggregate_dense_fedavg,
    aggregate_dense_updates,
    quantize_client_weights,
)
from plaintext_fl.federated import (
    dense_error_feedback_step,
)


class FederatedTest(unittest.TestCase):
    def test_exact_sample_weighted_fedavg(self):
        first = torch.tensor(
            [1.0, 1.0]
        )

        second = torch.tensor(
            [5.0, 5.0]
        )

        result = aggregate_dense_fedavg(
            dense_updates=[
                first,
                second,
            ],
            sample_counts=[
                1,
                3,
            ],
        )

        # 1/4 * 1 + 3/4 * 5 = 4
        torch.testing.assert_close(
            result,
            torch.tensor(
                [4.0, 4.0],
                dtype=torch.float64,
            ),
        )

    def test_dense_error_feedback_residual_is_zero(self):
        delta = torch.tensor(
            [1.0, 2.0, 3.0]
        )

        previous_residual = torch.zeros_like(
            delta
        )

        transmitted, new_residual = (
            dense_error_feedback_step(
                local_delta=delta,
                previous_residual=previous_residual,
                enabled=True,
            )
        )

        torch.testing.assert_close(
            transmitted,
            delta,
        )

        torch.testing.assert_close(
            new_residual,
            torch.zeros_like(delta),
        )

    def test_existing_residual_is_transmitted_and_cleared(self):
        delta = torch.tensor(
            [1.0, 2.0]
        )

        previous_residual = torch.tensor(
            [0.5, -0.5]
        )

        transmitted, new_residual = (
            dense_error_feedback_step(
                local_delta=delta,
                previous_residual=previous_residual,
                enabled=True,
            )
        )

        torch.testing.assert_close(
            transmitted,
            torch.tensor(
                [1.5, 1.5]
            ),
        )

        torch.testing.assert_close(
            new_residual,
            torch.zeros_like(delta),
        )
    
    def test_quantized_weighted_fedavg(self):
        updates = [
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        ]

        sample_counts = [
            1,
            2,
            3,
        ]

        alphas = quantize_client_weights(
            example_counts=sample_counts,
            denominator=128,
        )

        torch.testing.assert_close(
            alphas,
            torch.tensor(
                [21, 43, 64],
                dtype=torch.int64,
            ),
        )

        quantized_result = (
            aggregate_dense_updates(
                dense_updates=updates,
                alphas=alphas,
                weight_denominator=128,
            )
        )

        expected = torch.tensor(
            [
                (
                    21 * 0.0
                    + 43 * 1.0
                    + 64 * 2.0
                )
                / 128
            ],
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            quantized_result,
            expected,
        )

        exact_result = (
            aggregate_dense_fedavg(
                dense_updates=updates,
                sample_counts=sample_counts,
            )
        )

        self.assertFalse(
            torch.allclose(
                quantized_result,
                exact_result,
                rtol=0,
                atol=0,
            )
        )


if __name__ == "__main__":
    unittest.main()