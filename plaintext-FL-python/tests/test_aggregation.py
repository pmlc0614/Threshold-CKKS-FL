import unittest

import torch

from plaintext_fl.aggregation import (
    ClientSparseUpdate,
    aggregate_dense_updates,
    aggregate_sparse_fedavg,
    aggregate_sparse_updates,
    quantize_client_weights,
    uniform_client_weights,
)
from plaintext_fl.sparsify import (
    SparseSelection,
    select_topk_chunks,
)

from typing import Sequence


def make_selection(
    total_numel,
    chunk_size,
    chunk_ids,
    values,
):
    total_chunk_count = (
        total_numel + chunk_size - 1
    ) // chunk_size

    selection = SparseSelection(
        total_numel=total_numel,
        chunk_size=chunk_size,
        total_chunk_count=total_chunk_count,
        chunk_ids=torch.tensor(
            chunk_ids,
            dtype=torch.int64,
        ),
        values=torch.tensor(
            values,
            dtype=torch.float32,
        ),
        scores=torch.zeros(
            total_chunk_count,
            dtype=torch.float64,
        ),
    )

    selection.validate()

    return selection


def make_client_update(
    client_id,
    num_examples,
    chunk_ids,
    values,
    round_id=0,
):
    return ClientSparseUpdate(
        client_id=client_id,
        round_id=round_id,
        num_examples=num_examples,
        selection=make_selection(
            total_numel=8,
            chunk_size=4,
            chunk_ids=chunk_ids,
            values=values,
        ),
    )


class AggregationTest(unittest.TestCase):
    def test_vectorized_sparse_aggregation_matches_chunk_reference(self):
        selections = (
            ([4, 0], [[9.0, 91.0, 92.0, 93.0], [1.0, 2.0, 3.0, 4.0]]),
            ([0, 1], [[5.0, 6.0, 7.0, 8.0], [2.0, 3.0, 4.0, 5.0]]),
            ([4, 1], [[7.0, 71.0, 72.0, 73.0], [6.0, 5.0, 4.0, 3.0]]),
            ([0, 2], [[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]),
        )
        updates = [
            ClientSparseUpdate(
                client_id=client_id,
                round_id=7,
                num_examples=client_id + 1,
                selection=make_selection(
                    total_numel=17,
                    chunk_size=4,
                    chunk_ids=chunk_ids,
                    values=values,
                ),
            )
            for client_id, (chunk_ids, values) in enumerate(selections)
        ]
        alphas = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

        result = aggregate_sparse_updates(
            updates=updates,
            minimum_cohort_size=2,
            weight_denominator=10,
            alphas=alphas,
        )

        expected_cohorts = torch.zeros(5, dtype=torch.int64)
        expected_padded = torch.zeros(20, dtype=torch.float64)
        expected_chunks = expected_padded.reshape(5, 4)
        for update in updates:
            for chunk_id in update.selection.chunk_ids.tolist():
                expected_cohorts[int(chunk_id)] += 1
        expected_accepted = tuple(
            chunk_id
            for chunk_id in range(5)
            if int(expected_cohorts[chunk_id]) >= 2
        )
        for client_position, update in enumerate(updates):
            for row_position, chunk_id in enumerate(
                update.selection.chunk_ids.tolist()
            ):
                if int(chunk_id) not in expected_accepted:
                    continue
                expected_chunks[int(chunk_id)] += (
                    int(alphas[client_position])
                    * update.selection.values[row_position].to(torch.float64)
                )
        expected = (expected_padded / 10.0)[:17].contiguous()

        self.assertEqual(result.accepted_chunk_ids, expected_accepted)
        torch.testing.assert_close(
            result.cohort_sizes,
            expected_cohorts,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.dense_update,
            expected,
            rtol=0,
            atol=0,
        )

    def test_vectorized_exact_sparse_fedavg_matches_dense_reference(self):
        updates = [
            ClientSparseUpdate(
                client_id=0,
                round_id=3,
                num_examples=1,
                selection=make_selection(
                    total_numel=10,
                    chunk_size=4,
                    chunk_ids=[2, 0],
                    values=[
                        [9.0, 90.0, 91.0, 92.0],
                        [1.0, 2.0, 3.0, 4.0],
                    ],
                ),
            ),
            ClientSparseUpdate(
                client_id=1,
                round_id=3,
                num_examples=3,
                selection=make_selection(
                    total_numel=10,
                    chunk_size=4,
                    chunk_ids=[1, 2],
                    values=[
                        [4.0, 3.0, 2.0, 1.0],
                        [5.0, 6.0, 70.0, 80.0],
                    ],
                ),
            ),
        ]
        expected = aggregate_dense_updates(
            dense_updates=[
                update.selection.to_dense(dtype=torch.float64)
                for update in updates
            ],
            alphas=torch.tensor([1, 3], dtype=torch.int64),
            weight_denominator=4,
        )

        actual = aggregate_sparse_fedavg(updates)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_largest_remainder_weights(self):
        alphas = quantize_client_weights(
            example_counts=[
                1,
                1,
                1,
            ],
            denominator=128,
        )

        torch.testing.assert_close(
            alphas,
            torch.tensor(
                [43, 43, 42],
                dtype=torch.int64,
            ),
        )

        self.assertEqual(
            int(alphas.sum().item()),
            128,
        )

    def test_weighted_client_weights(self):
        alphas = quantize_client_weights(
            example_counts=[
                1,
                3,
            ],
            denominator=4,
        )

        torch.testing.assert_close(
            alphas,
            torch.tensor(
                [1, 3],
                dtype=torch.int64,
            ),
        )

    def test_uniform_client_weights(self):
        alphas = uniform_client_weights(
            client_count=3,
            denominator=128,
        )

        torch.testing.assert_close(
            alphas,
            torch.tensor(
                [43, 43, 42],
                dtype=torch.int64,
            ),
        )

    def test_minimum_cohort_filter(self):
        updates = [
            make_client_update(
                client_id=0,
                num_examples=1,
                chunk_ids=[0, 1],
                values=[
                    [1.0, 1.0, 1.0, 1.0],
                    [9.0, 9.0, 9.0, 9.0],
                ],
            ),
            make_client_update(
                client_id=1,
                num_examples=3,
                chunk_ids=[0],
                values=[
                    [5.0, 5.0, 5.0, 5.0],
                ],
            ),
        ]

        result = aggregate_sparse_updates(
            updates=updates,
            minimum_cohort_size=2,
            weight_denominator=4,
            alphas=torch.tensor(
                [1, 3],
                dtype=torch.int64,
            ),
        )

        # Chunk 0:
        # (1 * 1 + 3 * 5) / 4 = 4
        torch.testing.assert_close(
            result.dense_update[:4],
            torch.full(
                (4,),
                4.0,
                dtype=torch.float64,
            ),
        )

        # Chunk 1 was selected by one client only.
        torch.testing.assert_close(
            result.dense_update[4:],
            torch.zeros(
                4,
                dtype=torch.float64,
            ),
        )

        self.assertEqual(
            result.accepted_chunk_ids,
            (0,),
        )

        torch.testing.assert_close(
            result.cohort_sizes,
            torch.tensor(
                [2, 1],
                dtype=torch.int64,
            ),
        )

    def test_weights_are_not_renormalized_per_chunk(self):
        updates = [
            make_client_update(
                client_id=0,
                num_examples=1,
                chunk_ids=[1],
                values=[
                    [8.0, 8.0, 8.0, 8.0],
                ],
            ),
            make_client_update(
                client_id=1,
                num_examples=3,
                chunk_ids=[0],
                values=[
                    [4.0, 4.0, 4.0, 4.0],
                ],
            ),
        ]

        result = aggregate_sparse_updates(
            updates=updates,
            minimum_cohort_size=1,
            weight_denominator=4,
            alphas=torch.tensor(
                [1, 3],
                dtype=torch.int64,
            ),
        )

        # Chunk 1 receives only client 0's global weight:
        # 1/4 * 8 = 2
        torch.testing.assert_close(
            result.dense_update[4:],
            torch.full(
                (4,),
                2.0,
                dtype=torch.float64,
            ),
        )

        # It is not renormalized to 8.
        self.assertFalse(
            torch.allclose(
                result.dense_update[4:],
                torch.full(
                    (4,),
                    8.0,
                    dtype=torch.float64,
                ),
            )
        )

    def test_full_sparse_selection_matches_dense(self):
        first_dense = torch.tensor(
            [
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ],
        )

        second_dense = torch.tensor(
            [
                8.0, 7.0, 6.0, 5.0,
                4.0, 3.0, 2.0, 1.0,
            ],
        )

        first_selection = select_topk_chunks(
            update=first_dense,
            chunk_size=4,
            selected_chunk_count=2,
        )

        second_selection = select_topk_chunks(
            update=second_dense,
            chunk_size=4,
            selected_chunk_count=2,
        )

        sparse_result = aggregate_sparse_updates(
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
            ],
            minimum_cohort_size=2,
            weight_denominator=4,
            alphas=torch.tensor(
                [1, 3],
                dtype=torch.int64,
            ),
        )

        dense_result = aggregate_dense_updates(
            dense_updates=[
                first_dense,
                second_dense,
            ],
            alphas=torch.tensor(
                [1, 3],
                dtype=torch.int64,
            ),
            weight_denominator=4,
        )

        torch.testing.assert_close(
            sparse_result.dense_update,
            dense_result,
        )

    def test_rejected_chunks_remain_zero(self):
        updates = [
            make_client_update(
                client_id=0,
                num_examples=1,
                chunk_ids=[0],
                values=[
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ),
            make_client_update(
                client_id=1,
                num_examples=1,
                chunk_ids=[1],
                values=[
                    [2.0, 2.0, 2.0, 2.0],
                ],
            ),
        ]

        result = aggregate_sparse_updates(
            updates=updates,
            minimum_cohort_size=2,
            weight_denominator=2,
            alphas=torch.tensor(
                [1, 1],
                dtype=torch.int64,
            ),
        )

        self.assertEqual(
            result.accepted_chunk_ids,
            (),
        )

        torch.testing.assert_close(
            result.dense_update,
            torch.zeros(
                8,
                dtype=torch.float64,
            ),
        )

    def test_duplicate_client_id_is_rejected(self):
        updates = [
            make_client_update(
                client_id=0,
                num_examples=1,
                chunk_ids=[0],
                values=[
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ),
            make_client_update(
                client_id=0,
                num_examples=1,
                chunk_ids=[0],
                values=[
                    [2.0, 2.0, 2.0, 2.0],
                ],
            ),
        ]

        with self.assertRaises(ValueError):
            aggregate_sparse_updates(
                updates=updates,
                minimum_cohort_size=1,
                weight_denominator=2,
            )

    def test_round_mismatch_is_rejected(self):
        updates = [
            make_client_update(
                client_id=0,
                num_examples=1,
                chunk_ids=[0],
                values=[
                    [1.0, 1.0, 1.0, 1.0],
                ],
                round_id=0,
            ),
            make_client_update(
                client_id=1,
                num_examples=1,
                chunk_ids=[0],
                values=[
                    [2.0, 2.0, 2.0, 2.0],
                ],
                round_id=1,
            ),
        ]

        with self.assertRaises(ValueError):
            aggregate_sparse_updates(
                updates=updates,
                minimum_cohort_size=1,
                weight_denominator=2,
            )


if __name__ == "__main__":
    unittest.main()

def aggregate_dense_fedavg(
    dense_updates: Sequence[torch.Tensor],
    sample_counts: Sequence[int],
) -> torch.Tensor:
    """Exact sample-count-weighted dense FedAvg.

    Computes:

        sum_i (D_i / sum_j D_j) * update_i

    No alpha/128 quantization is applied.
    """

    if len(dense_updates) == 0:
        raise ValueError(
            "at least one dense update is required"
        )

    if len(dense_updates) != len(sample_counts):
        raise ValueError(
            "dense_updates and sample_counts must have equal lengths"
        )

    counts = torch.tensor(
        list(sample_counts),
        dtype=torch.float64,
        device="cpu",
    )

    if torch.any(counts <= 0):
        raise ValueError(
            "every sample count must be positive"
        )

    flattened_updates = [
        update.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
        for update in dense_updates
    ]

    expected_numel = flattened_updates[0].numel()

    for update in flattened_updates:
        if update.numel() != expected_numel:
            raise ValueError(
                "all dense updates must have equal dimensions"
            )

        if not torch.isfinite(update).all():
            raise ValueError(
                "dense update contains NaN or infinity"
            )

    weights = counts / counts.sum()

    aggregate = torch.zeros(
        expected_numel,
        dtype=torch.float64,
        device="cpu",
    )

    for weight, update in zip(
        weights,
        flattened_updates,
    ):
        aggregate += weight * update

    return aggregate.contiguous()
