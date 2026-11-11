import json
import tempfile
import unittest
from pathlib import Path

import torch

from plaintext_fl.model import resnet20
from plaintext_fl.state import (
    build_float_state_layout,
    flatten_float_state,
    layout_numel,
)
from plaintext_fl.stats import (
    append_jsonl,
    difference_stats,
    model_storage_stats,
    statewise_stats,
    tensor_stats,
    write_json,
)


class StatisticsTest(unittest.TestCase):
    def test_tensor_stats(self):
        values = torch.tensor(
            [-1.0, 0.0, 1.0],
            dtype=torch.float32,
        )

        result = tensor_stats(values)

        self.assertEqual(
            result["count"],
            3,
        )

        self.assertEqual(
            result["bytes_float32"],
            12,
        )

        self.assertEqual(
            result["bytes_float64"],
            24,
        )

        self.assertAlmostEqual(
            result["mean"],
            0.0,
        )

        self.assertAlmostEqual(
            result["zero_fraction"],
            1.0 / 3.0,
        )

        self.assertAlmostEqual(
            result["l1"],
            2.0,
        )

        self.assertAlmostEqual(
            result["l2"],
            2.0 ** 0.5,
        )

    def test_difference_stats_identity(self):
        values = torch.tensor(
            [1.0, 2.0, 3.0],
        )

        result = difference_stats(
            reference=values,
            candidate=values,
        )

        self.assertEqual(
            result["rmse"],
            0.0,
        )

        self.assertEqual(
            result["relative_l2_error"],
            0.0,
        )

        self.assertAlmostEqual(
            result["cosine_similarity"],
            1.0,
        )

    def test_difference_stats_orthogonal(self):
        reference = torch.tensor(
            [1.0, 0.0],
        )

        candidate = torch.tensor(
            [0.0, 1.0],
        )

        result = difference_stats(
            reference=reference,
            candidate=candidate,
        )

        self.assertAlmostEqual(
            result["cosine_similarity"],
            0.0,
        )

        self.assertAlmostEqual(
            result["relative_l2_error"],
            2.0 ** 0.5,
        )

    def test_statewise_stats(self):
        model = resnet20()
        layout = build_float_state_layout(
            model
        )

        flat = flatten_float_state(
            model,
            layout,
        )

        result = statewise_stats(
            flat_values=flat,
            layout=layout,
        )

        self.assertEqual(
            set(result),
            {
                entry.name
                for entry in layout
            },
        )

        total_count = sum(
            entry_stats["count"]
            for entry_stats in result.values()
        )

        self.assertEqual(
            total_count,
            layout_numel(layout),
        )

    def test_model_storage_stats(self):
        model = resnet20()
        layout = build_float_state_layout(
            model
        )

        result = model_storage_stats(
            model=model,
            layout=layout,
        )

        self.assertEqual(
            result["trainable_parameter_count"],
            272_474,
        )

        self.assertEqual(
            result["float_state_numel"],
            274_042,
        )

        self.assertEqual(
            result["float_state_bytes_float32"],
            274_042 * 4,
        )

        self.assertEqual(
            result["float_state_bytes_float64"],
            274_042 * 8,
        )

        # ResNet-20 contains 21 BatchNorm
        # num_batches_tracked scalar buffers.
        self.assertEqual(
            result["nonfloat_state_numel"],
            21,
        )

    def test_write_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "stats.json"
            )

            write_json(
                path,
                {
                    "value": 123,
                },
            )

            document = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                document,
                {
                    "value": 123,
                },
            )

    def test_append_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "updates.jsonl"
            )

            append_jsonl(
                path,
                {
                    "round": 0,
                },
            )

            append_jsonl(
                path,
                {
                    "round": 1,
                },
            )

            records = [
                json.loads(line)
                for line
                in path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(
                records,
                [
                    {
                        "round": 0,
                    },
                    {
                        "round": 1,
                    },
                ],
            )

    def test_nonfinite_values_are_rejected(self):
        values = torch.tensor(
            [1.0, float("nan")],
        )

        with self.assertRaises(ValueError):
            tensor_stats(values)


if __name__ == "__main__":
    unittest.main()