import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from plaintext_fl.federated_full import (
    add_full_protocol_noise,
    build_full_noise_masks,
    main,
    parse_args,
    validate_full_experiment,
)
from plaintext_fl.model import resnet32
from plaintext_fl.plaintext_communication import (
    measure_plaintext_full_round_communication,
)
from plaintext_fl.state import build_float_state_layout


class FullFederatedTest(unittest.TestCase):
    @staticmethod
    def _tiny_model():
        return nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2, 10),
        )

    def test_resnet32_full_layout_has_expected_protocol_dimensions(self):
        layout = build_float_state_layout(resnet32())
        parameter, running_mean, running_var = build_full_noise_masks(layout)
        self.assertEqual(int(parameter.sum().item()), 472_756)
        self.assertEqual(int(running_mean.sum().item()), 1_232)
        self.assertEqual(int(running_var.sum().item()), 1_232)
        self.assertEqual(parameter.numel(), 475_220)
        self.assertTrue(torch.all(parameter | running_mean | running_var))

    def test_full_noise_uses_disjoint_parameter_and_auxiliary_streams(self):
        values = torch.zeros(6, dtype=torch.float64)
        parameter = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
        running_mean = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)
        running_var = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
        noisy, noises = add_full_protocol_noise(
            values,
            parameter_mask=parameter,
            running_mean_mask=running_mean,
            running_var_mask=running_var,
            parameter_sigma=1.0,
            auxiliary_sigma=2.0,
            seed=123,
        )
        self.assertTrue(torch.equal(noisy, sum(noises.values())))
        self.assertTrue(torch.equal(noises["parameter"][~parameter], torch.zeros(4)))
        self.assertTrue(torch.equal(noises["running_mean"][~running_mean], torch.zeros(4)))
        self.assertTrue(torch.equal(noises["running_var"][~running_var], torch.zeros(4)))

    def test_full_communication_is_dense_unpadded_fp32(self):
        measured = measure_plaintext_full_round_communication(
            client_count=8,
            parameter_numel=472_756,
            auxiliary_numel=2_464,
        )
        expected_per_client = (472_756 + 2_464) * 4
        self.assertEqual(measured["accepted_only_fp32_uplink_bytes"], 8 * expected_per_client)
        self.assertEqual(measured["accepted_only_logical_downlink_bytes"], expected_per_client)
        self.assertEqual(measured["accepted_only_unicast_downlink_bytes"], 8 * expected_per_client)
        self.assertEqual(measured["transmitted_chunk_index_bits"], 0)
        self.assertFalse(measured["padding_included"])

    def test_full_cli_has_no_sparse_controls(self):
        args = parse_args([])
        self.assertEqual(args.dataset, "cifar100")
        self.assertFalse(hasattr(args, "rho"))
        self.assertFalse(hasattr(args, "chunk_size"))

    def test_dataset_model_pair_is_restricted(self):
        validate_full_experiment("cifar100", "resnet32")
        with self.assertRaisesRegex(ValueError, "supported pairs"):
            validate_full_experiment("cifar100", "resnet20")

    def test_one_round_smoke_writes_full_summary(self):
        train = TensorDataset(
            torch.randn(4, 3, 4, 4),
            torch.tensor([0, 1, 0, 1]),
        )
        test = TensorDataset(
            torch.randn(2, 3, 4, 4),
            torch.tensor([0, 1]),
        )
        train.targets = [0, 1, 0, 1]
        test.targets = [0, 1]
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "federated-full",
                "--dataset", "cifar10",
                "--model", "resnet20",
                "--rounds", "1",
                "--clients", "2",
                "--batch-size", "2",
                "--eval-batch-size", "2",
                "--device", "cpu",
                "--noise-p", "2",
                "--u-max", "1",
                "--aux-noise-p", "3",
                "--u-max-aux", "1",
                "--output-dir", directory,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "plaintext_fl.federated_full.load_full_dataset",
                    return_value=(train, test),
                ),
                patch(
                    "plaintext_fl.federated_full.build_full_model",
                    side_effect=lambda _model_name: self._tiny_model(),
                ),
            ):
                main()
            summary = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["communication_profile"], "FULL")
            self.assertFalse(summary["plaintext_payload_padding_included"])
            self.assertEqual(summary["parameter_noise_sigma"], 0.25)
            self.assertEqual(summary["auxiliary_noise_sigma"], 0.125)
            self.assertGreater(summary["total_accepted_only_fp32_uplink_bytes"], 0)
            self.assertIn("total_online_runtime_parallel_seconds", summary)
            self.assertEqual(
                summary["per_round_mean_client_fp32_uplink_bytes"],
                summary["per_round_accepted_only_fp32_uplink_bytes"] / 2,
            )


if __name__ == "__main__":
    unittest.main()
