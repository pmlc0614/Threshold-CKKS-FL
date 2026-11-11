import unittest

from plaintext_fl.centralized import (
    build_centralized_model,
    validate_experiment,
)
from plaintext_fl.model import ResNet20, ResNet32


class CentralizedExperimentTest(unittest.TestCase):
    def test_supported_dataset_model_pairs(self):
        validate_experiment("cifar10", "resnet20")
        validate_experiment("cifar100", "resnet32")

    def test_rejects_mismatched_dataset_model_pair(self):
        with self.assertRaises(ValueError):
            validate_experiment("cifar100", "resnet20")

    def test_builds_requested_model(self):
        self.assertIsInstance(
            build_centralized_model("resnet20"),
            ResNet20,
        )
        self.assertIsInstance(
            build_centralized_model("resnet32"),
            ResNet32,
        )


if __name__ == "__main__":
    unittest.main()
