import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from plaintext_fl.data import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    build_cifar100_transforms,
    load_or_create_partitions,
    load_cifar100,
    partition_indices,
)


class Cifar100DataTest(unittest.TestCase):
    def test_transforms_use_cifar100_normalization(self):
        train_transform, test_transform = (
            build_cifar100_transforms()
        )

        train_normalize = train_transform.transforms[-1]
        test_normalize = test_transform.transforms[-1]

        self.assertEqual(tuple(train_normalize.mean), CIFAR100_MEAN)
        self.assertEqual(tuple(train_normalize.std), CIFAR100_STD)
        self.assertEqual(tuple(test_normalize.mean), CIFAR100_MEAN)
        self.assertEqual(tuple(test_normalize.std), CIFAR100_STD)

    def test_loader_constructs_train_and_test_splits(self):
        with TemporaryDirectory() as temporary_directory:
            with patch(
                "torchvision.datasets.CIFAR100",
                side_effect=["train-set", "test-set"],
            ) as cifar100:
                train_set, test_set = load_cifar100(
                    data_dir=temporary_directory,
                    download=False,
                )

        self.assertEqual(train_set, "train-set")
        self.assertEqual(test_set, "test-set")
        self.assertEqual(cifar100.call_count, 2)
        self.assertEqual(
            Path(cifar100.call_args_list[0].kwargs["root"]),
            Path(temporary_directory),
        )
        self.assertTrue(cifar100.call_args_list[0].kwargs["train"])
        self.assertFalse(cifar100.call_args_list[1].kwargs["train"])


class DatasetPartitionTest(unittest.TestCase):
    def setUp(self):
        # 10 classes, 100 samples per class.
        self.targets = np.repeat(
            np.arange(10),
            100,
        )

    def assert_valid_partition(
        self,
        partitions,
        expected_sample_count,
    ):
        flattened = np.concatenate(partitions)

        self.assertEqual(
            flattened.size,
            expected_sample_count,
        )

        self.assertEqual(
            np.unique(flattened).size,
            expected_sample_count,
        )

        np.testing.assert_array_equal(
            np.sort(flattened),
            np.arange(expected_sample_count),
        )

    def test_iid_partition_covers_all_samples(self):
        partitions = partition_indices(
            targets=self.targets,
            num_clients=8,
            partition="iid",
            seed=42,
        )

        self.assertEqual(
            len(partitions),
            8,
        )

        self.assert_valid_partition(
            partitions,
            expected_sample_count=self.targets.size,
        )

        sizes = [
            len(partition)
            for partition in partitions
        ]

        self.assertLessEqual(
            max(sizes) - min(sizes),
            1,
        )

    def test_iid_partition_is_reproducible(self):
        first = partition_indices(
            targets=self.targets,
            num_clients=8,
            partition="iid",
            seed=42,
        )

        second = partition_indices(
            targets=self.targets,
            num_clients=8,
            partition="iid",
            seed=42,
        )

        for left, right in zip(
            first,
            second,
        ):
            np.testing.assert_array_equal(
                left,
                right,
            )

    def test_noniid_partition_covers_all_samples(self):
        partitions = partition_indices(
            targets=self.targets,
            num_clients=8,
            partition="noniid",
            seed=42,
            dirichlet_alpha=0.5,
            minimum_client_samples=1,
        )

        self.assertEqual(
            len(partitions),
            8,
        )

        self.assert_valid_partition(
            partitions,
            expected_sample_count=self.targets.size,
        )

        self.assertTrue(
            all(
                len(partition) >= 1
                for partition in partitions
            )
        )

    def test_noniid_partition_is_reproducible(self):
        first = partition_indices(
            targets=self.targets,
            num_clients=8,
            partition="noniid",
            seed=42,
            dirichlet_alpha=0.5,
        )

        second = partition_indices(
            targets=self.targets,
            num_clients=8,
            partition="noniid",
            seed=42,
            dirichlet_alpha=0.5,
        )

        for left, right in zip(
            first,
            second,
        ):
            np.testing.assert_array_equal(
                left,
                right,
            )

    def test_partition_file_is_created_and_reused_exactly(self):
        with TemporaryDirectory() as temporary_directory:
            partition_file = (
                Path(temporary_directory)
                / "cifar10-n8-alpha1.json"
            )
            created, created_info = load_or_create_partitions(
                targets=self.targets,
                dataset_name="cifar10",
                num_clients=8,
                partition="noniid",
                dirichlet_alpha=1.0,
                partition_seed=42,
                partition_file=partition_file,
            )
            loaded, loaded_info = load_or_create_partitions(
                targets=self.targets,
                dataset_name="cifar10",
                num_clients=8,
                partition="noniid",
                dirichlet_alpha=1.0,
                partition_seed=999,
                partition_file=partition_file,
                validate_partition_seed=False,
            )

            self.assertTrue(partition_file.is_file())
            self.assertEqual(created_info["source"], "created")
            self.assertEqual(loaded_info["source"], "loaded")
            self.assertEqual(loaded_info["partition_seed"], 42)
            self.assertEqual(
                created_info["partition_sha256"],
                loaded_info["partition_sha256"],
            )
            for left, right in zip(created, loaded, strict=True):
                np.testing.assert_array_equal(left, right)

    def test_explicit_partition_seed_must_match_existing_file(self):
        with TemporaryDirectory() as temporary_directory:
            partition_file = Path(temporary_directory) / "fixed.json"
            load_or_create_partitions(
                targets=self.targets,
                dataset_name="cifar10",
                num_clients=8,
                partition="noniid",
                dirichlet_alpha=1.0,
                partition_seed=42,
                partition_file=partition_file,
            )

            with self.assertRaisesRegex(ValueError, "partition_seed"):
                load_or_create_partitions(
                    targets=self.targets,
                    dataset_name="cifar10",
                    num_clients=8,
                    partition="noniid",
                    dirichlet_alpha=1.0,
                    partition_seed=7,
                    partition_file=partition_file,
                    validate_partition_seed=True,
                )

    def test_partition_file_rejects_different_dataset_targets(self):
        with TemporaryDirectory() as temporary_directory:
            partition_file = Path(temporary_directory) / "fixed.json"
            load_or_create_partitions(
                targets=self.targets,
                dataset_name="cifar10",
                num_clients=8,
                partition="noniid",
                dirichlet_alpha=1.0,
                partition_seed=42,
                partition_file=partition_file,
            )
            changed_targets = self.targets.copy()
            changed_targets[[0, -1]] = changed_targets[[-1, 0]]

            with self.assertRaisesRegex(ValueError, "targets_sha256"):
                load_or_create_partitions(
                    targets=changed_targets,
                    dataset_name="cifar10",
                    num_clients=8,
                    partition="noniid",
                    dirichlet_alpha=1.0,
                    partition_seed=42,
                    partition_file=partition_file,
                )

    def test_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            partition_indices(
                targets=self.targets,
                num_clients=8,
                partition="noniid",
                seed=42,
                dirichlet_alpha=0.0,
            )


if __name__ == "__main__":
    unittest.main()
