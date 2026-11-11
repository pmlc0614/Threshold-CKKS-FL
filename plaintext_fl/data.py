"""CIFAR loading and client data partitioning."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2023,
    0.1994,
    0.2010,
)

CIFAR100_MEAN = (
    0.5071,
    0.4867,
    0.4408,
)

CIFAR100_STD = (
    0.2675,
    0.2565,
    0.2761,
)

PartitionName = Literal["iid", "noniid"]
PARTITION_FILE_FORMAT = "plaintext-fl-partition-v1"


def _build_cifar_transforms(
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    return train_transform, test_transform


def build_cifar10_transforms():
    return _build_cifar_transforms(
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )


def build_cifar100_transforms():
    return _build_cifar_transforms(
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
    )


def load_cifar10(
    data_dir: str | Path,
    download: bool = True,
):
    from torchvision import datasets

    data_path = Path(data_dir)
    data_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_transform, test_transform = (
        build_cifar10_transforms()
    )

    train_set = datasets.CIFAR10(
        root=str(data_path),
        train=True,
        download=download,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=str(data_path),
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_set, test_set


def load_cifar100(
    data_dir: str | Path,
    download: bool = True,
):
    from torchvision import datasets

    data_path = Path(data_dir)
    data_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_transform, test_transform = (
        build_cifar100_transforms()
    )

    train_set = datasets.CIFAR100(
        root=str(data_path),
        train=True,
        download=download,
        transform=train_transform,
    )

    test_set = datasets.CIFAR100(
        root=str(data_path),
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_set, test_set


def partition_indices(
    targets: Sequence[int],
    num_clients: int,
    partition: PartitionName = "iid",
    seed: int = 42,
    dirichlet_alpha: float = 0.5,
    minimum_client_samples: int = 1,
    maximum_retries: int = 1000,
) -> list[np.ndarray]:
    """Partition dataset indices among clients.

    IID partitioning creates nearly equal-sized shuffled partitions.

    Non-IID partitioning samples a client distribution independently
    for every class from a Dirichlet distribution. The partition is
    retried when any client receives fewer than
    minimum_client_samples.
    """

    target_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    if target_array.ndim != 1:
        raise ValueError(
            "targets must be one-dimensional"
        )

    if target_array.size == 0:
        raise ValueError(
            "targets must not be empty"
        )

    if num_clients <= 0:
        raise ValueError(
            "num_clients must be positive"
        )

    if target_array.size < num_clients:
        raise ValueError(
            "dataset must contain at least one sample per client"
        )

    if minimum_client_samples < 0:
        raise ValueError(
            "minimum_client_samples must be non-negative"
        )

    if (
        minimum_client_samples * num_clients
        > target_array.size
    ):
        raise ValueError(
            "minimum_client_samples is impossible for this dataset"
        )

    if partition not in {"iid", "noniid"}:
        raise ValueError(
            f"unsupported partition: {partition}"
        )

    random_generator = np.random.default_rng(seed)

    if partition == "iid":
        shuffled_indices = random_generator.permutation(
            target_array.size
        )

        return [
            client_indices.astype(
                np.int64,
                copy=False,
            )
            for client_indices in np.array_split(
                shuffled_indices,
                num_clients,
            )
        ]

    if dirichlet_alpha <= 0:
        raise ValueError(
            "dirichlet_alpha must be positive"
        )

    class_labels = np.unique(target_array)

    for _attempt in range(maximum_retries):
        client_indices: list[list[int]] = [
            []
            for _ in range(num_clients)
        ]

        for class_label in class_labels:
            class_indices = np.flatnonzero(
                target_array == class_label
            )
            random_generator.shuffle(class_indices)

            proportions = random_generator.dirichlet(
                np.full(
                    num_clients,
                    dirichlet_alpha,
                    dtype=np.float64,
                )
            )

            client_counts = random_generator.multinomial(
                n=class_indices.size,
                pvals=proportions,
            )

            split_positions = np.cumsum(
                client_counts,
                dtype=np.int64,
            )[:-1]

            class_partitions = np.split(
                class_indices,
                split_positions,
            )

            for client_id, indices in enumerate(
                class_partitions
            ):
                client_indices[client_id].extend(
                    indices.tolist()
                )

        client_sizes = [
            len(indices)
            for indices in client_indices
        ]

        if min(client_sizes) < minimum_client_samples:
            continue

        result: list[np.ndarray] = []

        for indices in client_indices:
            random_generator.shuffle(indices)

            result.append(
                np.asarray(
                    indices,
                    dtype=np.int64,
                )
            )

        return result

    raise RuntimeError(
        "failed to construct a non-IID partition with the requested "
        "minimum client size; increase dirichlet_alpha or lower "
        "minimum_client_samples"
    )


def _targets_sha256(targets: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        targets,
        dtype=np.dtype("<i8"),
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _partition_sha256(partitions: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for client_id, indices in enumerate(partitions):
        canonical = np.ascontiguousarray(
            indices,
            dtype=np.dtype("<i8"),
        )
        digest.update(client_id.to_bytes(8, "little", signed=False))
        digest.update(canonical.size.to_bytes(8, "little", signed=False))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _validate_complete_partitions(
    partitions: Sequence[np.ndarray],
    *,
    dataset_size: int,
    num_clients: int,
    minimum_client_samples: int,
) -> list[np.ndarray]:
    if len(partitions) != num_clients:
        raise ValueError(
            "partition client count does not match num_clients"
        )

    normalized: list[np.ndarray] = []
    for client_id, indices in enumerate(partitions):
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError(
                f"partition client {client_id} indices must be one-dimensional"
            )
        if values.size < minimum_client_samples:
            raise ValueError(
                f"partition client {client_id} has fewer than "
                "minimum_client_samples"
            )
        normalized.append(values.copy())

    combined = np.concatenate(normalized)
    if combined.size != dataset_size:
        raise ValueError(
            "partition must contain exactly one entry per dataset sample"
        )
    if not np.array_equal(
        np.sort(combined),
        np.arange(dataset_size, dtype=np.int64),
    ):
        raise ValueError(
            "partition indices must cover every dataset sample exactly once"
        )
    return normalized


def _partition_file_document(
    *,
    dataset_name: str,
    targets: np.ndarray,
    partitions: Sequence[np.ndarray],
    partition: PartitionName,
    dirichlet_alpha: float,
    partition_seed: int,
    minimum_client_samples: int,
) -> dict:
    stored_alpha = (
        float(dirichlet_alpha)
        if partition == "noniid"
        else None
    )
    return {
        "format": PARTITION_FILE_FORMAT,
        "dataset": dataset_name,
        "dataset_size": int(targets.size),
        "targets_sha256": _targets_sha256(targets),
        "num_clients": len(partitions),
        "partition": partition,
        "dirichlet_alpha": stored_alpha,
        "partition_seed": int(partition_seed),
        "minimum_client_samples": int(minimum_client_samples),
        "partition_sha256": _partition_sha256(partitions),
        "clients": [
            {
                "client_id": client_id,
                "sample_count": int(indices.size),
                "indices": indices.tolist(),
            }
            for client_id, indices in enumerate(partitions)
        ],
    }


def _write_partition_file(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_or_create_partitions(
    *,
    targets: Sequence[int],
    dataset_name: str,
    num_clients: int,
    partition: PartitionName,
    dirichlet_alpha: float,
    partition_seed: int,
    partition_file: str | Path | None = None,
    validate_partition_seed: bool = False,
    minimum_client_samples: int = 1,
    maximum_retries: int = 1000,
) -> tuple[list[np.ndarray], dict]:
    """Create or reuse an exact client assignment.

    When ``partition_file`` exists, its sample indices are loaded and checked
    against the current dataset and partition contract.  Otherwise a new
    assignment is generated and atomically saved when a path was supplied.
    """

    target_array = np.asarray(targets, dtype=np.int64)
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional sequence")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if minimum_client_samples < 0:
        raise ValueError("minimum_client_samples must be non-negative")
    if partition not in {"iid", "noniid"}:
        raise ValueError(f"unsupported partition: {partition}")
    if partition == "noniid" and (
        not math.isfinite(float(dirichlet_alpha))
        or dirichlet_alpha <= 0
    ):
        raise ValueError("dirichlet_alpha must be finite and positive")
    if not dataset_name:
        raise ValueError("dataset_name must not be empty")
    file_path = None if partition_file is None else Path(partition_file)

    if file_path is not None and file_path.exists():
        try:
            document = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"failed to read partition file: {file_path}"
            ) from error
        if not isinstance(document, dict):
            raise ValueError("partition file root must be an object")

        expected = {
            "format": PARTITION_FILE_FORMAT,
            "dataset": dataset_name,
            "dataset_size": int(target_array.size),
            "targets_sha256": _targets_sha256(target_array),
            "num_clients": num_clients,
            "partition": partition,
            "minimum_client_samples": minimum_client_samples,
        }
        for name, value in expected.items():
            if document.get(name) != value:
                raise ValueError(
                    f"partition file {name} does not match the experiment"
                )

        expected_alpha = (
            float(dirichlet_alpha)
            if partition == "noniid"
            else None
        )
        stored_alpha = document.get("dirichlet_alpha")
        alpha_matches = (
            stored_alpha is None
            if expected_alpha is None
            else isinstance(stored_alpha, (int, float))
            and math.isclose(
                float(stored_alpha),
                expected_alpha,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        if not alpha_matches:
            raise ValueError(
                "partition file dirichlet_alpha does not match the experiment"
            )
        if validate_partition_seed and document.get(
            "partition_seed"
        ) != partition_seed:
            raise ValueError(
                "partition file partition_seed does not match --partition-seed"
            )

        clients = document.get("clients")
        if not isinstance(clients, list) or len(clients) != num_clients:
            raise ValueError("partition file clients are invalid")
        raw_partitions: list[np.ndarray] = []
        for client_id, client in enumerate(clients):
            if not isinstance(client, dict) or client.get(
                "client_id"
            ) != client_id:
                raise ValueError("partition file client IDs are invalid")
            raw_indices = client.get("indices")
            if not isinstance(raw_indices, list) or any(
                type(value) is not int for value in raw_indices
            ):
                raise ValueError("partition file indices must be integer lists")
            if client.get("sample_count") != len(raw_indices):
                raise ValueError("partition file sample_count is inconsistent")
            raw_partitions.append(
                np.asarray(raw_indices, dtype=np.int64)
            )

        partitions = _validate_complete_partitions(
            raw_partitions,
            dataset_size=target_array.size,
            num_clients=num_clients,
            minimum_client_samples=minimum_client_samples,
        )
        partition_sha256 = _partition_sha256(partitions)
        if document.get("partition_sha256") != partition_sha256:
            raise ValueError("partition file checksum does not match its indices")
        source = "loaded"
        stored_seed = document.get("partition_seed")
        if type(stored_seed) is not int:
            raise ValueError("partition file partition_seed must be an integer")
    else:
        partitions = partition_indices(
            targets=target_array,
            num_clients=num_clients,
            partition=partition,
            seed=partition_seed,
            dirichlet_alpha=dirichlet_alpha,
            minimum_client_samples=minimum_client_samples,
            maximum_retries=maximum_retries,
        )
        partitions = _validate_complete_partitions(
            partitions,
            dataset_size=target_array.size,
            num_clients=num_clients,
            minimum_client_samples=minimum_client_samples,
        )
        document = _partition_file_document(
            dataset_name=dataset_name,
            targets=target_array,
            partitions=partitions,
            partition=partition,
            dirichlet_alpha=dirichlet_alpha,
            partition_seed=partition_seed,
            minimum_client_samples=minimum_client_samples,
        )
        if file_path is not None:
            _write_partition_file(file_path, document)
            source = "created"
        else:
            source = "generated"
        partition_sha256 = document["partition_sha256"]
        stored_seed = partition_seed

    return partitions, {
        "format": PARTITION_FILE_FORMAT,
        "source": source,
        "partition_file": (
            None
            if file_path is None
            else str(file_path.resolve())
        ),
        "partition_seed": stored_seed,
        "partition_sha256": partition_sha256,
        "targets_sha256": document["targets_sha256"],
    }


def build_client_datasets(
    train_set: Dataset,
    client_indices: list[np.ndarray],
) -> list[Subset]:
    return [
        Subset(
            train_set,
            indices.tolist(),
        )
        for indices in client_indices
    ]


def build_client_loaders(
    client_datasets: list[Dataset],
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> list[DataLoader]:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers must be non-negative"
        )

    loaders: list[DataLoader] = []

    for client_id, client_dataset in enumerate(
        client_datasets
    ):
        if len(client_dataset) == 0:
            raise ValueError(
                f"client {client_id} has no training data"
            )

        generator = torch.Generator()
        generator.manual_seed(
            seed + client_id
        )

        loaders.append(
            DataLoader(
                dataset=client_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=num_workers > 0,
                generator=generator,
            )
        )

    return loaders


def build_test_loader(
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers must be non-negative"
        )

    return DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def client_sample_counts(
    client_datasets: list[Dataset],
) -> list[int]:
    return [
        len(client_dataset)
        for client_dataset in client_datasets
    ]
