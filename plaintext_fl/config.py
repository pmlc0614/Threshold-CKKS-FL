"""Configuration shared by plaintext training and the OpenFHE boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


OptimizerName = Literal["sgd", "adam"]
DataSplit = Literal["iid", "noniid"]
WireDtype = Literal["float32", "float64"]


@dataclass(frozen=True)
class ExperimentConfig:
    # Dataset
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./runs/default")

    # Federated learning
    rounds: int = 50
    num_clients: int = 8
    local_epochs: int = 1

    # DataLoader
    batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 2

    # Local optimizer
    #
    # These defaults match the reference FL_RLWE2MLWE experiment.
    optimizer: OptimizerName = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Data partition
    split: DataSplit = "iid"
    dirichlet_alpha: float = 0.5

    # OpenFHE-compatible chunk configuration
    ring_dimension: int = 8192
    mlwe_rank: int = 4
    selected_chunk_count: int = 4
    minimum_cohort_size: int = 3
    weight_denominator: int = 128

    # Compression
    use_error_feedback: bool = False
    residual_beta: float = 1.0

    # Gaussian noise
    noise_std_ratio: float = 0.0

    # Python/C++ binary representation
    wire_dtype: WireDtype = "float64"

    # Runtime
    seed: int = 42
    device: str = "cuda"

    @property
    def chunk_size(self) -> int:
        """Return ring_dimension / mlwe_rank.

        The current OpenFHE setting gives 8192 / 4 = 2048.
        """

        if self.mlwe_rank <= 0:
            raise ValueError("mlwe_rank must be positive")

        if self.ring_dimension % self.mlwe_rank != 0:
            raise ValueError(
                "ring_dimension must be divisible by mlwe_rank"
            )

        return self.ring_dimension // self.mlwe_rank

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")

        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")

        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.momentum < 0:
            raise ValueError("momentum must be non-negative")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        if self.optimizer not in {"sgd", "adam"}:
            raise ValueError(
                f"unsupported optimizer: {self.optimizer}"
            )

        if self.split not in {"iid", "noniid"}:
            raise ValueError(
                f"unsupported split: {self.split}"
            )

        if self.split == "noniid" and self.dirichlet_alpha <= 0:
            raise ValueError(
                "dirichlet_alpha must be positive for non-IID partitioning"
            )

        if self.ring_dimension <= 0:
            raise ValueError("ring_dimension must be positive")

        if self.mlwe_rank <= 0:
            raise ValueError("mlwe_rank must be positive")

        if self.ring_dimension % self.mlwe_rank != 0:
            raise ValueError(
                "ring_dimension must be divisible by mlwe_rank"
            )

        if self.selected_chunk_count <= 0:
            raise ValueError(
                "selected_chunk_count must be positive"
            )

        if not 1 <= self.minimum_cohort_size <= self.num_clients:
            raise ValueError(
                "minimum_cohort_size must be between 1 and num_clients"
            )

        if self.weight_denominator <= 0:
            raise ValueError(
                "weight_denominator must be positive"
            )

        if self.residual_beta < 0:
            raise ValueError(
                "residual_beta must be non-negative"
            )

        if self.noise_std_ratio < 0:
            raise ValueError(
                "noise_std_ratio must be non-negative"
            )

        if self.wire_dtype not in {"float32", "float64"}:
            raise ValueError(
                f"unsupported wire_dtype: {self.wire_dtype}"
            )

        # Evaluate the property so its validation also runs.
        _ = self.chunk_size

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        # pathlib.Path is not directly JSON serializable.
        result["data_dir"] = str(self.data_dir)
        result["output_dir"] = str(self.output_dir)

        # chunk_size is derived rather than stored.
        result["chunk_size"] = self.chunk_size

        return result