"""Common training and evaluation utilities."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


OptimizerName = Literal["sgd", "adam"]
ScheduleName = Literal["constant", "cosine"]


@dataclass(frozen=True)
class TrainMetrics:
    average_loss: float
    sample_count: int
    batch_count: int
    duration_seconds: float


@dataclass(frozen=True)
class EvaluationMetrics:
    average_loss: float
    accuracy_percent: float
    correct_count: int
    sample_count: int
    duration_seconds: float


def seed_everything(
    seed: int,
    deterministic: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def resolve_device(
    requested_device: str,
) -> torch.device:
    device = torch.device(requested_device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot use CUDA. "
            "Check torch.version.cuda, the NVIDIA driver, "
            "and torch.cuda.is_available()."
        )

    return device


def build_optimizer(
    model: nn.Module,
    optimizer_name: OptimizerName,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if momentum < 0:
        raise ValueError(
            "momentum must be non-negative"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay must be non-negative"
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"unsupported optimizer: {optimizer_name}"
    )


def learning_rate_at_step(
    initial_learning_rate: float,
    minimum_learning_rate: float,
    step: int,
    total_steps: int,
    schedule: ScheduleName,
) -> float:
    if initial_learning_rate <= 0:
        raise ValueError(
            "initial_learning_rate must be positive"
        )

    if minimum_learning_rate < 0:
        raise ValueError(
            "minimum_learning_rate must be non-negative"
        )

    if minimum_learning_rate > initial_learning_rate:
        raise ValueError(
            "minimum_learning_rate cannot exceed "
            "initial_learning_rate"
        )

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    if not 0 <= step < total_steps:
        raise ValueError(
            "step must be in [0, total_steps)"
        )

    if schedule == "constant":
        return initial_learning_rate

    if schedule == "cosine":
        cosine = 0.5 * (
            1.0
            + math.cos(
                math.pi * step / total_steps
            )
        )

        return (
            minimum_learning_rate
            + (
                initial_learning_rate
                - minimum_learning_rate
            )
            * cosine
        )

    raise ValueError(
        f"unsupported schedule: {schedule}"
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainMetrics:
    model.train()

    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    sample_count = 0
    batch_count = 0

    started = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

        targets = targets.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(inputs)
        loss = criterion(
            logits,
            targets,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "training loss became NaN or infinity"
            )

        loss.backward()
        optimizer.step()

        current_batch_size = int(
            targets.shape[0]
        )

        loss_sum += (
            float(loss.detach().item())
            * current_batch_size
        )

        sample_count += current_batch_size
        batch_count += 1

    if sample_count == 0:
        raise RuntimeError(
            "training loader produced no samples"
        )

    return TrainMetrics(
        average_loss=loss_sum / sample_count,
        sample_count=sample_count,
        batch_count=batch_count,
        duration_seconds=(
            time.perf_counter() - started
        ),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> EvaluationMetrics:
    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    started = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

        targets = targets.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

        logits = model(inputs)

        loss_sum += float(
            criterion(
                logits,
                targets,
            ).item()
        )

        predictions = logits.argmax(
            dim=1
        )

        correct_count += int(
            (
                predictions == targets
            ).sum().item()
        )

        sample_count += int(
            targets.shape[0]
        )

    if sample_count == 0:
        raise RuntimeError(
            "evaluation loader produced no samples"
        )

    return EvaluationMetrics(
        average_loss=loss_sum / sample_count,
        accuracy_percent=(
            100.0
            * correct_count
            / sample_count
        ),
        correct_count=correct_count,
        sample_count=sample_count,
        duration_seconds=(
            time.perf_counter() - started
        ),
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: dict,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    cpu_state = {
        name: tensor.detach().cpu()
        for name, tensor
        in model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": cpu_state,
            "metadata": metadata,
        },
        temporary_path,
    )

    temporary_path.replace(destination)