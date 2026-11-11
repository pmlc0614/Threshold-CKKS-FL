"""Centralized CIFAR ResNet baselines."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    build_test_loader,
    load_cifar10,
    load_cifar100,
)
from .model import resnet20, resnet32
from .model_adapter import build_model_protocol_layouts
from .state import (
    build_float_state_layout,
    layout_sha256,
    write_layout_json,
)
from .stats import (
    model_storage_stats,
    write_json,
)
from .training import (
    build_optimizer,
    evaluate,
    learning_rate_at_step,
    resolve_device,
    save_checkpoint,
    seed_everything,
    set_optimizer_learning_rate,
    train_one_epoch,
)


SUPPORTED_EXPERIMENTS = {
    ("cifar10", "resnet20"),
    ("cifar100", "resnet32"),
}


def validate_experiment(
    dataset_name: str,
    model_name: str,
) -> None:
    if (dataset_name, model_name) not in SUPPORTED_EXPERIMENTS:
        supported = ", ".join(
            f"{dataset}/{model}"
            for dataset, model in sorted(SUPPORTED_EXPERIMENTS)
        )
        raise ValueError(
            f"unsupported dataset/model pair: {dataset_name}/{model_name}; "
            f"supported pairs: {supported}"
        )


def load_centralized_dataset(
    *,
    dataset_name: str,
    data_dir: str,
    download: bool,
):
    if dataset_name == "cifar10":
        return load_cifar10(
            data_dir=data_dir,
            download=download,
        )
    if dataset_name == "cifar100":
        return load_cifar100(
            data_dir=data_dir,
            download=download,
        )
    raise ValueError(f"unsupported dataset: {dataset_name}")


def build_centralized_model(
    model_name: str,
) -> torch.nn.Module:
    if model_name == "resnet20":
        return resnet20()
    if model_name == "resnet32":
        return resnet32()
    raise ValueError(f"unsupported model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Centralized CIFAR ResNet baseline"
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )

    parser.add_argument(
        "--model",
        choices=["resnet20", "resnet32"],
        default="resnet20",
    )

    parser.add_argument(
        "--data-dir",
        default="./data",
    )

    parser.add_argument(
        "--output-dir",
        default="./runs/centralized",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--schedule",
        choices=["constant", "cosine"],
        default="cosine",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
    )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    device = resolve_device(
        args.device
    )

    train_set, test_set = load_centralized_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        download=not args.no_download,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        args.seed
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=loader_generator,
    )

    test_loader = build_test_loader(
        test_set=test_set,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_centralized_model(
        args.model
    ).to(
        device
    )

    experiment_name = (
        f"{args.model}-{args.dataset}"
    )

    protocol_layouts = build_model_protocol_layouts(
        model,
        model_name=experiment_name,
    )

    layout = build_float_state_layout(
        model
    )

    write_layout_json(
        path=output_dir / "model_layout.json",
        layout=layout,
    )

    configuration = vars(args).copy()
    configuration["mode"] = "centralized"
    configuration["experiment_name"] = experiment_name
    configuration["train_samples"] = len(
        train_set
    )
    configuration["test_samples"] = len(
        test_set
    )
    configuration["layout_sha256"] = (
        layout_sha256(layout)
    )
    configuration["parameter_numel"] = (
        protocol_layouts.parameter.total_numel
    )
    configuration["running_mean_numel"] = (
        protocol_layouts.running_mean.total_numel
    )
    configuration["running_variance_numel"] = (
        protocol_layouts.running_variance.total_numel
    )
    configuration["auxiliary_numel"] = (
        protocol_layouts.running_mean.total_numel
        + protocol_layouts.running_variance.total_numel
    )

    write_json(
        output_dir / "config.json",
        configuration,
    )

    write_json(
        output_dir / "model_stats.json",
        model_storage_stats(
            model=model,
            layout=layout,
        ),
    )

    optimizer = build_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    metrics_path = (
        output_dir / "metrics.csv"
    )

    best_accuracy = float("-inf")
    best_epoch = -1

    total_started = time.perf_counter()

    print(
        (
            f"[START] {args.dataset.upper()}/{args.model.upper()}"
            f" | epochs={args.epochs}"
            f" | train={len(train_set)}"
            f" | test={len(test_set)}"
            f" | parameters={protocol_layouts.parameter.total_numel:,}"
            f" | auxiliary={configuration['auxiliary_numel']:,}"
            f" | device={device}"
            f" | seed={args.seed}"
        ),
        flush=True,
    )

    with metrics_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "test_loss",
            "accuracy_percent",
            "train_seconds",
            "eval_seconds",
            "epoch_seconds",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for epoch_index in range(
            args.epochs
        ):
            epoch_started = (
                time.perf_counter()
            )

            learning_rate = (
                learning_rate_at_step(
                    initial_learning_rate=(
                        args.learning_rate
                    ),
                    minimum_learning_rate=(
                        args.minimum_learning_rate
                    ),
                    step=epoch_index,
                    total_steps=args.epochs,
                    schedule=args.schedule,
                )
            )

            set_optimizer_learning_rate(
                optimizer,
                learning_rate,
            )

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

            evaluation = evaluate(
                model=model,
                loader=test_loader,
                device=device,
            )

            epoch_number = (
                epoch_index + 1
            )

            row = {
                "epoch": epoch_number,
                "learning_rate": learning_rate,
                "train_loss": (
                    train_metrics.average_loss
                ),
                "test_loss": (
                    evaluation.average_loss
                ),
                "accuracy_percent": (
                    evaluation.accuracy_percent
                ),
                "train_seconds": (
                    train_metrics.duration_seconds
                ),
                "eval_seconds": (
                    evaluation.duration_seconds
                ),
                "epoch_seconds": (
                    time.perf_counter()
                    - epoch_started
                ),
            }

            writer.writerow(row)
            handle.flush()

            if (
                evaluation.accuracy_percent
                > best_accuracy
            ):
                best_accuracy = (
                    evaluation.accuracy_percent
                )
                best_epoch = epoch_number

                save_checkpoint(
                    path=(
                        output_dir
                        / "best_model.pt"
                    ),
                    model=model,
                    metadata={
                        "mode": "centralized",
                        "epoch": epoch_number,
                        "accuracy_percent": (
                            best_accuracy
                        ),
                    },
                )

            print(
                (
                    f"[EPOCH {epoch_number:03d}/{args.epochs:03d}]"
                    f" lr={learning_rate:.6g}"
                    f" | train_loss={train_metrics.average_loss:.4f}"
                    f" | test_loss={evaluation.average_loss:.4f}"
                    f" | test_acc={evaluation.accuracy_percent:.2f}%"
                    f" | best={best_accuracy:.2f}%"
                    f" | seconds={row['epoch_seconds']:.1f}"
                ),
                flush=True,
            )

    total_seconds = (
        time.perf_counter()
        - total_started
    )

    save_checkpoint(
        path=output_dir / "final_model.pt",
        model=model,
        metadata={
            "mode": "centralized",
            "epoch": args.epochs,
            "best_epoch": best_epoch,
            "best_accuracy_percent": (
                best_accuracy
            ),
        },
    )

    summary = {
        "mode": "centralized",
        "dataset": args.dataset,
        "model": args.model,
        "experiment_name": experiment_name,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_accuracy_percent": (
            best_accuracy
        ),
        "total_seconds": total_seconds,
    }

    write_json(
        output_dir / "summary.json",
        summary,
    )

    print(
        (
            f"[COMPLETE] {args.dataset.upper()}/{args.model.upper()}"
            f" | best_epoch={best_epoch}"
            f" | best_accuracy={best_accuracy:.2f}%"
            f" | total_seconds={total_seconds:.1f}"
            f" | output={output_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
