"""Dense FedAvg CIFAR-10 ResNet-20 experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .aggregation import (
    aggregate_dense_fedavg,
    aggregate_dense_updates,
    quantize_client_weights,
)
from .data import (
    build_client_datasets,
    build_client_loaders,
    build_test_loader,
    client_sample_counts,
    load_cifar10,
    partition_indices,
)
from .model import resnet20
from .state import (
    apply_model_delta,
    build_float_state_layout,
    compute_model_delta,
    flatten_float_state,
    layout_numel,
    layout_sha256,
    write_layout_json,
)
from .stats import (
    model_storage_stats,
    tensor_stats,
    write_json,
)
from .training import (
    build_optimizer,
    evaluate,
    learning_rate_at_step,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


def dense_error_feedback_step(
    local_delta: torch.Tensor,
    previous_residual: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply error feedback to a dense uncompressed update.

    Since no compression is applied:

        transmitted = local_delta + residual
        new_residual = 0

    With an initially zero residual, the residual remains zero for
    every round.
    """

    delta = (
        local_delta.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    residual = (
        previous_residual.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    if delta.numel() != residual.numel():
        raise ValueError(
            "local_delta and previous_residual "
            "must have equal dimensions"
        )

    if enabled:
        compression_input = (
            delta + residual
        )
    else:
        compression_input = delta

    # Dense baseline: nothing is dropped.
    transmitted = (
        compression_input.clone()
    )

    new_residual = (
        compression_input
        - transmitted
    )

    return (
        transmitted.contiguous(),
        new_residual.contiguous(),
    )


def partition_document(
    targets,
    partitions: list[np.ndarray],
) -> dict:
    target_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    clients = []

    for client_id, indices in enumerate(
        partitions
    ):
        labels = target_array[indices]

        histogram = np.bincount(
            labels,
            minlength=10,
        )

        clients.append(
            {
                "client_id": client_id,
                "sample_count": int(
                    len(indices)
                ),
                "class_histogram": (
                    histogram.tolist()
                ),
            }
        )

    return {
        "total_samples": int(
            sum(
                len(indices)
                for indices in partitions
            )
        ),
        "clients": clients,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dense FedAvg CIFAR-10 ResNet-20"
        )
    )

    parser.add_argument(
        "--data-dir",
        default="./data",
    )

    parser.add_argument(
        "--output-dir",
        default="./runs/federated",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
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
        "--split",
        choices=["iid", "noniid"],
        default="iid",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--weight-mode",
        choices=[
            "exact",
            "quantized",
        ],
        default="exact",
        help=(
            "exact uses Di/sum(Di); "
            "quantized uses alpha_i/denominator"
        ),
    )

    parser.add_argument(
        "--weight-denominator",
        type=int,
        default=128,
        help=(
            "Integer weight denominator used "
            "when weight-mode is quantized"
        ),
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

    # Sequential client execution creates one DataLoader per client.
    # Zero avoids creating many persistent worker processes.
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
    )

    parser.set_defaults(
        use_error_feedback=True
    )

    parser.add_argument(
        "--no-error-feedback",
        action="store_false",
        dest="use_error_feedback",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rounds <= 0:
        raise ValueError(
            "rounds must be positive"
        )

    if args.clients <= 0:
        raise ValueError(
            "clients must be positive"
        )

    if args.local_epochs <= 0:
        raise ValueError(
            "local_epochs must be positive"
        )

    if (
        args.split == "noniid"
        and args.alpha <= 0
    ):
        raise ValueError(
            "alpha must be positive"
        )
    if args.weight_denominator <= 0:
        raise ValueError(
            "weight_denominator must be positive"
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

    pin_memory = (
        device.type == "cuda"
    )

    train_set, test_set = load_cifar10(
        data_dir=args.data_dir,
        download=not args.no_download,
    )

    partitions = partition_indices(
        targets=train_set.targets,
        num_clients=args.clients,
        partition=args.split,
        seed=args.seed,
        dirichlet_alpha=args.alpha,
        minimum_client_samples=1,
    )

    client_datasets = build_client_datasets(
        train_set=train_set,
        client_indices=partitions,
    )

    client_loaders = build_client_loaders(
        client_datasets=client_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    test_loader = build_test_loader(
        test_set=test_set,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sample_counts = client_sample_counts(
        client_datasets
    )

    total_sample_count = sum(
        sample_counts
    )

    exact_client_weights = [
        count / total_sample_count
        for count in sample_counts
    ]

    if args.weight_mode == "quantized":
        aggregation_alphas = (
            quantize_client_weights(
                example_counts=sample_counts,
                denominator=(
                    args.weight_denominator
                ),
            )
        )

        effective_client_weights = [
            int(alpha.item())
            / args.weight_denominator
            for alpha in aggregation_alphas
        ]

        aggregation_name = (
            f"quantized-alpha-over-"
            f"{args.weight_denominator}"
        )
    else:
        aggregation_alphas = None

        effective_client_weights = (
            exact_client_weights.copy()
        )

        aggregation_name = (
            "exact-Di-over-sum-Di"
        )

    max_weight_error = max(
        abs(exact - effective)
        for exact, effective in zip(
            exact_client_weights,
            effective_client_weights,
        )
    )

    write_json(
        output_dir / "partition.json",
        partition_document(
            targets=train_set.targets,
            partitions=partitions,
        ),
    )

    global_model = resnet20().to(
        device
    )

    layout = build_float_state_layout(
        global_model
    )

    write_layout_json(
        path=output_dir / "model_layout.json",
        layout=layout,
    )

    configuration = vars(args).copy()
    configuration["mode"] = "dense-fedavg"
    configuration["aggregation"] = (
        aggregation_name
    )

    configuration["exact_client_weights"] = (
        exact_client_weights
    )

    configuration["effective_client_weights"] = (
        effective_client_weights
    )

    configuration["aggregation_alphas"] = (
        aggregation_alphas.tolist()
        if aggregation_alphas is not None
        else None
    )

    configuration["max_abs_weight_error"] = (
        max_weight_error
    )
    configuration["compression"] = "none"
    configuration["layout_sha256"] = (
        layout_sha256(layout)
    )
    configuration["client_sample_counts"] = (
        sample_counts
    )

    write_json(
        output_dir / "config.json",
        configuration,
    )

    write_json(
        output_dir / "model_stats.json",
        model_storage_stats(
            model=global_model,
            layout=layout,
        ),
    )

    residuals = [
        torch.zeros(
            layout_numel(layout),
            dtype=torch.float32,
            device="cpu",
        )
        for _ in range(args.clients)
    ]

    metrics_path = (
        output_dir / "metrics.csv"
    )

    best_accuracy = float("-inf")
    best_round = -1

    total_started = time.perf_counter()

    with metrics_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "round",
            "learning_rate",
            "weighted_train_loss",
            "test_loss",
            "accuracy_percent",
            "aggregate_l2",
            "max_residual_l2",
            "round_seconds",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for round_index in range(
            args.rounds
        ):
            round_started = (
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
                    step=round_index,
                    total_steps=args.rounds,
                    schedule=args.schedule,
                )
            )

            client_updates = []
            client_average_losses = []

            for client_id, loader in enumerate(
                client_loaders
            ):
                # deepcopy avoids consuming additional RNG state
                # through a fresh random model initialization.
                local_model = copy.deepcopy(
                    global_model
                )

                local_optimizer = (
                    build_optimizer(
                        model=local_model,
                        optimizer_name=(
                            args.optimizer
                        ),
                        learning_rate=(
                            learning_rate
                        ),
                        momentum=args.momentum,
                        weight_decay=(
                            args.weight_decay
                        ),
                    )
                )

                local_loss_sum = 0.0
                processed_samples = 0

                for _local_epoch in range(
                    args.local_epochs
                ):
                    train_metrics = (
                        train_one_epoch(
                            model=local_model,
                            loader=loader,
                            optimizer=(
                                local_optimizer
                            ),
                            device=device,
                        )
                    )

                    local_loss_sum += (
                        train_metrics.average_loss
                        * train_metrics.sample_count
                    )

                    processed_samples += (
                        train_metrics.sample_count
                    )

                client_average_loss = (
                    local_loss_sum
                    / processed_samples
                )

                local_delta = (
                    compute_model_delta(
                        local_model=local_model,
                        global_model=global_model,
                        layout=layout,
                        device="cpu",
                    )
                )

                (
                    transmitted_update,
                    new_residual,
                ) = dense_error_feedback_step(
                    local_delta=local_delta,
                    previous_residual=(
                        residuals[client_id]
                    ),
                    enabled=(
                        args.use_error_feedback
                    ),
                )

                residuals[client_id] = (
                    new_residual
                )

                client_updates.append(
                    transmitted_update
                )

                client_average_losses.append(
                    client_average_loss
                )

                del local_optimizer
                del local_model

            if args.weight_mode == "exact":
                aggregate = (
                    aggregate_dense_fedavg(
                        dense_updates=(
                            client_updates
                        ),
                        sample_counts=(
                            sample_counts
                        ),
                    )
                )
            else:
                if aggregation_alphas is None:
                    raise RuntimeError(
                        "quantized alphas "
                        "were not initialized"
                    )

                aggregate = (
                    aggregate_dense_updates(
                        dense_updates=(
                            client_updates
                        ),
                        alphas=(
                            aggregation_alphas
                        ),
                        weight_denominator=(
                            args.weight_denominator
                        ),
                    )
                )

            apply_model_delta(
                model=global_model,
                delta=aggregate,
                layout=layout,
                scale=1.0,
            )

            evaluation = evaluate(
                model=global_model,
                loader=test_loader,
                device=device,
            )

            total_samples = sum(
                sample_counts
            )

            weighted_train_loss = sum(
                client_loss * client_size
                for client_loss, client_size
                in zip(
                    client_average_losses,
                    sample_counts,
                )
            ) / total_samples

            max_residual_l2 = max(
                float(
                    torch.linalg.vector_norm(
                        residual.to(
                            dtype=torch.float64
                        )
                    ).item()
                )
                for residual in residuals
            )

            round_number = (
                round_index + 1
            )

            row = {
                "round": round_number,
                "learning_rate": learning_rate,
                "weighted_train_loss": (
                    weighted_train_loss
                ),
                "test_loss": (
                    evaluation.average_loss
                ),
                "accuracy_percent": (
                    evaluation.accuracy_percent
                ),
                "aggregate_l2": (
                    tensor_stats(
                        aggregate
                    )["l2"]
                ),
                "max_residual_l2": (
                    max_residual_l2
                ),
                "round_seconds": (
                    time.perf_counter()
                    - round_started
                ),
            }

            writer.writerow(row)
            handle.flush()

            print(
                json.dumps(row),
                flush=True,
            )

            if (
                evaluation.accuracy_percent
                > best_accuracy
            ):
                best_accuracy = (
                    evaluation.accuracy_percent
                )
                best_round = round_number

                save_checkpoint(
                    path=(
                        output_dir
                        / "best_model.pt"
                    ),
                    model=global_model,
                    metadata={
                        "mode": "dense-fedavg",
                        "round": round_number,
                        "split": args.split,
                        "alpha": args.alpha,
                        "accuracy_percent": (
                            best_accuracy
                        ),
                        "weight_mode": (
                            args.weight_mode
                        ),
                        "weight_denominator": (
                            args.weight_denominator
                        ),
                        "aggregation_alphas": (
                            aggregation_alphas.tolist()
                            if aggregation_alphas
                            is not None
                            else None
                        ),
                    },
                )

    total_seconds = (
        time.perf_counter()
        - total_started
    )

    save_checkpoint(
        path=output_dir / "final_model.pt",
        model=global_model,
        metadata={
            "mode": "dense-fedavg",
            "round": args.rounds,
            "split": args.split,
            "alpha": args.alpha,
            "best_round": best_round,
            "best_accuracy_percent": (
                best_accuracy
            ),
            "weight_mode": (
                args.weight_mode
            ),
            "weight_denominator": (
                args.weight_denominator
            ),
            "aggregation_alphas": (
                aggregation_alphas.tolist()
                if aggregation_alphas
                is not None
                else None
            ),
        },
    )

    summary = {
        "mode": "dense-fedavg",
        "rounds": args.rounds,
        "clients": args.clients,
        "local_epochs": args.local_epochs,
        "split": args.split,
        "alpha": (
            args.alpha
            if args.split == "noniid"
            else None
        ),
        "best_round": best_round,
        "best_accuracy_percent": (
            best_accuracy
        ),
        "total_seconds": total_seconds,
        "error_feedback_enabled": (
            args.use_error_feedback
        ),
        "expected_dense_residual": 0.0,
        "weight_mode": args.weight_mode,
        "weight_denominator": (
            args.weight_denominator
        ),
        "aggregation": aggregation_name,
        "aggregation_alphas": (
            aggregation_alphas.tolist()
            if aggregation_alphas is not None
            else None
        ),
        "exact_client_weights": (
            exact_client_weights
        ),
        "effective_client_weights": (
            effective_client_weights
        ),
        "max_abs_weight_error": (
            max_weight_error
        ),
    }

    write_json(
        output_dir / "summary.json",
        summary,
    )

    print(
        json.dumps(summary),
        flush=True,
    )


if __name__ == "__main__":
    main()