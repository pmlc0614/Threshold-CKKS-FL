"""Dense plaintext FedAvg for CIFAR-10/100 ResNet experiments.

FULL mode deliberately has no sparse chunks, indices, padding, residuals, or
cohort filtering.  Every client transmits its complete FP32 floating-point
state delta in the canonical state-layout order.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Sequence

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
    load_cifar100,
    load_or_create_partitions,
)
from .federated import partition_document
from .model import resnet20, resnet32
from .plaintext_communication import measure_plaintext_full_round_communication
from .state import (
    StateEntry,
    apply_model_delta,
    build_float_state_layout,
    compute_model_delta,
    layout_numel,
    layout_sha256,
    write_layout_json,
)
from .stats import model_storage_stats, tensor_stats, write_json
from .timing import maximum_phase_seconds, synchronized_perf_counter
from .training import (
    build_optimizer,
    evaluate,
    learning_rate_at_step,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from .update_analysis import (
    add_masked_gaussian_noise,
    gaussian_noise_std_from_exponent,
)


SUPPORTED_FULL_EXPERIMENTS = {
    ("cifar10", "resnet20"),
    ("cifar100", "resnet32"),
}


def validate_full_experiment(dataset_name: str, model_name: str) -> None:
    if (dataset_name, model_name) not in SUPPORTED_FULL_EXPERIMENTS:
        supported = ", ".join(
            f"{dataset}/{model}"
            for dataset, model in sorted(SUPPORTED_FULL_EXPERIMENTS)
        )
        raise ValueError(
            f"unsupported FULL dataset/model pair: {dataset_name}/{model_name}; "
            f"supported pairs: {supported}"
        )


def load_full_dataset(*, dataset_name: str, data_dir: str, download: bool):
    if dataset_name == "cifar10":
        return load_cifar10(data_dir=data_dir, download=download)
    if dataset_name == "cifar100":
        return load_cifar100(data_dir=data_dir, download=download)
    raise ValueError(f"unsupported FULL dataset: {dataset_name}")


def build_full_model(model_name: str) -> torch.nn.Module:
    if model_name == "resnet20":
        return resnet20()
    if model_name == "resnet32":
        return resnet32()
    raise ValueError(f"unsupported FULL model: {model_name}")


def build_full_noise_masks(
    layout: Sequence[StateEntry],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return disjoint parameter, running-mean, and running-var masks."""

    total_numel = layout_numel(layout)
    parameter_mask = torch.zeros(total_numel, dtype=torch.bool)
    running_mean_mask = torch.zeros(total_numel, dtype=torch.bool)
    running_var_mask = torch.zeros(total_numel, dtype=torch.bool)
    unsupported_buffers: list[str] = []

    for entry in layout:
        target = slice(entry.offset, entry.offset + entry.numel)
        if entry.is_parameter:
            parameter_mask[target] = True
        elif entry.name.endswith("running_mean"):
            running_mean_mask[target] = True
        elif entry.name.endswith("running_var"):
            running_var_mask[target] = True
        else:
            unsupported_buffers.append(entry.name)

    if unsupported_buffers:
        raise ValueError(
            "FULL layout has unsupported floating-point buffers: "
            + ", ".join(unsupported_buffers)
        )
    if not parameter_mask.any():
        raise ValueError("FULL layout contains no trainable parameters")
    if not running_mean_mask.any() or not running_var_mask.any():
        raise ValueError("FULL layout must contain running_mean and running_var")
    if torch.any(parameter_mask & (running_mean_mask | running_var_mask)):
        raise RuntimeError("FULL noise masks overlap")
    if not torch.all(parameter_mask | running_mean_mask | running_var_mask):
        raise RuntimeError("FULL noise masks do not cover the floating state")

    return parameter_mask, running_mean_mask, running_var_mask


def add_full_protocol_noise(
    aggregate: torch.Tensor,
    *,
    parameter_mask: torch.Tensor,
    running_mean_mask: torch.Tensor,
    running_var_mask: torch.Tensor,
    parameter_sigma: float,
    auxiliary_sigma: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply independent p and p_aux Gaussian streams to a FULL update."""

    noisy, parameter_noise = add_masked_gaussian_noise(
        aggregate, parameter_mask, parameter_sigma, seed
    )
    noisy, running_mean_noise = add_masked_gaussian_noise(
        noisy, running_mean_mask, auxiliary_sigma, seed + 1
    )
    noisy, running_var_noise = add_masked_gaussian_noise(
        noisy, running_var_mask, auxiliary_sigma, seed + 2
    )
    return noisy, {
        "parameter": parameter_noise,
        "running_mean": running_mean_noise,
        "running_var": running_var_noise,
    }


def _maximum_absolute(values: torch.Tensor) -> float:
    selected = values.detach().reshape(-1)
    return float(selected.abs().max().item())


def _safety_bound(observed: float, factor: float = 1.5) -> float:
    return 0.0 if observed == 0.0 else float(observed) * factor


def _validate_positive(name: str, value: int | float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FULL dense FedAvg CIFAR-10/100 ResNet experiment"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--model", choices=["resnet20", "resnet32"], default="resnet32")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./runs/federated-full")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--split", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--partition-file", default=None)
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--weight-mode", choices=["exact", "quantized"], default="exact")
    parser.add_argument("--weight-denominator", type=int, default=128)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--parameter-update-bound", "--u-max", dest="parameter_update_bound", type=float, default=None)
    parser.add_argument("--running-update-bound", "--u-max-aux", dest="running_update_bound", type=float, default=None)
    parser.add_argument("--noise-p", "--parameter-noise-p", dest="noise_p", type=float, default=None)
    parser.add_argument("--aux-noise-p", "--running-noise-p", dest="aux_noise_p", type=float, default=None)
    parser.add_argument("--noise-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    validate_full_experiment(args.dataset, args.model)
    for name, value in (
        ("rounds", args.rounds),
        ("clients", args.clients),
        ("local_epochs", args.local_epochs),
        ("batch_size", args.batch_size),
        ("eval_batch_size", args.eval_batch_size),
        ("learning_rate", args.learning_rate),
        ("weight_denominator", args.weight_denominator),
    ):
        _validate_positive(name, value)
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.split == "noniid":
        _validate_positive("alpha", args.alpha)
    for name, bound in (
        ("parameter_update_bound", args.parameter_update_bound),
        ("running_update_bound", args.running_update_bound),
    ):
        if bound is not None:
            _validate_positive(name, bound)

    parameter_sigma = gaussian_noise_std_from_exponent(
        args.noise_p, reference_bound=args.parameter_update_bound
    )
    auxiliary_sigma = gaussian_noise_std_from_exponent(
        args.aux_noise_p, reference_bound=args.running_update_bound
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed=args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    train_set, test_set = load_full_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        download=not args.no_download,
    )
    partition_seed = args.seed if args.partition_seed is None else args.partition_seed
    partitions, partition_info = load_or_create_partitions(
        targets=train_set.targets,
        dataset_name=args.dataset,
        num_clients=args.clients,
        partition=args.split,
        dirichlet_alpha=args.alpha,
        partition_seed=partition_seed,
        partition_file=args.partition_file,
        validate_partition_seed=args.partition_seed is not None,
        minimum_client_samples=1,
    )
    client_datasets = build_client_datasets(train_set=train_set, client_indices=partitions)
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
    sample_counts = client_sample_counts(client_datasets)
    total_sample_count = sum(sample_counts)
    exact_weights = [count / total_sample_count for count in sample_counts]
    if args.weight_mode == "quantized":
        alphas = quantize_client_weights(sample_counts, args.weight_denominator)
        effective_weights = [
            int(alpha.item()) / args.weight_denominator for alpha in alphas
        ]
    else:
        alphas = None
        effective_weights = exact_weights.copy()

    partition_report = partition_document(targets=train_set.targets, partitions=partitions)
    partition_report["assignment"] = partition_info
    write_json(output_dir / "partition.json", partition_report)

    global_model = build_full_model(args.model).to(device)
    layout = build_float_state_layout(global_model)
    parameter_mask, running_mean_mask, running_var_mask = build_full_noise_masks(layout)
    parameter_numel = int(parameter_mask.sum().item())
    auxiliary_numel = int((running_mean_mask | running_var_mask).sum().item())
    communication = measure_plaintext_full_round_communication(
        client_count=args.clients,
        parameter_numel=parameter_numel,
        auxiliary_numel=auxiliary_numel,
    )
    write_layout_json(
        output_dir / "model_layout.json",
        layout,
        model_name=f"{args.model}-{args.dataset}-full",
    )

    configuration = vars(args).copy()
    configuration.update(
        {
            "mode": "full-fedavg",
            "communication_profile": "FULL",
            "communication_accounting": "dense-fixed-order-fp32-no-indices-no-padding",
            "plaintext_payload_padding_included": False,
            "parameter_numel": parameter_numel,
            "auxiliary_numel": auxiliary_numel,
            "float_state_numel": layout_numel(layout),
            "model_layout_sha256": layout_sha256(layout),
            "parameter_noise_sigma": parameter_sigma,
            "auxiliary_noise_sigma": auxiliary_sigma,
            "client_sample_counts": sample_counts,
            "exact_client_weights": exact_weights,
            "effective_client_weights": effective_weights,
            "aggregation_alphas": (
                None
                if alphas is None
                else [int(alpha.item()) for alpha in alphas]
            ),
            "partition_source": partition_info["source"],
            "partition_sha256": partition_info["partition_sha256"],
        }
    )
    write_json(output_dir / "config.json", configuration)

    metrics_path = output_dir / "metrics.csv"
    jsonl_path = output_dir / "round_metrics.jsonl"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "round", "learning_rate", "train_loss", "test_loss",
            "test_accuracy_percent", "sequential_client_seconds",
            "maximum_client_seconds", "server_seconds", "evaluation_seconds",
            "round_seconds", "online_runtime_parallel_seconds",
            "fp32_uplink_mb", "unicast_downlink_mb",
        ],
    )
    metrics_writer.writeheader()

    started = synchronized_perf_counter(device)
    total_online_parallel = 0.0
    total_sequential_clients = 0.0
    total_server = 0.0
    total_evaluation = 0.0
    observed_parameter_max = 0.0
    observed_mean_max = 0.0
    observed_var_max = 0.0
    best_accuracy = -math.inf
    best_round = 0
    final_evaluation = None

    print("\n=== PLAINTEXT FULL FL RUN ===", flush=True)
    print(
        f"dataset={args.dataset} | model={args.model} | rounds={args.rounds} "
        f"| clients={args.clients} | device={device}",
        flush=True,
    )
    print(
        "communication_profile=FULL | indices=0 | padding=0 | "
        f"d={parameter_numel} | d_aux={auxiliary_numel}",
        flush=True,
    )

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for round_index in range(args.rounds):
                round_number = round_index + 1
                round_started = synchronized_perf_counter(device)
                learning_rate = learning_rate_at_step(
                    initial_learning_rate=args.learning_rate,
                    minimum_learning_rate=args.minimum_learning_rate,
                    step=round_index,
                    total_steps=args.rounds,
                    schedule=args.schedule,
                )
                dense_updates: list[torch.Tensor] = []
                client_seconds: list[float] = []
                weighted_loss_sum = 0.0
                for loader, count in zip(client_loaders, sample_counts):
                    client_started = synchronized_perf_counter(device)
                    local_model = copy.deepcopy(global_model).to(device)
                    optimizer = build_optimizer(
                        model=local_model,
                        optimizer_name=args.optimizer,
                        learning_rate=learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                    )
                    client_loss_sum = 0.0
                    for _ in range(args.local_epochs):
                        train_metrics = train_one_epoch(local_model, loader, optimizer, device)
                        client_loss_sum += train_metrics.average_loss
                    dense_updates.append(
                        compute_model_delta(local_model, global_model, layout, device="cpu")
                    )
                    weighted_loss_sum += (client_loss_sum / args.local_epochs) * count
                    client_seconds.append(
                        synchronized_perf_counter(device) - client_started
                    )
                    del local_model, optimizer

                sequential_client_seconds = sum(client_seconds)
                maximum_client_seconds = maximum_phase_seconds(client_seconds)
                server_started = synchronized_perf_counter(device)
                if args.weight_mode == "quantized":
                    clean_aggregate = aggregate_dense_updates(
                        dense_updates, alphas, args.weight_denominator
                    )
                else:
                    clean_aggregate = aggregate_dense_fedavg(dense_updates, sample_counts)

                observed_parameter_max = max(
                    observed_parameter_max,
                    _maximum_absolute(clean_aggregate[parameter_mask]),
                )
                observed_mean_max = max(
                    observed_mean_max,
                    _maximum_absolute(clean_aggregate[running_mean_mask]),
                )
                observed_var_max = max(
                    observed_var_max,
                    _maximum_absolute(clean_aggregate[running_var_mask]),
                )
                noise_seed = args.seed + args.noise_seed_offset + 3 * round_index
                noisy_aggregate, noises = add_full_protocol_noise(
                    clean_aggregate,
                    parameter_mask=parameter_mask,
                    running_mean_mask=running_mean_mask,
                    running_var_mask=running_var_mask,
                    parameter_sigma=parameter_sigma,
                    auxiliary_sigma=auxiliary_sigma,
                    seed=noise_seed,
                )
                apply_model_delta(global_model, noisy_aggregate, layout)
                server_seconds = synchronized_perf_counter(device) - server_started

                evaluation_started = synchronized_perf_counter(device)
                final_evaluation = evaluate(global_model, test_loader, device)
                evaluation_seconds = synchronized_perf_counter(device) - evaluation_started
                round_seconds = synchronized_perf_counter(device) - round_started
                online_parallel = maximum_client_seconds + server_seconds
                total_online_parallel += online_parallel
                total_sequential_clients += sequential_client_seconds
                total_server += server_seconds
                total_evaluation += evaluation_seconds
                train_loss = weighted_loss_sum / total_sample_count
                best_accuracy = max(best_accuracy, final_evaluation.accuracy_percent)
                if final_evaluation.accuracy_percent == best_accuracy:
                    best_round = round_number

                round_metrics = {
                    "round": round_number,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "test_loss": final_evaluation.average_loss,
                    "test_accuracy_percent": final_evaluation.accuracy_percent,
                    "client_seconds": client_seconds,
                    "sequential_client_seconds": sequential_client_seconds,
                    "maximum_client_seconds": maximum_client_seconds,
                    "server_seconds": server_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "round_seconds": round_seconds,
                    "online_runtime_parallel_seconds": online_parallel,
                    "communication": communication,
                    "clean_parameter_update": tensor_stats(clean_aggregate[parameter_mask]),
                    "clean_running_mean_update": tensor_stats(clean_aggregate[running_mean_mask]),
                    "clean_running_var_update": tensor_stats(clean_aggregate[running_var_mask]),
                    "parameter_noise": tensor_stats(noises["parameter"][parameter_mask]),
                    "running_mean_noise": tensor_stats(noises["running_mean"][running_mean_mask]),
                    "running_var_noise": tensor_stats(noises["running_var"][running_var_mask]),
                }
                json.dump(round_metrics, jsonl_file, sort_keys=True, allow_nan=False)
                jsonl_file.write("\n")
                jsonl_file.flush()
                csv_row = {
                    key: round_metrics[key]
                    for key in metrics_writer.fieldnames
                    if key in round_metrics
                }
                csv_row["fp32_uplink_mb"] = communication["accepted_only_fp32_uplink_bytes"] / 1_000_000
                csv_row["unicast_downlink_mb"] = communication["accepted_only_unicast_downlink_bytes"] / 1_000_000
                metrics_writer.writerow(csv_row)
                metrics_file.flush()

                print(
                    f"\n=== ROUND {round_number:03d}/{args.rounds} "
                    f"| ACCURACY: {final_evaluation.accuracy_percent:.2f}% ===",
                    flush=True,
                )
                print(f"train_loss={train_loss:.6f} | lr={learning_rate:.7g}", flush=True)
                print(
                    f"online_comm={(communication['accepted_only_fp32_uplink_bytes'] + communication['accepted_only_unicast_downlink_bytes']) / 1_000_000:.3f} MB "
                    f"all-clients total (uplink={communication['accepted_only_fp32_uplink_bytes'] / 1_000_000:.3f} MB, "
                    f"downlink={communication['accepted_only_unicast_downlink_bytes'] / 1_000_000:.3f} MB)",
                    flush=True,
                )
                print(
                    f"round_time={round_seconds:.1f}s | client_phase_sequential={sequential_client_seconds:.1f}s "
                    f"| client_phase_parallel={maximum_client_seconds:.1f}s | server={server_seconds:.1f}s "
                    f"| evaluation={evaluation_seconds:.1f}s | online_runtime_parallel={online_parallel:.1f}s "
                    "(evaluation_excluded, network_excluded)",
                    flush=True,
                )
    finally:
        metrics_file.close()

    if final_evaluation is None:
        raise RuntimeError("FULL experiment completed without evaluation")
    total_seconds = synchronized_perf_counter(device) - started
    rounds = args.rounds
    uplink_bytes = communication["accepted_only_fp32_uplink_bytes"] * rounds
    downlink_bytes = communication["accepted_only_unicast_downlink_bytes"] * rounds
    summary = {
        **configuration,
        "final_test_accuracy_percent": final_evaluation.accuracy_percent,
        "final_test_loss": final_evaluation.average_loss,
        "best_accuracy_percent": best_accuracy,
        "best_round": best_round,
        "total_seconds": total_seconds,
        "total_sequential_client_seconds": total_sequential_clients,
        "total_server_seconds": total_server,
        "total_evaluation_seconds": total_evaluation,
        "total_online_runtime_parallel_seconds": total_online_parallel,
        "average_online_runtime_parallel_seconds_per_round": total_online_parallel / rounds,
        "observed_parameter_update_max_abs": observed_parameter_max,
        "observed_running_mean_update_max_abs": observed_mean_max,
        "observed_running_var_update_max_abs": observed_var_max,
        "observed_running_update_max_abs": max(observed_mean_max, observed_var_max),
        "recommended_parameter_update_bound_1_5x": _safety_bound(observed_parameter_max),
        "recommended_running_update_bound_1_5x": _safety_bound(max(observed_mean_max, observed_var_max)),
        "per_round_accepted_only_fp32_uplink_bytes": communication["accepted_only_fp32_uplink_bytes"],
        "per_round_accepted_only_logical_downlink_bytes": communication["accepted_only_logical_downlink_bytes"],
        "per_round_accepted_only_unicast_downlink_bytes": communication["accepted_only_unicast_downlink_bytes"],
        "per_round_mean_client_fp32_uplink_bytes": communication["accepted_only_fp32_uplink_bytes"] / args.clients,
        "per_round_mean_client_unicast_downlink_bytes": communication["accepted_only_unicast_downlink_bytes"] / args.clients,
        "total_accepted_only_fp32_uplink_bytes": uplink_bytes,
        "total_accepted_only_logical_downlink_bytes": communication["accepted_only_logical_downlink_bytes"] * rounds,
        "total_accepted_only_unicast_downlink_bytes": downlink_bytes,
        "total_accepted_only_fp32_uplink_mb": uplink_bytes / 1_000_000,
        "total_accepted_only_logical_downlink_mb": communication["accepted_only_logical_downlink_bytes"] * rounds / 1_000_000,
        "total_accepted_only_unicast_downlink_mb": downlink_bytes / 1_000_000,
        "total_mean_client_fp32_uplink_mb": uplink_bytes / args.clients / 1_000_000,
        "total_mean_client_unicast_downlink_mb": downlink_bytes / args.clients / 1_000_000,
        "total_all_clients_communication_mb": (uplink_bytes + downlink_bytes) / 1_000_000,
        "model_storage": model_storage_stats(global_model, layout),
    }
    write_json(output_dir / "observed_bounds.json", {
        key: value for key, value in summary.items()
        if key.startswith("observed_") or key.startswith("recommended_")
    })
    write_json(output_dir / "summary.json", summary)
    save_checkpoint(output_dir / "final.pt", global_model, summary)

    print("\n=== TRAINING COMPLETE ===", flush=True)
    print(
        f"final_accuracy={final_evaluation.accuracy_percent:.2f}% | best={best_accuracy:.2f}% at round {best_round}",
        flush=True,
    )
    print(
        f"online_communication={(uplink_bytes + downlink_bytes) / 1_000_000:.3f} MB all-clients cumulative "
        f"(uplink={uplink_bytes / 1_000_000:.3f} MB, downlink={downlink_bytes / 1_000_000:.3f} MB)",
        flush=True,
    )
    print(
        f"online_runtime_parallel={total_online_parallel:.1f}s total | mean/round={total_online_parallel / rounds:.1f}s "
        "(max_client + server, evaluation/network excluded)",
        flush=True,
    )
    print(f"total_time={total_seconds:.1f}s | summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
