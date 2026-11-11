"""Sparse FedAvg CIFAR-10/100 ResNet experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import torch

from .aggregation import (
    ClientSparseUpdate,
    aggregate_sparse_fedavg,
    aggregate_sparse_updates,
    quantize_client_weights,
    sparse_communication_counts,
)
from .data import (
    build_client_datasets,
    build_client_loaders,
    build_test_loader,
    client_sample_counts,
    load_or_create_partitions,
    load_cifar10,
    load_cifar100,
)
from .federated import partition_document
from .model import resnet20, resnet32
from .plaintext_communication import measure_plaintext_round_communication
from .support_bitmap import (
    SUPPORT_CODEC_NAME,
    decode_support_bitmap,
    encode_support_bitmap,
    intersect_support_bitmaps,
)
from .sparsify import (
    chunk_count,
    compress_with_error_feedback,
    effective_sparsity_metrics,
    restore_rejected_chunks_to_residual,
    selected_chunk_count_from_ratio,
)
from .state import (
    apply_model_delta,
    build_float_state_layout,
    clone_float_buffers,
    compact_parameter_layout,
    compute_model_delta,
    layout_numel,
    layout_sha256,
    load_weighted_average_float_buffers,
    write_layout_json,
    zero_nonparameter_entries,
)
from .stats import (
    model_storage_stats,
    tensor_stats,
    write_json,
)
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
    apply_buffer_delta_vector,
    buffer_delta_vector,
    buffer_values_vector,
    build_selected_parameter_mask,
    extract_parameter_values,
    gaussian_noise_std_from_exponent,
)


SUPPORTED_SPARSE_EXPERIMENTS = {
    ("cifar10", "resnet20"),
    ("cifar100", "resnet32"),
}

ROUND_METRIC_CONTEXT_FIELDS = (
    "round",
    "dataset",
    "model",
)

PLAINTEXT_CLIENT_RUNTIME_PHASES = (
    "local_sgd",
    "delta_generation",
    "sparsification_prepare",
    "chunk_score",
    "topk_selection",
    "selection_validation",
    "residual_generation",
    "selection_bitmap",
    "sparsification_total",
    "other_client_processing",
)

PLAINTEXT_SERVER_RUNTIME_PHASES = (
    "sparse_aggregation",
    "admission_bitmap",
    "rejected_residual_restore",
    "other_server_processing",
)


def validate_sparse_experiment(
    dataset_name: str,
    model_name: str,
) -> None:
    if (dataset_name, model_name) not in SUPPORTED_SPARSE_EXPERIMENTS:
        supported = ", ".join(
            f"{dataset}/{model}"
            for dataset, model in sorted(SUPPORTED_SPARSE_EXPERIMENTS)
        )
        raise ValueError(
            f"unsupported sparse dataset/model pair: "
            f"{dataset_name}/{model_name}; supported pairs: {supported}"
        )


def load_sparse_dataset(
    *,
    dataset_name: str,
    data_dir: str,
    download: bool,
):
    if dataset_name == "cifar10":
        return load_cifar10(data_dir=data_dir, download=download)
    if dataset_name == "cifar100":
        return load_cifar100(data_dir=data_dir, download=download)
    raise ValueError(f"unsupported sparse dataset: {dataset_name}")


def build_sparse_model(model_name: str) -> torch.nn.Module:
    if model_name == "resnet20":
        return resnet20()
    if model_name == "resnet32":
        return resnet32()
    raise ValueError(f"unsupported sparse model: {model_name}")


def validate_sparse_rho(rho: float) -> None:
    """Require genuine sparsity; FULL experiments use federated_full."""

    value = float(rho)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "sparse rho must be greater than zero and less than one; "
            "use plaintext_fl.federated_full for FULL training"
        )


def validate_sparse_chunk_selection(
    *, selected_chunk_count: int, total_chunk_count: int
) -> None:
    if selected_chunk_count >= total_chunk_count:
        raise ValueError(
            "sparse mode must select fewer than all model chunks; "
            "use plaintext_fl.federated_full for FULL training"
        )


def round_metric_context(
    *,
    round_number: int,
    dataset_name: str,
    model_name: str,
) -> dict[str, int | str]:
    return {
        "round": round_number,
        "dataset": dataset_name,
        "model": model_name,
    }


def basic_vector_stats(
    values: torch.Tensor,
) -> tuple[float, float]:
    """Return population standard deviation and L2 norm."""

    vector = (
        values.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.float64,
        )
    )

    standard_deviation = float(
        vector.std(
            unbiased=False
        ).item()
    )

    l2_norm = float(
        torch.linalg.vector_norm(
            vector
        ).item()
    )

    return standard_deviation, l2_norm


def recommended_safety_scaled_bound(
    observed_max_abs: float,
    safety_factor: float = 1.5,
) -> float:
    """Scale an empirical maximum by a multiplicative safety margin."""

    observed = float(observed_max_abs)
    factor = float(safety_factor)

    if not math.isfinite(observed) or observed < 0:
        raise ValueError(
            "observed_max_abs must be finite and non-negative"
        )

    if not math.isfinite(factor) or factor <= 1:
        raise ValueError(
            "safety_factor must be finite and greater than one"
        )

    if observed == 0:
        return 0.0

    return observed * factor


def _print_run_header(
    *,
    args: argparse.Namespace,
    parameter_numel: int,
    auxiliary_numel: int,
    total_chunk_count: int,
    selected_chunk_count: int,
    partition_info: dict,
    output_dir: Path,
) -> None:
    print(
        (
            "\n=== PLAINTEXT SPARSE FL RUN ===\n"
            f"dataset={args.dataset} | model={args.model}\n"
            f"rounds={args.rounds} | clients={args.clients}"
            f" | device={args.device} | seed={args.seed}"
            f" | deterministic={args.deterministic}\n"
            f"d={parameter_numel} | d_aux={auxiliary_numel}"
            f" | chunk_size={args.chunk_size}"
            f" | selected_chunks={selected_chunk_count}/{total_chunk_count}"
            f" | rho_requested={args.rho:g}"
            f" | rho_selected={selected_chunk_count / total_chunk_count:.6f}"
            f" | tau={args.minimum_cohort_size}\n"
            f"split={args.split} | alpha="
            f"{args.alpha if args.split == 'noniid' else 'N/A'}"
            f" | partition_source={partition_info['source']}"
            f" | partition_sha256={partition_info['partition_sha256']}\n"
            f"outputs={output_dir}"
        ),
        flush=True,
    )


def _print_round_progress(row: dict, requested_rounds: int) -> None:
    auxiliary_max_abs = max(
        row["running_mean_update_max_abs"],
        row["running_var_update_max_abs"],
    )
    print(
        (
            f"\n=== ROUND {row['round']:03d}/{requested_rounds:03d}"
            f" | TEST ACCURACY: {row['accuracy_percent']:.2f}% ==="
        ),
        flush=True,
    )
    print(
        (
            f"train_loss={row['weighted_train_loss']:.6f}"
            f" | test_loss={row['test_loss']:.6f}"
            f" | lr={row['learning_rate']:.6g}"
            f" | accepted_chunks={row['accepted_chunk_count']}"
            f"/{row['total_chunk_count']}"
        ),
        flush=True,
    )
    print(
        (
            f"rho_requested={row['rho_requested']:.6f}"
            f" | rho_selected={row['rho_selected']:.6f}"
            f" | rho_eff_round={row['rho_eff_round']:.6f}"
            f" | rho_eff_running={row['rho_eff_running']:.6f}"
            f" | cohort_retention={row['cohort_retention_ratio']:.6f}"
        ),
        flush=True,
    )
    print(
        (
            "PARAMETER UPDATE (clean aggregate):"
            f" range=[{row['parameter_update_min']:.6g},"
            f" {row['parameter_update_max']:.6g}]"
            f" | max_abs={row['parameter_update_max_abs']:.6g}"
            f" | U_max_so_far="
            f"{row['observed_parameter_update_max_abs_so_far']:.6g}"
        ),
        flush=True,
    )
    print(
        (
            "RUNNING MEAN UPDATE (clean aggregate):"
            f" range=[{row['running_mean_update_min']:.6g},"
            f" {row['running_mean_update_max']:.6g}]"
            f" | max_abs={row['running_mean_update_max_abs']:.6g}"
        ),
        flush=True,
    )
    print(
        (
            "RUNNING VAR UPDATE (clean aggregate):"
            f" range=[{row['running_var_update_min']:.6g},"
            f" {row['running_var_update_max']:.6g}]"
            f" | max_abs={row['running_var_update_max_abs']:.6g}"
            f" | U_max_aug_this_round={auxiliary_max_abs:.6g}"
            f" | U_max_aug_so_far="
            f"{row['observed_running_update_max_abs_so_far']:.6g}"
        ),
        flush=True,
    )
    print(f"round_time={row['round_seconds']:.1f}s", flush=True)
    print(
        (
            "parallel_estimate="
            f"{row['estimated_parallel_round_seconds']:.1f}s"
            " (independent_clients, barrier_critical_path, network_excluded)"
            f" | protocol_only="
            f"{row['estimated_parallel_protocol_seconds']:.1f}s"
        ),
        flush=True,
    )


def _print_final_summary(summary: dict, output_dir: Path) -> None:
    print(
        (
            f"\n=== PLAINTEXT TRAINING COMPLETE | ROUNDS: {summary['rounds']}"
            f" | FINAL TEST ACCURACY:"
            f" {summary['final_accuracy_percent']:.2f}% ==="
        ),
        flush=True,
    )
    print(
        (
            f"best_accuracy={summary['best_accuracy_percent']:.2f}%"
            f" at round {summary['best_round']}"
        ),
        flush=True,
    )
    print(
        (
            f"rho_requested={summary['rho_requested']:.6f}"
            f" | rho_selected={summary['rho_selected']:.6f}"
            f" | rho_eff={summary['rho_eff']:.6f}"
            f" | cohort_retention={summary['cohort_retention_ratio']:.6f}"
        ),
        flush=True,
    )
    print(
        (
            f"U_max: observed={summary['observed_parameter_update_max_abs']:.6g}"
            f" | recommended_1.5x="
            f"{summary['recommended_parameter_update_bound']:.6g}"
        ),
        flush=True,
    )
    print(
        (
            f"U_max_aug: observed="
            f"{summary['observed_running_update_max_abs']:.6g}"
            f" | running_mean="
            f"{summary['observed_running_mean_update_max_abs']:.6g}"
            f" | running_var="
            f"{summary['observed_running_var_update_max_abs']:.6g}"
            f" | recommended_1.5x="
            f"{summary['recommended_running_update_bound']:.6g}"
        ),
        flush=True,
    )
    print(
        (
            f"total_time={summary['total_seconds']:.1f}s"
            f" | summary={output_dir / 'summary.json'}"
            f" | round_details={output_dir / 'round_metrics.jsonl'}"
        ),
        flush=True,
    )
    print(
        (
            "estimated_parallel_protocol_total="
            f"{summary['total_estimated_parallel_protocol_seconds']:.1f}s"
            " | mean/round="
            f"{summary['average_estimated_parallel_protocol_seconds_per_round']:.1f}s"
            " (independent_clients, network_excluded)"
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sparse FedAvg CIFAR-10/100 ResNet experiment"
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
        default="./runs/federated-sparse",
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
        "--partition-file",
        default=None,
        help=(
            "Create or reuse an exact client sample-index assignment JSON"
        ),
    )

    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help=(
            "Seed used only when creating a partition file; defaults to "
            "--seed. When explicitly supplied, an existing file must match it"
        ),
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
        "--rho",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--residual-beta",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--minimum-cohort-size",
        type=int,
        default=1,
        help=(
            "Accept a sparse chunk only when at least this many "
            "clients selected it"
        ),
    )

    parser.add_argument(
        "--parameter-update-bound",
        "--u-max",
        dest="parameter_update_bound",
        type=float,
        default=None,
        help=(
            "Fixed design U_max for aggregated trainable-parameter "
            "updates; required when --noise-p is used"
        ),
    )

    parser.add_argument(
        "--running-update-bound",
        "--u-max-aux",
        dest="running_update_bound",
        type=float,
        default=None,
        help=(
            "Fixed design U'_max shared by aggregated running_mean "
            "and running_var updates; required when "
            "--aux-noise-p is used"
        ),
    )

    parser.add_argument(
        "--noise-p",
        dest="noise_p",
        type=float,
        default=None,
        help=(
            "Add Gaussian noise with sigma=U_max*2^(-p) "
            "to the aggregated trainable-parameter update; "
            "omit to disable"
        ),
    )

    parser.add_argument(
        "--parameter-noise-p",
        dest="noise_p",
        type=float,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--aux-noise-p",
        dest="aux_noise_p",
        type=float,
        default=None,
        help=(
            "Add independent Gaussian noise with "
            "sigma=U'_max*2^(-p_aux) "
            "to the aggregated running_mean and running_var "
            "updates; omit to disable"
        ),
    )

    parser.add_argument(
        "--running-noise-p",
        dest="aux_noise_p",
        type=float,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--noise-seed-offset",
        type=int,
        default=1_000_000,
        help=(
            "Offset used with --seed to derive reproducible, "
            "independent parameter/running noise streams"
        ),
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

    validate_sparse_experiment(
        args.dataset,
        args.model,
    )

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

    if args.chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    validate_sparse_rho(args.rho)

    if args.residual_beta < 0:
        raise ValueError(
            "residual_beta must be non-negative"
        )

    if not 1 <= args.minimum_cohort_size <= args.clients:
        raise ValueError(
            "minimum_cohort_size must be between 1 and clients"
        )

    if (
        args.weight_mode == "exact"
        and args.minimum_cohort_size != 1
    ):
        raise ValueError(
            "minimum_cohort_size greater than one currently "
            "requires weight_mode=quantized"
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

    for bound_name, bound_value in (
        (
            "parameter_update_bound",
            args.parameter_update_bound,
        ),
        (
            "running_update_bound",
            args.running_update_bound,
        ),
    ):
        if bound_value is None:
            continue

        if (
            not math.isfinite(float(bound_value))
            or float(bound_value) <= 0
        ):
            raise ValueError(
                f"{bound_name} must be finite and positive"
            )

    parameter_noise_sigma = (
        gaussian_noise_std_from_exponent(
            args.noise_p,
            reference_bound=(
                args.parameter_update_bound
            ),
        )
    )

    running_noise_sigma = (
        gaussian_noise_std_from_exponent(
            args.aux_noise_p,
            reference_bound=(
                args.running_update_bound
            ),
        )
    )

    output_dir = Path(args.output_dir)
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

    train_set, test_set = load_sparse_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        download=not args.no_download,
    )

    partition_seed = (
        args.seed
        if args.partition_seed is None
        else args.partition_seed
    )
    partitions, partition_info = load_or_create_partitions(
        targets=train_set.targets,
        dataset_name=args.dataset,
        num_clients=args.clients,
        partition=args.split,
        dirichlet_alpha=args.alpha,
        partition_seed=partition_seed,
        partition_file=args.partition_file,
        validate_partition_seed=(
            args.partition_seed is not None
        ),
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

    partition_report = partition_document(
        targets=train_set.targets,
        partitions=partitions,
    )
    partition_report["assignment"] = partition_info
    write_json(output_dir / "partition.json", partition_report)

    global_model = build_sparse_model(args.model).to(device)

    full_float_layout = build_float_state_layout(
        global_model
    )

    # Sparse chunks contain trainable parameters only. BatchNorm running
    # statistics are aggregated through the independent dense auxiliary path.
    layout = compact_parameter_layout(
        full_float_layout
    )

    model_numel = layout_numel(layout)

    total_chunk_count = chunk_count(
        total_numel=model_numel,
        chunk_size=args.chunk_size,
    )

    selected_chunk_count = (
        selected_chunk_count_from_ratio(
            total_numel=model_numel,
            chunk_size=args.chunk_size,
            rho=args.rho,
        )
    )

    effective_chunk_ratio = (
        selected_chunk_count
        / total_chunk_count
    )
    validate_sparse_chunk_selection(
        selected_chunk_count=selected_chunk_count,
        total_chunk_count=total_chunk_count,
    )

    communication_profile = "SPARSE"

    wire_value_numel_per_client = (
        selected_chunk_count
        * args.chunk_size
    )

    # v17 uses one MSB-first bit per actual chunk position. Model/update
    # values remain FP32 and sparse chunks retain their B-padding on wire.
    chunk_index_bits = max(
        1,
        (total_chunk_count - 1).bit_length(),
    )
    transmitted_chunk_index_bits = 1
    plaintext_value_bits = 32
    dense_aux_numel_per_client = sum(
        entry.numel
        for entry in full_float_layout
        if not entry.is_parameter
    )

    write_layout_json(
        path=output_dir / "model_layout.json",
        layout=layout,
    )

    configuration = vars(args).copy()

    configuration.update(
        {
            "mode": "sparse-fedavg",
            "aggregation": (
                aggregation_name
            ),
            "compression": (
                "chunk-top-k-error-feedback"
            ),
            "model_numel": model_numel,
            "parameter_numel": model_numel,
            "auxiliary_numel": dense_aux_numel_per_client,
            "selection_count_formula": (
                "ceil(rho*parameter_numel/chunk_size)"
            ),
            "total_chunk_count": (
                total_chunk_count
            ),
            "selected_chunk_count": (
                selected_chunk_count
            ),
            "effective_chunk_ratio": (
                effective_chunk_ratio
            ),
            "wire_value_numel_per_client": (
                wire_value_numel_per_client
            ),
            "candidate_wire_value_numel_per_client": (
                wire_value_numel_per_client
            ),
            "communication_accounting": (
                "v17-bitmap-accepted-values-only-b-padded"
            ),
            "communication_profile": communication_profile,
            "plaintext_payload_padding_included": True,
            "support_codec": SUPPORT_CODEC_NAME,
            "legacy_wire_fields_semantics": (
                "internal-candidate-values-before-cohort-filter-"
                "may-include-padding"
            ),
            "downlink_accounting": (
                "persistent-synchronized-clients-receive-"
                "accepted-sparse-and-dense-aux-aggregate"
            ),
            "chunk_index_bits": transmitted_chunk_index_bits,
            "canonical_chunk_index_bits": chunk_index_bits,
            "plaintext_value_bits": (
                plaintext_value_bits
            ),
            "dense_aux_numel_per_client": (
                dense_aux_numel_per_client
            ),
            "layout_sha256": (
                layout_sha256(layout)
            ),
            "client_sample_counts": (
                sample_counts
            ),
            "parameter_compression": (
                "chunk-top-k-error-feedback"
            ),
            "float_buffer_aggregation": (
                aggregation_name
            ),
            "exact_client_weights": (
                exact_client_weights
            ),
            "effective_client_weights": (
                effective_client_weights
            ),
            "aggregation_alphas": (
                aggregation_alphas.tolist()
                if aggregation_alphas
                is not None
                else None
            ),
            "max_abs_weight_error": (
                max_weight_error
            ),
            "noise_p": (
                args.noise_p
            ),
            "parameter_noise_sigma": (
                parameter_noise_sigma
            ),
            "aux_noise_p": (
                args.aux_noise_p
            ),
            "running_noise_sigma": (
                running_noise_sigma
            ),
            "parameter_noise_target": (
                "trainable-parameter-slots-"
                "inside-selected-chunks"
            ),
            "running_noise_target": (
                "dense-aggregated-running-mean-"
                "and-running-var-deltas"
            ),
        }
    )

    write_json(
        output_dir / "config.json",
        configuration,
    )

    write_json(
        output_dir / "model_stats.json",
        model_storage_stats(
            model=global_model,
            layout=full_float_layout,
        ),
    )

    residuals = [
        torch.zeros(
            model_numel,
            dtype=torch.float32,
            device="cpu",
        )
        for _ in range(args.clients)
    ]

    metrics_path = (
        output_dir / "metrics.csv"
    )

    client_metrics_path = (
        output_dir / "client_metrics.csv"
    )

    round_metrics_jsonl_path = (
        output_dir / "round_metrics.jsonl"
    )

    best_accuracy = float("-inf")
    best_round = -1
    final_accuracy = float("nan")

    observed_parameter_update_max_abs = 0.0
    observed_running_mean_update_max_abs = 0.0
    observed_running_var_update_max_abs = 0.0
    observed_running_update_max_abs = 0.0

    total_candidate_contribution_count = 0
    total_accepted_contribution_count = 0
    total_rejected_contribution_count = 0
    total_accepted_chunk_count = 0
    total_rejected_chunk_count = 0
    total_candidate_index_bits = 0
    total_candidate_index_bytes = 0
    total_selection_bitmap_bytes = 0
    total_admission_bitmap_bytes = 0
    total_upload_support_bitmap_bytes = 0
    total_result_bitmap_bytes = 0
    total_accepted_wire_value_numel = 0
    total_accepted_sparse_fp32_uplink_bytes = 0
    total_dense_aux_fp32_uplink_bytes = 0
    total_accepted_only_fp32_uplink_bytes = 0
    total_accepted_only_logical_downlink_bytes = 0
    total_accepted_only_unicast_downlink_bytes = 0
    total_sequential_round_seconds = 0.0
    total_estimated_parallel_protocol_seconds = 0.0
    total_estimated_parallel_round_seconds = 0.0
    total_client_runtime_sequential_seconds = {
        phase: 0.0 for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
    }
    total_client_runtime_parallel_estimate_seconds = {
        phase: 0.0 for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
    }
    total_server_runtime_seconds = {
        phase: 0.0 for phase in PLAINTEXT_SERVER_RUNTIME_PHASES
    }

    total_started = time.perf_counter()

    _print_run_header(
        args=args,
        parameter_numel=model_numel,
        auxiliary_numel=dense_aux_numel_per_client,
        total_chunk_count=total_chunk_count,
        selected_chunk_count=selected_chunk_count,
        partition_info=partition_info,
        output_dir=output_dir,
    )

    with (
        metrics_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as metrics_handle,
        client_metrics_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as client_handle,
        round_metrics_jsonl_path.open(
            mode="w",
            encoding="utf-8",
        ) as round_jsonl_handle,
    ):
        metric_fields = [
            *ROUND_METRIC_CONTEXT_FIELDS,
            "learning_rate",
            "weighted_train_loss",
            "test_loss",
            "accuracy_percent",
            "rho_requested",
            "rho_selected",
            "rho_eff_round",
            "rho_eff_running",
            "cohort_retention_ratio",
            "effective_chunk_ratio",
            "selected_chunk_count",
            "total_chunk_count",
            "union_chunk_count",
            "union_chunk_ratio",
            "minimum_cohort_size",
            "accepted_chunk_count",
            "accepted_chunk_ratio",
            "rejected_chunk_count",
            "mean_nonzero_cohort_size",

            # Two-phase communication: announce support bitmaps first, then
            # send values only for contributions in accepted cohorts.
            "candidate_contribution_count",
            "accepted_contribution_count",
            "rejected_contribution_count",
            "accepted_contribution_ratio",
            "candidate_index_bits",
            "candidate_index_bytes_per_client",
            "candidate_index_bytes",
            "support_codec",
            "selection_bitmap_bytes",
            "admission_bitmap_bytes",
            "upload_support_bitmap_bytes",
            "result_bitmap_bytes",
            "accepted_wire_value_numel",
            "rejected_wire_value_numel",
            "accepted_sparse_fp32_uplink_bytes",
            "dense_aux_fp32_uplink_bytes",
            "accepted_only_fp32_uplink_bytes",
            "accepted_only_logical_downlink_bytes",
            "accepted_only_unicast_downlink_bytes",

            # Statistics of top-k values sent by all clients.
            "wire_update_count",
            "candidate_wire_update_count",
            "wire_update_mean",
            "wire_update_std",
            "wire_update_min",
            "wire_update_max",
            "wire_update_max_abs",
            "wire_update_l2",
            "wire_update_zero_fraction",
            "wire_update_q01",
            "wire_update_q50",
            "wire_update_q99",

            # Aggregate statistics before noise injection.
            "aggregate_std",
            "aggregate_l2",
            "aggregate_zero_fraction",

            # Trainable-parameter update statistics before noise.
            "parameter_update_mean",
            "parameter_update_std",
            "parameter_update_min",
            "parameter_update_max",
            "parameter_update_max_abs",
            "parameter_update_l2",
            "parameter_update_bound",
            "parameter_bound_ratio",
            "parameter_bound_exceeded",
            "observed_parameter_update_max_abs_so_far",

            # Noise added to the aggregated trainable-parameter update.
            "noise_p",
            "parameter_noise_sigma",
            "parameter_noise_realized_std",
            "parameter_noise_realized_min",
            "parameter_noise_realized_max",
            "parameter_noise_realized_l2",

            # Parameter update statistics after noise.
            "noisy_parameter_update_std",
            "noisy_parameter_update_min",
            "noisy_parameter_update_max",
            "noisy_parameter_update_max_abs",
            "noisy_parameter_update_l2",

            # Dense-aggregated running deltas before noise.
            "running_mean_update_std",
            "running_mean_update_min",
            "running_mean_update_max",
            "running_mean_update_max_abs",
            "running_mean_update_l2",
            "running_var_update_std",
            "running_var_update_min",
            "running_var_update_max",
            "running_var_update_max_abs",
            "running_var_update_l2",
            "running_update_bound",
            "running_mean_bound_ratio",
            "running_var_bound_ratio",
            "running_bound_exceeded",
            "observed_running_mean_update_max_abs_so_far",
            "observed_running_var_update_max_abs_so_far",
            "observed_running_update_max_abs_so_far",

            # Independent Gaussian noise added to running deltas.
            "aux_noise_p",
            "running_noise_sigma",
            "running_mean_noise_realized_std",
            "running_mean_noise_realized_min",
            "running_mean_noise_realized_max",
            "running_mean_noise_realized_l2",
            "running_var_noise_realized_std",
            "running_var_noise_realized_min",
            "running_var_noise_realized_max",
            "running_var_noise_realized_l2",

            # Running deltas after noise injection.
            "noisy_running_mean_update_std",
            "noisy_running_mean_update_min",
            "noisy_running_mean_update_max",
            "noisy_running_mean_update_max_abs",
            "noisy_running_mean_update_l2",
            "noisy_running_var_update_std",
            "noisy_running_var_update_min",
            "noisy_running_var_update_max",
            "noisy_running_var_update_max_abs",
            "noisy_running_var_update_l2",

            # Validate the final running variance.
            "noisy_running_var_value_min",
            "noisy_running_var_negative_count",
            "noisy_running_var_nonpositive_fraction",

            # Local-update and residual statistics.
            "mean_local_delta_std",
            "mean_local_delta_l2",
            "mean_compression_input_std",
            "mean_compression_input_l2",
            "mean_residual_std",
            "mean_residual_l2",
            "max_residual_l2",
            *(
                f"sequential_{phase}_seconds"
                for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
            ),
            *(
                f"parallel_estimate_{phase}_seconds"
                for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
            ),
            *(
                f"server_{phase}_seconds"
                for phase in PLAINTEXT_SERVER_RUNTIME_PHASES
            ),
            "round_seconds",
            "sequential_round_seconds",
            "sequential_client_phase_seconds",
            "maximum_client_processing_seconds",
            "server_processing_seconds",
            "evaluation_seconds",
            "estimated_parallel_protocol_seconds",
            "estimated_parallel_round_seconds",
        ]

        client_fields = [
            "round",
            "client_id",
            "num_examples",
            "learning_rate",
            "average_train_loss",
            "local_delta_std",
            "local_delta_l2",
            "compression_input_std",
            "compression_input_l2",
            "selected_value_l2",
            "wire_update_mean",
            "wire_update_std",
            "wire_update_min",
            "wire_update_max",
            "wire_update_max_abs",
            "wire_update_zero_fraction",
            "wire_update_q01",
            "wire_update_q50",
            "wire_update_q99",

            "residual_std",
            "residual_l2",
            "selected_chunk_count",
            "selected_valid_numel",
            "wire_value_numel",
            "candidate_wire_value_numel",
            *(
                f"{phase}_seconds"
                for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
            ),
            "processing_seconds",
        ]

        metrics_writer = csv.DictWriter(
            metrics_handle,
            fieldnames=metric_fields,
        )

        client_writer = csv.DictWriter(
            client_handle,
            fieldnames=client_fields,
        )

        metrics_writer.writeheader()
        client_writer.writeheader()

        for round_index in range(
            args.rounds
        ):
            round_started = (
                synchronized_perf_counter(device)
            )

            round_number = (
                round_index + 1
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

            # Each client's sparse top-k update.
            sparse_updates = []
            selection_bitmaps: list[bytes] = []
            selection_chunk_ids_by_client: list[tuple[int, ...]] = []

            client_average_losses = []
            client_float_buffers = []

            # Top-k values in the transmitted payload.
            round_wire_values = []

            local_delta_stds = []
            local_delta_l2s = []
            compression_input_stds = []
            compression_input_l2s = []
            residual_stds = []
            residual_l2s = []
            client_runtime_seconds = {
                phase: [] for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
            }

            cohort_sizes = torch.zeros(
                total_chunk_count,
                dtype=torch.int64,
                device="cpu",
            )

            client_processing_seconds = []
            client_phase_started = synchronized_perf_counter(device)

            for client_id, loader in enumerate(
                client_loaders
            ):
                client_started = synchronized_perf_counter(device)
                local_model = copy.deepcopy(
                    global_model
                )

                local_optimizer = build_optimizer(
                    model=local_model,
                    optimizer_name=args.optimizer,
                    learning_rate=learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )

                local_loss_sum = 0.0
                processed_samples = 0
                local_sgd_seconds = 0.0
                delta_generation_seconds = 0.0

                for _local_epoch in range(
                    args.local_epochs
                ):
                    local_sgd_started = synchronized_perf_counter(device)
                    train_metrics = train_one_epoch(
                        model=local_model,
                        loader=loader,
                        optimizer=local_optimizer,
                        device=device,
                    )
                    local_sgd_seconds += (
                        synchronized_perf_counter(device) - local_sgd_started
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

                    # Store local BatchNorm running means and variances.
                    # Aggregate these densely with FedAvg, without sparsification.
                    client_float_buffers.append(
                        clone_float_buffers(
                            local_model
                        )
                    )

                    delta_started = synchronized_perf_counter(device)
                    raw_local_delta = compute_model_delta(
                        local_model=local_model,
                        global_model=global_model,
                        layout=layout,
                        device="cpu",
                    )

                    # Keep only trainable-parameter deltas.
                    # Zero the running-mean and running-variance positions.
                    local_delta = zero_nonparameter_entries(
                        flat_values=raw_local_delta,
                        layout=layout,
                    )

                    delta_generation_seconds += (
                        synchronized_perf_counter(device) - delta_started
                    )

                previous_residual = (
                    residuals[client_id]
                    if args.use_error_feedback
                    else None
                )

                compression_timing: dict[str, float] = {}
                (
                    selection,
                    new_residual,
                    compression_input,
                ) = compress_with_error_feedback(
                    update=local_delta,
                    previous_residual=(
                        previous_residual
                    ),
                    chunk_size=args.chunk_size,
                    selected_chunk_count=(
                        selected_chunk_count
                    ),
                    residual_beta=(
                        args.residual_beta
                    ),
                    timing=compression_timing,
                )

                if not args.use_error_feedback:
                    new_residual = (
                        torch.zeros_like(
                            local_delta
                        )
                    )

                residuals[client_id] = (
                    new_residual
                )


                # Collect candidate top-k statistics. Two-phase accounting
                # announces chunk IDs first, then sends admitted values.
                # Only admitted contributions enter that value payload.
                # The error-feedback residual was already computed by
                # compress_with_error_feedback().
                sparse_update = ClientSparseUpdate(
                    client_id=client_id,
                    round_id=round_index,
                    num_examples=(
                        sample_counts[client_id]
                    ),
                    selection=selection,
                )

                sparse_update.validate()

                sparse_updates.append(
                    sparse_update
                )

                bitmap_started = time.perf_counter()
                selected_chunk_ids = tuple(
                    int(value) for value in selection.chunk_ids.tolist()
                )
                selection_bitmap = encode_support_bitmap(
                    total_chunk_count,
                    selected_chunk_ids,
                )
                decoded_selection = decode_support_bitmap(
                    total_chunk_count,
                    selection_bitmap,
                    expected_popcount=selected_chunk_count,
                )
                if decoded_selection != tuple(sorted(selected_chunk_ids)):
                    raise RuntimeError("selection bitmap round-trip mismatch")
                selection_bitmaps.append(selection_bitmap)
                selection_chunk_ids_by_client.append(selected_chunk_ids)
                selection_bitmap_seconds = (
                    time.perf_counter() - bitmap_started
                )

                cohort_sizes[
                    selection.chunk_ids
                ] += 1

                (
                    local_delta_std,
                    local_delta_l2,
                ) = basic_vector_stats(
                    local_delta
                )

                (
                    compression_input_std,
                    compression_input_l2,
                ) = basic_vector_stats(
                    compression_input
                )

                (
                    residual_std,
                    residual_l2,
                ) = basic_vector_stats(
                    new_residual
                )

                # selection.values is the pre-admission candidate payload.
                #
                # shape:
                #   [selected_chunk_count, chunk_size]
                #
                # Payload shape depends on the active experiment:
                # selected_chunk_count by chunk_size.
                #
                # Includes padding zeros for an incomplete final chunk.
                wire_values = (
                    selection.values
                    .detach()
                    .reshape(-1)
                    .to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    .contiguous()
                )

                wire_statistics = tensor_stats(
                    wire_values
                )

                selected_value_l2 = (
                    wire_statistics["l2"]
                )

                round_wire_values.append(
                    wire_values
                )

                local_delta_stds.append(
                    local_delta_std
                )
                local_delta_l2s.append(
                    local_delta_l2
                )
                compression_input_stds.append(
                    compression_input_std
                )
                compression_input_l2s.append(
                    compression_input_l2
                )
                residual_stds.append(
                    residual_std
                )
                residual_l2s.append(
                    residual_l2
                )

                client_average_losses.append(
                    client_average_loss
                )

                client_seconds = (
                    synchronized_perf_counter(device)
                    - client_started
                )
                client_processing_seconds.append(
                    client_seconds
                )
                client_runtime_values = {
                    "local_sgd": local_sgd_seconds,
                    "delta_generation": delta_generation_seconds,
                    "sparsification_prepare": (
                        compression_timing["compression_prepare_seconds"]
                        + compression_timing[
                            "selection_input_prepare_seconds"
                        ]
                        + compression_timing[
                            "score_input_prepare_seconds"
                        ]
                    ),
                    "chunk_score": compression_timing[
                        "chunk_score_seconds"
                    ],
                    "topk_selection": compression_timing[
                        "topk_seconds"
                    ],
                    "selection_validation": compression_timing[
                        "selection_validation_seconds"
                    ],
                    "residual_generation": compression_timing[
                        "residual_seconds"
                    ],
                    "selection_bitmap": selection_bitmap_seconds,
                    "sparsification_total": compression_timing[
                        "total_seconds"
                    ],
                    "other_client_processing": 0.0,
                }
                accounted_client_seconds = (
                    local_sgd_seconds
                    + delta_generation_seconds
                    + compression_timing["total_seconds"]
                    + selection_bitmap_seconds
                )
                client_runtime_values["other_client_processing"] = max(
                    0.0,
                    client_seconds - accounted_client_seconds,
                )
                for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES:
                    client_runtime_seconds[phase].append(
                        client_runtime_values[phase]
                    )

                client_writer.writerow(
                    {
                        "round": round_number,
                        "client_id": client_id,
                        "num_examples": (
                            sample_counts[
                                client_id
                            ]
                        ),
                        "learning_rate": (
                            learning_rate
                        ),
                        "average_train_loss": (
                            client_average_loss
                        ),
                        "local_delta_std": (
                            local_delta_std
                        ),
                        "local_delta_l2": (
                            local_delta_l2
                        ),
                        "compression_input_std": (
                            compression_input_std
                        ),
                        "compression_input_l2": (
                            compression_input_l2
                        ),
                        "selected_value_l2": (
                            selected_value_l2
                        ),

                        "wire_update_mean": (
                            wire_statistics["mean"]
                        ),
                        "wire_update_std": (
                            wire_statistics["std"]
                        ),
                        "wire_update_min": (
                            wire_statistics["min"]
                        ),
                        "wire_update_max": (
                            wire_statistics["max"]
                        ),
                        "wire_update_max_abs": (
                            wire_statistics["max_abs"]
                        ),
                        "wire_update_zero_fraction": (
                            wire_statistics[
                                "zero_fraction"
                            ]
                        ),
                        "wire_update_q01": (
                            wire_statistics["q01"]
                        ),
                        "wire_update_q50": (
                            wire_statistics["q50"]
                        ),
                        "wire_update_q99": (
                            wire_statistics["q99"]
                        ),
                        "residual_std": (
                            residual_std
                        ),
                        "residual_l2": (
                            residual_l2
                        ),
                        "selected_chunk_count": (
                            selection.selected_chunk_count
                        ),
                        "selected_valid_numel": (
                            selection.selected_valid_numel
                        ),
                        "wire_value_numel": (
                            selection.wire_value_numel
                        ),
                        "candidate_wire_value_numel": (
                            selection.wire_value_numel
                        ),
                        **{
                            f"{phase}_seconds": client_runtime_values[phase]
                            for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
                        },
                        "processing_seconds": (
                            client_seconds
                        ),
                    }
                )

                del local_optimizer
                del local_model

            sequential_client_phase_seconds = (
                synchronized_perf_counter(device)
                - client_phase_started
            )
            server_phase_started = synchronized_perf_counter(device)

            # Concatenate transmitted values from all clients
            # to compute per-round statistics.
            if len(round_wire_values) != args.clients:
                raise RuntimeError(
                    "wire payload count does not match "
                    "the number of clients"
                )

            all_wire_values = torch.cat(
                round_wire_values,
                dim=0,
            )

            expected_wire_value_count = (
                args.clients
                * selected_chunk_count
                * args.chunk_size
            )

            if (
                all_wire_values.numel()
                != expected_wire_value_count
            ):
                raise RuntimeError(
                    "unexpected wire payload size: "
                    f"{all_wire_values.numel()} != "
                    f"{expected_wire_value_count}"
                )

            wire_update_statistics = (
                tensor_stats(
                    all_wire_values
                )
            )


            # =====================================================
            # 1. Aggregate sparse client updates.
            # =====================================================

            server_sparse_aggregation_seconds = 0.0
            server_admission_bitmap_seconds = 0.0
            server_rejected_residual_restore_seconds = 0.0

            if args.weight_mode == "exact":
                # Weight sparse updates by Di/sum(Di).
                aggregation_started = time.perf_counter()
                aggregate = (
                    aggregate_sparse_fedavg(
                        updates=sparse_updates
                    )
                )
                server_sparse_aggregation_seconds = (
                    time.perf_counter() - aggregation_started
                )

                # Aggregate BatchNorm running means and variances
                # with the same Di/sum(Di) weights.
                buffer_states_for_aggregation = (
                    client_float_buffers
                )

                buffer_weights_for_aggregation = (
                    sample_counts
                )

            else:
                # Use integer weights alpha_i/weight_denominator.
                if aggregation_alphas is None:
                    raise RuntimeError(
                        "quantized alphas "
                        "were not initialized"
                    )

                aggregation_started = time.perf_counter()
                sparse_result = (
                    aggregate_sparse_updates(
                        updates=sparse_updates,

                        minimum_cohort_size=(
                            args.minimum_cohort_size
                        ),

                        weight_denominator=(
                            args.weight_denominator
                        ),

                        alphas=(
                            aggregation_alphas
                        ),
                    )
                )
                server_sparse_aggregation_seconds = (
                    time.perf_counter() - aggregation_started
                )

                admission_bitmap_started = time.perf_counter()
                accepted_ids_sorted = tuple(
                    sorted(int(value) for value in sparse_result.accepted_chunk_ids)
                )
                accepted_id_set = set(accepted_ids_sorted)
                admission_bitmap = encode_support_bitmap(
                    total_chunk_count,
                    accepted_ids_sorted,
                )
                if decode_support_bitmap(
                    total_chunk_count,
                    admission_bitmap,
                ) != accepted_ids_sorted:
                    raise RuntimeError("admission bitmap round-trip mismatch")
                for selection_bitmap, selected_ids in zip(
                    selection_bitmaps,
                    selection_chunk_ids_by_client,
                    strict=True,
                ):
                    upload_bitmap = intersect_support_bitmaps(
                        total_chunk_count,
                        selection_bitmap,
                        admission_bitmap,
                    )
                    expected_upload = tuple(
                        sorted(
                            chunk_id
                            for chunk_id in selected_ids
                            if chunk_id in accepted_id_set
                        )
                    )
                    if decode_support_bitmap(
                        total_chunk_count,
                        upload_bitmap,
                    ) != expected_upload:
                        raise RuntimeError("upload-support bitmap mismatch")
                server_admission_bitmap_seconds = (
                    time.perf_counter() - admission_bitmap_started
                )

                if args.use_error_feedback:
                    accepted_chunk_ids = set(
                        sparse_result.accepted_chunk_ids
                    )

                    for sparse_update in sparse_updates:
                        client_id = sparse_update.client_id

                        residual_restore_started = time.perf_counter()
                        residuals[client_id] = (
                            restore_rejected_chunks_to_residual(
                                residual=residuals[client_id],
                                selection=sparse_update.selection,
                                accepted_chunk_ids=(
                                    accepted_chunk_ids
                                ),
                            )
                        )
                        server_rejected_residual_restore_seconds += (
                            time.perf_counter() - residual_restore_started
                        )

                        (
                            residual_stds[client_id],
                            residual_l2s[client_id],
                        ) = basic_vector_stats(
                            residuals[client_id]
                        )

                # The following operations have already been applied.
                #
                # sum(alpha_i * sparse_update_i)
                # --------------------------------
                #        weight_denominator
                aggregate = (
                    sparse_result.dense_update
                )

                # Aggregate BatchNorm running means and variances
                # with the same alpha_i/weight_denominator weights.
                buffer_states_for_aggregation = []
                buffer_weights_for_aggregation = []

                for buffers, alpha in zip(
                    client_float_buffers,
                    aggregation_alphas.tolist(),
                ):
                    integer_alpha = int(alpha)

                    # Clients with zero integer weight contribute nothing in this plaintext experiment.
                    if integer_alpha == 0:
                        continue

                    buffer_states_for_aggregation.append(
                        buffers
                    )

                    buffer_weights_for_aggregation.append(
                        integer_alpha
                    )

            # =====================================================
            # 2. Parameter statistics before Gaussian noise.
            # =====================================================

            # aggregate contains the sum of sparse top-k updates.
            # These are the admitted client contributions.
            # Aggregate Gaussian noise has not yet been added.
            parameter_update = (
                extract_parameter_values(
                    flat_values=aggregate,
                    layout=layout,
                )
            )

            parameter_update_statistics = (
                tensor_stats(
                    parameter_update
                )
            )

            parameter_update_max_abs = float(
                parameter_update_statistics["max_abs"]
            )

            observed_parameter_update_max_abs = max(
                observed_parameter_update_max_abs,
                parameter_update_max_abs,
            )

            if args.parameter_update_bound is None:
                parameter_bound_ratio = None
                parameter_bound_exceeded = False
            else:
                parameter_bound_ratio = (
                    parameter_update_max_abs
                    / args.parameter_update_bound
                )
                parameter_bound_exceeded = (
                    parameter_bound_ratio > 1.0
                )

            if parameter_bound_exceeded:
                print(
                    "Warning: clean parameter update exceeds "
                    f"U_max at round {round_number}: "
                    f"ratio={parameter_bound_ratio:.6g}",
                    flush=True,
                )

            # In the quantized plaintext experiment, only alpha_i>0
            # clients contribute to the aggregate.
            if args.weight_mode == "quantized":
                if aggregation_alphas is None:
                    raise RuntimeError(
                        "quantized alphas "
                        "were not initialized"
                    )

                positive_client_positions = [
                    client_position
                    for client_position, alpha
                    in enumerate(
                        aggregation_alphas.tolist()
                    )
                    if int(alpha) > 0
                ]
            else:
                positive_client_positions = list(
                    range(
                        len(sparse_updates)
                    )
                )

            # Form the union of chunks selected by
            # clients that contribute to the aggregate.
            active_chunk_ids = tuple(
                int(chunk_id)
                for chunk_id in torch.unique(
                    torch.cat(
                        [
                            sparse_updates[
                                client_position
                            ].selection.chunk_ids
                            for client_position in positive_client_positions
                        ]
                    ),
                    sorted=True,
                ).tolist()
            )

            # Inject Gaussian noise only into trainable-parameter
            # coordinates of the selected chunks.
            # Exclude running means and variances from this mask.
            noise_mask = (
                build_selected_parameter_mask(
                    layout=layout,
                    chunk_size=args.chunk_size,
                    selected_chunk_ids=(
                        active_chunk_ids
                    ),
                )
            )

            # =====================================================
            # 3. Add parameter noise with sigma=U_max*2^(-p).
            # =====================================================

            round_noise_seed = (
                args.seed
                + args.noise_seed_offset
                + round_index * 3
            )

            parameter_noise_seed = (
                round_noise_seed
            )

            running_mean_noise_seed = (
                round_noise_seed + 1
            )

            running_var_noise_seed = (
                round_noise_seed + 2
            )

            (
                noisy_aggregate,
                parameter_noise,
            ) = add_masked_gaussian_noise(
                flat_values=aggregate,
                coordinate_mask=noise_mask,
                standard_deviation=(
                    parameter_noise_sigma
                ),
                seed=parameter_noise_seed,
            )

            # Compute noise statistics only on affected coordinates.
            active_parameter_noise = (
                parameter_noise[
                    noise_mask
                ]
            )

            parameter_noise_statistics = (
                tensor_stats(
                    active_parameter_noise
                )
            )

            # Post-noise statistics of all trainable parameters.
            noisy_parameter_update = (
                extract_parameter_values(
                    flat_values=noisy_aggregate,
                    layout=layout,
                )
            )

            noisy_parameter_statistics = (
                tensor_stats(
                    noisy_parameter_update
                )
            )

            # =====================================================
            # 4. Save the previous BatchNorm state.
            # =====================================================

            # Save global running means and variances before aggregation
            # to compute current-minus-previous deltas later.
            previous_float_buffers = (
                clone_float_buffers(
                    global_model
                )
            )

            # =====================================================
            # 5. Apply the sparse parameter update including noise.
            # =====================================================

            apply_model_delta(
                model=global_model,
                delta=noisy_aggregate,
                layout=layout,
                scale=1.0,
            )

            # =====================================================
            # 6. Aggregate BatchNorm running statistics densely.
            # =====================================================

            # Exact plaintext mode:
            #   buffer_weights = Di
            # the helper weights by Di/sum(Di).
            #
            # Quantized plaintext mode:
            #   buffer_weights = alpha_i
            # the helper uses its configured integer-weight normalization.
            #   sum(alpha_i) = weight_denominator
            load_weighted_average_float_buffers(
                model=global_model,
                client_buffers=(
                    buffer_states_for_aggregation
                ),
                sample_counts=(
                    buffer_weights_for_aggregation
                ),
            )

            # =====================================================
            # 7. Aggregated running-delta statistics.
            #    sigma=U'_max*2^(-p_aux) noise
            # =====================================================

            clean_float_buffers = (
                clone_float_buffers(
                    global_model
                )
            )

            # The encrypted quantity is the dense current-minus-previous
            # running delta, not the absolute running state.
            running_mean_update = (
                buffer_delta_vector(
                    previous_buffers=(
                        previous_float_buffers
                    ),
                    current_buffers=(
                        clean_float_buffers
                    ),
                    name_suffix="running_mean",
                )
            )

            running_var_update = (
                buffer_delta_vector(
                    previous_buffers=(
                        previous_float_buffers
                    ),
                    current_buffers=(
                        clean_float_buffers
                    ),
                    name_suffix="running_var",
                )
            )

            running_mean_statistics = tensor_stats(
                running_mean_update
            )

            running_var_statistics = tensor_stats(
                running_var_update
            )

            running_mean_update_max_abs = float(
                running_mean_statistics["max_abs"]
            )

            running_var_update_max_abs = float(
                running_var_statistics["max_abs"]
            )

            observed_running_mean_update_max_abs = max(
                observed_running_mean_update_max_abs,
                running_mean_update_max_abs,
            )

            observed_running_var_update_max_abs = max(
                observed_running_var_update_max_abs,
                running_var_update_max_abs,
            )

            observed_running_update_max_abs = max(
                observed_running_mean_update_max_abs,
                observed_running_var_update_max_abs,
            )

            if args.running_update_bound is None:
                running_mean_bound_ratio = None
                running_var_bound_ratio = None
                running_bound_exceeded = False
            else:
                running_mean_bound_ratio = (
                    running_mean_update_max_abs
                    / args.running_update_bound
                )
                running_var_bound_ratio = (
                    running_var_update_max_abs
                    / args.running_update_bound
                )
                running_bound_exceeded = (
                    max(
                        running_mean_bound_ratio,
                        running_var_bound_ratio,
                    ) > 1.0
                )

            if running_bound_exceeded:
                print(
                    "Warning: clean running update exceeds "
                    f"U'_max at round {round_number}: "
                    f"mean_ratio={running_mean_bound_ratio:.6g}, "
                    f"var_ratio={running_var_bound_ratio:.6g}",
                    flush=True,
                )

            (
                noisy_running_mean_update,
                running_mean_noise,
            ) = add_masked_gaussian_noise(
                flat_values=running_mean_update,
                coordinate_mask=torch.ones_like(
                    running_mean_update,
                    dtype=torch.bool,
                ),
                standard_deviation=(
                    running_noise_sigma
                ),
                seed=running_mean_noise_seed,
            )

            (
                noisy_running_var_update,
                running_var_noise,
            ) = add_masked_gaussian_noise(
                flat_values=running_var_update,
                coordinate_mask=torch.ones_like(
                    running_var_update,
                    dtype=torch.bool,
                ),
                standard_deviation=(
                    running_noise_sigma
                ),
                seed=running_var_noise_seed,
            )

            running_mean_noise_statistics = tensor_stats(
                running_mean_noise
            )

            running_var_noise_statistics = tensor_stats(
                running_var_noise
            )

            noisy_running_mean_statistics = tensor_stats(
                noisy_running_mean_update
            )

            noisy_running_var_statistics = tensor_stats(
                noisy_running_var_update
            )

            # Final buffer = previous global buffer + noisy delta.
            apply_buffer_delta_vector(
                model=global_model,
                previous_buffers=(
                    previous_float_buffers
                ),
                delta=noisy_running_mean_update,
                name_suffix="running_mean",
            )

            apply_buffer_delta_vector(
                model=global_model,
                previous_buffers=(
                    previous_float_buffers
                ),
                delta=noisy_running_var_update,
                name_suffix="running_var",
            )

            final_float_buffers = clone_float_buffers(
                global_model
            )

            noisy_running_var_values = (
                buffer_values_vector(
                    buffers=final_float_buffers,
                    name_suffix="running_var",
                )
            )

            noisy_running_var_value_min = float(
                noisy_running_var_values.min().item()
            )

            noisy_running_var_negative_count = int(
                (
                    noisy_running_var_values < 0
                ).sum().item()
            )

            noisy_running_var_nonpositive_fraction = float(
                (
                    noisy_running_var_values <= 0
                ).to(torch.float64).mean().item()
            )

            if noisy_running_var_negative_count > 0:
                print(
                    "Warning: noisy running_var contains "
                    f"{noisy_running_var_negative_count} "
                    "negative values",
                    flush=True,
                )

            # =====================================================
            # 8. Evaluate the model after both noise paths.
            # =====================================================

            server_before_evaluation_seconds = (
                synchronized_perf_counter(device)
                - server_phase_started
            )

            evaluation = evaluate(
                model=global_model,
                loader=test_loader,
                device=device,
            )
            server_after_evaluation_started = synchronized_perf_counter(
                device
            )

            final_accuracy = (
                evaluation.accuracy_percent
            )

            total_samples = sum(
                sample_counts
            )

            weighted_train_loss = sum(
                loss * count
                for loss, count in zip(
                    client_average_losses,
                    sample_counts,
                )
            ) / total_samples

            nonzero_cohorts = cohort_sizes[
                cohort_sizes > 0
            ]

            communication_counts = (
                sparse_communication_counts(
                    cohort_sizes=cohort_sizes,
                    minimum_cohort_size=(
                        args.minimum_cohort_size
                    ),
                )
            )

            union_chunk_count = (
                communication_counts.union_chunk_count
            )
            accepted_chunk_count = (
                communication_counts.accepted_chunk_count
            )
            rejected_chunk_count = (
                communication_counts.rejected_chunk_count
            )
            candidate_contribution_count = (
                communication_counts.candidate_contribution_count
            )
            accepted_contribution_count = (
                communication_counts.accepted_contribution_count
            )
            rejected_contribution_count = (
                communication_counts.rejected_contribution_count
            )

            expected_candidate_contribution_count = (
                args.clients
                * selected_chunk_count
            )
            if (
                candidate_contribution_count
                != expected_candidate_contribution_count
            ):
                raise RuntimeError(
                    "candidate contribution count mismatch: "
                    f"{candidate_contribution_count} != "
                    f"{expected_candidate_contribution_count}"
                )

            accepted_contribution_ratio = (
                accepted_contribution_count
                / candidate_contribution_count
            )
            sparsity_metrics = effective_sparsity_metrics(
                accepted_contribution_count=accepted_contribution_count,
                client_count=args.clients,
                selected_chunk_count=selected_chunk_count,
                total_chunk_count=total_chunk_count,
            )
            rho_eff_round = float(sparsity_metrics["rho_eff"])
            rho_eff_running = (
                total_accepted_contribution_count
                + accepted_contribution_count
            ) / (
                round_number
                * args.clients
                * total_chunk_count
            )
            plaintext_communication = (
                measure_plaintext_round_communication(
                    profile=communication_profile,
                    cohort_sizes=cohort_sizes.tolist(),
                    minimum_cohort_size=args.minimum_cohort_size,
                    client_count=args.clients,
                    selected_chunk_count=selected_chunk_count,
                    parameter_numel=model_numel,
                    chunk_size=args.chunk_size,
                    auxiliary_numel=dense_aux_numel_per_client,
                )
            )
            candidate_index_bits = plaintext_communication[
                "candidate_index_bits"
            ]
            candidate_index_bytes_per_client = plaintext_communication[
                "candidate_index_bytes_per_client"
            ]
            candidate_index_bytes = plaintext_communication[
                "candidate_index_bytes"
            ]
            selection_bitmap_bytes = plaintext_communication[
                "selection_bitmap_bytes"
            ]
            admission_bitmap_bytes = plaintext_communication[
                "admission_bitmap_bytes"
            ]
            upload_support_bitmap_bytes = plaintext_communication[
                "upload_support_bitmap_bytes"
            ]
            result_bitmap_bytes = plaintext_communication[
                "result_bitmap_bytes"
            ]
            accepted_wire_value_numel = plaintext_communication[
                "accepted_wire_value_numel"
            ]
            rejected_wire_value_numel = plaintext_communication[
                "rejected_wire_value_numel"
            ]
            accepted_sparse_fp32_uplink_bytes = plaintext_communication[
                "accepted_model_fp32_uplink_bytes"
            ]
            dense_aux_fp32_uplink_bytes = plaintext_communication[
                "dense_aux_fp32_uplink_bytes"
            ]
            accepted_only_fp32_uplink_bytes = plaintext_communication[
                "accepted_only_fp32_uplink_bytes"
            ]
            accepted_only_logical_downlink_bytes = plaintext_communication[
                "accepted_only_logical_downlink_bytes"
            ]
            accepted_only_unicast_downlink_bytes = plaintext_communication[
                "accepted_only_unicast_downlink_bytes"
            ]

            total_candidate_contribution_count += (
                candidate_contribution_count
            )
            total_accepted_contribution_count += (
                accepted_contribution_count
            )
            total_rejected_contribution_count += (
                rejected_contribution_count
            )
            total_accepted_chunk_count += (
                accepted_chunk_count
            )
            total_rejected_chunk_count += (
                rejected_chunk_count
            )
            total_candidate_index_bits += (
                candidate_index_bits
            )
            total_candidate_index_bytes += (
                candidate_index_bytes
            )
            total_selection_bitmap_bytes += selection_bitmap_bytes
            total_admission_bitmap_bytes += admission_bitmap_bytes
            total_upload_support_bitmap_bytes += upload_support_bitmap_bytes
            total_result_bitmap_bytes += result_bitmap_bytes
            total_accepted_wire_value_numel += (
                accepted_wire_value_numel
            )
            total_accepted_sparse_fp32_uplink_bytes += (
                accepted_sparse_fp32_uplink_bytes
            )
            total_dense_aux_fp32_uplink_bytes += (
                dense_aux_fp32_uplink_bytes
            )
            total_accepted_only_fp32_uplink_bytes += (
                accepted_only_fp32_uplink_bytes
            )
            total_accepted_only_logical_downlink_bytes += (
                accepted_only_logical_downlink_bytes
            )
            total_accepted_only_unicast_downlink_bytes += (
                accepted_only_unicast_downlink_bytes
            )

            mean_nonzero_cohort_size = float(
                nonzero_cohorts.to(
                    torch.float64
                ).mean().item()
            )

            aggregate_statistics = tensor_stats(
                aggregate
            )

            server_after_evaluation_seconds = (
                synchronized_perf_counter(device)
                - server_after_evaluation_started
            )
            server_processing_seconds = (
                server_before_evaluation_seconds
                + server_after_evaluation_seconds
            )
            maximum_client_processing_seconds = maximum_phase_seconds(
                client_processing_seconds
            )
            round_client_runtime_sequential_seconds = {
                phase: sum(client_runtime_seconds[phase])
                for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
            }
            round_client_runtime_parallel_estimate_seconds = {
                phase: maximum_phase_seconds(client_runtime_seconds[phase])
                for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
            }
            round_server_runtime_seconds = {
                "sparse_aggregation": server_sparse_aggregation_seconds,
                "admission_bitmap": server_admission_bitmap_seconds,
                "rejected_residual_restore": (
                    server_rejected_residual_restore_seconds
                ),
                "other_server_processing": max(
                    0.0,
                    server_processing_seconds
                    - server_sparse_aggregation_seconds
                    - server_admission_bitmap_seconds
                    - server_rejected_residual_restore_seconds,
                ),
            }
            for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES:
                total_client_runtime_sequential_seconds[phase] += (
                    round_client_runtime_sequential_seconds[phase]
                )
                total_client_runtime_parallel_estimate_seconds[phase] += (
                    round_client_runtime_parallel_estimate_seconds[phase]
                )
            for phase in PLAINTEXT_SERVER_RUNTIME_PHASES:
                total_server_runtime_seconds[phase] += (
                    round_server_runtime_seconds[phase]
                )
            estimated_parallel_protocol_seconds = (
                maximum_client_processing_seconds
                + server_processing_seconds
            )
            estimated_parallel_round_seconds = (
                estimated_parallel_protocol_seconds
                + evaluation.duration_seconds
            )
            sequential_round_seconds = (
                synchronized_perf_counter(device)
                - round_started
            )
            total_sequential_round_seconds += sequential_round_seconds
            total_estimated_parallel_protocol_seconds += (
                estimated_parallel_protocol_seconds
            )
            total_estimated_parallel_round_seconds += (
                estimated_parallel_round_seconds
            )

            row = {
                **round_metric_context(
                    round_number=round_number,
                    dataset_name=args.dataset,
                    model_name=args.model,
                ),
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
                "rho_requested": args.rho,
                "rho_selected": sparsity_metrics["rho_selected"],
                "rho_eff_round": rho_eff_round,
                "rho_eff_running": rho_eff_running,
                "cohort_retention_ratio": sparsity_metrics[
                    "cohort_retention_ratio"
                ],
                "effective_chunk_ratio": (
                    effective_chunk_ratio
                ),
                "selected_chunk_count": (
                    selected_chunk_count
                ),
                "total_chunk_count": (
                    total_chunk_count
                ),
                "union_chunk_count": (
                    union_chunk_count
                ),
                "union_chunk_ratio": (
                    union_chunk_count
                    / total_chunk_count
                ),
                "minimum_cohort_size": (
                    args.minimum_cohort_size
                ),
                "accepted_chunk_count": (
                    accepted_chunk_count
                ),
                "accepted_chunk_ratio": (
                    accepted_chunk_count
                    / total_chunk_count
                ),
                "rejected_chunk_count": (
                    rejected_chunk_count
                ),
                "mean_nonzero_cohort_size": (
                    mean_nonzero_cohort_size
                ),
                "candidate_contribution_count": (
                    candidate_contribution_count
                ),
                "accepted_contribution_count": (
                    accepted_contribution_count
                ),
                "rejected_contribution_count": (
                    rejected_contribution_count
                ),
                "accepted_contribution_ratio": (
                    accepted_contribution_ratio
                ),
                "candidate_index_bits": (
                    candidate_index_bits
                ),
                "candidate_index_bytes_per_client": (
                    candidate_index_bytes_per_client
                ),
                "candidate_index_bytes": (
                    candidate_index_bytes
                ),
                "support_codec": SUPPORT_CODEC_NAME,
                "selection_bitmap_bytes": selection_bitmap_bytes,
                "admission_bitmap_bytes": admission_bitmap_bytes,
                "upload_support_bitmap_bytes": upload_support_bitmap_bytes,
                "result_bitmap_bytes": result_bitmap_bytes,
                "accepted_wire_value_numel": (
                    accepted_wire_value_numel
                ),
                "rejected_wire_value_numel": (
                    rejected_wire_value_numel
                ),
                "accepted_sparse_fp32_uplink_bytes": (
                    accepted_sparse_fp32_uplink_bytes
                ),
                "dense_aux_fp32_uplink_bytes": (
                    dense_aux_fp32_uplink_bytes
                ),
                "accepted_only_fp32_uplink_bytes": (
                    accepted_only_fp32_uplink_bytes
                ),
                "accepted_only_logical_downlink_bytes": (
                    accepted_only_logical_downlink_bytes
                ),
                "accepted_only_unicast_downlink_bytes": (
                    accepted_only_unicast_downlink_bytes
                ),

                "wire_update_count": (
                    wire_update_statistics["count"]
                ),
                "candidate_wire_update_count": (
                    wire_update_statistics["count"]
                ),
                "wire_update_mean": (
                    wire_update_statistics["mean"]
                ),
                "wire_update_std": (
                    wire_update_statistics["std"]
                ),
                "wire_update_min": (
                    wire_update_statistics["min"]
                ),
                "wire_update_max": (
                    wire_update_statistics["max"]
                ),
                "wire_update_max_abs": (
                    wire_update_statistics["max_abs"]
                ),
                "wire_update_l2": (
                    wire_update_statistics["l2"]
                ),
                "wire_update_zero_fraction": (
                    wire_update_statistics[
                        "zero_fraction"
                    ]
                ),
                "wire_update_q01": (
                    wire_update_statistics["q01"]
                ),
                "wire_update_q50": (
                    wire_update_statistics["q50"]
                ),
                "wire_update_q99": (
                    wire_update_statistics["q99"]
                ),

                "aggregate_std": (
                    aggregate_statistics["std"]
                ),
                "aggregate_l2": (
                    aggregate_statistics["l2"]
                ),
                "aggregate_zero_fraction": (
                    aggregate_statistics[
                        "zero_fraction"
                    ]
                ),
                "mean_local_delta_std": (
                    sum(local_delta_stds)
                    / len(local_delta_stds)
                ),
                "mean_local_delta_l2": (
                    sum(local_delta_l2s)
                    / len(local_delta_l2s)
                ),
                "mean_compression_input_std": (
                    sum(compression_input_stds)
                    / len(compression_input_stds)
                ),
                "mean_compression_input_l2": (
                    sum(compression_input_l2s)
                    / len(compression_input_l2s)
                ),
                "mean_residual_std": (
                    sum(residual_stds)
                    / len(residual_stds)
                ),
                "mean_residual_l2": (
                    sum(residual_l2s)
                    / len(residual_l2s)
                ),
                "max_residual_l2": max(
                    residual_l2s
                ),
                **{
                    f"sequential_{phase}_seconds": (
                        round_client_runtime_sequential_seconds[phase]
                    )
                    for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
                },
                **{
                    f"parallel_estimate_{phase}_seconds": (
                        round_client_runtime_parallel_estimate_seconds[phase]
                    )
                    for phase in PLAINTEXT_CLIENT_RUNTIME_PHASES
                },
                **{
                    f"server_{phase}_seconds": (
                        round_server_runtime_seconds[phase]
                    )
                    for phase in PLAINTEXT_SERVER_RUNTIME_PHASES
                },
                "round_seconds": (
                    sequential_round_seconds
                ),
                "sequential_round_seconds": sequential_round_seconds,
                "sequential_client_phase_seconds": (
                    sequential_client_phase_seconds
                ),
                "maximum_client_processing_seconds": (
                    maximum_client_processing_seconds
                ),
                "server_processing_seconds": server_processing_seconds,
                "evaluation_seconds": evaluation.duration_seconds,
                "estimated_parallel_protocol_seconds": (
                    estimated_parallel_protocol_seconds
                ),
                "estimated_parallel_round_seconds": (
                    estimated_parallel_round_seconds
                ),
                "parameter_update_mean": (
                    parameter_update_statistics[
                        "mean"
                    ]
                ),
                "parameter_update_std": (
                    parameter_update_statistics[
                        "std"
                    ]
                ),
                "parameter_update_min": (
                    parameter_update_statistics[
                        "min"
                    ]
                ),
                "parameter_update_max": (
                    parameter_update_statistics[
                        "max"
                    ]
                ),
                "parameter_update_max_abs": (
                    parameter_update_statistics[
                        "max_abs"
                    ]
                ),
                "parameter_update_l2": (
                    parameter_update_statistics[
                        "l2"
                    ]
                ),
                "parameter_update_bound": (
                    args.parameter_update_bound
                ),
                "parameter_bound_ratio": (
                    parameter_bound_ratio
                ),
                "parameter_bound_exceeded": (
                    parameter_bound_exceeded
                ),
                "observed_parameter_update_max_abs_so_far": (
                    observed_parameter_update_max_abs
                ),

                "noise_p": (
                    args.noise_p
                ),
                "parameter_noise_sigma": (
                    parameter_noise_sigma
                ),
                "parameter_noise_realized_std": (
                    parameter_noise_statistics["std"]
                ),
                "parameter_noise_realized_min": (
                    parameter_noise_statistics["min"]
                ),
                "parameter_noise_realized_max": (
                    parameter_noise_statistics["max"]
                ),
                "parameter_noise_realized_l2": (
                    parameter_noise_statistics["l2"]
                ),

                "noisy_parameter_update_std": (
                    noisy_parameter_statistics[
                        "std"
                    ]
                ),
                "noisy_parameter_update_min": (
                    noisy_parameter_statistics[
                        "min"
                    ]
                ),
                "noisy_parameter_update_max": (
                    noisy_parameter_statistics[
                        "max"
                    ]
                ),
                "noisy_parameter_update_max_abs": (
                    noisy_parameter_statistics[
                        "max_abs"
                    ]
                ),
                "noisy_parameter_update_l2": (
                    noisy_parameter_statistics[
                        "l2"
                    ]
                ),

                "running_mean_update_std": (
                    running_mean_statistics[
                        "std"
                    ]
                ),
                "running_mean_update_min": (
                    running_mean_statistics[
                        "min"
                    ]
                ),
                "running_mean_update_max": (
                    running_mean_statistics[
                        "max"
                    ]
                ),
                "running_mean_update_max_abs": (
                    running_mean_statistics[
                        "max_abs"
                    ]
                ),
                "running_mean_update_l2": (
                    running_mean_statistics[
                        "l2"
                    ]
                ),

                "running_var_update_std": (
                    running_var_statistics[
                        "std"
                    ]
                ),
                "running_var_update_min": (
                    running_var_statistics[
                        "min"
                    ]
                ),
                "running_var_update_max": (
                    running_var_statistics[
                        "max"
                    ]
                ),
                "running_var_update_max_abs": (
                    running_var_statistics[
                        "max_abs"
                    ]
                ),
                "running_var_update_l2": (
                    running_var_statistics[
                        "l2"
                    ]
                ),
                "running_update_bound": (
                    args.running_update_bound
                ),
                "running_mean_bound_ratio": (
                    running_mean_bound_ratio
                ),
                "running_var_bound_ratio": (
                    running_var_bound_ratio
                ),
                "running_bound_exceeded": (
                    running_bound_exceeded
                ),
                "observed_running_mean_update_max_abs_so_far": (
                    observed_running_mean_update_max_abs
                ),
                "observed_running_var_update_max_abs_so_far": (
                    observed_running_var_update_max_abs
                ),
                "observed_running_update_max_abs_so_far": (
                    observed_running_update_max_abs
                ),

                "aux_noise_p": (
                    args.aux_noise_p
                ),
                "running_noise_sigma": (
                    running_noise_sigma
                ),
                "running_mean_noise_realized_std": (
                    running_mean_noise_statistics["std"]
                ),
                "running_mean_noise_realized_min": (
                    running_mean_noise_statistics["min"]
                ),
                "running_mean_noise_realized_max": (
                    running_mean_noise_statistics["max"]
                ),
                "running_mean_noise_realized_l2": (
                    running_mean_noise_statistics["l2"]
                ),
                "running_var_noise_realized_std": (
                    running_var_noise_statistics["std"]
                ),
                "running_var_noise_realized_min": (
                    running_var_noise_statistics["min"]
                ),
                "running_var_noise_realized_max": (
                    running_var_noise_statistics["max"]
                ),
                "running_var_noise_realized_l2": (
                    running_var_noise_statistics["l2"]
                ),

                "noisy_running_mean_update_std": (
                    noisy_running_mean_statistics["std"]
                ),
                "noisy_running_mean_update_min": (
                    noisy_running_mean_statistics["min"]
                ),
                "noisy_running_mean_update_max": (
                    noisy_running_mean_statistics["max"]
                ),
                "noisy_running_mean_update_max_abs": (
                    noisy_running_mean_statistics["max_abs"]
                ),
                "noisy_running_mean_update_l2": (
                    noisy_running_mean_statistics["l2"]
                ),
                "noisy_running_var_update_std": (
                    noisy_running_var_statistics["std"]
                ),
                "noisy_running_var_update_min": (
                    noisy_running_var_statistics["min"]
                ),
                "noisy_running_var_update_max": (
                    noisy_running_var_statistics["max"]
                ),
                "noisy_running_var_update_max_abs": (
                    noisy_running_var_statistics["max_abs"]
                ),
                "noisy_running_var_update_l2": (
                    noisy_running_var_statistics["l2"]
                ),

                "noisy_running_var_value_min": (
                    noisy_running_var_value_min
                ),
                "noisy_running_var_negative_count": (
                    noisy_running_var_negative_count
                ),
                "noisy_running_var_nonpositive_fraction": (
                    noisy_running_var_nonpositive_fraction
                ),
            }

            metrics_writer.writerow(row)
            round_jsonl_handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            metrics_handle.flush()
            client_handle.flush()
            round_jsonl_handle.flush()

            _print_round_progress(row, args.rounds)

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
                        "mode": "sparse-fedavg",
                        "dataset": args.dataset,
                        "model": args.model,
                        "round": round_number,
                        "split": args.split,
                        "alpha": args.alpha,
                        "partition_seed": partition_info[
                            "partition_seed"
                        ],
                        "partition_file": partition_info[
                            "partition_file"
                        ],
                        "partition_sha256": partition_info[
                            "partition_sha256"
                        ],
                        "rho": args.rho,
                        "chunk_size": (
                            args.chunk_size
                        ),
                        "error_feedback": (
                            args.use_error_feedback
                        ),
                        "minimum_cohort_size": (
                            args.minimum_cohort_size
                        ),
                        "accuracy_percent": (
                            best_accuracy
                        ),
                        "weight_mode": args.weight_mode,
                        "weight_denominator": (
                            args.weight_denominator
                        ),
                        "aggregation_alphas": (
                            aggregation_alphas.tolist()
                            if aggregation_alphas
                            is not None
                            else None
                        ),
                        "noise_p": (
                            args.noise_p
                        ),
                        "parameter_update_bound": (
                            args.parameter_update_bound
                        ),
                        "aux_noise_p": (
                            args.aux_noise_p
                        ),
                        "running_update_bound": (
                            args.running_update_bound
                        ),
                        "seed": args.seed,
                    },
                )

    total_seconds = (
        time.perf_counter()
        - total_started
    )

    profiling_safety_factor = 1.5

    recommended_parameter_update_bound = (
        recommended_safety_scaled_bound(
            observed_parameter_update_max_abs,
            safety_factor=profiling_safety_factor,
        )
    )

    recommended_running_update_bound = (
        recommended_safety_scaled_bound(
            observed_running_update_max_abs,
            safety_factor=profiling_safety_factor,
        )
    )

    save_checkpoint(
        path=output_dir / "final_model.pt",
        model=global_model,
        metadata={
            "mode": "sparse-fedavg",
            "dataset": args.dataset,
            "model": args.model,
            "round": args.rounds,
            "split": args.split,
            "alpha": args.alpha,
            "partition_seed": partition_info["partition_seed"],
            "partition_file": partition_info["partition_file"],
            "partition_sha256": partition_info["partition_sha256"],
            "rho": args.rho,
            "chunk_size": args.chunk_size,
            "error_feedback": (
                args.use_error_feedback
            ),
            "minimum_cohort_size": (
                args.minimum_cohort_size
            ),
            "best_round": best_round,
            "best_accuracy_percent": (
                best_accuracy
            ),
            "final_accuracy_percent": (
                final_accuracy
            ),
            "weight_mode": args.weight_mode,
            "weight_denominator": (
                args.weight_denominator
            ),
            "aggregation_alphas": (
                aggregation_alphas.tolist()
                if aggregation_alphas
                is not None
                else None
            ),
            "noise_p": (
                args.noise_p
            ),
            "parameter_update_bound": (
                args.parameter_update_bound
            ),
            "aux_noise_p": (
                args.aux_noise_p
            ),
            "running_update_bound": (
                args.running_update_bound
            ),
            "seed": args.seed,
        },
    )

    summary = {
        "mode": "sparse-fedavg",
        "dataset": args.dataset,
        "model": args.model,
        "rounds": args.rounds,
        "clients": args.clients,
        "local_epochs": args.local_epochs,
        "split": args.split,
        "alpha": (
            args.alpha
            if args.split == "noniid"
            else None
        ),
        "partition_seed": partition_info["partition_seed"],
        "partition_file": partition_info["partition_file"],
        "partition_source": partition_info["source"],
        "partition_sha256": partition_info["partition_sha256"],
        "rho_requested": args.rho,
        "chunk_size": args.chunk_size,
        "parameter_numel": model_numel,
        "auxiliary_numel": dense_aux_numel_per_client,
        "selection_count_formula": (
            "ceil(rho*parameter_numel/chunk_size)"
        ),
        "total_chunk_count": (
            total_chunk_count
        ),
        "selected_chunk_count": (
            selected_chunk_count
        ),
        "effective_chunk_ratio": (
            effective_chunk_ratio
        ),
        "rho_selected": effective_chunk_ratio,
        "rho_eff": (
            total_accepted_contribution_count
            / args.rounds
            / args.clients
            / total_chunk_count
        ),
        "cohort_retention_ratio": (
            total_accepted_contribution_count
            / total_candidate_contribution_count
        ),
        "wire_value_numel_per_client": (
            wire_value_numel_per_client
        ),
        "candidate_wire_value_numel_per_client": (
            wire_value_numel_per_client
        ),
        "communication_accounting": (
            "v17-bitmap-accepted-values-only-b-padded"
        ),
        "communication_profile": communication_profile,
        "plaintext_payload_padding_included": True,
        "support_codec": SUPPORT_CODEC_NAME,
        "legacy_wire_fields_semantics": (
            "internal-candidate-values-before-cohort-filter-"
            "may-include-padding"
        ),
        "downlink_accounting": (
            "persistent-synchronized-clients-receive-"
            "accepted-sparse-and-dense-aux-aggregate"
        ),
        "communication_scope": (
            "online payload only; excludes serialization headers, "
            "one-time setup, and initial full-model distribution"
        ),
        "chunk_index_bits": transmitted_chunk_index_bits,
        "canonical_chunk_index_bits": chunk_index_bits,
        "plaintext_value_bits": (
            plaintext_value_bits
        ),
        "dense_aux_numel_per_client": (
            dense_aux_numel_per_client
        ),
        "total_candidate_contribution_count": (
            total_candidate_contribution_count
        ),
        "total_accepted_contribution_count": (
            total_accepted_contribution_count
        ),
        "total_rejected_contribution_count": (
            total_rejected_contribution_count
        ),
        "accepted_contribution_ratio": (
            total_accepted_contribution_count
            / total_candidate_contribution_count
        ),
        "average_accepted_contribution_count": (
            total_accepted_contribution_count
            / args.rounds
        ),
        "average_accepted_contribution_count_per_client": (
            total_accepted_contribution_count
            / args.rounds
            / args.clients
        ),
        "total_accepted_chunk_count": (
            total_accepted_chunk_count
        ),
        "average_accepted_chunk_count": (
            total_accepted_chunk_count
            / args.rounds
        ),
        "total_rejected_chunk_count": (
            total_rejected_chunk_count
        ),
        "total_candidate_index_bits": (
            total_candidate_index_bits
        ),
        "total_candidate_index_bytes": (
            total_candidate_index_bytes
        ),
        "total_selection_bitmap_bytes": total_selection_bitmap_bytes,
        "total_admission_bitmap_bytes": total_admission_bitmap_bytes,
        "total_upload_support_bitmap_bytes": (
            total_upload_support_bitmap_bytes
        ),
        "total_result_bitmap_bytes": total_result_bitmap_bytes,
        "total_support_bitmap_uplink_bytes": (
            total_selection_bitmap_bytes + total_upload_support_bitmap_bytes
        ),
        "total_support_bitmap_downlink_bytes": (
            total_admission_bitmap_bytes + total_result_bitmap_bytes
        ),
        "total_accepted_wire_value_numel": (
            total_accepted_wire_value_numel
        ),
        "total_accepted_sparse_fp32_uplink_bytes": (
            total_accepted_sparse_fp32_uplink_bytes
        ),
        "total_dense_aux_fp32_uplink_bytes": (
            total_dense_aux_fp32_uplink_bytes
        ),
        "total_accepted_only_fp32_uplink_bytes": (
            total_accepted_only_fp32_uplink_bytes
        ),
        "total_accepted_only_fp32_uplink_mb": (
            total_accepted_only_fp32_uplink_bytes
            / 1_000_000
        ),
        "average_accepted_only_fp32_uplink_mb_per_round": (
            total_accepted_only_fp32_uplink_bytes
            / args.rounds
            / 1_000_000
        ),
        "total_accepted_only_logical_downlink_bytes": (
            total_accepted_only_logical_downlink_bytes
        ),
        "total_accepted_only_logical_downlink_mb": (
            total_accepted_only_logical_downlink_bytes
            / 1_000_000
        ),
        "total_accepted_only_unicast_downlink_bytes": (
            total_accepted_only_unicast_downlink_bytes
        ),
        "total_accepted_only_unicast_downlink_mb": (
            total_accepted_only_unicast_downlink_bytes
            / 1_000_000
        ),
        "error_feedback_enabled": (
            args.use_error_feedback
        ),
        "residual_beta": (
            args.residual_beta
        ),
        "minimum_cohort_size": (
            args.minimum_cohort_size
        ),
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

        "noise_p": (
            args.noise_p
        ),
        "parameter_update_bound": (
            args.parameter_update_bound
        ),
        "parameter_noise_sigma": (
            parameter_noise_sigma
        ),
        "aux_noise_p": (
            args.aux_noise_p
        ),
        "running_update_bound": (
            args.running_update_bound
        ),
        "running_noise_sigma": (
            running_noise_sigma
        ),
        "noise_seed_offset": (
            args.noise_seed_offset
        ),
        "parameter_noise_target": (
            "trainable-parameter-slots-"
            "inside-selected-chunks"
        ),
        "running_noise_target": (
            "dense-aggregated-running-mean-"
            "and-running-var-deltas"
        ),
        "noise_seed_derivation": (
            "seed+offset+round_index*3+{0,1,2}"
        ),
        "noise_formula": (
            "parameter_update_bound*2^(-noise_p)"
        ),
        "aux_noise_formula": (
            "running_update_bound*2^(-aux_noise_p)"
        ),

        "observed_parameter_update_max_abs": (
            observed_parameter_update_max_abs
        ),
        "observed_running_mean_update_max_abs": (
            observed_running_mean_update_max_abs
        ),
        "observed_running_var_update_max_abs": (
            observed_running_var_update_max_abs
        ),
        "observed_running_update_max_abs": (
            observed_running_update_max_abs
        ),
        "profiling_safety_factor": (
            profiling_safety_factor
        ),
        "recommended_parameter_update_bound": (
            recommended_parameter_update_bound
        ),
        "recommended_running_update_bound": (
            recommended_running_update_bound
        ),

        "best_round": best_round,
        "best_accuracy_percent": (
            best_accuracy
        ),
        "final_accuracy_percent": (
            final_accuracy
        ),
        "timing_model": (
            "barrier-critical-path-independent-clients-v1"
        ),
        "timing_network_latency_included": False,
        "plaintext_sparse_runtime_breakdown": {
            "version": "client-server-phase-v1",
            "client_sequential_total_seconds": dict(
                total_client_runtime_sequential_seconds
            ),
            "client_sequential_average_per_round_seconds": {
                phase: seconds / args.rounds
                for phase, seconds in (
                    total_client_runtime_sequential_seconds.items()
                )
            },
            "client_parallel_estimate_total_seconds": dict(
                total_client_runtime_parallel_estimate_seconds
            ),
            "client_parallel_estimate_average_per_round_seconds": {
                phase: seconds / args.rounds
                for phase, seconds in (
                    total_client_runtime_parallel_estimate_seconds.items()
                )
            },
            "server_total_seconds": dict(total_server_runtime_seconds),
            "server_average_per_round_seconds": {
                phase: seconds / args.rounds
                for phase, seconds in total_server_runtime_seconds.items()
            },
            "semantics": {
                "client_sequential": (
                    "sum across all logical clients and rounds"
                ),
                "client_parallel_estimate": (
                    "per-phase maximum client time summed across rounds; "
                    "diagnostic and not an additive critical path"
                ),
                "sparsification_total": (
                    "wall time of error-feedback preparation, chunk scoring, "
                    "top-k selection, validation, and residual generation"
                ),
                "other_client_processing": (
                    "client processing time not covered by SGD, delta, "
                    "sparsification total, or selection bitmap timing"
                ),
                "other_server_processing": (
                    "server processing time excluding sparse aggregation, "
                    "admission bitmap, rejected residual restoration, and "
                    "model evaluation"
                ),
            },
        },
        "total_sequential_round_seconds": (
            total_sequential_round_seconds
        ),
        "total_estimated_parallel_protocol_seconds": (
            total_estimated_parallel_protocol_seconds
        ),
        "total_estimated_parallel_round_seconds": (
            total_estimated_parallel_round_seconds
        ),
        "average_estimated_parallel_protocol_seconds_per_round": (
            total_estimated_parallel_protocol_seconds
            / args.rounds
        ),
        "average_estimated_parallel_round_seconds": (
            total_estimated_parallel_round_seconds
            / args.rounds
        ),
        "total_seconds": total_seconds,
    }

    observed_bounds = {
        "dataset": args.dataset,
        "model": args.model,
        "rounds": args.rounds,
        "seed": args.seed,
        "noise_p": (
            args.noise_p
        ),
        "aux_noise_p": (
            args.aux_noise_p
        ),
        "observed_parameter_update_max_abs": (
            observed_parameter_update_max_abs
        ),
        "observed_running_mean_update_max_abs": (
            observed_running_mean_update_max_abs
        ),
        "observed_running_var_update_max_abs": (
            observed_running_var_update_max_abs
        ),
        "observed_running_update_max_abs": (
            observed_running_update_max_abs
        ),
        "profiling_safety_factor": (
            profiling_safety_factor
        ),
        "recommended_parameter_update_bound": (
            recommended_parameter_update_bound
        ),
        "recommended_running_update_bound": (
            recommended_running_update_bound
        ),
        "recommendation": (
            "Profile without p/p_aux, then reuse the recommended "
            "fixed bounds in a same-seed noise experiment."
        ),
    }

    write_json(
        output_dir / "observed_bounds.json",
        observed_bounds,
    )

    write_json(
        output_dir / "summary.json",
        summary,
    )

    _print_final_summary(summary, output_dir)



if __name__ == "__main__":
    main()
