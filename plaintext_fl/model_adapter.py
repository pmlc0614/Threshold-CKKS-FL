"""PyTorch adapter for the engine-neutral v17 model representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .crypto_engine import DecryptedRoundResult
from .protocol_contract import LayoutEntry, LayoutKind, TensorLayout
from .round_orchestrator import ClientRoundCandidate
from .sparsify import SparseSelection


@dataclass(frozen=True)
class ModelProtocolLayouts:
    parameter: TensorLayout
    running_mean: TensorLayout
    running_variance: TensorLayout

    def validate(self) -> None:
        self.parameter.validate()
        self.running_mean.validate()
        self.running_variance.validate()
        model_names = {
            self.parameter.model_name,
            self.running_mean.model_name,
            self.running_variance.model_name,
        }
        if len(model_names) != 1:
            raise ValueError("all layouts must describe the same model")


def _make_layout(
    *,
    model_name: str,
    kind: LayoutKind,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> TensorLayout:
    entries: list[LayoutEntry] = []
    offset = 0
    for name, tensor in named_tensors:
        if not tensor.is_floating_point():
            raise TypeError(f"{name} is not floating point")
        numel = tensor.numel()
        entries.append(
            LayoutEntry(
                name=name,
                shape=tuple(int(dimension) for dimension in tensor.shape),
                offset=offset,
                numel=numel,
            )
        )
        offset += numel
    layout = TensorLayout(kind=kind, model_name=model_name, entries=tuple(entries))
    layout.validate()
    return layout


def build_model_protocol_layouts(
    model: nn.Module,
    model_name: str = "resnet20-cifar10",
) -> ModelProtocolLayouts:
    """Split trainable parameters and BatchNorm statistics deterministically."""

    parameters = list(model.named_parameters())
    running_means: list[tuple[str, torch.Tensor]] = []
    running_variances: list[tuple[str, torch.Tensor]] = []
    unsupported_float_buffers: list[str] = []
    for name, buffer in model.named_buffers():
        if not buffer.is_floating_point():
            continue
        if name.endswith(".running_mean"):
            running_means.append((name, buffer))
        elif name.endswith(".running_var"):
            running_variances.append((name, buffer))
        else:
            unsupported_float_buffers.append(name)

    if unsupported_float_buffers:
        raise ValueError(
            "unsupported floating model buffers: "
            + ", ".join(unsupported_float_buffers)
        )
    layouts = ModelProtocolLayouts(
        parameter=_make_layout(
            model_name=model_name,
            kind=LayoutKind.PARAMETER,
            named_tensors=parameters,
        ),
        running_mean=_make_layout(
            model_name=model_name,
            kind=LayoutKind.RUNNING_MEAN,
            named_tensors=running_means,
        ),
        running_variance=_make_layout(
            model_name=model_name,
            kind=LayoutKind.RUNNING_VARIANCE,
            named_tensors=running_variances,
        ),
    )
    layouts.validate()
    return layouts


@torch.no_grad()
def flatten_layout(
    model: nn.Module,
    layout: TensorLayout,
) -> np.ndarray:
    """Copy one protocol layout to a contiguous CPU float32 array."""

    layout.validate()
    state = model.state_dict()
    parts: list[np.ndarray] = []
    for entry in layout.entries:
        if entry.name not in state:
            raise KeyError(f"model has no state entry named {entry.name}")
        tensor = state[entry.name]
        if tuple(tensor.shape) != entry.shape:
            raise ValueError(f"shape mismatch for {entry.name}")
        part = (
            tensor.detach()
            .reshape(-1)
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .copy()
        )
        parts.append(part)
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def compute_layout_delta(
    local_model: nn.Module,
    global_model: nn.Module,
    layout: TensorLayout,
) -> np.ndarray:
    return np.ascontiguousarray(
        flatten_layout(local_model, layout) - flatten_layout(global_model, layout),
        dtype=np.float32,
    )


@torch.no_grad()
def apply_layout_delta(
    model: nn.Module,
    layout: TensorLayout,
    delta: np.ndarray,
) -> None:
    """Apply a decrypted delta to exactly one protocol layout."""

    if not isinstance(delta, np.ndarray) or delta.dtype != np.float32:
        raise TypeError("delta must be a float32 numpy array")
    if delta.shape != (layout.total_numel,) or not delta.flags.c_contiguous:
        raise ValueError("delta has an invalid shape or memory layout")
    if not np.isfinite(delta).all():
        raise ValueError("delta contains NaN or infinity")

    state = model.state_dict()
    updated = dict(state)
    flat_delta = torch.from_numpy(delta)
    for entry in layout.entries:
        target = state[entry.name]
        piece = flat_delta[entry.offset : entry.offset + entry.numel].reshape(
            entry.shape
        )
        updated[entry.name] = target + piece.to(
            device=target.device,
            dtype=target.dtype,
        )
    model.load_state_dict(updated, strict=True)


def candidate_from_models(
    *,
    client_id: int,
    selection: SparseSelection,
    local_model: nn.Module,
    global_model: nn.Module,
    layouts: ModelProtocolLayouts,
) -> ClientRoundCandidate:
    """Package local training output without exposing PyTorch to the engine."""

    selection.validate()
    if selection.total_numel != layouts.parameter.total_numel:
        raise ValueError("selection does not use the parameter layout")
    return ClientRoundCandidate(
        client_id=client_id,
        selected_actual_chunk_ids=tuple(
            int(chunk_id) for chunk_id in selection.chunk_ids.tolist()
        ),
        selected_values=np.ascontiguousarray(
            selection.values.detach().to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        ),
        running_mean_delta=compute_layout_delta(
            local_model,
            global_model,
            layouts.running_mean,
        ),
        running_variance_delta=compute_layout_delta(
            local_model,
            global_model,
            layouts.running_variance,
        ),
    )


def apply_decrypted_result(
    model: nn.Module,
    layouts: ModelProtocolLayouts,
    result: DecryptedRoundResult,
) -> None:
    """Apply all three independently decrypted streams to the global model."""

    apply_layout_delta(model, layouts.parameter, result.parameter_delta)
    apply_layout_delta(model, layouts.running_mean, result.running_mean_delta)
    apply_layout_delta(
        model,
        layouts.running_variance,
        result.running_variance_delta,
    )
