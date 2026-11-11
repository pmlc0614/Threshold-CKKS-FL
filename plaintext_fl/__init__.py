"""Plaintext CIFAR-10 federated-learning package.

Public objects are imported lazily so the engine-neutral orchestration and
pybind11 boundary can run on a crypto-only host without importing PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CryptoEngineSetup",
    "CryptoParameterProfile",
    "AggregateReference",
    "AggregateStreamMaxima",
    "EncryptedSparseRoundConfig",
    "EncryptedSparseRoundOutcome",
    "EncryptedSparseTrainingOutcome",
    "PairedEncryptedSparseRoundOutcome",
    "ExperimentConfig",
    "FederatedCryptoEngine",
    "FederatedRoundOrchestrator",
    "NativeBindingUnavailableError",
    "NativeCryptoEngine",
    "NativeCryptoEngineOptions",
    "PlaintextCryptoEngine",
    "ResNet20",
    "ResNet32",
    "resnet20",
    "resnet32",
    "native_binding_available",
    "load_crypto_parameter_profile",
    "run_encrypted_sparse_round",
    "run_encrypted_sparse_rounds",
    "run_paired_encrypted_sparse_round",
]

__version__ = "0.1.0"


_LAZY_EXPORTS = {
    "AggregateReference": (
        ".encrypted_sparse",
        "AggregateReference",
    ),
    "AggregateStreamMaxima": (
        ".encrypted_sparse",
        "AggregateStreamMaxima",
    ),
    "CryptoEngineSetup": (".crypto_engine", "CryptoEngineSetup"),
    "CryptoParameterProfile": (
        ".crypto_parameter_profile",
        "CryptoParameterProfile",
    ),
    "EncryptedSparseRoundConfig": (
        ".encrypted_sparse",
        "EncryptedSparseRoundConfig",
    ),
    "EncryptedSparseRoundOutcome": (
        ".encrypted_sparse",
        "EncryptedSparseRoundOutcome",
    ),
    "EncryptedSparseTrainingOutcome": (
        ".encrypted_sparse",
        "EncryptedSparseTrainingOutcome",
    ),
    "PairedEncryptedSparseRoundOutcome": (
        ".encrypted_sparse",
        "PairedEncryptedSparseRoundOutcome",
    ),
    "ExperimentConfig": (".config", "ExperimentConfig"),
    "FederatedCryptoEngine": (".crypto_engine", "FederatedCryptoEngine"),
    "FederatedRoundOrchestrator": (
        ".round_orchestrator",
        "FederatedRoundOrchestrator",
    ),
    "NativeBindingUnavailableError": (
        ".native_engine",
        "NativeBindingUnavailableError",
    ),
    "NativeCryptoEngine": (".native_engine", "NativeCryptoEngine"),
    "NativeCryptoEngineOptions": (
        ".native_engine",
        "NativeCryptoEngineOptions",
    ),
    "PlaintextCryptoEngine": (".plaintext_engine", "PlaintextCryptoEngine"),
    "ResNet20": (".model", "ResNet20"),
    "ResNet32": (".model", "ResNet32"),
    "native_binding_available": (
        ".native_engine",
        "native_binding_available",
    ),
    "load_crypto_parameter_profile": (
        ".crypto_parameter_profile",
        "load_crypto_parameter_profile",
    ),
    "resnet20": (".model", "resnet20"),
    "resnet32": (".model", "resnet32"),
    "run_encrypted_sparse_round": (
        ".encrypted_sparse",
        "run_encrypted_sparse_round",
    ),
    "run_encrypted_sparse_rounds": (
        ".encrypted_sparse",
        "run_encrypted_sparse_rounds",
    ),
    "run_paired_encrypted_sparse_round": (
        ".encrypted_sparse",
        "run_paired_encrypted_sparse_round",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
