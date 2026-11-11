"""Language-neutral protocol-v17 interoperability contract.

This module deliberately has no PyTorch dependency.  It defines the stable
types and canonical encodings shared by the Python FL orchestrator and the C++
cryptographic engine.  The runtime transport will be pybind11; the binary
header is retained for golden vectors, replay, and debugging.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Sequence


PROTOCOL_VERSION = "v17"
SCHEMA_VERSION_MAJOR = 1
SCHEMA_VERSION_MINOR = 0
BINDING_ABI_VERSION = 22
WEIGHT_DENOMINATOR = 128
WIRE_ENDIANNESS = "little"
WIRE_FLOAT_DTYPE = "float32"
BINARY_MAGIC = b"FLV17BIN"
NO_CLIENT_ID = (1 << 64) - 1


class LayoutKind(str, Enum):
    PARAMETER = "parameter"
    RUNNING_MEAN = "running_mean"
    RUNNING_VARIANCE = "running_variance"


class MessageType(IntEnum):
    SETUP_MANIFEST = 1
    CLIENT_SELECTION = 2
    ADMISSION_RESULT = 3
    ACCEPTED_CHUNK_VALUES = 4
    AUXILIARY_DELTA = 5
    ROUND_RESULT = 6


class RoundPhase(str, Enum):
    CREATED = "created"
    SELECTION_COLLECTING = "selection_collecting"
    ADMISSION_FINALIZED = "admission_finalized"
    VALUES_COLLECTING = "values_collecting"
    ENCRYPTED = "encrypted"
    DECRYPTING = "decrypting"
    COMMITTED = "committed"
    ABORTED = "aborted"


_ALLOWED_ROUND_TRANSITIONS = {
    RoundPhase.CREATED: {
        RoundPhase.SELECTION_COLLECTING,
        RoundPhase.VALUES_COLLECTING,
        RoundPhase.ABORTED,
    },
    RoundPhase.SELECTION_COLLECTING: {
        RoundPhase.ADMISSION_FINALIZED,
        RoundPhase.ABORTED,
    },
    RoundPhase.ADMISSION_FINALIZED: {
        RoundPhase.VALUES_COLLECTING,
        RoundPhase.ABORTED,
    },
    RoundPhase.VALUES_COLLECTING: {
        RoundPhase.ENCRYPTED,
        RoundPhase.ABORTED,
    },
    RoundPhase.ENCRYPTED: {
        RoundPhase.DECRYPTING,
        RoundPhase.ABORTED,
    },
    RoundPhase.DECRYPTING: {
        RoundPhase.COMMITTED,
        RoundPhase.ABORTED,
    },
    RoundPhase.COMMITTED: set(),
    RoundPhase.ABORTED: set(),
}


def is_allowed_round_transition(
    current: RoundPhase,
    following: RoundPhase,
) -> bool:
    return following in _ALLOWED_ROUND_TRANSITIONS[current]


@dataclass(frozen=True)
class LayoutEntry:
    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int
    dtype: str = WIRE_FLOAT_DTYPE

    def validate(self) -> None:
        if not self.name:
            raise ValueError("layout entry name must not be empty")
        if self.dtype != WIRE_FLOAT_DTYPE:
            raise ValueError("v17 wire layouts require float32")
        if self.offset < 0:
            raise ValueError("layout entry offset must be non-negative")
        if self.numel <= 0:
            raise ValueError("layout entry numel must be positive")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("layout entry shape must contain positive dimensions")
        if math.prod(self.shape) != self.numel:
            raise ValueError("layout entry shape/numel mismatch")


@dataclass(frozen=True)
class TensorLayout:
    kind: LayoutKind
    model_name: str
    entries: tuple[LayoutEntry, ...]

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not isinstance(self.kind, LayoutKind):
            raise ValueError("unsupported layout kind")
        if not self.entries:
            raise ValueError("layout must contain at least one entry")

        expected_offset = 0
        names: set[str] = set()
        for entry in self.entries:
            entry.validate()
            if entry.name in names:
                raise ValueError(f"duplicate layout entry: {entry.name}")
            if entry.offset != expected_offset:
                raise ValueError(
                    f"invalid offset for {entry.name}: "
                    f"expected {expected_offset}, got {entry.offset}"
                )
            names.add(entry.name)
            expected_offset += entry.numel

    @property
    def total_numel(self) -> int:
        self.validate()
        return sum(entry.numel for entry in self.entries)

    def canonical_bytes(self) -> bytes:
        """Return the canonical little-endian layout representation."""

        self.validate()
        kind_code = {
            LayoutKind.PARAMETER: 1,
            LayoutKind.RUNNING_MEAN: 2,
            LayoutKind.RUNNING_VARIANCE: 3,
        }[self.kind]

        model_bytes = self.model_name.encode("utf-8")
        result = bytearray(b"FLV17LAY")
        result.extend(struct.pack("<HHB", SCHEMA_VERSION_MAJOR, SCHEMA_VERSION_MINOR, kind_code))
        result.extend(struct.pack("<I", len(model_bytes)))
        result.extend(model_bytes)
        result.extend(struct.pack("<Q", len(self.entries)))

        for entry in self.entries:
            name_bytes = entry.name.encode("utf-8")
            result.extend(struct.pack("<I", len(name_bytes)))
            result.extend(name_bytes)
            result.extend(struct.pack("<B", 1))  # float32
            result.extend(struct.pack("<I", len(entry.shape)))
            for dimension in entry.shape:
                result.extend(struct.pack("<Q", dimension))
            result.extend(struct.pack("<QQ", entry.offset, entry.numel))

        return bytes(result)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class RegisteredClientWeight:
    client_id: int
    example_count: int

    def validate(self) -> None:
        if not 0 <= self.client_id <= NO_CLIENT_ID - 1:
            raise ValueError("client_id must fit uint64 and not use the sentinel")
        if not 0 < self.example_count <= NO_CLIENT_ID:
            raise ValueError("example_count must be a positive uint64")


def _validate_uint64(value: int, name: str, *, allow_sentinel: bool = False) -> None:
    maximum = NO_CLIENT_ID if allow_sentinel else NO_CLIENT_ID - 1
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must fit uint64")


def _validate_unique_chunk_ids(chunk_ids: Sequence[int], name: str) -> None:
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"{name} must not contain duplicates")
    for chunk_id in chunk_ids:
        _validate_uint64(chunk_id, name)


@dataclass(frozen=True)
class ClientSelectionRequest:
    setup_id: bytes
    round_id: int
    client_id: int
    parameter_layout_sha256: str
    selected_actual_chunk_ids: tuple[int, ...]

    def validate(self) -> None:
        if len(self.setup_id) != 16:
            raise ValueError("setup_id must contain 16 bytes")
        _validate_uint64(self.round_id, "round_id", allow_sentinel=True)
        _validate_uint64(self.client_id, "client_id")
        _decode_hash(self.parameter_layout_sha256, "parameter_layout_sha256")
        if not self.selected_actual_chunk_ids:
            raise ValueError("selected_actual_chunk_ids must not be empty")
        _validate_unique_chunk_ids(
            self.selected_actual_chunk_ids,
            "selected_actual_chunk_ids",
        )


@dataclass(frozen=True)
class ClientAdmissionResult:
    setup_id: bytes
    round_id: int
    client_id: int
    accepted_actual_chunk_ids: tuple[int, ...]

    def validate(self) -> None:
        if len(self.setup_id) != 16:
            raise ValueError("setup_id must contain 16 bytes")
        _validate_uint64(self.round_id, "round_id", allow_sentinel=True)
        _validate_uint64(self.client_id, "client_id")
        _validate_unique_chunk_ids(
            self.accepted_actual_chunk_ids,
            "accepted_actual_chunk_ids",
        )


@dataclass(frozen=True)
class AcceptedChunkValues:
    setup_id: bytes
    round_id: int
    client_id: int
    chunk_size: int
    actual_chunk_ids: tuple[int, ...]
    values: tuple[float, ...]

    def validate(self) -> None:
        if len(self.setup_id) != 16:
            raise ValueError("setup_id must contain 16 bytes")
        _validate_uint64(self.round_id, "round_id", allow_sentinel=True)
        _validate_uint64(self.client_id, "client_id")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        _validate_unique_chunk_ids(self.actual_chunk_ids, "actual_chunk_ids")
        if len(self.values) != len(self.actual_chunk_ids) * self.chunk_size:
            raise ValueError("accepted chunk value count does not match IDs and chunk_size")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("accepted chunk values must be finite")


@dataclass(frozen=True)
class AuxiliaryDeltaPayload:
    setup_id: bytes
    round_id: int
    client_id: int
    running_mean_delta: tuple[float, ...]
    running_variance_delta: tuple[float, ...]

    def validate(self) -> None:
        if len(self.setup_id) != 16:
            raise ValueError("setup_id must contain 16 bytes")
        _validate_uint64(self.round_id, "round_id", allow_sentinel=True)
        _validate_uint64(self.client_id, "client_id")
        if not self.running_mean_delta or not self.running_variance_delta:
            raise ValueError("both auxiliary delta streams must be present")
        if any(
            not math.isfinite(value)
            for value in self.running_mean_delta + self.running_variance_delta
        ):
            raise ValueError("auxiliary deltas must be finite")


def quantize_registered_client_weights(
    clients: Sequence[RegisteredClientWeight],
    denominator: int = WEIGHT_DENOMINATOR,
) -> tuple[int, ...]:
    """Quantize registered sample ratios using exact largest remainder.

    The result stays aligned with ``clients``.  Remainder ties use ascending
    client ID.  The v17 profile assumes every largest-remainder result is
    strictly positive; inputs violating that setup condition are rejected.
    """

    if not 0 < denominator <= 0xFFFFFFFF:
        raise ValueError("denominator must be a positive uint32")
    if not clients:
        raise ValueError("clients must not be empty")
    if len(clients) > denominator:
        raise ValueError("positive weights are impossible when clients exceed denominator")

    seen: set[int] = set()
    for client in clients:
        client.validate()
        if client.client_id in seen:
            raise ValueError(f"duplicate client_id: {client.client_id}")
        seen.add(client.client_id)

    total_examples = sum(client.example_count for client in clients)
    if total_examples > NO_CLIENT_ID:
        raise ValueError("total example count overflows uint64")
    if any(client.example_count > NO_CLIENT_ID // denominator for client in clients):
        raise ValueError("weight numerator overflows uint64")
    numerators = [client.example_count * denominator for client in clients]
    alphas = [numerator // total_examples for numerator in numerators]
    remainders = [numerator % total_examples for numerator in numerators]
    remaining = denominator - sum(alphas)

    order = sorted(
        range(len(clients)),
        key=lambda position: (
            -remainders[position],
            clients[position].client_id,
        ),
    )
    for position in order[:remaining]:
        alphas[position] += 1

    if sum(alphas) != denominator:
        raise RuntimeError("quantized weights do not sum to denominator")
    if any(alpha < 1 for alpha in alphas):
        raise ValueError(
            "v17 requires alpha_i >= 1; increase the denominator "
            "or revise the registered client set"
        )

    return tuple(alphas)


_BINARY_HEADER = struct.Struct("<8sHHHH16sQQ32s32s32sQ32s")
BINARY_HEADER_SIZE = _BINARY_HEADER.size


def _decode_hash(value: str, field_name: str) -> bytes:
    if len(value) != 64:
        raise ValueError(f"{field_name} must contain 64 hexadecimal characters")
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{field_name} is not hexadecimal") from error
    if len(result) != 32:
        raise ValueError(f"{field_name} must decode to 32 bytes")
    return result


@dataclass(frozen=True)
class BinaryHeader:
    message_type: MessageType
    setup_id: bytes
    round_id: int
    client_id: int
    parameter_layout_sha256: str
    running_mean_layout_sha256: str
    running_variance_layout_sha256: str
    payload_length: int
    payload_sha256: str
    flags: int = 0

    def validate(self) -> None:
        if len(self.setup_id) != 16:
            raise ValueError("setup_id must contain 16 bytes")
        for name, value in (("round_id", self.round_id), ("client_id", self.client_id)):
            if not 0 <= value <= NO_CLIENT_ID:
                raise ValueError(f"{name} must fit uint64")
        if self.flags != 0:
            raise ValueError("schema 1.0 requires flags=0")
        if not 0 <= self.payload_length <= NO_CLIENT_ID:
            raise ValueError("payload_length must fit uint64")
        _decode_hash(self.parameter_layout_sha256, "parameter_layout_sha256")
        _decode_hash(self.running_mean_layout_sha256, "running_mean_layout_sha256")
        _decode_hash(self.running_variance_layout_sha256, "running_variance_layout_sha256")
        _decode_hash(self.payload_sha256, "payload_sha256")

    def to_bytes(self) -> bytes:
        self.validate()
        return _BINARY_HEADER.pack(
            BINARY_MAGIC,
            SCHEMA_VERSION_MAJOR,
            SCHEMA_VERSION_MINOR,
            int(self.message_type),
            self.flags,
            self.setup_id,
            self.round_id,
            self.client_id,
            _decode_hash(self.parameter_layout_sha256, "parameter_layout_sha256"),
            _decode_hash(self.running_mean_layout_sha256, "running_mean_layout_sha256"),
            _decode_hash(self.running_variance_layout_sha256, "running_variance_layout_sha256"),
            self.payload_length,
            _decode_hash(self.payload_sha256, "payload_sha256"),
        )

    @classmethod
    def for_payload(
        cls,
        *,
        message_type: MessageType,
        setup_id: bytes,
        round_id: int,
        client_id: int,
        parameter_layout_sha256: str,
        running_mean_layout_sha256: str,
        running_variance_layout_sha256: str,
        payload: bytes,
        flags: int = 0,
    ) -> "BinaryHeader":
        return cls(
            message_type=message_type,
            setup_id=setup_id,
            round_id=round_id,
            client_id=client_id,
            parameter_layout_sha256=parameter_layout_sha256,
            running_mean_layout_sha256=running_mean_layout_sha256,
            running_variance_layout_sha256=running_variance_layout_sha256,
            payload_length=len(payload),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            flags=flags,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BinaryHeader":
        if len(data) != BINARY_HEADER_SIZE:
            raise ValueError(f"binary header must contain {BINARY_HEADER_SIZE} bytes")
        (
            magic,
            schema_major,
            schema_minor,
            message_type,
            flags,
            setup_id,
            round_id,
            client_id,
            parameter_hash,
            mean_hash,
            variance_hash,
            payload_length,
            payload_hash,
        ) = _BINARY_HEADER.unpack(data)
        if magic != BINARY_MAGIC:
            raise ValueError("invalid binary magic")
        if (schema_major, schema_minor) != (SCHEMA_VERSION_MAJOR, SCHEMA_VERSION_MINOR):
            raise ValueError("unsupported binary schema version")
        try:
            parsed_type = MessageType(message_type)
        except ValueError as error:
            raise ValueError("unsupported message type") from error
        result = cls(
            message_type=parsed_type,
            setup_id=setup_id,
            round_id=round_id,
            client_id=client_id,
            parameter_layout_sha256=parameter_hash.hex(),
            running_mean_layout_sha256=mean_hash.hex(),
            running_variance_layout_sha256=variance_hash.hex(),
            payload_length=payload_length,
            payload_sha256=payload_hash.hex(),
            flags=flags,
        )
        result.validate()
        return result
