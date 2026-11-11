"""SD v17 byte reference and public setup binding (no private setup material).

The documents fix tags/BE64/MSB0. The SDV17PUB envelope and MSB-packed
coefficient container are implementation conventions, not a network transport.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import struct


def be64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError("expected unsigned 64-bit integer")
    return value.to_bytes(8, "big")


def _fixed(value: bytes, size: int) -> bytes:
    value = bytes(value)
    if len(value) != size:
        raise ValueError(f"expected {size} bytes")
    return value


def local_a_tag(setup_id: bytes, round_id: int, client_id: int, block: int, path: str) -> bytes:
    domains = {"FULL-MODEL": b"ENC-FULL-A", "SP-MODEL": b"ENC-SP-A", "AUX": b"ENC-AUX-A"}
    if path not in domains:
        raise ValueError("invalid path")
    return domains[path] + _fixed(setup_id, 16) + be64(round_id) + be64(client_id) + be64(block)


def dd_tag(setup_id: bytes, round_id: int, instance: int, path: str, client_id: int | None = None) -> bytes:
    if path not in {"FULL-MODEL", "SP-MODEL", "AUX"}:
        raise ValueError("invalid path")
    context = _fixed(setup_id, 16) + be64(round_id) + path.encode("ascii") + be64(instance)
    return (b"C2-OFFSET" + context if client_id is None
            else b"C2-DITH" + context + be64(client_id))


def _words(address: bytes):
    xof = hashlib.shake_128(address)
    length = 256
    data = xof.digest(length)
    offset = 0
    while True:
        if offset == length:
            length *= 2
            data = xof.digest(length)
        yield int.from_bytes(data[offset:offset + 8], "big")
        offset += 8


def uniform_polynomial(seed: bytes, tag: bytes, dimension: int, prime: int) -> tuple[int, ...]:
    if dimension <= 0 or not 1 < prime < 2**64:
        raise ValueError("invalid uniform polynomial parameters")
    stream = _words(b"UPOLY" + _fixed(seed, 16) + tag + be64(dimension) + be64(prime))
    limit = (2**64 // prime) * prime
    result = []
    while len(result) < dimension:
        word = next(stream)
        if word < limit:
            result.append(word % prime)
    return tuple(result)


def index_seed(key: bytes, setup_id: bytes, round_id: int, chunks: int) -> bytes:
    if chunks <= 0:
        raise ValueError("chunk count must be positive")
    return hmac.digest(_fixed(key, 16), b"IDX-PERM" + _fixed(setup_id, 16)
                       + be64(round_id) + be64(chunks), "sha256")[:16]


def index_permutation(key: bytes, setup_id: bytes, round_id: int, chunks: int) -> tuple[int, ...]:
    stream = _words(b"IDX-FY" + index_seed(key, setup_id, round_id, chunks) + be64(chunks))
    values = list(range(chunks))
    for upper in range(chunks, 1, -1):
        limit = (2**64 // upper) * upper
        word = next(stream)
        while word >= limit:
            word = next(stream)
        j = word % upper
        values[upper - 1], values[j] = values[j], values[upper - 1]
    return tuple(values)


def pack_codes(values, bits: int) -> bytes:
    if not 1 <= bits <= 63:
        raise ValueError("invalid code width")
    values = tuple(values)
    accumulator = 0
    for value in values:
        if not isinstance(value, int) or not 0 <= value < 2**bits:
            raise ValueError("code exceeds width")
        accumulator = (accumulator << bits) | value
    padding = (-len(values) * bits) % 8
    return (accumulator << padding).to_bytes((len(values) * bits + 7) // 8, "big")


def unpack_codes(payload: bytes, count: int, bits: int) -> tuple[int, ...]:
    if count < 0 or not 1 <= bits <= 63 or len(payload) != (count * bits + 7) // 8:
        raise ValueError("noncanonical code length")
    padding = (-count * bits) % 8
    integer = int.from_bytes(payload, "big")
    if integer & ((1 << padding) - 1):
        raise ValueError("nonzero code padding")
    integer >>= padding
    return tuple((integer >> (bits * (count - i - 1))) & ((1 << bits) - 1) for i in range(count))


def canonical_parameter_descriptor(effective: dict) -> bytes:
    """Independent encoder of the native parameter descriptor, in field order."""
    e = effective
    if e["protocol_version"] != "v17" or e["protocol_mode"] not in {"SPARSE", "FULL"}:
        raise ValueError("unsupported protocol")
    def text(s):
        data = s.encode("ascii")
        return be64(len(data)) + data
    def real(v):
        if not math.isfinite(v):
            raise ValueError("nonfinite descriptor real")
        return struct.pack(">d", v)
    out = text("FEDERATED-SETUP-DESCRIPTOR-2") + text("v17")
    sparse = e["protocol_mode"] == "SPARSE"
    out += be64(0 if sparse else 1) + bytes([int(sparse)])
    for name in ("client_count", "model_dimension", "ring_dimension"):
        out += be64(e[name])
    for name in ("q_primes", "p_primes"):
        out += be64(len(e[name])) + b"".join(be64(v) for v in e[name])
    for name in ("hybrid_large_digits", "maximum_qp_bits", "mlwe_rank", "weight_denominator"):
        out += be64(e[name])
    if len(e["setup_noise_sigmas"]) != 4:
        raise ValueError("four setup sigmas required")
    out += b"".join(real(v) for v in e["setup_noise_sigmas"]) + text(e["sampler_id"])
    out += be64(e["auxiliary_dimension"])
    for name in ("model_scaling_factor", "aux_scaling_factor"):
        value = e[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**256:
            raise ValueError("invalid exact integer scale")
        out += text(str(value))
    for name in ("minimum_cohort_size",
                 "collusion_bound", "error_feedback_enabled", "precision_bits", "aux_precision_bits",
                 "compression_c1", "compression_ca", "compression_c2", "compression_cd",
                 "aux_compression_c1", "aux_compression_ca", "aux_compression_c2", "aux_compression_cd"):
        out += be64(int(e[name]))
    for name in ("selection_ratio", "update_bound", "aux_update_bound", "smudging_sigma", "aux_smudging_sigma",
                 "model_delivery_scale", "aux_delivery_scale"):
        out += real(e[name])
    if e.get("experimental_dynamic_cohort", False):
        out += text("v17-post-upload-pre-request-dropout")
        out += be64(e["dropout_attempt_cap"]) + be64(e["dropout_target_cap"])
    return out


def serialize_public_setup(setup_id: bytes, crs_seed: bytes, descriptor: bytes, external_ids, layout_hashes) -> bytes:
    ids = tuple(external_ids)
    hashes = tuple(layout_hashes)
    if not descriptor or not ids or len(set(ids)) != len(ids) or len(hashes) != 3:
        raise ValueError("invalid public setup binding")
    out = b"SDV17PUB" + be64(1) + _fixed(setup_id, 16) + _fixed(crs_seed, 16)
    out += be64(len(descriptor)) + bytes(descriptor) + be64(len(ids))
    out += b"".join(be64(position) + be64(external_id) for position, external_id in enumerate(ids))
    return out + b"".join(_fixed(h, 32) for h in hashes)


def parse_public_setup(payload: bytes) -> dict:
    payload = bytes(payload)
    offset = 0
    def take(n):
        nonlocal offset
        if n < 0 or n > len(payload) - offset:
            raise ValueError("truncated public setup binding")
        result = payload[offset:offset + n]
        offset += n
        return result
    def integer(): return int.from_bytes(take(8), "big")
    if take(8) != b"SDV17PUB" or integer() != 1:
        raise ValueError("unsupported public setup binding")
    setup_id, crs_seed = take(16), take(16)
    descriptor = take(integer())
    count = integer()
    if not descriptor or count == 0 or count > (len(payload) - offset - 96) // 16:
        raise ValueError("invalid public setup roster")
    ids = []
    for position in range(count):
        if integer() != position:
            raise ValueError("noncanonical crypto client ID")
        ids.append(integer())
    hashes = tuple(take(32) for _ in range(3))
    if offset != len(payload) or len(set(ids)) != len(ids):
        raise ValueError("noncanonical public setup binding")
    return {"setup_id": setup_id, "crs_seed": crs_seed, "descriptor": descriptor,
            "external_ids": tuple(ids), "layout_hashes": hashes}
