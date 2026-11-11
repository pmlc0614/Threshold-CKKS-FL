"""Canonical v17 sparse-support bitmap codec.

``CHUNK-BITMAP-MSB0-V1`` represents the public chunk/label universe with one
bit per position.  The bit for position ``y`` is the MSB-first bit
``0x80 >> (y % 8)`` in byte ``y // 8``.  Unused low bits in the final byte
must be zero.
"""

from __future__ import annotations

from collections.abc import Iterable


SUPPORT_CODEC_NAME = "CHUNK-BITMAP-MSB0-V1"
SUPPORT_CODEC_CODE = 0x01
_BYTE_SET_BIT_OFFSETS_MSB0 = tuple(
    tuple(
        bit_index
        for bit_index in range(8)
        if byte_value & (0x80 >> bit_index)
    )
    for byte_value in range(256)
)


def bitmap_byte_count(chunk_count: int) -> int:
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int):
        raise TypeError("chunk_count must be an integer")
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    return (chunk_count + 7) // 8


def _validated_bitmap_bytes(
    chunk_count: int,
    bitmap: bytes | bytearray | memoryview,
) -> bytes:
    raw = bytes(bitmap)
    expected_length = bitmap_byte_count(chunk_count)
    if len(raw) != expected_length:
        raise ValueError(
            f"support bitmap must contain exactly {expected_length} bytes"
        )
    used_bits = chunk_count % 8
    if used_bits and raw[-1] & ((1 << (8 - used_bits)) - 1):
        raise ValueError("support bitmap has nonzero unused low bits")
    return raw


def encode_support_bitmap(
    chunk_count: int,
    positions: Iterable[int],
) -> bytes:
    result = bytearray(bitmap_byte_count(chunk_count))
    for raw_position in positions:
        if isinstance(raw_position, bool) or not isinstance(raw_position, int):
            raise TypeError("support positions must be integers")
        if not 0 <= raw_position < chunk_count:
            raise ValueError("support position is out of range")
        byte_index, bit_index = divmod(raw_position, 8)
        mask = 0x80 >> bit_index
        if result[byte_index] & mask:
            raise ValueError("support positions must not contain duplicates")
        result[byte_index] |= mask
    return bytes(result)


def decode_support_bitmap(
    chunk_count: int,
    bitmap: bytes | bytearray | memoryview,
    *,
    expected_popcount: int | None = None,
) -> tuple[int, ...]:
    raw = _validated_bitmap_bytes(chunk_count, bitmap)
    positions = tuple(
        byte_index * 8 + bit_index
        for byte_index, byte_value in enumerate(raw)
        for bit_index in _BYTE_SET_BIT_OFFSETS_MSB0[byte_value]
    )
    if expected_popcount is not None:
        if (
            isinstance(expected_popcount, bool)
            or not isinstance(expected_popcount, int)
            or expected_popcount < 0
        ):
            raise ValueError("expected_popcount must be a non-negative integer")
        if len(positions) != expected_popcount:
            raise ValueError("support bitmap popcount mismatch")
    return positions


def intersect_support_bitmaps(
    chunk_count: int,
    left: bytes | bytearray | memoryview,
    right: bytes | bytearray | memoryview,
) -> bytes:
    left_raw = _validated_bitmap_bytes(chunk_count, left)
    right_raw = _validated_bitmap_bytes(chunk_count, right)
    result = bytes(a & b for a, b in zip(left_raw, right_raw, strict=True))
    _validated_bitmap_bytes(chunk_count, result)
    return result
