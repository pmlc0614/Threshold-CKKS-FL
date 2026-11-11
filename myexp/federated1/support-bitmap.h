#ifndef FEDERATED1_SUPPORT_BITMAP_H
#define FEDERATED1_SUPPORT_BITMAP_H

#include "protocol-types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace federated {

inline constexpr std::uint8_t kFullSupportCodec = 0x00;
inline constexpr std::uint8_t kChunkBitmapMsb0V1 = 0x01;
inline constexpr const char* kChunkBitmapMsb0V1Name =
    "CHUNK-BITMAP-MSB0-V1";

std::size_t SupportBitmapByteCount(std::size_t chunkCount);

ByteVector EncodeSupportBitmapMsb0(
    std::size_t chunkCount,
    const std::vector<ShuffledChunkLabel>& labels);

std::vector<ShuffledChunkLabel> DecodeSupportBitmapMsb0(
    std::size_t chunkCount,
    const ByteVector& bitmap,
    std::optional<std::size_t> expectedPopcount = std::nullopt);

void ValidateSupportBitmapMsb0(
    std::size_t chunkCount,
    const ByteVector& bitmap,
    std::optional<std::size_t> expectedPopcount = std::nullopt);

bool SupportBitmapContains(
    const ByteVector& bitmap,
    std::size_t chunkCount,
    ShuffledChunkLabel label);

ByteVector IntersectSupportBitmapsMsb0(
    std::size_t chunkCount,
    const ByteVector& left,
    const ByteVector& right);

}  // namespace federated

#endif  // FEDERATED1_SUPPORT_BITMAP_H
