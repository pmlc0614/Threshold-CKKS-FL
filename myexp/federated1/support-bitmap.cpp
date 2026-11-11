#include "support-bitmap.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace federated {
namespace {

std::uint8_t LabelMask(std::size_t label) {
    return static_cast<std::uint8_t>(0x80U >> (label % 8U));
}

}  // namespace

std::size_t SupportBitmapByteCount(std::size_t chunkCount) {
    if (chunkCount == 0) {
        throw std::invalid_argument("support bitmap chunk count must be positive");
    }
    if (chunkCount > std::numeric_limits<std::size_t>::max() - 7U) {
        throw std::overflow_error("support bitmap byte count overflows size_t");
    }
    return (chunkCount + 7U) / 8U;
}

void ValidateSupportBitmapMsb0(
    std::size_t chunkCount,
    const ByteVector& bitmap,
    std::optional<std::size_t> expectedPopcount) {

    if (bitmap.size() != SupportBitmapByteCount(chunkCount)) {
        throw std::invalid_argument("support bitmap has an invalid byte length");
    }
    const std::size_t usedBitsInLastByte = chunkCount % 8U;
    if (usedBitsInLastByte != 0) {
        const std::uint8_t unusedMask = static_cast<std::uint8_t>(
            (1U << (8U - usedBitsInLastByte)) - 1U);
        if ((bitmap.back() & unusedMask) != 0) {
            throw std::invalid_argument(
                "support bitmap has nonzero unused low bits");
        }
    }
    if (expectedPopcount.has_value()) {
        std::size_t popcount = 0;
        for (std::size_t label = 0; label < chunkCount; ++label) {
            if ((bitmap[label / 8U] & LabelMask(label)) != 0) {
                ++popcount;
            }
        }
        if (popcount != *expectedPopcount) {
            throw std::invalid_argument("support bitmap popcount mismatch");
        }
    }
}

ByteVector EncodeSupportBitmapMsb0(
    std::size_t chunkCount,
    const std::vector<ShuffledChunkLabel>& labels) {

    ByteVector bitmap(SupportBitmapByteCount(chunkCount), 0);
    for (const auto label : labels) {
        if (label.value >= chunkCount) {
            throw std::invalid_argument("support bitmap label is out of range");
        }
        const std::size_t byteIndex = static_cast<std::size_t>(label.value / 8U);
        const std::uint8_t mask = LabelMask(static_cast<std::size_t>(label.value));
        if ((bitmap[byteIndex] & mask) != 0) {
            throw std::invalid_argument("support bitmap contains a duplicate label");
        }
        bitmap[byteIndex] = static_cast<std::uint8_t>(bitmap[byteIndex] | mask);
    }
    return bitmap;
}

std::vector<ShuffledChunkLabel> DecodeSupportBitmapMsb0(
    std::size_t chunkCount,
    const ByteVector& bitmap,
    std::optional<std::size_t> expectedPopcount) {

    ValidateSupportBitmapMsb0(chunkCount, bitmap, expectedPopcount);
    std::vector<ShuffledChunkLabel> labels;
    labels.reserve(expectedPopcount.value_or(0));
    for (std::size_t label = 0; label < chunkCount; ++label) {
        if ((bitmap[label / 8U] & LabelMask(label)) != 0) {
            labels.push_back(ShuffledChunkLabel{label});
        }
    }
    return labels;
}

bool SupportBitmapContains(
    const ByteVector& bitmap,
    std::size_t chunkCount,
    ShuffledChunkLabel label) {

    ValidateSupportBitmapMsb0(chunkCount, bitmap);
    if (label.value >= chunkCount) {
        throw std::out_of_range("support bitmap label is out of range");
    }
    return (bitmap[static_cast<std::size_t>(label.value / 8U)] &
            LabelMask(static_cast<std::size_t>(label.value))) != 0;
}

ByteVector IntersectSupportBitmapsMsb0(
    std::size_t chunkCount,
    const ByteVector& left,
    const ByteVector& right) {

    ValidateSupportBitmapMsb0(chunkCount, left);
    ValidateSupportBitmapMsb0(chunkCount, right);
    ByteVector result(left.size(), 0);
    std::transform(
        left.begin(), left.end(), right.begin(), result.begin(),
        [](std::uint8_t a, std::uint8_t b) {
            return static_cast<std::uint8_t>(a & b);
        });
    ValidateSupportBitmapMsb0(chunkCount, result);
    return result;
}

}  // namespace federated
