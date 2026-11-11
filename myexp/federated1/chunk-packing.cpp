#include "chunk-packing.h"

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace federated {
namespace {

size_t CeilDivide(
    size_t numerator,
    size_t denominator) {

    if (denominator == 0) {
        throw std::invalid_argument(
            "CeilDivide: denominator is zero");
    }

    return numerator / denominator +
           (numerator % denominator != 0 ? 1 : 0);
}

void ValidatePackingConfig(
    const FederatedConfig& config,
    const char* functionName) {

    if (config.modelDimension == 0) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": modelDimension is zero");
    }

    if (config.ringDimension == 0) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": ringDimension is zero");
    }

    if (config.mlweRank == 0) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": mlweRank is zero");
    }

    if (config.ringDimension %
            config.mlweRank !=
        0) {

        throw std::invalid_argument(
            std::string(functionName) +
            ": ringDimension is not divisible "
            "by mlweRank");
    }
}

size_t GetChunkSize(
    const FederatedConfig& config) {

    return config.ringDimension /
           config.mlweRank;
}

size_t GetChunkCount(
    const FederatedConfig& config) {

    return CeilDivide(
        config.modelDimension,
        GetChunkSize(config));
}

void ValidatePackingGroup(
    const ChunkPackingGroup& packingGroup,
    const FederatedConfig& config,
    const char* functionName) {

    if (packingGroup.chunkIds.size() !=
        config.mlweRank) {

        throw std::invalid_argument(
            std::string(functionName) +
            ": packing-group size must equal "
            "mlweRank");
    }

    const size_t chunkCount =
        GetChunkCount(config);

    std::set<size_t> validChunkIds;

    bool paddingSeen = false;

    for (const size_t chunkId :
         packingGroup.chunkIds) {

        if (chunkId ==
            kInvalidChunkId) {

            paddingSeen = true;
            continue;
        }

        if (paddingSeen) {
            throw std::invalid_argument(
                std::string(functionName) +
                ": valid chunk appears after "
                "zero-padding");
        }

        if (chunkId >= chunkCount) {
            throw std::invalid_argument(
                std::string(functionName) +
                ": invalid chunk ID");
        }

        if (!validChunkIds
                 .insert(chunkId)
                 .second) {

            throw std::invalid_argument(
                std::string(functionName) +
                ": duplicate chunk ID");
        }
    }
}

}  // namespace

std::vector<double> ExtractPaddedChunk(
    const std::vector<double>& update,
    size_t chunkId,
    const FederatedConfig& config) {

    ValidatePackingConfig(
        config,
        "ExtractPaddedChunk");

    if (update.size() !=
        config.modelDimension) {

        throw std::invalid_argument(
            "ExtractPaddedChunk: update size "
            "does not match modelDimension");
    }

    const size_t chunkSize =
        GetChunkSize(config);

    const size_t chunkCount =
        GetChunkCount(config);

    if (chunkId >= chunkCount) {
        throw std::out_of_range(
            "ExtractPaddedChunk: invalid chunk ID");
    }

    std::vector<double> chunk(
        chunkSize,
        0.0);

    const size_t begin =
        chunkId * chunkSize;

    for (size_t offset = 0;
         offset < chunkSize;
         ++offset) {

        const size_t sourceIndex =
            begin + offset;

        if (sourceIndex >=
            config.modelDimension) {

            break;
        }

        chunk[offset] =
            update[sourceIndex];
    }

    return chunk;
}

std::vector<double> PackInterleavedChunks(
    const std::vector<double>& update,
    const ChunkPackingGroup& packingGroup,
    const FederatedConfig& config) {

    ValidatePackingConfig(
        config,
        "PackInterleavedChunks");

    if (update.size() !=
        config.modelDimension) {

        throw std::invalid_argument(
            "PackInterleavedChunks: update size "
            "does not match modelDimension");
    }

    ValidatePackingGroup(
        packingGroup,
        config,
        "PackInterleavedChunks");

    const size_t chunkSize =
        GetChunkSize(config);

    const size_t rank =
        config.mlweRank;

    std::vector<double> packed(
        config.ringDimension,
        0.0);

    for (size_t slot = 0;
         slot < rank;
         ++slot) {

        const size_t chunkId =
            packingGroup.chunkIds[slot];

        if (chunkId ==
            kInvalidChunkId) {

            continue;
        }

        const size_t chunkBegin =
            chunkId * chunkSize;

        for (size_t offset = 0;
             offset < chunkSize;
             ++offset) {

            const size_t sourceIndex =
                chunkBegin + offset;

            if (sourceIndex >=
                config.modelDimension) {

                break;
            }

            const size_t packedIndex =
                offset * rank + slot;

            packed[packedIndex] =
                update[sourceIndex];
        }
    }

    return packed;
}

std::vector<std::vector<double>>
UnpackInterleavedChunks(
    const std::vector<double>& packed,
    const FederatedConfig& config) {

    ValidatePackingConfig(
        config,
        "UnpackInterleavedChunks");

    if (packed.size() !=
        config.ringDimension) {

        throw std::invalid_argument(
            "UnpackInterleavedChunks: packed "
            "vector size must equal ringDimension");
    }

    const size_t chunkSize =
        GetChunkSize(config);

    const size_t rank =
        config.mlweRank;

    std::vector<std::vector<double>> chunks(
        rank,
        std::vector<double>(
            chunkSize,
            0.0));

    for (size_t slot = 0;
         slot < rank;
         ++slot) {

        for (size_t offset = 0;
             offset < chunkSize;
             ++offset) {

            const size_t packedIndex =
                offset * rank + slot;

            chunks[slot][offset] =
                packed[packedIndex];
        }
    }

    return chunks;
}

void PlaceUnpackedChunks(
    const std::vector<
        std::vector<double>>& chunks,
    const ChunkPackingGroup& packingGroup,
    const FederatedConfig& config,
    std::vector<double>& destination) {

    ValidatePackingConfig(
        config,
        "PlaceUnpackedChunks");

    ValidatePackingGroup(
        packingGroup,
        config,
        "PlaceUnpackedChunks");

    if (destination.size() !=
        config.modelDimension) {

        throw std::invalid_argument(
            "PlaceUnpackedChunks: destination "
            "size does not match modelDimension");
    }

    if (chunks.size() !=
        config.mlweRank) {

        throw std::invalid_argument(
            "PlaceUnpackedChunks: number of chunks "
            "must equal mlweRank");
    }

    const size_t chunkSize =
        GetChunkSize(config);

    for (const auto& chunk : chunks) {
        if (chunk.size() != chunkSize) {
            throw std::invalid_argument(
                "PlaceUnpackedChunks: chunk size "
                "mismatch");
        }
    }

    for (size_t slot = 0;
         slot < config.mlweRank;
         ++slot) {

        const size_t chunkId =
            packingGroup.chunkIds[slot];

        if (chunkId ==
            kInvalidChunkId) {

            continue;
        }

        const size_t destinationBegin =
            chunkId * chunkSize;

        for (size_t offset = 0;
             offset < chunkSize;
             ++offset) {

            const size_t destinationIndex =
                destinationBegin + offset;

            if (destinationIndex >=
                config.modelDimension) {

                break;
            }

            destination[destinationIndex] =
                chunks[slot][offset];
        }
    }
}

}  // namespace federated