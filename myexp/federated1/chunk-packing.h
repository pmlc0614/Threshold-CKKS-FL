#ifndef FEDERATED1_CHUNK_PACKING_H
#define FEDERATED1_CHUNK_PACKING_H

#include "federated-types.h"

#include <cstddef>
#include <vector>

namespace federated {

/* Extract one chunk of N/k entries; zero-pad the incomplete final chunk. */
std::vector<double> ExtractPaddedChunk(
    const std::vector<double>& update,
    size_t chunkId,
    const FederatedConfig& config);

/* Interleave at most k chunks: packed[t*k+slot] = chunk[slot][t]. Invalid slots contain zeros. */
std::vector<double> PackInterleavedChunks(
    const std::vector<double>& update,
    const ChunkPackingGroup& packingGroup,
    const FederatedConfig& config);

/* Undo interleaving: chunks[slot][t] = packed[t*k+slot]. */
std::vector<std::vector<double>>
UnpackInterleavedChunks(
    const std::vector<double>& packed,
    const FederatedConfig& config);

/* Scatter unpacked chunks into a modelDimension-sized destination; ignore invalid chunk IDs. */
void PlaceUnpackedChunks(
    const std::vector<
        std::vector<double>>& chunks,
    const ChunkPackingGroup& packingGroup,
    const FederatedConfig& config,
    std::vector<double>& destination);

}  // namespace federated

#endif  // FEDERATED1_CHUNK_PACKING_H