#ifndef FEDERATED1_FEDERATED_TYPES_H
#define FEDERATED1_FEDERATED_TYPES_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace federated {

/* Sentinel for a zero-padding slot without an actual chunk. */
constexpr size_t kInvalidChunkId =
    std::numeric_limits<size_t>::max();

/* Configuration of the functional FL simulation fixture. */
struct FederatedConfig {
    size_t clientCount = 8;
    size_t modelDimension = 0;

    size_t ringDimension = 1u << 13;  // 8192
    size_t mlweRank = 4;

    /* Number of chunks selected per client by L2 norm. */
    size_t selectedChunkCount = 4;

    /* Minimum cohort size tau. */
    size_t minimumCohortSize = 3;

    /* Client weights are alpha_i/128. */
    uint32_t weightDenominator = 128;

    uint32_t scaleBits = 60;

    double sigmaShare =    281474976710656.0;  // 2^48

    /* Synthetic update parameters used by simulation tests. */
    double updateStandardDeviation = 0.01;
    uint64_t randomSeed = 20250818;
};

/* One client's input to one round. */
struct ClientRoundInput {
    size_t clientId = 0;

    /* The effective weight is alpha/weightDenominator. */
    uint32_t alpha = 0;

    /* Flattened update with config.modelDimension entries. */
    std::vector<double> update;
};

/* Interleaving group with exactly mlweRank slots; fill unused slots with kInvalidChunkId. */
struct ChunkPackingGroup {
    std::vector<size_t> chunkIds;
};

/* A client's fixed sparsification result. */
struct ClientSparsePlan {
    size_t clientId = 0;

    /* Chunks selected by decreasing squared L2 norm; ties use increasing chunk ID. */
    std::vector<size_t> selectedChunkIds;

    /* Selected chunks grouped in batches of mlweRank. */
    std::vector<ChunkPackingGroup> packingGroups;
};

/* Cohort of clients selecting a particular chunk. */
struct ChunkCohort {
    size_t chunkId = 0;

    /* Selecting client IDs in increasing order. */
    std::vector<size_t> clientIds;

    /* Whether clientIds.size() >= minimumCohortSize. */
    bool accepted = false;
};

/* Fixed plan shared by plaintext and encrypted functional test paths. */
struct SparseRoundPlan {
    size_t chunkSize = 0;
    size_t chunkCount = 0;

    /* Ordered by client ID. */
    std::vector<ClientSparsePlan> clientPlans;

    /* One entry per chunk, ordered by chunk ID. */
    std::vector<ChunkCohort> cohorts;

    /* Admitted chunk IDs in increasing order. */
    std::vector<size_t> acceptedChunkIds;

    /* Aggregated chunks regrouped in batches of mlweRank for M2R. */
    std::vector<ChunkPackingGroup>
        serverPackingGroups;

    /* Deterministic hash used to compare the two test paths' plans. */
    uint64_t planHash = 0;
};

}  // namespace federated

#endif  // FEDERATED1_FEDERATED_TYPES_H