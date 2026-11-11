#ifndef FEDERATED1_PROTOCOL_SPARSE_ROUND_H
#define FEDERATED1_PROTOCOL_SPARSE_ROUND_H

#include "protocol-contract.h"
#include "protocol-parameters.h"
#include "protocol-setup.h"
#include "protocol-types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace federated {

struct ProtocolClientRoundInput {
    ClientId clientId;
    std::uint32_t integerWeight = 0;
    std::vector<double> update;
};

struct ClientSparseChunkLabelMapping {
    ActualChunkId actualChunk;
    ShuffledChunkLabel label;
};

/* This packing metadata never crosses the client/server boundary. */
struct ProtocolClientPackingGroup {
    ClientId clientId;
    std::size_t groupIndex = 0;
    std::vector<ActualChunkId> actualChunks;
    PackedShuffledLabels shuffledLabels;
};

/* Kidx, pi_t, pi_t^{-1}, and actual identifiers are client-only. */
struct ClientSparsePrivateRoundState {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::size_t chunkSize = 0;
    std::size_t chunkCount = 0;
    std::size_t mlweRank = 0;

    Seed128 kidx{};
    std::vector<std::size_t> permutation;
    std::vector<std::size_t> inversePermutation;
    std::vector<ActualChunkId> actualSelectedChunks;
    std::vector<ClientSparseChunkLabelMapping> selectedMappings;
    /* Immutable upload-support header for this local collection. */
    ByteVector uploadSupportBitmap;
    std::vector<ProtocolClientPackingGroup> packingGroups;
};

/* Public server state deliberately contains shuffled labels only. */
struct ServerSparseAdmissionPlan {
    SetupId setupId;
    RoundId roundId;
    std::size_t chunkSize = 0;
    std::size_t chunkCount = 0;
    std::size_t mlweRank = 0;

    std::vector<SparseIndexAnnouncement> announcements;
    std::vector<ShuffledChunkCohort> cohorts;
    std::vector<ShuffledChunkLabel> acceptedLabels;
    /* Admission/final-context/result support, encoded once per collection. */
    ByteVector acceptedSupportBitmap;
    std::vector<PackedShuffledLabels> packingGroups;
    std::uint64_t publicPlanHash = 0;
};

/* Privileged single-process simulation bundle; never pass this to a server. */
struct ProtocolSparseRoundArtifacts {
    std::vector<ClientSparsePrivateRoundState> clientPrivateStates;
    ServerSparseAdmissionPlan serverPlan;
};

ClientSparsePrivateRoundState BuildClientSparsePrivateRoundState(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    RoundId roundId,
    const ProtocolClientRoundInput& client);

/* External-input variant: Python owns top-k and supplies actual chunk IDs. */
ClientSparsePrivateRoundState BuildClientSparsePrivateRoundStateFromSelection(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    const ClientSelectionRequest& selection);

SparseIndexAnnouncement BuildSparseIndexAnnouncement(
    const ClientSparsePrivateRoundState& state);

ServerSparseAdmissionPlan BuildServerSparseAdmissionPlan(
    const ProtocolExperimentConfig& config,
    SetupId setupId,
    RoundId roundId,
    const SurvivorSet& survivors,
    const std::vector<SparseIndexAnnouncement>& announcements);

void FinalizeClientSparsePrivateRoundState(
    const ServerSparseAdmissionPlan& serverPlan,
    ClientSparsePrivateRoundState* state);

ProtocolSparseRoundArtifacts BuildProtocolSparseRoundArtifacts(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    RoundId roundId,
    const SurvivorSet& survivors,
    const std::vector<ProtocolClientRoundInput>& clients);

/* Build the same private/server artifacts without recomputing top-k in C++. */
ProtocolSparseRoundArtifacts BuildProtocolSparseRoundArtifactsFromSelections(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    RoundId roundId,
    const SurvivorSet& survivors,
    const std::vector<ClientSelectionRequest>& selections);

void ValidateClientSparsePrivateRoundState(
    const ProtocolExperimentConfig& config,
    const ProtocolClientRoundInput& client,
    const SparseIndexAnnouncement& announcement,
    const ServerSparseAdmissionPlan& serverPlan,
    const ClientSparsePrivateRoundState& state);

void ValidateServerSparseAdmissionPlan(
    const ProtocolExperimentConfig& config,
    const SurvivorSet& survivors,
    const ServerSparseAdmissionPlan& plan);

ActualChunkId ActualChunkForLabel(
    const ClientSparsePrivateRoundState& state,
    ShuffledChunkLabel label);

const ShuffledChunkCohort& FindShuffledCohort(
    const ServerSparseAdmissionPlan& plan,
    ShuffledChunkLabel label);

const ClientSparsePrivateRoundState& FindClientSparsePrivateState(
    const std::vector<ClientSparsePrivateRoundState>& states,
    ClientId clientId);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_SPARSE_ROUND_H
