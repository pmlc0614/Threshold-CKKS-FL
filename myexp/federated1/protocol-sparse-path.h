#ifndef FEDERATED1_PROTOCOL_SPARSE_PATH_H
#define FEDERATED1_PROTOCOL_SPARSE_PATH_H

#include "crypto-execution.h"
#include "protocol-pre-smudging-noise.h"
#include "protocol-ring-conversion.h"
#include "protocol-sparse-round.h"
#include "protocol-setup.h"
#include "protocol-v17-decryption.h"
#include "survivor-keys.h"
#include "seeded-rlwe.h"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace federated {

struct SparsePathCompressionProfile {
    bool compressClientB = false;
    bool compressAggregateA = false;
    bool compressPartialDecryption = false;
    bool quantizeFinalResult = false;
    bool measurePreSmudgingNoise = false;
    // Standalone diagnostic harnesses retain references; runtime opts out.
    bool computeReference = true;
    std::uint32_t c1 = 0;
    std::uint32_t ca = 0;
    std::uint32_t c2 = 0;
    std::uint32_t cd = 0;
    long double deliveryScale = 0.0L;
    std::shared_ptr<V17DecryptionSeedGate> v17SeedGate;
};

SparsePathCompressionProfile MakeUncompressedSparseProfile();

SparsePathCompressionProfile MakeConfiguredSparseProfile(
    const ProtocolExperimentConfig& config);

/* The sparse server returns labels and chunk payloads, never actual indices. */
struct LabeledAggregateChunk {
    ShuffledChunkLabel label;
    std::vector<double> values;
    std::vector<std::string> phaseResidues;
};

struct SparseServerPathResult {
    std::vector<LabeledAggregateChunk> aggregateChunks;
    std::uint64_t planHash = 0;
    std::size_t clientRLWECiphertextCount = 0;
    std::size_t survivorClientRLWECiphertextCount = 0;
    std::size_t discardedClientRLWECiphertextCount = 0;
    std::size_t acceptedContributionCount = 0;
    std::size_t aggregatedMLWECiphertextCount = 0;
    std::size_t serverRLWECiphertextCount = 0;
};

/* Explicitly privileged simulation-only view used for accuracy evaluation. */
struct SparsePrivilegedObserverResult {
    ProtocolPhaseAudit phaseAudit;
    std::vector<double> idealReference;
    std::vector<double> quantizedReference;
    std::vector<double> aggregate;
    std::vector<std::uint8_t> acceptedCoordinateMask;
};

struct SparsePathResult {
    SparseServerPathResult server;
    SparsePrivilegedObserverResult observer;
    long double scalingFactor = 0.0L;
    double sequentialSeconds = 0.0;
    std::vector<double> clientEncryptionSeconds;
    std::vector<double> clientKeySwitchSeconds;
    std::vector<double> clientPartialDecryptionSeconds;
    std::vector<CryptoStageTimings> clientStageTimings;
    CryptoStageTimings serverStageTimings{};
    CryptoStageTimings cryptoStageTimings{};
    ProtocolPreSmudgingNoiseMeasurement preSmudgingNoise;
    ProtocolNoiseBreakdownMeasurement preSmudgingNoiseBreakdown;
};

struct SparseUploadCiphertextPacket {
    ClientId clientId;
    std::uint32_t integerWeight = 0;
    std::size_t groupIndex = 0;
    /* Server-derived from the collection bitmap; not serialized per packet. */
    PackedShuffledLabels derivedLabels;
    SeededSparseRLWEPacket ciphertext;
};

struct SparseUploadSupportHeader {
    ClientId clientId;
    std::uint64_t admissionRevision = 0;
    ByteVector supportBitmap;
};

struct SparseUploadPacketPair {
    SparseUploadCiphertextPacket wire;
    SparseUploadCiphertextPacket bctShadow;
};

struct SparsePathUploadBatch {
    SetupId setupId;
    RoundId roundId;
    std::vector<SparseUploadSupportHeader> supportHeaders;
    std::vector<SparseUploadPacketPair> packetPairs;
    std::vector<double> clientEncryptionSeconds;
    std::vector<CryptoStageTimings> clientStageTimings;
    double sequentialSeconds = 0.0;
};

SparsePathUploadBatch EncryptProtocolSparseUploads(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& uploadRound,
    const ProtocolSetupResult& setup,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr);

SparsePathResult FinalizeProtocolSparsePathAfterDropout(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const SparsePathUploadBatch& uploads,
    const ProtocolSparseRoundArtifacts& survivorRound,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

/* Client-side inverse-permutation application of a label-only server result. */
std::vector<double> ApplyLabeledAggregateChunksForClient(
    const ProtocolExperimentConfig& config,
    const ClientSparsePrivateRoundState& clientState,
    const std::vector<LabeledAggregateChunk>& chunks);

/* Only labels present in the finalized result are removed from residual. */
void UpdateClientResidualFromFinalizedChunks(
    const ProtocolExperimentConfig& config,
    const ProtocolClientRoundInput& client,
    const ClientSparsePrivateRoundState& clientState,
    const std::vector<LabeledAggregateChunk>& finalizedChunks,
    bool errorFeedbackEnabled,
    std::vector<double>* residual);

/* Privileged single-process wrapper around client and label-only server steps. */
SparsePathResult RunProtocolSparsePath(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& round,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

/* Encrypt against the upload-time all-client plan, then reuse only survivor
   packets under a separately rebuilt post-upload admission plan. */
SparsePathResult RunProtocolSparsePathAfterDropout(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& uploadRound,
    const ProtocolSparseRoundArtifacts& survivorRound,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

/* Serially initialize the Q/QP and small-ring transform tables before sparse
   ciphertext workers can reach R2M/M2R concurrently. */
void PrewarmProtocolSparseTransforms(const ProtocolSetupResult& setup);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_SPARSE_PATH_H
