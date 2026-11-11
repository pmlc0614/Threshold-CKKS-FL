#ifndef FEDERATED1_DIRECT_RLWE_AGGREGATION_H
#define FEDERATED1_DIRECT_RLWE_AGGREGATION_H

#include "crypto-execution.h"
#include "protocol-parameters.h"
#include "protocol-pre-smudging-noise.h"
#include "protocol-setup.h"
#include "protocol-types.h"
#include "protocol-v17-decryption.h"
#include "survivor-keys.h"
#include "seeded-dense-rlwe.h"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace federated {

struct DirectRlweClientInput {
    ClientId clientId;
    std::uint32_t integerWeight = 0;
    std::vector<double> values;
};

struct DirectRlweCompressionProfile {
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

struct DirectRlweAggregationParameters {
    std::string seedDomain;
    std::string v17Path;
    ScalingParameters scaling;
    /* Bound on the weighted aggregate, not on an individual client value. */
    long double updateBound = 0.0L;
    long double sigmaSmg = 0.0L;
};

using DirectRlweStageTimings = CryptoStageTimings;

struct DirectRlweUploadBatch {
    SetupId setupId;
    RoundId roundId;
    std::size_t valueCount = 0;
    std::size_t blockCount = 0;
    long double scalingFactor = 0.0L;
    std::vector<std::vector<std::unique_ptr<SeededDenseRLWEPacket>>>
        wirePacketsByClient;
    std::vector<std::vector<std::unique_ptr<SeededDenseRLWEPacket>>>
        shadowPacketsByClient;
    std::vector<double> clientEncryptionSeconds;
    std::vector<DirectRlweStageTimings> clientStageTimings;
    double sequentialSeconds = 0.0;
};

struct DirectRlweAggregationResult {
    ProtocolPhaseAudit phaseAudit;
    std::vector<double> idealReference;
    std::vector<double> quantizedReference;
    std::vector<double> aggregate;

    long double scalingFactor = 0.0L;
    std::size_t valueCount = 0;
    std::size_t blockCount = 0;
    std::size_t clientCiphertextCount = 0;
    std::size_t survivorClientCiphertextCount = 0;
    std::size_t discardedClientCiphertextCount = 0;
    std::size_t aggregateCiphertextCount = 0;

    /* Direct-RLWE never creates sparse routing or residual state. */
    std::size_t r2mCiphertextCount = 0;
    std::size_t m2rCiphertextCount = 0;
    std::size_t residualValueCount = 0;
    /* Clients are executed one at a time on the emulation host while each
       client's ciphertext blocks use the bounded executor.  The per-client
       arrays therefore contain isolated wall times suitable for a maximum-
       client critical-path estimate. */
    double sequentialSeconds = 0.0;
    std::vector<double> clientEncryptionSeconds;
    std::vector<double> clientKeySwitchSeconds;
    std::vector<double> clientPartialDecryptionSeconds;
    std::vector<DirectRlweStageTimings> clientStageTimings;
    DirectRlweStageTimings serverStageTimings{};
    ProtocolPreSmudgingNoiseMeasurement preSmudgingNoise;
    ProtocolNoiseBreakdownMeasurement preSmudgingNoiseBreakdown;
    DirectRlweStageTimings cryptoStageTimings{};
};

class DirectRlweAggregationEngine {
public:
    static DirectRlweUploadBatch EncryptUploads(
        const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
        const ProtocolExperimentConfig& config,
        RoundId roundId,
        const std::vector<DirectRlweClientInput>& clients,
        const ProtocolSetupResult& setup,
        const DirectRlweAggregationParameters& parameters,
        const DirectRlweCompressionProfile& compression,
        CryptoTaskExecutor* executor = nullptr);

    static DirectRlweAggregationResult AggregateUploaded(
        const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
        const ProtocolExperimentConfig& config,
        RoundId roundId,
        const std::vector<DirectRlweClientInput>& clients,
        const DirectRlweUploadBatch& uploads,
        const ProtocolSetupResult& setup,
        const SurvivorKeySet& survivorKeys,
        const DirectRlweAggregationParameters& parameters,
        const DirectRlweCompressionProfile& compression,
        CryptoTaskExecutor* executor = nullptr,
        CryptoStageMetrics* aggregateMetrics = nullptr);

    static DirectRlweAggregationResult Aggregate(
        const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
        const ProtocolExperimentConfig& config,
        RoundId roundId,
        const std::vector<DirectRlweClientInput>& clients,
        const ProtocolSetupResult& setup,
        const SurvivorKeySet& survivorKeys,
        const DirectRlweAggregationParameters& parameters,
        const DirectRlweCompressionProfile& compression,
        CryptoTaskExecutor* executor = nullptr,
        CryptoStageMetrics* aggregateMetrics = nullptr);
};

long double DirectRlweCompressionModelBound(
    const ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount,
    const DirectRlweAggregationParameters& parameters,
    const DirectRlweCompressionProfile& compression);

}  // namespace federated

#endif  // FEDERATED1_DIRECT_RLWE_AGGREGATION_H
