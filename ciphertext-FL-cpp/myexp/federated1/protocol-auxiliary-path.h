#ifndef FEDERATED1_PROTOCOL_AUXILIARY_PATH_H
#define FEDERATED1_PROTOCOL_AUXILIARY_PATH_H

#include "direct-rlwe-aggregation.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace federated {

struct AuxiliaryClientRoundInput {
    ClientId clientId;
    std::uint32_t integerWeight = 0;
    std::vector<double> runningMeanDelta;
    std::vector<double> runningVarianceDelta;
};

using AuxiliaryCompressionProfile = DirectRlweCompressionProfile;
using AuxiliaryUploadBatch = DirectRlweUploadBatch;

AuxiliaryCompressionProfile MakeUncompressedAuxiliaryProfile();

AuxiliaryCompressionProfile MakeConfiguredAuxiliaryProfile(
    const ProtocolExperimentConfig& config);

struct AuxiliaryPathResult {
    ProtocolPhaseAudit phaseAudit;
    std::vector<double> idealRunningMeanDelta;
    std::vector<double> idealRunningVarianceDelta;
    std::vector<double> quantizedRunningMeanDelta;
    std::vector<double> quantizedRunningVarianceDelta;
    std::vector<double> runningMeanDelta;
    std::vector<double> runningVarianceDelta;

    long double scalingFactor = 0.0L;
    std::size_t auxiliaryValueCount = 0;
    std::size_t blockCount = 0;
    std::size_t clientCiphertextCount = 0;
    std::size_t survivorClientCiphertextCount = 0;
    std::size_t discardedClientCiphertextCount = 0;
    std::size_t aggregateCiphertextCount = 0;
    std::size_t r2mCiphertextCount = 0;
    std::size_t m2rCiphertextCount = 0;
    std::size_t residualValueCount = 0;
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

AuxiliaryUploadBatch EncryptProtocolAuxiliaryUploads(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const AuxiliaryCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr);

AuxiliaryPathResult FinalizeProtocolAuxiliaryPathAfterDropout(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const AuxiliaryUploadBatch& uploads,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const AuxiliaryCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

AuxiliaryPathResult RunProtocolAuxiliaryPath(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const AuxiliaryCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

long double AuxiliaryCompressionModelBound(
    const ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_AUXILIARY_PATH_H
