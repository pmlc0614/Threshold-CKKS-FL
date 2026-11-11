#ifndef FEDERATED1_PROTOCOL_FULL_MODEL_PATH_H
#define FEDERATED1_PROTOCOL_FULL_MODEL_PATH_H

#include "direct-rlwe-aggregation.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace federated {

struct FullModelClientRoundInput {
    ClientId clientId;
    std::uint32_t integerWeight = 0;
    std::vector<double> modelDelta;
};

using FullModelCompressionProfile = DirectRlweCompressionProfile;
using FullModelPathResult = DirectRlweAggregationResult;
using FullModelUploadBatch = DirectRlweUploadBatch;

FullModelCompressionProfile MakeUncompressedFullModelProfile();

FullModelCompressionProfile MakeConfiguredFullModelProfile(
    const ProtocolExperimentConfig& config);

FullModelUploadBatch EncryptProtocolFullModelUploads(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<FullModelClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const FullModelCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr);

FullModelPathResult FinalizeProtocolFullModelPathAfterDropout(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<FullModelClientRoundInput>& clients,
    const FullModelUploadBatch& uploads,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const FullModelCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

FullModelPathResult RunProtocolFullModelPath(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<FullModelClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const FullModelCompressionProfile& compression,
    CryptoTaskExecutor* executor = nullptr,
    CryptoStageMetrics* aggregateMetrics = nullptr);

long double FullModelCompressionModelBound(
    const ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_FULL_MODEL_PATH_H
