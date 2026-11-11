#include "protocol-full-model-path.h"

#include <stdexcept>
#include <utility>

namespace federated {
namespace {

DirectRlweAggregationParameters MakeFullModelParameters(
    const ProtocolExperimentConfig& config) {

    DirectRlweAggregationParameters result;
    result.seedDomain = "ROUND-FULL-MODEL-ENC-SEED";
    result.v17Path = "FULL-MODEL";
    result.scaling = config.scaling;
    result.updateBound = config.updateBound;
    result.sigmaSmg = config.smudging.sigmaSmg;
    return result;
}

std::vector<DirectRlweClientInput> MakeDirectClients(
    const ProtocolExperimentConfig& config,
    const std::vector<FullModelClientRoundInput>& clients) {

    if (config.mode != ProtocolMode::FULL) {
        throw std::invalid_argument("FULL model path requires FULL mode");
    }
    std::vector<DirectRlweClientInput> directClients;
    directClients.reserve(clients.size());
    for (const auto& client : clients) {
        if (client.modelDelta.size() != config.modelDimension) {
            throw std::invalid_argument(
                "FULL model delta dimension does not match the profile");
        }
        DirectRlweClientInput direct;
        direct.clientId = client.clientId;
        direct.integerWeight = client.integerWeight;
        direct.values = client.modelDelta;
        directClients.push_back(std::move(direct));
    }
    return directClients;
}

}  // namespace

FullModelCompressionProfile MakeUncompressedFullModelProfile() {
    return FullModelCompressionProfile{};
}

FullModelCompressionProfile MakeConfiguredFullModelProfile(
    const ProtocolExperimentConfig& config) {

    FullModelCompressionProfile result;
    result.compressClientB = true;
    result.compressAggregateA = true;
    result.compressPartialDecryption = true;
    result.quantizeFinalResult = true;
    result.c1 = config.compression.c1;
    result.ca = config.compression.ca;
    result.c2 = config.compression.c2;
    result.cd = config.compression.cd;
    result.deliveryScale = config.modelDeliveryScale;
    return result;
}

FullModelUploadBatch EncryptProtocolFullModelUploads(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<FullModelClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const FullModelCompressionProfile& compression,
    CryptoTaskExecutor* executor) {

    return DirectRlweAggregationEngine::EncryptUploads(
        cc, config, roundId, MakeDirectClients(config, clients),
        setup, MakeFullModelParameters(config), compression, executor);
}

FullModelPathResult FinalizeProtocolFullModelPathAfterDropout(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<FullModelClientRoundInput>& clients,
    const FullModelUploadBatch& uploads,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const FullModelCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    if (config.mode != ProtocolMode::FULL ||
        setup.mode != ProtocolMode::FULL ||
        setup.sparse.has_value()) {

        throw std::invalid_argument(
            "FULL model path requires a FULL setup without M2R material");
    }

    return DirectRlweAggregationEngine::AggregateUploaded(
        cc,
        config,
        roundId,
        MakeDirectClients(config, clients),
        uploads,
        setup,
        survivorKeys,
        MakeFullModelParameters(config),
        compression,
        executor,
        aggregateMetrics);
}

FullModelPathResult RunProtocolFullModelPath(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<FullModelClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const FullModelCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    const auto uploads = EncryptProtocolFullModelUploads(
        cc, config, roundId, clients, setup, compression, executor);
    return FinalizeProtocolFullModelPathAfterDropout(
        cc, config, roundId, clients, uploads, setup, survivorKeys,
        compression, executor, aggregateMetrics);
}

long double FullModelCompressionModelBound(
    const ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount) {

    return DirectRlweCompressionModelBound(
        config,
        q,
        scalingFactor,
        survivorCount,
        MakeFullModelParameters(config),
        MakeConfiguredFullModelProfile(config));
}

}  // namespace federated
