#include "protocol-auxiliary-path.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace federated {
namespace {

DirectRlweAggregationParameters MakeAuxiliaryParameters(
    const ProtocolExperimentConfig& config) {

    DirectRlweAggregationParameters result;
    result.seedDomain = "ROUND-AUXILIARY-ENC-SEED";
    result.v17Path = "AUX";
    result.scaling = config.auxScaling;
    result.updateBound = config.auxUpdateBound;
    result.sigmaSmg = config.smudging.auxSigmaSmg;
    return result;
}

std::vector<double> FlattenAuxiliaryInput(
    const AuxiliaryClientRoundInput& client,
    const ProtocolExperimentConfig& config) {

    if (client.runningMeanDelta.size() !=
            config.syntheticUpdates.runningMeanDimension ||
        client.runningVarianceDelta.size() !=
            config.syntheticUpdates.runningVarianceDimension) {

        throw std::invalid_argument(
            "auxiliary running-statistic dimensions are invalid");
    }

    std::vector<double> result;
    result.reserve(
        client.runningMeanDelta.size() +
        client.runningVarianceDelta.size());
    result.insert(
        result.end(),
        client.runningMeanDelta.begin(),
        client.runningMeanDelta.end());
    result.insert(
        result.end(),
        client.runningVarianceDelta.begin(),
        client.runningVarianceDelta.end());
    return result;
}

std::vector<double> Slice(
    const std::vector<double>& values,
    std::size_t begin,
    std::size_t count) {

    return std::vector<double>(
        values.begin() + begin,
        values.begin() + begin + count);
}

std::vector<DirectRlweClientInput> MakeDirectClients(
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const ProtocolExperimentConfig& config) {

    std::vector<DirectRlweClientInput> result;
    result.reserve(clients.size());
    for (const auto& client : clients) {
        DirectRlweClientInput direct;
        direct.clientId = client.clientId;
        direct.integerWeight = client.integerWeight;
        direct.values = FlattenAuxiliaryInput(client, config);
        result.push_back(std::move(direct));
    }
    return result;
}

AuxiliaryPathResult MakeAuxiliaryResult(
    DirectRlweAggregationResult direct,
    const ProtocolExperimentConfig& config) {

    const std::size_t meanCount =
        config.syntheticUpdates.runningMeanDimension;
    const std::size_t varianceCount =
        config.syntheticUpdates.runningVarianceDimension;

    AuxiliaryPathResult result;
    if (!direct.idealReference.empty()) {
        result.idealRunningMeanDelta = Slice(
            direct.idealReference,
            0,
            meanCount);
        result.idealRunningVarianceDelta = Slice(
            direct.idealReference,
            meanCount,
            varianceCount);
        result.quantizedRunningMeanDelta = Slice(
            direct.quantizedReference,
            0,
            meanCount);
        result.quantizedRunningVarianceDelta = Slice(
            direct.quantizedReference,
            meanCount,
            varianceCount);
    }
    result.runningMeanDelta = Slice(direct.aggregate, 0, meanCount);
    result.runningVarianceDelta = Slice(
        direct.aggregate,
        meanCount,
        varianceCount);
    result.phaseAudit = std::move(direct.phaseAudit);
    result.scalingFactor = direct.scalingFactor;
    result.auxiliaryValueCount = direct.valueCount;
    result.blockCount = direct.blockCount;
    result.clientCiphertextCount = direct.clientCiphertextCount;
    result.survivorClientCiphertextCount =
        direct.survivorClientCiphertextCount;
    result.discardedClientCiphertextCount =
        direct.discardedClientCiphertextCount;
    result.aggregateCiphertextCount = direct.aggregateCiphertextCount;
    result.r2mCiphertextCount = direct.r2mCiphertextCount;
    result.m2rCiphertextCount = direct.m2rCiphertextCount;
    result.residualValueCount = direct.residualValueCount;
    result.sequentialSeconds = direct.sequentialSeconds;
    result.clientEncryptionSeconds =
        std::move(direct.clientEncryptionSeconds);
    result.clientKeySwitchSeconds =
        std::move(direct.clientKeySwitchSeconds);
    result.clientPartialDecryptionSeconds =
        std::move(direct.clientPartialDecryptionSeconds);
    result.clientStageTimings =
        std::move(direct.clientStageTimings);
    result.serverStageTimings = direct.serverStageTimings;
    result.preSmudgingNoise = std::move(direct.preSmudgingNoise);
    result.preSmudgingNoiseBreakdown =
        std::move(direct.preSmudgingNoiseBreakdown);
    result.cryptoStageTimings = direct.cryptoStageTimings;
    return result;
}

}  // namespace

AuxiliaryCompressionProfile MakeUncompressedAuxiliaryProfile() {
    return AuxiliaryCompressionProfile{};
}

AuxiliaryCompressionProfile MakeConfiguredAuxiliaryProfile(
    const ProtocolExperimentConfig& config) {

    AuxiliaryCompressionProfile result;
    result.compressClientB = true;
    result.compressAggregateA = true;
    result.compressPartialDecryption = true;
    result.quantizeFinalResult = true;
    result.c1 = config.compression.auxC1;
    result.ca = config.compression.auxCa;
    result.c2 = config.compression.auxC2;
    result.cd = config.compression.auxCd;
    result.deliveryScale = config.auxDeliveryScale;
    return result;
}

AuxiliaryUploadBatch EncryptProtocolAuxiliaryUploads(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const AuxiliaryCompressionProfile& compression,
    CryptoTaskExecutor* executor) {

    return DirectRlweAggregationEngine::EncryptUploads(
        cc,
        config,
        roundId,
        MakeDirectClients(clients, config),
        setup,
        MakeAuxiliaryParameters(config),
        compression,
        executor);
}

AuxiliaryPathResult FinalizeProtocolAuxiliaryPathAfterDropout(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const AuxiliaryUploadBatch& uploads,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const AuxiliaryCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    std::vector<DirectRlweClientInput> directClients =
        MakeDirectClients(clients, config);

    DirectRlweAggregationResult direct =
        DirectRlweAggregationEngine::AggregateUploaded(
            cc,
            config,
            roundId,
            directClients,
            uploads,
            setup,
            survivorKeys,
            MakeAuxiliaryParameters(config),
            compression,
            executor,
            aggregateMetrics);
    return MakeAuxiliaryResult(std::move(direct), config);
}

AuxiliaryPathResult RunProtocolAuxiliaryPath(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<AuxiliaryClientRoundInput>& clients,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const AuxiliaryCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    const AuxiliaryUploadBatch uploads = EncryptProtocolAuxiliaryUploads(
        cc,
        config,
        roundId,
        clients,
        setup,
        compression,
        executor);
    return FinalizeProtocolAuxiliaryPathAfterDropout(
        cc,
        config,
        roundId,
        clients,
        uploads,
        setup,
        survivorKeys,
        compression,
        executor,
        aggregateMetrics);
}

long double AuxiliaryCompressionModelBound(
    const ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount) {

    return DirectRlweCompressionModelBound(
        config,
        q,
        scalingFactor,
        survivorCount,
        MakeAuxiliaryParameters(config),
        MakeConfiguredAuxiliaryProfile(config));
}

}  // namespace federated
