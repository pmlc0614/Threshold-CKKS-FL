#include "protocol-rounding.h"
#include "direct-rlwe-aggregation.h"

#include "protocol-randomness.h"

#include "client-to-collective-kswitch.h"
#include "coefficient-codec.h"
#include "coefficient-compression.h"
#include "deterministic-randomness.h"
#include "rns-polynomial.h"
#include "seeded-dense-rlwe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

struct DirectAggregateCiphertext {
    std::size_t blockIndex = 0;
    std::size_t validValueCount = 0;
    DCRTPoly b;
    DCRTPoly a;
};

void AccumulateStageTiming(
    CryptoStageTiming* destination,
    const CryptoStageTiming& source) {

    destination->wallSeconds += source.wallSeconds;
    destination->executorWaitSeconds += source.executorWaitSeconds;
    destination->workSeconds += source.workSeconds;
    destination->taskCount += source.taskCount;
}

void AccumulateStageTimings(
    CryptoStageTimings* destination,
    const CryptoStageTimings& source) {

    for (std::size_t index = 0; index < kCryptoStageCount; ++index) {
        AccumulateStageTiming(&destination->at(index), source.at(index));
    }
}

const DirectRlweClientInput& FindClient(
    const std::vector<DirectRlweClientInput>& clients,
    ClientId clientId) {

    const auto found = std::find_if(
        clients.begin(),
        clients.end(),
        [clientId](const DirectRlweClientInput& client) {
            return client.clientId == clientId;
        });
    if (found == clients.end()) {
        throw std::invalid_argument("direct-RLWE survivor input is missing");
    }
    return *found;
}

const ClientSecretSetupMaterial& FindSecret(
    const ProtocolSetupResult& setup,
    ClientId clientId) {

    const auto found = std::find_if(
        setup.clientSecrets.begin(),
        setup.clientSecrets.end(),
        [clientId](const ClientSecretSetupMaterial& secret) {
            return secret.clientId == clientId;
        });
    if (found == setup.clientSecrets.end()) {
        throw std::invalid_argument("direct-RLWE client secret is missing");
    }
    return *found;
}

ByteVector DirectSeedContext(
    RoundId roundId,
    ClientId clientId,
    std::size_t blockIndex) {

    ByteVector result;
    const auto append = [&result](std::uint64_t value) {
        const ByteVector encoded =
            deterministic_randomness::EncodeBigEndian64(value);
        result.insert(result.end(), encoded.begin(), encoded.end());
    };
    append(roundId.value);
    append(clientId.value);
    append(blockIndex);
    return result;
}

void ValidateClientValues(
    const DirectRlweClientInput& client,
    std::size_t valueCount) {

    if (client.values.size() != valueCount) {

        throw std::invalid_argument(
            "direct-RLWE client value dimensions are inconsistent");
    }

    for (const double value : client.values) {
        if (!std::isfinite(value)) {

            throw std::invalid_argument(
                "direct-RLWE input must be finite");
        }
    }
}

void ValidateInputs(
    const ProtocolExperimentConfig& config,
    const std::vector<DirectRlweClientInput>& clients,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const DirectRlweAggregationParameters& parameters) {

    ValidateProtocolParameters(config).ThrowIfInvalid();
    if (config.mode != setup.mode ||
        setup.mode != setup.common.descriptor.mode ||
        !(survivorKeys.setupId == setup.common.setupId) ||
        survivorKeys.publicKey.survivors.clients.empty()) {

        throw std::invalid_argument(
            "direct-RLWE survivor key metadata is invalid");
    }

    if (survivorKeys.publicKey.survivors.clients.size() <
        config.sparsification.minimumCohort) {

        throw std::invalid_argument(
            "direct-RLWE path requires at least minimumCohort survivors");
    }

    if (clients.empty() || clients.front().values.empty() ||
        parameters.seedDomain.empty() ||
        !std::isfinite(parameters.updateBound) ||
        parameters.updateBound <= 0.0L ||
        !std::isfinite(parameters.sigmaSmg) ||
        parameters.sigmaSmg < 0.0L) {

        throw std::invalid_argument(
            "direct-RLWE aggregation parameters are invalid");
    }

    const std::size_t valueCount = clients.front().values.size();
    std::vector<bool> seen(config.clientCount, false);
    for (const auto& client : clients) {
        if (client.clientId.value >= config.clientCount ||
            seen[client.clientId.value] ||
            client.integerWeight != config.integerWeights[client.clientId.value]) {

            throw std::invalid_argument(
                "direct-RLWE client metadata is invalid");
        }
        seen[client.clientId.value] = true;
        ValidateClientValues(client, valueCount);
    }

    for (const auto clientId : survivorKeys.publicKey.survivors.clients) {
        (void)FindClient(clients, clientId);
        (void)FindSecret(setup, clientId);
    }
}

void MultiplyPolynomialByInteger(
    DCRTPoly* polynomial,
    std::uint32_t scalar) {

    const Format format = polynomial->GetFormat();
    const lbcrypto::NativeInteger factor(scalar);
    for (auto& tower : polynomial->GetAllElements()) {
        const lbcrypto::NativeInteger modulus = tower.GetModulus();
        lbcrypto::NativeVector values = tower.GetValues();
        for (std::size_t index = 0; index < values.GetLength(); ++index) {
            values[index] = values[index].ModMul(factor, modulus);
        }
        tower.SetValues(std::move(values), format);
    }
}

DCRTPoly SampleDirectSmudging(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    long double sigma, bool certifiedV17) {

    if (sigma == 0.0L) {
        return DCRTPoly(parameters, Format::EVALUATION, true);
    }
    if (!std::isfinite(sigma) || sigma < 0.0L ||
        sigma > std::ldexp(1.0L, certifiedV17 ? 128 : 59)) {

        throw std::invalid_argument("invalid direct-RLWE smudging sigma");
    }
    return SampleProtocolGaussianPolynomial(parameters, sigma, certifiedV17);
}

DCRTPoly MaybeCompressDirectAggregateA(
    DCRTPoly a,
    const ProtocolSetupResult& setup,
    const DirectRlweCompressionProfile& profile) {

    if (profile.compressAggregateA) {
        const auto payload = (setup.mode == ProtocolMode::SP
            ? coefficient_compression::CompressAggregateCoefficients
            : coefficient_compression::CompressAllCoefficients)(
            a,
            profile.ca);
        a = coefficient_compression::DecompressAllCoefficients(
            setup.common.rns.paramsQ,
            payload);
    }
    a.SetFormat(Format::EVALUATION);
    return a;
}

lbcrypto::DCRTPoly ComputeDirectPartialShare(
    const DCRTPoly& compressedA,
    const ProtocolSetupResult& setup,
    ClientId clientId,
    const DirectRlweAggregationParameters& parameters,
    const DirectRlweCompressionProfile& profile) {

    DCRTPoly secret = FindSecret(setup, clientId)
        .secretKeyQ->GetPrivateElement();
    secret.SetFormat(Format::EVALUATION);
    DCRTPoly partial = compressedA * secret + SampleDirectSmudging(
        setup.common.rns.paramsQ,
        parameters.sigmaSmg, setup.common.certifiedV17Gaussian);

    if (profile.compressPartialDecryption) {
        const auto payload = coefficient_compression::CompressAllCoefficients(
            partial,
            profile.c2);
        partial = coefficient_compression::DecompressAllCoefficients(
            setup.common.rns.paramsQ,
            payload);
    }
    partial.SetFormat(Format::EVALUATION);
    return partial;
}

std::vector<long double> DecodeDirectPhase(
    DCRTPoly phase,
    std::size_t validValueCount,
    const ProtocolSetupResult& setup) {

    const lbcrypto::BigInteger modulus =
        setup.common.rns.paramsQ->GetModulus();
    const lbcrypto::BigInteger halfModulus = modulus >> 1;
    const auto phaseValues =
        rns_polynomial::InterpolateCoefficients(std::move(phase));
    std::vector<long double> result(validValueCount);
    for (std::size_t index = 0; index < validValueCount; ++index) {
        const bool negative = phaseValues[index] > halfModulus;
        const lbcrypto::BigInteger magnitude = negative
            ? modulus - phaseValues[index]
            : phaseValues[index];
        const long double converted = std::stold(magnitude.ToString());
        result[index] = negative ? -converted : converted;
    }
    return result;
}

struct DirectReference {
    std::vector<double> ideal;
    std::vector<double> quantized;
    std::vector<exact_integer::Integer> integerNumerator;
};

DirectReference ComputeReference(
    const ProtocolExperimentConfig& config,
    const std::vector<DirectRlweClientInput>& clients,
    const SurvivorSet& survivors,
    std::size_t valueCount,
    const exact_integer::Integer& exactScale) {

    const long double scalingFactor = exactScale.convert_to<long double>();
    std::vector<long double> idealNumerator(valueCount, 0.0L);
    std::vector<exact_integer::Integer> integerNumerator(valueCount, 0);

    for (const auto clientId : survivors.clients) {
        const auto& client = FindClient(clients, clientId);
        for (std::size_t index = 0; index < valueCount; ++index) {
            idealNumerator[index] +=
                static_cast<long double>(client.integerWeight) *
                client.values[index];
            const exact_integer::Integer term = client.integerWeight *
                exact_integer::RoundProduct(client.values[index], exactScale);
            integerNumerator[index] += term;
        }
    }

    DirectReference result;
    result.ideal.resize(valueCount);
    result.quantized.resize(valueCount);
    result.integerNumerator = integerNumerator;
    const long double idealDenominator = config.weightDenominator;
    const long double integerDenominator =
        idealDenominator * scalingFactor;
    for (std::size_t index = 0; index < valueCount; ++index) {
        result.ideal[index] = static_cast<double>(
            idealNumerator[index] / idealDenominator);
        result.quantized[index] = static_cast<double>(
            integerNumerator[index].convert_to<long double>() / integerDenominator);
    }
    return result;
}

void ValidateAggregateReference(
    const DirectReference& reference,
    long double aggregateBound) {

    for (const double value : reference.ideal) {
        if (!std::isfinite(value) ||
            std::abs(static_cast<long double>(value)) > aggregateBound) {
            throw std::invalid_argument(
                "direct-RLWE aggregate exceeds its configured bound");
        }
    }
}

lbcrypto::BigInteger SignedIntegerToResidue(
    const exact_integer::Integer& value, const lbcrypto::BigInteger& modulus) {
    return exact_integer::Residue(value, modulus);
}

long double CenteredModularDifference(
    const lbcrypto::BigInteger& actual,
    const lbcrypto::BigInteger& expected,
    const lbcrypto::BigInteger& modulus) {

    const lbcrypto::BigInteger residue =
        (actual + modulus - expected).Mod(modulus);
    const lbcrypto::BigInteger halfModulus = modulus >> 1;
    const bool negative = residue > halfModulus;
    const lbcrypto::BigInteger magnitude = negative
        ? modulus - residue
        : residue;
    const long double converted = std::stold(magnitude.ToString());
    return negative ? -converted : converted;
}

ProtocolNoiseComponentResiduals MeasureDirectNoiseBreakdown(
    const DirectAggregateCiphertext& ciphertext,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const DirectRlweCompressionProfile& profile,
    const DirectReference& reference,
    std::size_t blockBegin,
    DCRTPoly encryptionPhase,
    DCRTPoly preKeySwitchModDownPhaseQP,
    DCRTPoly postKeySwitchModDownPhase) {

    DCRTPoly a = MaybeCompressDirectAggregateA(
        ciphertext.a,
        setup,
        profile);
    DCRTPoly collectiveSecret(
        setup.common.rns.paramsQ,
        Format::EVALUATION,
        true);
    for (const auto clientId : survivorKeys.publicKey.survivors.clients) {
        DCRTPoly secret = FindSecret(setup, clientId)
            .secretKeyQ->GetPrivateElement();
        secret.SetFormat(Format::EVALUATION);
        collectiveSecret += secret;
    }

    DCRTPoly phase = ciphertext.b + a * collectiveSecret;
    const lbcrypto::BigInteger modulus =
        setup.common.rns.paramsQ->GetModulus();
    const lbcrypto::BigInteger modulusQP =
        setup.common.rns.paramsQP->GetModulus();
    const long double specialModulus = std::stold(
        setup.common.rns.paramsP->GetModulus().ToString());
    DCRTPoly liftedEncryptionPhaseQP = LiftQPolynomialTimesPToQP(
        encryptionPhase,
        setup.common.rns);
    const auto totalValues =
        rns_polynomial::InterpolateCoefficients(std::move(phase));
    const auto encryptionValues = rns_polynomial::InterpolateCoefficients(
        std::move(encryptionPhase));
    const auto preKeySwitchValuesQP =
        rns_polynomial::InterpolateCoefficients(
            std::move(preKeySwitchModDownPhaseQP));
    const auto postKeySwitchValues = rns_polynomial::InterpolateCoefficients(
        std::move(postKeySwitchModDownPhase));
    const auto liftedEncryptionValuesQP =
        rns_polynomial::InterpolateCoefficients(
            std::move(liftedEncryptionPhaseQP));
    ProtocolNoiseComponentResiduals residuals;
    residuals.encryption.reserve(totalValues.size());
    residuals.keySwitchCore.reserve(totalValues.size());
    residuals.keySwitchModDown.reserve(totalValues.size());
    residuals.m2rCore.assign(totalValues.size(), 0.0L);
    residuals.m2rModDown.assign(totalValues.size(), 0.0L);
    residuals.aggregateACompression.reserve(totalValues.size());
    residuals.total.reserve(totalValues.size());
    for (std::size_t index = 0; index < totalValues.size(); ++index) {
        const std::size_t coordinate = blockBegin + index;
        const exact_integer::Integer expectedInteger =
            coordinate < reference.integerNumerator.size()
                ? reference.integerNumerator[coordinate]
                : exact_integer::Integer(0);
        const lbcrypto::BigInteger expected =
            SignedIntegerToResidue(expectedInteger, modulus);
        residuals.encryption.push_back(CenteredModularDifference(
            encryptionValues[index], expected, modulus));
        const long double keySwitchCore = CenteredModularDifference(
            preKeySwitchValuesQP[index],
            liftedEncryptionValuesQP[index],
            modulusQP) / specialModulus;
        const long double keySwitchTotal = CenteredModularDifference(
            postKeySwitchValues[index], encryptionValues[index], modulus);
        residuals.keySwitchCore.push_back(keySwitchCore);
        residuals.keySwitchModDown.push_back(
            keySwitchTotal - keySwitchCore);
        residuals.aggregateACompression.push_back(
            CenteredModularDifference(
                totalValues[index],
                postKeySwitchValues[index],
                modulus));
        residuals.total.push_back(CenteredModularDifference(
            totalValues[index], expected, modulus));
    }
    return residuals;
}

std::vector<double> Slice(
    const std::vector<double>& values,
    std::size_t begin,
    std::size_t count) {

    return std::vector<double>(
        values.begin() + begin,
        values.begin() + begin + count);
}

}  // namespace

DirectRlweUploadBatch DirectRlweAggregationEngine::EncryptUploads(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<DirectRlweClientInput>& clients,
    const ProtocolSetupResult& setup,
    const DirectRlweAggregationParameters& parameters,
    const DirectRlweCompressionProfile& compression,
    CryptoTaskExecutor* executor) {

    const auto started = std::chrono::steady_clock::now();
    ValidateProtocolParameters(config).ThrowIfInvalid();
    if (!cc || config.mode != setup.mode ||
        setup.mode != setup.common.descriptor.mode || clients.empty() ||
        clients.front().values.empty() || parameters.seedDomain.empty()) {
        throw std::invalid_argument("direct-RLWE upload input is invalid");
    }
    const std::size_t valueCount = clients.front().values.size();
    std::vector<bool> seen(config.clientCount, false);
    for (const auto& client : clients) {
        if (client.clientId.value >= config.clientCount ||
            seen[client.clientId.value] ||
            client.integerWeight != config.integerWeights[client.clientId.value]) {
            throw std::invalid_argument("direct-RLWE upload client metadata is invalid");
        }
        seen[client.clientId.value] = true;
        ValidateClientValues(client, valueCount);
        (void)FindSecret(setup, client.clientId);
    }
    const std::size_t ringDimension =
        setup.common.rns.paramsQ->GetRingDimension();
    const std::size_t blockCount =
        (valueCount + ringDimension - 1) / ringDimension;
    const long double scalingFactor = ScalingFactor(parameters.scaling);
    CryptoTaskExecutor sequentialExecutor(1);
    CryptoTaskExecutor& activeExecutor =
        executor == nullptr ? sequentialExecutor : *executor;

    DirectRlweUploadBatch result;
    result.setupId = setup.common.setupId;
    result.roundId = roundId;
    result.valueCount = valueCount;
    result.blockCount = blockCount;
    result.scalingFactor = scalingFactor;
    result.wirePacketsByClient.resize(config.clientCount);
    if (compression.measurePreSmudgingNoise) {
        result.shadowPacketsByClient.resize(config.clientCount);
    }
    result.clientEncryptionSeconds.assign(config.clientCount, 0.0);
    result.clientStageTimings.assign(config.clientCount, {});

    for (const auto& client : clients) {
        const auto clientId = client.clientId;
        const auto& secret = FindSecret(setup, clientId);
        auto& wirePackets = result.wirePacketsByClient.at(clientId.value);
        wirePackets.resize(blockCount);
        if (compression.measurePreSmudgingNoise) {
            result.shadowPacketsByClient.at(clientId.value).resize(blockCount);
        }
        CryptoStageMetrics encryptionMetrics;
        activeExecutor.ParallelFor(
            blockCount,
            CryptoStage::ENCRYPTION,
            &encryptionMetrics,
            [&](std::size_t block) {
                const std::size_t begin = block * ringDimension;
                const std::size_t validCount = std::min(
                    ringDimension,
                    valueCount - begin);
                DCRTPoly message = coefficient_codec::EncodeExact(
                    cc,
                    Slice(client.values, begin, validCount),
                    parameters.scaling.factor);
                const Seed128 seed = federated::UsesV17WireProtocol(config.protocolVersion)
                    ? SampleFreshProtocolSeed128()
                    : deterministic_randomness::DeriveSeed128(
                          config.masterSeed, parameters.seedDomain,
                          DirectSeedContext(roundId, clientId, block));
                SeededDenseRLWEPacket uncompressed =
                    EncryptSeededDenseRLWEUncompressed(
                        std::move(message),
                        secret,
                        setup.common,
                        roundId,
                        block,
                        validCount,
                        seed,
                        config.setupNoise.encryptionStandardDeviation,
                        federated::UsesV17WireProtocol(config.protocolVersion)
                            ? parameters.v17Path : std::string{});
                if (compression.measurePreSmudgingNoise) {
                    result.shadowPacketsByClient.at(clientId.value).at(block) =
                        std::make_unique<SeededDenseRLWEPacket>(uncompressed);
                }
                SeededDenseRLWEPacket wire = compression.compressClientB
                    ? CompressSeededDenseRLWEPacket(
                          uncompressed,
                          setup.common,
                          compression.c1,
                          federated::UsesV17WireProtocol(config.protocolVersion))
                    : std::move(uncompressed);
                wirePackets.at(block) =
                    std::make_unique<SeededDenseRLWEPacket>(std::move(wire));
            });
        const auto timing = encryptionMetrics.Snapshot().at(
            static_cast<std::size_t>(CryptoStage::ENCRYPTION));
        result.clientEncryptionSeconds.at(clientId.value) = timing.wallSeconds;
        AccumulateStageTiming(
            &result.clientStageTimings.at(clientId.value).at(
                static_cast<std::size_t>(CryptoStage::ENCRYPTION)),
            timing);
    }
    result.sequentialSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

DirectRlweAggregationResult DirectRlweAggregationEngine::AggregateUploaded(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<DirectRlweClientInput>& clients,
    const DirectRlweUploadBatch& uploads,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const DirectRlweAggregationParameters& parameters,
    const DirectRlweCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    const auto aggregateStarted = std::chrono::steady_clock::now();

    if (!cc) {
        throw std::invalid_argument(
            "direct-RLWE aggregation crypto context is null");
    }
    ValidateInputs(config, clients, setup, survivorKeys, parameters);
    if (!(uploads.setupId == setup.common.setupId) ||
        !(uploads.roundId == roundId)) {
        throw std::invalid_argument("direct-RLWE upload metadata mismatch");
    }
    if (compression.quantizeFinalResult) {
        coefficient_compression::ValidateNoClippingDeliveryRange(
            compression.cd,
            compression.deliveryScale,
            parameters.updateBound);
    }

    const std::size_t ringDimension =
        setup.common.rns.paramsQ->GetRingDimension();
    const std::size_t valueCount = clients.front().values.size();
    const std::size_t blockCount =
        (valueCount + ringDimension - 1) / ringDimension;
    const long double scalingFactor = ScalingFactor(parameters.scaling);
    // EncodeExact already validated each original upload, inside its client's
    // encryption batch. Do not repeat exact encoding serially at the server.
    DirectReference reference;
    if (compression.computeReference || compression.measurePreSmudgingNoise) {
        reference = ComputeReference(config, clients,
            survivorKeys.publicKey.survivors, valueCount, parameters.scaling.factor);
        ValidateAggregateReference(reference, parameters.updateBound);
    }

    CryptoTaskExecutor sequentialExecutor(1);
    CryptoTaskExecutor& activeExecutor =
        executor == nullptr ? sequentialExecutor : *executor;
    CryptoStageMetrics serverMetrics;

    std::vector<std::unique_ptr<DirectAggregateCiphertext>> aggregates(blockCount);
    std::vector<std::unique_ptr<DirectAggregateCiphertext>> shadowAggregates(
        compression.measurePreSmudgingNoise ? blockCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> encryptionPhaseAggregates(
        compression.measurePreSmudgingNoise ? blockCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> preKeySwitchPhaseAggregates(
        compression.measurePreSmudgingNoise ? blockCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> postKeySwitchPhaseAggregates(
        compression.measurePreSmudgingNoise ? blockCount : 0);
    DCRTPoly collectiveSecretQ(
        setup.common.rns.paramsQ,
        Format::EVALUATION,
        true);
    DCRTPoly collectiveSecretQP(
        setup.common.rns.paramsQP,
        Format::EVALUATION,
        true);
    if (compression.measurePreSmudgingNoise) {
        for (const auto clientId : survivorKeys.publicKey.survivors.clients) {
            DCRTPoly secretQ = FindSecret(setup, clientId)
                .secretKeyQ->GetPrivateElement();
            secretQ.SetFormat(Format::EVALUATION);
            collectiveSecretQ += secretQ;
            DCRTPoly secretQP = FindSecret(setup, clientId).secretKeyQP;
            secretQP.SetFormat(Format::EVALUATION);
            collectiveSecretQP += secretQP;
        }
    }
    DirectRlweAggregationResult result;
    result.scalingFactor = scalingFactor;
    result.valueCount = valueCount;
    result.blockCount = blockCount;
    result.clientEncryptionSeconds = uploads.clientEncryptionSeconds;
    result.clientKeySwitchSeconds.assign(config.clientCount, 0.0);
    result.clientPartialDecryptionSeconds.assign(config.clientCount, 0.0);
    result.clientStageTimings = uploads.clientStageTimings;
    const auto& survivorIds = survivorKeys.publicKey.survivors.clients;
    const std::size_t survivorCount = survivorIds.size();
    const std::size_t jobCount = survivorCount * blockCount;
    std::vector<const DirectRlweClientInput*> survivorClients;
    std::vector<const ClientSecretSetupMaterial*> survivorSecrets;
    survivorClients.reserve(survivorCount);
    survivorSecrets.reserve(survivorCount);
    for (const auto clientId : survivorIds) {
        survivorClients.push_back(&FindClient(clients, clientId));
        survivorSecrets.push_back(&FindSecret(setup, clientId));
    }

    if (uploads.valueCount != valueCount ||
        uploads.blockCount != blockCount ||
        uploads.scalingFactor != scalingFactor ||
        uploads.wirePacketsByClient.size() != config.clientCount ||
        uploads.clientEncryptionSeconds.size() != config.clientCount ||
        uploads.clientStageTimings.size() != config.clientCount ||
        (compression.measurePreSmudgingNoise &&
         uploads.shadowPacketsByClient.size() != config.clientCount)) {
        throw std::invalid_argument("direct-RLWE upload shape mismatch");
    }
    const auto& wirePacketsByClient = uploads.wirePacketsByClient;
    const auto& shadowPacketsByClient = uploads.shadowPacketsByClient;

    std::vector<std::unique_ptr<CollectiveRLWECiphertext>> switchedJobs(
        jobCount);
    std::vector<std::unique_ptr<CollectiveRLWECiphertext>> shadowSwitchedJobs(
        compression.measurePreSmudgingNoise ? jobCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> encryptionPhaseJobs(
        compression.measurePreSmudgingNoise ? jobCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> preKeySwitchPhaseJobs(
        compression.measurePreSmudgingNoise ? jobCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> postKeySwitchPhaseJobs(
        compression.measurePreSmudgingNoise ? jobCount : 0);
    for (std::size_t clientIndex = 0;
         clientIndex < survivorCount;
         ++clientIndex) {
        const auto clientId = survivorIds[clientIndex];
        const auto& client = *survivorClients[clientIndex];
        CryptoStageMetrics keySwitchMetrics;
        activeExecutor.ParallelFor(
            blockCount,
            CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH,
            &keySwitchMetrics,
            [&](std::size_t block) {
                const std::size_t job = clientIndex * blockCount + block;
                const RestoredDenseRLWECiphertext restored =
                    RestoreSeededDenseRLWE(
                        *wirePacketsByClient.at(clientId.value).at(block),
                        setup.common);
                CollectiveRLWECiphertext switched =
                    KeySwitchClientToCollective(
                        restored,
                        survivorKeys,
                        setup.common);
                MultiplyPolynomialByInteger(
                    &switched.b,
                    client.integerWeight);
                MultiplyPolynomialByInteger(
                    &switched.a,
                    client.integerWeight);
                switchedJobs[job] =
                    std::make_unique<CollectiveRLWECiphertext>(
                        std::move(switched));

                if (compression.measurePreSmudgingNoise) {
                    const RestoredDenseRLWECiphertext shadowRestored =
                        RestoreSeededDenseRLWE(
                            *shadowPacketsByClient.at(clientId.value).at(block),
                            setup.common);
                    HybridModDownTrace modDownTrace;
                    CollectiveRLWECiphertext shadowSwitched =
                        KeySwitchClientToCollective(
                            shadowRestored,
                            FindClientToCollectiveKey(
                                survivorKeys,
                                clientId),
                            setup.common,
                            &modDownTrace);
                    DCRTPoly sourceSecretQ = survivorSecrets[clientIndex]
                        ->encryptionKeyQ->GetPrivateElement();
                    sourceSecretQ.SetFormat(Format::EVALUATION);
                    DCRTPoly encryptionPhase =
                        shadowRestored.b +
                        shadowRestored.a * sourceSecretQ;
                    DCRTPoly preKeySwitchPhase = BuildPreModDownPhaseQP(
                        shadowRestored.b,
                        modDownTrace,
                        collectiveSecretQP,
                        setup.common.rns);
                    DCRTPoly postKeySwitchPhase =
                        shadowSwitched.b +
                        shadowSwitched.a * collectiveSecretQ;
                    MultiplyPolynomialByInteger(
                        &shadowSwitched.b,
                        client.integerWeight);
                    MultiplyPolynomialByInteger(
                        &shadowSwitched.a,
                        client.integerWeight);
                    MultiplyPolynomialByInteger(
                        &encryptionPhase,
                        client.integerWeight);
                    MultiplyPolynomialByInteger(
                        &preKeySwitchPhase,
                        client.integerWeight);
                    MultiplyPolynomialByInteger(
                        &postKeySwitchPhase,
                        client.integerWeight);
                    shadowSwitchedJobs[job] =
                        std::make_unique<CollectiveRLWECiphertext>(
                            std::move(shadowSwitched));
                    encryptionPhaseJobs[job] =
                        std::make_unique<DCRTPoly>(
                            std::move(encryptionPhase));
                    preKeySwitchPhaseJobs[job] =
                        std::make_unique<DCRTPoly>(
                            std::move(preKeySwitchPhase));
                    postKeySwitchPhaseJobs[job] =
                        std::make_unique<DCRTPoly>(
                            std::move(postKeySwitchPhase));
                }
            });
        const auto keySwitchTiming = keySwitchMetrics.Snapshot().at(
            static_cast<std::size_t>(
                CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH));
        result.clientKeySwitchSeconds.at(clientId.value) =
            keySwitchTiming.wallSeconds;
        AccumulateStageTiming(
            &result.clientStageTimings.at(clientId.value).at(
                static_cast<std::size_t>(
                    CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH)),
            keySwitchTiming);
    }
    result.clientCiphertextCount = clients.size() * blockCount;
    result.survivorClientCiphertextCount = jobCount;
    result.discardedClientCiphertextCount =
        result.clientCiphertextCount - result.survivorClientCiphertextCount;

    std::vector<std::unique_ptr<DCRTPoly>> compressedAggregateA(blockCount);
    std::vector<std::unique_ptr<ProtocolNoiseComponentResiduals>> noiseResiduals(
        compression.measurePreSmudgingNoise ? blockCount : 0);
    activeExecutor.ParallelFor(
        blockCount,
        CryptoStage::RLWE_AGGREGATION,
        &serverMetrics,
        [&](std::size_t block) {
            const std::size_t validCount = std::min(
                ringDimension,
                valueCount - block * ringDimension);
            auto aggregate = std::make_unique<DirectAggregateCiphertext>();
            aggregate->blockIndex = block;
            aggregate->validValueCount = validCount;
            for (std::size_t clientIndex = 0;
                 clientIndex < survivorCount;
                 ++clientIndex) {
                auto& switched = *switchedJobs[
                    clientIndex * blockCount + block];
                if (clientIndex == 0) {
                    aggregate->b = std::move(switched.b);
                    aggregate->a = std::move(switched.a);
                }
                else {
                    aggregate->b += switched.b;
                    aggregate->a += switched.a;
                }
            }
            aggregates[block] = std::move(aggregate);

            if (compression.measurePreSmudgingNoise) {
                auto shadowAggregate =
                    std::make_unique<DirectAggregateCiphertext>();
                shadowAggregate->blockIndex = block;
                shadowAggregate->validValueCount = validCount;
                auto reducePhase = [&](
                    std::vector<std::unique_ptr<DCRTPoly>>& jobs) {
                    std::unique_ptr<DCRTPoly> phase;
                    for (std::size_t clientIndex = 0;
                         clientIndex < survivorCount;
                         ++clientIndex) {
                        auto& contribution = jobs[
                            clientIndex * blockCount + block];
                        if (clientIndex == 0) {
                            phase = std::make_unique<DCRTPoly>(
                                std::move(*contribution));
                        }
                        else {
                            *phase += *contribution;
                        }
                    }
                    return phase;
                };
                for (std::size_t clientIndex = 0;
                     clientIndex < survivorCount;
                     ++clientIndex) {
                    auto& switched = *shadowSwitchedJobs[
                        clientIndex * blockCount + block];
                    if (clientIndex == 0) {
                        shadowAggregate->b = std::move(switched.b);
                        shadowAggregate->a = std::move(switched.a);
                    }
                    else {
                        shadowAggregate->b += switched.b;
                        shadowAggregate->a += switched.a;
                    }
                }
                shadowAggregates[block] = std::move(shadowAggregate);
                encryptionPhaseAggregates[block] =
                    reducePhase(encryptionPhaseJobs);
                preKeySwitchPhaseAggregates[block] =
                    reducePhase(preKeySwitchPhaseJobs);
                postKeySwitchPhaseAggregates[block] =
                    reducePhase(postKeySwitchPhaseJobs);
                noiseResiduals[block] =
                    std::make_unique<ProtocolNoiseComponentResiduals>(
                        MeasureDirectNoiseBreakdown(
                            *shadowAggregates[block],
                            setup,
                            survivorKeys,
                            compression,
                            reference,
                            block * ringDimension,
                            *encryptionPhaseAggregates[block],
                            *preKeySwitchPhaseAggregates[block],
                            *postKeySwitchPhaseAggregates[block]));
            }
            compressedAggregateA[block] = std::make_unique<DCRTPoly>(
                MaybeCompressDirectAggregateA(
                    aggregates[block]->a,
                    setup,
                    compression));
        });
    if (compression.measurePreSmudgingNoise) {
        for (std::size_t block = 0; block < blockCount; ++block) {
            AppendProtocolNoiseBreakdownGroup(
                &result.preSmudgingNoiseBreakdown,
                *noiseResiduals[block]);
            AppendProtocolNoiseGroup(
                &result.preSmudgingNoise,
                MeasureProtocolNoiseGroup(noiseResiduals[block]->total));
        }
    }
    switchedJobs.clear();
    shadowSwitchedJobs.clear();
    encryptionPhaseJobs.clear();
    preKeySwitchPhaseJobs.clear();
    postKeySwitchPhaseJobs.clear();

    const bool v17C2 = federated::UsesV17WireProtocol(config.protocolVersion) &&
        compression.compressPartialDecryption;
    const Seed128 seedD = v17C2
        ? (compression.v17SeedGate
              ? compression.v17SeedGate->ArriveAfterAggregateA()
              : FreshV17DecryptionSeed128())
        : Seed128{};

    std::vector<std::unique_ptr<DCRTPoly>> partialShares(jobCount);
    std::vector<coefficient_compression::CompressedCoefficientPayload>
        v17ResponseCodes(v17C2 ? jobCount : 0);
    for (std::size_t clientIndex = 0;
         clientIndex < survivorCount;
         ++clientIndex) {
        const auto clientId = survivorIds[clientIndex];
        CryptoStageMetrics partialDecryptionMetrics;
        activeExecutor.ParallelFor(
            blockCount,
            CryptoStage::PARTIAL_DECRYPTION,
            &partialDecryptionMetrics,
            [&](std::size_t block) {
                const std::size_t job = clientIndex * blockCount + block;
                DirectRlweCompressionProfile shareProfile = compression;
                if (v17C2) {
                    shareProfile.compressPartialDecryption = false;
                }
                DCRTPoly partial = ComputeDirectPartialShare(
                    *compressedAggregateA[block], setup, clientId,
                    parameters, shareProfile);
                if (v17C2) {
                    v17ResponseCodes[job] = MakeV17PartialResponse(
                        std::move(partial), setup.common.rns.paramsQ,
                        seedD, setup.common.setupId, roundId,
                        parameters.v17Path, block, clientId,
                        survivorIds.front(), compression.c2);
                }
                else {
                    partialShares[job] = std::make_unique<DCRTPoly>(
                        std::move(partial));
                }
            });
        const auto partialDecryptionTiming =
            partialDecryptionMetrics.Snapshot().at(
                static_cast<std::size_t>(
                    CryptoStage::PARTIAL_DECRYPTION));
        result.clientPartialDecryptionSeconds.at(clientId.value) =
            partialDecryptionTiming.wallSeconds;
        AccumulateStageTiming(
            &result.clientStageTimings.at(clientId.value).at(
                static_cast<std::size_t>(
                    CryptoStage::PARTIAL_DECRYPTION)),
            partialDecryptionTiming);
    }

    std::vector<std::unique_ptr<DCRTPoly>> recoveredPhases(blockCount);
    activeExecutor.ParallelFor(
        blockCount,
        CryptoStage::PARTIAL_DECRYPTION,
        &serverMetrics,
        [&](std::size_t block) {
            if (v17C2) {
                std::vector<
                    coefficient_compression::CompressedCoefficientPayload>
                    blockCodes;
                blockCodes.reserve(survivorCount);
                for (std::size_t clientIndex = 0;
                     clientIndex < survivorCount; ++clientIndex) {
                    blockCodes.push_back(std::move(v17ResponseCodes[
                        clientIndex * blockCount + block]));
                }
                recoveredPhases[block] = std::make_unique<DCRTPoly>(
                    RecoverV17Phase(
                        aggregates[block]->b, setup.common.rns.paramsQ,
                        blockCodes, survivorIds, seedD,
                        setup.common.setupId, roundId,
                        parameters.v17Path, block));
                return;
            }
            DCRTPoly phase = aggregates[block]->b;
            phase.SetFormat(Format::EVALUATION);
            for (std::size_t clientIndex = 0;
                 clientIndex < survivorCount;
                 ++clientIndex) {
                phase += *partialShares[
                    clientIndex * blockCount + block];
            }
            recoveredPhases[block] =
                std::make_unique<DCRTPoly>(std::move(phase));
        });
    partialShares.clear();
    compressedAggregateA.clear();

    std::vector<double> decoded(valueCount, 0.0);
    if (compression.measurePreSmudgingNoise) {
        result.phaseAudit.phaseResidues.resize(valueCount);
        result.phaseAudit.integerReferences.resize(valueCount);
    }
    const long double denominator =
        static_cast<long double>(config.weightDenominator) * scalingFactor;
    activeExecutor.ParallelFor(
        blockCount,
        CryptoStage::DECODING,
        &serverMetrics,
        [&](std::size_t block) {
            const std::size_t begin = block * ringDimension;
            const std::size_t validCount = std::min(
                ringDimension,
                valueCount - begin);
            if (compression.measurePreSmudgingNoise) {
                const auto residues = rns_polynomial::InterpolateCoefficients(*recoveredPhases[block]);
                for (std::size_t j=0;j<validCount;++j) {
                    const auto& ref=reference.integerNumerator[begin+j];
                    result.phaseAudit.phaseResidues[begin+j]=residues[j].ToString();
                    result.phaseAudit.integerReferences[begin+j]=ref.str();
                }
            }
            const auto integers = DecodeDirectPhase(
                std::move(*recoveredPhases[block]),
                validCount,
                setup);
            for (std::size_t offset = 0;
                 offset < integers.size();
                 ++offset) {
                const long double integer = integers[offset];
                double value = static_cast<double>(integer / denominator);
                if (compression.quantizeFinalResult) {
                    const std::int64_t quantized =
                        coefficient_compression::QuantizeSignedWithScale(
                            value,
                            compression.cd,
                            compression.deliveryScale);
                    value = static_cast<double>(
                        coefficient_compression::DequantizeSignedWithScale(
                            quantized,
                            compression.cd,
                            compression.deliveryScale));
                }
                decoded[begin + offset] = value;
            }
        });
    result.aggregateCiphertextCount = blockCount;

    result.idealReference = reference.ideal;
    result.quantizedReference = reference.quantized;
    result.aggregate = std::move(decoded);
    result.serverStageTimings = serverMetrics.Snapshot();
    result.cryptoStageTimings = result.serverStageTimings;
    for (const auto& clientTimings : result.clientStageTimings) {
        AccumulateStageTimings(
            &result.cryptoStageTimings,
            clientTimings);
    }
    if (aggregateMetrics != nullptr) {
        for (std::size_t index = 0; index < kCryptoStageCount; ++index) {
            const auto& timing = result.cryptoStageTimings[index];
            aggregateMetrics->Add(
                static_cast<CryptoStage>(index),
                timing.wallSeconds,
                timing.workSeconds,
                timing.taskCount, timing.executorWaitSeconds);
        }
    }
    result.sequentialSeconds = uploads.sequentialSeconds +
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - aggregateStarted).count();
    return result;
}

DirectRlweAggregationResult DirectRlweAggregationEngine::Aggregate(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    RoundId roundId,
    const std::vector<DirectRlweClientInput>& clients,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const DirectRlweAggregationParameters& parameters,
    const DirectRlweCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    DirectRlweUploadBatch uploads = EncryptUploads(
        cc, config, roundId, clients, setup, parameters, compression, executor);
    return AggregateUploaded(
        cc, config, roundId, clients, uploads, setup, survivorKeys,
        parameters, compression, executor, aggregateMetrics);
}

long double DirectRlweCompressionModelBound(
    const ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount,
    const DirectRlweAggregationParameters& parameters,
    const DirectRlweCompressionProfile& compression) {

    const long double denominator =
        static_cast<long double>(config.weightDenominator) * scalingFactor;
    const long double weightSum = config.weightDenominator;
    const long double c1 = compression.compressClientB
        ? weightSum * q /
            (denominator * std::ldexp(1.0L, compression.c1 + 1))
        : 0.0L;
    const long double c2 = compression.compressPartialDecryption
        ? survivorCount * q /
            (denominator * std::ldexp(1.0L, compression.c2 + 1))
        : 0.0L;
    const long double ca = compression.compressAggregateA
        ? static_cast<long double>(q) *
            config.modulus.ringDimension * survivorCount /
            (denominator * std::ldexp(1.0L, compression.ca + 1))
        : 0.0L;
    const long double cd = compression.quantizeFinalResult
        ? 1.0L / (2.0L * compression.deliveryScale)
        : 0.0L;
    const long double decodedSigmaSmg =
        std::sqrt(static_cast<long double>(survivorCount)) *
        parameters.sigmaSmg /
        denominator;
    return c1 + c2 + ca + cd +
        6.0L * decodedSigmaSmg + 1.0e-4L;
}

}  // namespace federated
