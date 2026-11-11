#include "protocol-rounding.h"
#include "protocol-sparse-path.h"

#include "protocol-randomness.h"
#include "support-bitmap.h"

#include "coefficient-codec.h"
#include "coefficient-compression.h"
#include "deterministic-randomness.h"
#include "rns-polynomial.h"
#include "ring-conversion.h"
#include "seeded-rlwe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

using SparseClientCiphertextPacket = SparseUploadCiphertextPacket;
using SparseClientPacketPair = SparseUploadPacketPair;

struct ProcessedSparsePacket {
    std::unique_ptr<CollectiveRLWECiphertext> switched;
    std::unique_ptr<DCRTPoly> encryptionPhase;
    std::unique_ptr<DCRTPoly> preKeySwitchPhase;
    std::unique_ptr<DCRTPoly> postKeySwitchPhase;
    std::vector<LabeledMLWECiphertext> mlwe;
    std::vector<LabeledMLWECiphertext> encryptionPhaseMlwe;
    std::vector<LabeledMLWECiphertext> preKeySwitchPhaseMlwe;
    std::vector<LabeledMLWECiphertext> postKeySwitchPhaseMlwe;
};

struct SparseServerExecution {
    SparseServerPathResult result;
    std::vector<PackedAggregateRLWECiphertext> finalCiphertexts;
    struct NoiseCheckpoints {
        DCRTPoly compressedA;
        DCRTPoly encryptionPhase;
        DCRTPoly preKeySwitchModDownPhaseQP;
        DCRTPoly postKeySwitchModDownPhase;
        DCRTPoly preM2RModDownPhaseQP;
        DCRTPoly postM2RModDownPhase;
    };
    std::vector<NoiseCheckpoints> noiseCheckpoints;
};

struct ReferenceValues {
    std::vector<double> ideal;
    std::vector<double> quantized;
    std::vector<std::uint8_t> mask;
    std::vector<exact_integer::Integer> integerNumerator;
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

const ProtocolClientRoundInput& FindClient(
    const std::vector<ProtocolClientRoundInput>& clients,
    ClientId clientId) {

    const auto found = std::find_if(
        clients.begin(),
        clients.end(),
        [clientId](const ProtocolClientRoundInput& client) {
            return client.clientId == clientId;
        });
    if (found == clients.end()) {
        throw std::invalid_argument("protocol sparse path client is missing");
    }
    return *found;
}

const SparseIndexAnnouncement& FindAnnouncement(
    const ServerSparseAdmissionPlan& plan,
    ClientId clientId) {

    const auto found = std::find_if(
        plan.announcements.begin(),
        plan.announcements.end(),
        [clientId](const SparseIndexAnnouncement& announcement) {
            return announcement.clientId == clientId;
        });
    if (found == plan.announcements.end()) {
        throw std::invalid_argument("server announcement is missing");
    }
    return *found;
}

const SparseUploadSupportHeader& FindUploadSupportHeader(
    const std::vector<SparseUploadSupportHeader>& headers,
    ClientId clientId) {

    const auto found = std::find_if(
        headers.begin(), headers.end(),
        [clientId](const SparseUploadSupportHeader& header) {
            return header.clientId == clientId;
        });
    if (found == headers.end()) {
        throw std::invalid_argument("missing sparse upload-support header");
    }
    return *found;
}

PackedShuffledLabels DeriveUploadGroupLabels(
    const SparseUploadSupportHeader& header,
    std::size_t chunkCount,
    std::size_t mlweRank,
    std::size_t groupIndex) {

    if (header.admissionRevision != 0 || mlweRank == 0) {
        throw std::invalid_argument("invalid sparse upload-support header");
    }
    const auto labels = DecodeSupportBitmapMsb0(
        chunkCount,
        header.supportBitmap);
    const std::size_t groupCount =
        labels.size() / mlweRank +
        (labels.size() % mlweRank == 0 ? 0U : 1U);
    if (groupIndex >= groupCount) {
        throw std::invalid_argument("sparse upload group index is out of range");
    }
    const std::size_t begin = groupIndex * mlweRank;
    const std::size_t end = std::min(begin + mlweRank, labels.size());
    PackedShuffledLabels result;
    result.labels.insert(
        result.labels.end(), labels.begin() + begin, labels.begin() + end);
    return result;
}

const ClientSecretSetupMaterial& FindClientSecret(
    const ProtocolSetupResult& setup,
    ClientId clientId) {

    const auto found = std::find_if(
        setup.clientSecrets.begin(),
        setup.clientSecrets.end(),
        [clientId](const ClientSecretSetupMaterial& secret) {
            return secret.clientId == clientId;
        });
    if (found == setup.clientSecrets.end()) {
        throw std::invalid_argument("protocol sparse path secret is missing");
    }
    return *found;
}

ByteVector CiphertextSeedContext(
    RoundId roundId,
    ClientId clientId,
    std::size_t groupIndex) {

    ByteVector result;
    const auto append = [&result](std::uint64_t value) {
        const ByteVector encoded =
            deterministic_randomness::EncodeBigEndian64(value);
        result.insert(result.end(), encoded.begin(), encoded.end());
    };
    append(roundId.value);
    append(clientId.value);
    append(groupIndex);
    return result;
}

std::vector<double> PackClientGroup(
    const ProtocolClientRoundInput& client,
    const ProtocolClientPackingGroup& group,
    const ClientSparsePrivateRoundState& state) {

    std::vector<double> packed(state.chunkSize * state.mlweRank, 0.0);
    for (std::size_t slot = 0; slot < group.actualChunks.size(); ++slot) {
        const std::size_t begin =
            group.actualChunks[slot].value * state.chunkSize;
        const std::size_t end = std::min(
            begin + state.chunkSize,
            client.update.size());
        for (std::size_t offset = 0; offset < end - begin; ++offset) {
            packed[offset * state.mlweRank + slot] =
                client.update[begin + offset];
        }
    }
    return packed;
}

std::vector<SparseClientPacketPair> EncryptClientPackets(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& round,
    const ProtocolSetupResult& setup,
    long double /*scalingFactor*/,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor& executor,
    std::vector<double>* clientSeconds,
    std::vector<CryptoStageTimings>* clientStageTimings) {

    std::vector<SparseClientPacketPair> result;
    for (const auto& state : round.clientPrivateStates) {
        const auto& client = FindClient(clients, state.clientId);
        ValidateClientSparsePrivateRoundState(
            config,
            client,
            FindAnnouncement(round.serverPlan, state.clientId),
            round.serverPlan,
            state);
        const auto& secret = FindClientSecret(setup, state.clientId);
        std::vector<std::unique_ptr<SparseClientPacketPair>> clientPackets(
            state.packingGroups.size());
        CryptoStageMetrics encryptionMetrics;
        executor.ParallelFor(
            state.packingGroups.size(),
            CryptoStage::ENCRYPTION,
            &encryptionMetrics,
            [&](std::size_t groupIndex) {
                const auto& group = state.packingGroups[groupIndex];
                const auto packed = PackClientGroup(client, group, state);
                DCRTPoly message = coefficient_codec::EncodeExact(
                    cc,
                    packed,
                    config.scaling.factor);
                const Seed128 seed = federated::UsesV17WireProtocol(config.protocolVersion)
                    ? SampleFreshProtocolSeed128()
                    : deterministic_randomness::DeriveSeed128(
                          config.masterSeed, "ROUND-ENC-A-SEED",
                          CiphertextSeedContext(state.roundId,
                              state.clientId, group.groupIndex));

                auto output = std::make_unique<SparseClientPacketPair>();
                output->bctShadow.clientId = state.clientId;
                output->bctShadow.integerWeight = client.integerWeight;
                output->bctShadow.groupIndex = group.groupIndex;
                output->bctShadow.derivedLabels = group.shuffledLabels;
                output->bctShadow.ciphertext =
                    EncryptSeededSparseRLWEUncompressed(
                        message,
                        secret,
                        setup.common,
                        state.roundId,
                        group.groupIndex,
                        group.actualChunks.size(),
                        seed,
                        config.setupNoise.encryptionStandardDeviation,
                        federated::UsesV17WireProtocol(config.protocolVersion));
                output->wire = output->bctShadow;
                output->wire.ciphertext = compression.compressClientB
                    ? CompressSeededSparseRLWEPacket(
                          output->bctShadow.ciphertext,
                          setup.common,
                          compression.c1,
                          federated::UsesV17WireProtocol(config.protocolVersion))
                    : output->bctShadow.ciphertext;
                clientPackets[groupIndex] = std::move(output);
            });
        const auto timing = encryptionMetrics.Snapshot().at(
            static_cast<std::size_t>(CryptoStage::ENCRYPTION));
        if (clientSeconds != nullptr) {
            clientSeconds->at(state.clientId.value) = timing.wallSeconds;
        }
        if (clientStageTimings != nullptr) {
            AccumulateStageTiming(
                &clientStageTimings->at(state.clientId.value).at(
                    static_cast<std::size_t>(CryptoStage::ENCRYPTION)),
                timing);
        }
        for (auto& packet : clientPackets) {
            result.push_back(std::move(*packet));
        }
    }
    return result;
}

DCRTPoly SampleSmudging(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    long double sigma, bool certifiedV17) {

    if (sigma == 0.0L) {
        return DCRTPoly(parameters, Format::EVALUATION, true);
    }
    if (!std::isfinite(sigma) || sigma < 0.0L ||
        sigma > std::ldexp(1.0L, certifiedV17 ? 128 : 59)) {
        throw std::invalid_argument("invalid sparse smudging sigma");
    }
    return SampleProtocolGaussianPolynomial(parameters, sigma, certifiedV17);
}

DCRTPoly MaybeCompressAggregateA(
    DCRTPoly a,
    const ProtocolRnsParameters& rns,
    const SparsePathCompressionProfile& profile) {

    if (!profile.compressAggregateA) {
        a.SetFormat(Format::EVALUATION);
        return a;
    }
    const auto payload = coefficient_compression::CompressAggregateCoefficients(
        a,
        profile.ca);
    DCRTPoly restored = coefficient_compression::DecompressAllCoefficients(
        rns.paramsQ,
        payload);
    restored.SetFormat(Format::EVALUATION);
    return restored;
}

std::vector<LabeledAggregateChunk> DecodeLabeledAggregateGroup(
    const PackedAggregateRLWECiphertext& ciphertext,
    const std::vector<long double>& selectedValues,
    const ProtocolExperimentConfig& config,
    const ServerSparseAdmissionPlan& plan,
    const SparsePathCompressionProfile& profile,
    long double scalingFactor) {

    const std::size_t activeCount = ciphertext.shuffledLabels.labels.size();
    if (selectedValues.size() != plan.chunkSize * activeCount) {
        throw std::runtime_error("threshold sparse result has invalid size");
    }
    const long double denominator =
        static_cast<long double>(config.weightDenominator) * scalingFactor;
    std::vector<LabeledAggregateChunk> result(activeCount);
    for (std::size_t slot = 0; slot < activeCount; ++slot) {
        result[slot].label = ciphertext.shuffledLabels.labels[slot];
        result[slot].values.resize(plan.chunkSize);
        for (std::size_t offset = 0; offset < plan.chunkSize; ++offset) {
            double value = static_cast<double>(
                selectedValues[offset * activeCount + slot] / denominator);
            if (profile.quantizeFinalResult) {
                const std::int64_t quantized =
                    coefficient_compression::QuantizeSignedWithScale(
                        value,
                        profile.cd,
                        profile.deliveryScale);
                value = static_cast<double>(
                    coefficient_compression::DequantizeSignedWithScale(
                        quantized,
                        profile.cd,
                        profile.deliveryScale));
            }
            result[slot].values[offset] = value;
        }
    }
    return result;
}

CollectiveRLWECiphertext MakePhaseOnlyCiphertext(
    const CollectiveRLWECiphertext& prototype,
    DCRTPoly phase) {

    CollectiveRLWECiphertext result = prototype;
    result.b = std::move(phase);
    result.a = DCRTPoly(
        result.b.GetParams(),
        Format::EVALUATION,
        true);
    return result;
}

LabeledMLWECiphertext MoveRankOneRLWEToLabeled(
    CollectiveRLWECiphertext ciphertext,
    const PackedShuffledLabels& labels) {

    if (labels.labels.size() != 1 || ciphertext.activeStreamCount != 1) {
        throw std::invalid_argument(
            "rank-one sparse RLWE requires exactly one shuffled label");
    }
    LabeledMLWECiphertext result;
    result.label = labels.labels.front();
    result.rank = 1;
    result.ringDimension = ciphertext.a.GetParams()->GetRingDimension();
    result.a.push_back(std::move(ciphertext.a));
    result.b = std::move(ciphertext.b);
    return result;
}

PackedAggregateRLWECiphertext MoveRankOneAggregateToRLWE(
    LabeledMLWECiphertext ciphertext,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId) {

    if (labels.labels.size() != 1 ||
        !(ciphertext.label == labels.labels.front()) ||
        ciphertext.rank != 1 || ciphertext.a.size() != 1 ||
        ciphertext.ringDimension !=
            publicMaterial.rns.paramsQ->GetRingDimension()) {

        throw std::invalid_argument("invalid rank-one sparse RLWE aggregate");
    }
    PackedAggregateRLWECiphertext result;
    result.setupId = publicMaterial.setupId;
    result.roundId = roundId;
    result.survivors = survivorKeys.publicKey.survivors;
    result.shuffledLabels = labels;
    result.meaningfulCoefficientIndices = SparseStreamCoefficientIndices(
        publicMaterial.rns.paramsQ->GetRingDimension(),
        1,
        1);
    result.b = std::move(ciphertext.b);
    result.a = std::move(ciphertext.a.front());
    return result;
}

DCRTPoly PackMLWEPhaseRows(
    const std::vector<LabeledMLWECiphertext>& inputs,
    std::size_t rank) {

    if (inputs.empty() || inputs.size() > rank) {
        throw std::invalid_argument("invalid phase-only MLWE packing input");
    }
    std::vector<DCRTPoly> rows;
    rows.reserve(rank);
    for (const auto& input : inputs) {
        rows.push_back(input.b);
    }
    while (rows.size() < rank) {
        DCRTPoly zero(
            inputs.front().b.GetParams(),
            Format::EVALUATION,
            true);
        rows.push_back(std::move(zero));
    }
    DCRTPoly result = ring_conversion::Combine(rows);
    result.SetFormat(Format::EVALUATION);
    return result;
}

SparseServerExecution RunLabelOnlyServer(
    const ProtocolExperimentConfig& config,
    const ServerSparseAdmissionPlan& plan,
    const std::vector<SparseClientCiphertextPacket>& packets,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    long double scalingFactor,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor& executor,
    CryptoStageMetrics* serverMetrics,
    std::vector<double>* clientKeySwitchSeconds,
    std::vector<double>* clientPartialDecryptionSeconds,
    std::vector<CryptoStageTimings>* clientStageTimings,
    bool collectNoiseBreakdown = false) {

    ValidateServerSparseAdmissionPlan(
        config,
        survivorKeys.publicKey.survivors,
        plan);
    SparseServerExecution execution;
    execution.result.planHash = plan.publicPlanHash;
    std::vector<ProcessedSparsePacket> processed(packets.size());
    std::vector<std::unique_ptr<LabeledMLWECiphertext>> encryptionPhases(
        collectNoiseBreakdown ? plan.chunkCount : 0);
    std::vector<std::unique_ptr<LabeledMLWECiphertext>> preKeySwitchPhases(
        collectNoiseBreakdown ? plan.chunkCount : 0);
    std::vector<std::unique_ptr<LabeledMLWECiphertext>> postKeySwitchPhases(
        collectNoiseBreakdown ? plan.chunkCount : 0);
    DCRTPoly collectiveSecretQ(
        setup.common.rns.paramsQ,
        Format::EVALUATION,
        true);
    DCRTPoly collectiveSecretQP(
        setup.common.rns.paramsQP,
        Format::EVALUATION,
        true);
    if (collectNoiseBreakdown) {
        for (const auto clientId : survivorKeys.publicKey.survivors.clients) {
            DCRTPoly secretQ = FindClientSecret(setup, clientId)
                .secretKeyQ->GetPrivateElement();
            secretQ.SetFormat(Format::EVALUATION);
            collectiveSecretQ += secretQ;
            DCRTPoly secretQP = FindClientSecret(setup, clientId).secretKeyQP;
            secretQP.SetFormat(Format::EVALUATION);
            collectiveSecretQP += secretQP;
        }
    }
    DCRTPoly modelSecretQ(setup.common.rns.paramsQ, Format::EVALUATION, true);
    DCRTPoly modelSecretQP(setup.common.rns.paramsQP, Format::EVALUATION, true);
    if (collectNoiseBreakdown) {
        for (const auto id : survivorKeys.publicKey.survivors.clients) {
            const auto& keys = FindClientSecret(setup, id);
            modelSecretQ += (keys.modelKeyQ ? keys.modelKeyQ : keys.secretKeyQ)
                ->GetPrivateElement();
            modelSecretQP += setup.common.mlweRank > 1
                ? keys.modelKeyQP : keys.secretKeyQP;
        }
    }
    std::set<std::pair<std::size_t, std::size_t>> receivedContributions;
    std::vector<std::vector<std::size_t>> packetIndicesByClient(
        config.clientCount);

    for (std::size_t packetIndex = 0;
         packetIndex < packets.size();
         ++packetIndex) {
        const auto& packet = packets[packetIndex];
        if (packet.clientId.value >= config.clientCount ||
            packet.integerWeight !=
                config.integerWeights[packet.clientId.value] ||
            packet.derivedLabels.labels.empty() ||
            packet.derivedLabels.labels.size() > plan.mlweRank ||
            !(packet.ciphertext.setupId == plan.setupId) ||
            !(packet.ciphertext.roundId == plan.roundId) ||
            !(packet.ciphertext.clientId == packet.clientId) ||
            packet.ciphertext.activeStreamCount !=
                packet.derivedLabels.labels.size() ||
            !std::is_sorted(
                packet.derivedLabels.labels.begin(),
                packet.derivedLabels.labels.end()) ||
            std::adjacent_find(
                packet.derivedLabels.labels.begin(),
                packet.derivedLabels.labels.end()) !=
                    packet.derivedLabels.labels.end()) {
            throw std::invalid_argument("server received invalid client metadata");
        }
        const auto& announcement = FindAnnouncement(plan, packet.clientId);
        for (const auto label : packet.derivedLabels.labels) {
            if (!SupportBitmapContains(
                    announcement.supportBitmap,
                    plan.chunkCount,
                    label) ||
                !receivedContributions.emplace(
                    packet.clientId.value,
                    label.value).second) {
                throw std::invalid_argument(
                    "ciphertext label is unannounced or duplicated");
            }
        }
        packetIndicesByClient.at(packet.clientId.value).push_back(packetIndex);
    }

    for (const auto clientId : survivorKeys.publicKey.survivors.clients) {
        const auto& clientPacketIndices = packetIndicesByClient.at(
            clientId.value);
        CryptoStageMetrics keySwitchMetrics;
        executor.ParallelFor(
            clientPacketIndices.size(),
            CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH,
            &keySwitchMetrics,
            [&](std::size_t localIndex) {
                const std::size_t packetIndex =
                    clientPacketIndices[localIndex];
                const auto& packet = packets[packetIndex];
                const auto restored = RestoreSeededSparseRLWE(
                    packet.ciphertext,
                    setup.common);
                HybridModDownTrace trace;
                auto switched = collectNoiseBreakdown
                    ? KeySwitchClientToCollective(
                          restored,
                          FindClientToCollectiveKey(
                              survivorKeys,
                              packet.clientId),
                          setup.common,
                          &trace)
                    : KeySwitchClientToCollective(
                          restored,
                          survivorKeys,
                          setup.common);
                if (collectNoiseBreakdown) {
                    DCRTPoly sourceSecretQ = FindClientSecret(
                        setup,
                        packet.clientId).encryptionKeyQ->GetPrivateElement();
                    sourceSecretQ.SetFormat(Format::EVALUATION);
                    processed[packetIndex].encryptionPhase =
                        std::make_unique<DCRTPoly>(
                            restored.b + restored.a * sourceSecretQ);
                    processed[packetIndex].preKeySwitchPhase =
                        std::make_unique<DCRTPoly>(BuildPreModDownPhaseQP(
                            restored.b,
                            trace,
                            collectiveSecretQP,
                            setup.common.rns));
                    processed[packetIndex].postKeySwitchPhase =
                        std::make_unique<DCRTPoly>(
                            switched.b + switched.a * collectiveSecretQ);
                }
                processed[packetIndex].switched =
                    std::make_unique<CollectiveRLWECiphertext>(
                        std::move(switched));
            });
        const auto timing = keySwitchMetrics.Snapshot().at(
            static_cast<std::size_t>(
                CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH));
        if (clientKeySwitchSeconds != nullptr) {
            clientKeySwitchSeconds->at(clientId.value) = timing.wallSeconds;
        }
        if (clientStageTimings != nullptr) {
            AccumulateStageTiming(
                &clientStageTimings->at(clientId.value).at(
                    static_cast<std::size_t>(
                        CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH)),
                timing);
        }
    }

    const bool requiresRingConversion = RequiresSparseRingConversion(
        config.mode,
        plan.mlweRank);
    executor.ParallelFor(
        packets.size(),
        requiresRingConversion
            ? CryptoStage::R2M
            : CryptoStage::RLWE_AGGREGATION,
        serverMetrics,
        [&](std::size_t packetIndex) {
            const auto& packet = packets[packetIndex];
            auto& item = processed[packetIndex];
            if (requiresRingConversion) {
                item.mlwe = ProtocolRLWEToMLWE(
                    *item.switched,
                    packet.derivedLabels,
                    plan.mlweRank);
                if (collectNoiseBreakdown) {
                    item.encryptionPhaseMlwe = ProtocolRLWEToMLWE(
                        MakePhaseOnlyCiphertext(
                            *item.switched,
                            *item.encryptionPhase),
                        packet.derivedLabels,
                        plan.mlweRank);
                    item.preKeySwitchPhaseMlwe = ProtocolRLWEToMLWE(
                        MakePhaseOnlyCiphertext(
                            *item.switched,
                            *item.preKeySwitchPhase),
                        packet.derivedLabels,
                        plan.mlweRank);
                    item.postKeySwitchPhaseMlwe = ProtocolRLWEToMLWE(
                        MakePhaseOnlyCiphertext(
                            *item.switched,
                            *item.postKeySwitchPhase),
                        packet.derivedLabels,
                        plan.mlweRank);
                }
            }
            else {
                if (collectNoiseBreakdown) {
                    item.encryptionPhaseMlwe.push_back(
                        MoveRankOneRLWEToLabeled(
                            MakePhaseOnlyCiphertext(
                                *item.switched,
                                std::move(*item.encryptionPhase)),
                            packet.derivedLabels));
                    item.preKeySwitchPhaseMlwe.push_back(
                        MoveRankOneRLWEToLabeled(
                            MakePhaseOnlyCiphertext(
                                *item.switched,
                                std::move(*item.preKeySwitchPhase)),
                            packet.derivedLabels));
                    item.postKeySwitchPhaseMlwe.push_back(
                        MoveRankOneRLWEToLabeled(
                            MakePhaseOnlyCiphertext(
                                *item.switched,
                                std::move(*item.postKeySwitchPhase)),
                            packet.derivedLabels));
                }
                item.mlwe.push_back(MoveRankOneRLWEToLabeled(
                    std::move(*item.switched),
                    packet.derivedLabels));
            }
            item.switched.reset();
            item.encryptionPhase.reset();
            item.preKeySwitchPhase.reset();
            item.postKeySwitchPhase.reset();
        });
    execution.result.clientRLWECiphertextCount = packets.size();

    using ContributionLocation = std::pair<std::size_t, std::size_t>;
    std::vector<std::vector<ContributionLocation>> contributionLocations(
        plan.chunkCount);
    for (std::size_t packetIndex = 0;
         packetIndex < packets.size();
         ++packetIndex) {
        for (std::size_t contributionIndex = 0;
             contributionIndex < processed[packetIndex].mlwe.size();
             ++contributionIndex) {
            const auto label =
                processed[packetIndex].mlwe[contributionIndex].label;
            if (!FindShuffledCohort(plan, label).accepted) {
                continue;
            }
            contributionLocations.at(label.value).emplace_back(
                packetIndex,
                contributionIndex);
            ++execution.result.acceptedContributionCount;
        }
    }

    std::vector<std::unique_ptr<LabeledMLWECiphertext>> aggregates(
        plan.chunkCount);
    executor.ParallelFor(
        plan.acceptedLabels.size(),
        requiresRingConversion
            ? CryptoStage::MLWE_AGGREGATION
            : CryptoStage::RLWE_AGGREGATION,
        serverMetrics,
        [&](std::size_t acceptedIndex) {
            const auto label = plan.acceptedLabels[acceptedIndex];
            const auto& locations = contributionLocations.at(label.value);
            if (locations.empty()) {
                throw std::runtime_error(
                    "an accepted label has no MLWE contribution");
            }
            const auto accumulate = [&locations, &processed, &packets](
                std::vector<std::unique_ptr<LabeledMLWECiphertext>>* target,
                const auto& sourceSelector) {
                std::unique_ptr<LabeledMLWECiphertext> sum;
                for (const auto& location : locations) {
                    const auto packetIndex = location.first;
                    const auto contributionIndex = location.second;
                    const auto& source = sourceSelector(
                        processed[packetIndex]);
                    LabeledMLWECiphertext weighted = MultiplyMLWEByInteger(
                        source.at(contributionIndex),
                        packets[packetIndex].integerWeight);
                    if (!sum) {
                        sum = std::make_unique<LabeledMLWECiphertext>(
                            std::move(weighted));
                    }
                    else {
                        AddMLWEInPlace(*sum, weighted);
                    }
                }
                target->at(sum->label.value) = std::move(sum);
            };
            accumulate(
                &aggregates,
                [](const ProcessedSparsePacket& item)
                    -> const std::vector<LabeledMLWECiphertext>& {
                    return item.mlwe;
                });
            if (collectNoiseBreakdown) {
                accumulate(
                    &encryptionPhases,
                    [](const ProcessedSparsePacket& item)
                        -> const std::vector<LabeledMLWECiphertext>& {
                        return item.encryptionPhaseMlwe;
                    });
                accumulate(
                    &preKeySwitchPhases,
                    [](const ProcessedSparsePacket& item)
                        -> const std::vector<LabeledMLWECiphertext>& {
                        return item.preKeySwitchPhaseMlwe;
                    });
                accumulate(
                    &postKeySwitchPhases,
                    [](const ProcessedSparsePacket& item)
                        -> const std::vector<LabeledMLWECiphertext>& {
                        return item.postKeySwitchPhaseMlwe;
                    });
            }
        });
    execution.result.aggregatedMLWECiphertextCount =
        requiresRingConversion ? plan.acceptedLabels.size() : 0;
    if (requiresRingConversion &&
        execution.result.aggregatedMLWECiphertextCount !=
            plan.acceptedLabels.size()) {
        throw std::runtime_error("an accepted label has no MLWE aggregate");
    }
    std::size_t expectedContributionCount = 0;
    for (const auto label : plan.acceptedLabels) {
        expectedContributionCount += FindShuffledCohort(plan, label).clients.size();
    }
    if (execution.result.acceptedContributionCount != expectedContributionCount) {
        throw std::runtime_error("accepted label cohort is missing a contribution");
    }
    processed.clear();
    contributionLocations.clear();

    const std::size_t groupCount = plan.packingGroups.size();
    std::vector<std::unique_ptr<PackedAggregateRLWECiphertext>> outputs(
        groupCount);
    std::vector<std::unique_ptr<SparseServerExecution::NoiseCheckpoints>>
        checkpointJobs(collectNoiseBreakdown ? groupCount : 0);
    std::vector<std::unique_ptr<DCRTPoly>> compressedAggregateA(groupCount);
    std::vector<PreparedM2R> preparedM2R(requiresRingConversion ? groupCount : 0);
    if (requiresRingConversion) {
        executor.ParallelFor(groupCount, CryptoStage::M2R, serverMetrics,
            [&](std::size_t groupIndex) {
                const auto& group = plan.packingGroups[groupIndex];
                std::vector<LabeledMLWECiphertext> inputs;
                inputs.reserve(group.labels.size());
                for (const auto label : group.labels) {
                    if (label.value >= aggregates.size() || !aggregates[label.value])
                        throw std::runtime_error("server group references missing MLWE aggregate");
                    inputs.push_back(std::move(*aggregates[label.value]));
                }
                preparedM2R[groupIndex] = PrepareM2R(
                    std::move(inputs), group, survivorKeys, setup.common, collectNoiseBreakdown);
            });
        // Flatten groups and columns into ONE server batch: nested ParallelFor
        // calls execute serially, and would leave most workers idle at high rank.
        executor.ParallelFor(groupCount * plan.mlweRank, CryptoStage::M2R, serverMetrics,
            [&](std::size_t taskIndex) {
                SwitchPreparedM2RComponent(preparedM2R[taskIndex / plan.mlweRank],
                    taskIndex % plan.mlweRank, survivorKeys, setup.common);
            });
    }

    executor.ParallelFor(
        groupCount,
        requiresRingConversion
            ? CryptoStage::M2R
            : CryptoStage::RLWE_AGGREGATION,
        serverMetrics,
        [&](std::size_t groupIndex) {
            const auto& serverGroup = plan.packingGroups[groupIndex];
            std::vector<LabeledMLWECiphertext> inputs;
            inputs.reserve(serverGroup.labels.size());
            std::vector<LabeledMLWECiphertext> encryptionInputs;
            std::vector<LabeledMLWECiphertext> preKeySwitchInputs;
            std::vector<LabeledMLWECiphertext> postKeySwitchInputs;
            if (collectNoiseBreakdown) {
                encryptionInputs.reserve(serverGroup.labels.size());
                preKeySwitchInputs.reserve(serverGroup.labels.size());
                postKeySwitchInputs.reserve(serverGroup.labels.size());
            }
            for (const auto label : serverGroup.labels) {
                if (label.value >= aggregates.size() ||
                    !aggregates[label.value]) {
                    throw std::runtime_error(
                        "server group references missing MLWE aggregate");
                }
                if (!requiresRingConversion)
                    inputs.push_back(std::move(*aggregates[label.value]));
                if (collectNoiseBreakdown) {
                    if (!encryptionPhases[label.value] ||
                        !preKeySwitchPhases[label.value] ||
                        !postKeySwitchPhases[label.value]) {
                        throw std::runtime_error(
                            "server group noise checkpoint is missing");
                    }
                    if (requiresRingConversion) {
                        encryptionInputs.push_back(
                            *encryptionPhases[label.value]);
                        preKeySwitchInputs.push_back(
                            *preKeySwitchPhases[label.value]);
                        postKeySwitchInputs.push_back(
                            *postKeySwitchPhases[label.value]);
                    }
                    else {
                        encryptionInputs.push_back(
                            std::move(*encryptionPhases[label.value]));
                        preKeySwitchInputs.push_back(
                            std::move(*preKeySwitchPhases[label.value]));
                        postKeySwitchInputs.push_back(
                            std::move(*postKeySwitchPhases[label.value]));
                    }
                }
            }
            M2RModDownTrace m2rTrace;
            auto output = std::make_unique<PackedAggregateRLWECiphertext>(
                requiresRingConversion
                    ? FinishM2R(
                          preparedM2R[groupIndex],
                          serverGroup,
                          survivorKeys,
                          setup.common,
                          plan.roundId,
                          collectNoiseBreakdown ? &m2rTrace : nullptr)
                    : MoveRankOneAggregateToRLWE(
                          std::move(inputs.front()),
                          serverGroup,
                          survivorKeys,
                          setup.common,
                          plan.roundId));
            if (collectNoiseBreakdown) {
                auto checkpoints = std::make_unique<
                    SparseServerExecution::NoiseCheckpoints>();
                checkpoints->encryptionPhase = PackMLWEPhaseRows(
                    encryptionInputs,
                    plan.mlweRank);
                checkpoints->preKeySwitchModDownPhaseQP = PackMLWEPhaseRows(
                    preKeySwitchInputs,
                    plan.mlweRank);
                checkpoints->postKeySwitchModDownPhase = PackMLWEPhaseRows(
                    postKeySwitchInputs,
                    plan.mlweRank);
                if (requiresRingConversion) {
                    checkpoints->preM2RModDownPhaseQP = BuildPreModDownPhaseQP(
                        m2rTrace.combinedB,
                        m2rTrace.switching,
                        modelSecretQP,
                        setup.common.rns);
                    checkpoints->postM2RModDownPhase =
                        output->b + output->a * modelSecretQ;
                }
                else {
                    checkpoints->preM2RModDownPhaseQP =
                        LiftQPolynomialTimesPToQP(
                            checkpoints->postKeySwitchModDownPhase,
                            setup.common.rns);
                    checkpoints->postM2RModDownPhase =
                        checkpoints->postKeySwitchModDownPhase;
                }
                checkpointJobs[groupIndex] = std::move(checkpoints);
            }
            compressedAggregateA[groupIndex] = std::make_unique<DCRTPoly>(
                MaybeCompressAggregateA(
                    output->a,
                    setup.common.rns,
                    compression));
            if (collectNoiseBreakdown)
                checkpointJobs[groupIndex]->compressedA = *compressedAggregateA[groupIndex];
            outputs[groupIndex] = std::move(output);
            // Release scratch storage inside the timed server batch.
            if (requiresRingConversion) preparedM2R[groupIndex] = PreparedM2R{};
        });
    preparedM2R.clear();
    execution.result.serverRLWECiphertextCount = groupCount;
    aggregates.clear();
    encryptionPhases.clear();
    preKeySwitchPhases.clear();
    postKeySwitchPhases.clear();

    const bool v17C2 = federated::UsesV17WireProtocol(config.protocolVersion) &&
        compression.compressPartialDecryption;
    const Seed128 seedD = v17C2
        ? (compression.v17SeedGate
              ? compression.v17SeedGate->ArriveAfterAggregateA()
              : FreshV17DecryptionSeed128())
        : Seed128{};

    const auto& survivorIds = survivorKeys.publicKey.survivors.clients;
    const std::size_t survivorCount = survivorIds.size();
    std::vector<std::unique_ptr<DCRTPoly>> partialShares(
        survivorCount * groupCount);
    std::vector<coefficient_compression::CompressedCoefficientPayload>
        v17ResponseCodes(v17C2 ? survivorCount * groupCount : 0);
    for (std::size_t clientIndex = 0;
         clientIndex < survivorCount;
         ++clientIndex) {
        const auto clientId = survivorIds[clientIndex];
        CryptoStageMetrics partialMetrics;
        executor.ParallelFor(
            groupCount,
            CryptoStage::PARTIAL_DECRYPTION,
            &partialMetrics,
            [&](std::size_t groupIndex) {
                const auto& keys = FindClientSecret(setup, clientId);
                DCRTPoly secret = (keys.modelKeyQ ? keys.modelKeyQ : keys.secretKeyQ)->GetPrivateElement();
                secret.SetFormat(Format::EVALUATION);
                DCRTPoly partial =
                    *compressedAggregateA[groupIndex] * secret +
                    SampleSmudging(
                        setup.common.rns.paramsQ,
                        config.smudging.sigmaSmg, setup.common.certifiedV17Gaussian);
                if (v17C2) {
                    v17ResponseCodes[clientIndex * groupCount + groupIndex] =
                        MakeV17PartialResponse(
                            std::move(partial), setup.common.rns.paramsQ,
                            seedD, setup.common.setupId, plan.roundId,
                            "SP-MODEL", groupIndex, clientId,
                            survivorIds.front(), compression.c2,
                            outputs[groupIndex]
                                ->meaningfulCoefficientIndices);
                    return;
                }
                if (compression.compressPartialDecryption) {
                    const auto payload =
                        coefficient_compression::CompressSelectedCoefficients(
                            partial,
                            compression.c2,
                            outputs[groupIndex]
                                ->meaningfulCoefficientIndices);
                    partial =
                        coefficient_compression::DecompressSelectedCoefficients(
                            setup.common.rns.paramsQ,
                            payload,
                            outputs[groupIndex]
                                ->meaningfulCoefficientIndices);
                }
                partial.SetFormat(Format::EVALUATION);
                partialShares[clientIndex * groupCount + groupIndex] =
                    std::make_unique<DCRTPoly>(std::move(partial));
            });
        const auto timing = partialMetrics.Snapshot().at(
            static_cast<std::size_t>(CryptoStage::PARTIAL_DECRYPTION));
        if (clientPartialDecryptionSeconds != nullptr) {
            clientPartialDecryptionSeconds->at(clientId.value) =
                timing.wallSeconds;
        }
        if (clientStageTimings != nullptr) {
            AccumulateStageTiming(
                &clientStageTimings->at(clientId.value).at(
                    static_cast<std::size_t>(
                        CryptoStage::PARTIAL_DECRYPTION)),
                timing);
        }
    }

    std::vector<std::unique_ptr<DCRTPoly>> recoveredPhases(groupCount);
    executor.ParallelFor(
        groupCount,
        CryptoStage::PARTIAL_DECRYPTION,
        serverMetrics,
        [&](std::size_t groupIndex) {
            if (v17C2) {
                std::vector<
                    coefficient_compression::CompressedCoefficientPayload>
                    groupCodes;
                groupCodes.reserve(survivorCount);
                for (std::size_t clientIndex = 0;
                     clientIndex < survivorCount; ++clientIndex) {
                    groupCodes.push_back(std::move(v17ResponseCodes[
                        clientIndex * groupCount + groupIndex]));
                }
                recoveredPhases[groupIndex] = std::make_unique<DCRTPoly>(
                    RecoverV17Phase(
                        outputs[groupIndex]->b,
                        setup.common.rns.paramsQ, groupCodes,
                        survivorIds, seedD, setup.common.setupId,
                        plan.roundId, "SP-MODEL", groupIndex,
                        outputs[groupIndex]
                            ->meaningfulCoefficientIndices));
                return;
            }
            DCRTPoly phase = outputs[groupIndex]->b;
            phase.SetFormat(Format::EVALUATION);
            for (std::size_t clientIndex = 0;
                 clientIndex < survivorCount;
                 ++clientIndex) {
                phase += *partialShares[
                    clientIndex * groupCount + groupIndex];
            }
            recoveredPhases[groupIndex] =
                std::make_unique<DCRTPoly>(std::move(phase));
        });

    std::vector<std::vector<LabeledAggregateChunk>> decodedGroups(groupCount);
    executor.ParallelFor(
        groupCount,
        CryptoStage::DECODING,
        serverMetrics,
        [&](std::size_t groupIndex) {
            const auto& ciphertext = *outputs[groupIndex];
            const lbcrypto::BigInteger modulus =
                setup.common.rns.paramsQ->GetModulus();
            const lbcrypto::BigInteger halfModulus = modulus >> 1;
            const auto phaseValues =
                rns_polynomial::InterpolateCoefficientsAtIndices(
                    std::move(*recoveredPhases[groupIndex]),
                    ciphertext.meaningfulCoefficientIndices);
            std::vector<long double> selected;
            selected.reserve(phaseValues.size());
            for (const auto& residue : phaseValues) {
                const bool negative = residue > halfModulus;
                const lbcrypto::BigInteger magnitude = negative
                    ? modulus - residue
                    : residue;
                const long double converted =
                    std::stold(magnitude.ToString());
                selected.push_back(negative ? -converted : converted);
            }
            decodedGroups[groupIndex] = DecodeLabeledAggregateGroup(
                ciphertext,
                selected,
                config,
                plan,
                compression,
                scalingFactor);
            if (compression.measurePreSmudgingNoise) {
                const auto active=decodedGroups[groupIndex].size();
                for (std::size_t slot=0;slot<active;++slot) {
                    auto& audit=decodedGroups[groupIndex][slot].phaseResidues;
                    audit.resize(plan.chunkSize);
                    for (std::size_t j=0;j<plan.chunkSize;++j)
                        audit[j]=phaseValues[j*active+slot].ToString();
                }
            }
        });

    for (std::size_t groupIndex = 0;
         groupIndex < groupCount;
         ++groupIndex) {
        auto& decoded = decodedGroups[groupIndex];
        execution.result.aggregateChunks.insert(
            execution.result.aggregateChunks.end(),
            std::make_move_iterator(decoded.begin()),
            std::make_move_iterator(decoded.end()));
        execution.finalCiphertexts.push_back(std::move(*outputs[groupIndex]));
        if (collectNoiseBreakdown) {
            execution.noiseCheckpoints.push_back(
                std::move(*checkpointJobs[groupIndex]));
        }
    }
    if (execution.result.aggregateChunks.size() != plan.acceptedLabels.size()) {
        throw std::runtime_error("server finalized-label count mismatch");
    }
    for (std::size_t index = 0;
         index < plan.acceptedLabels.size();
         ++index) {
        if (!(execution.result.aggregateChunks[index].label ==
              plan.acceptedLabels[index])) {
            throw std::runtime_error("server finalized-label ordering mismatch");
        }
    }
    return execution;
}

ReferenceValues ComputePrivilegedReference(
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& round,
    const ClientSparsePrivateRoundState& observerState,
    long double scalingFactor) {

    ReferenceValues result;
    result.ideal.assign(config.modelDimension, 0.0);
    result.quantized.assign(config.modelDimension, 0.0);
    result.mask.assign(config.modelDimension, 0U);
    result.integerNumerator.assign(config.modelDimension, 0);
    std::vector<long double> idealNumerator(config.modelDimension, 0.0L);

    for (const auto label : round.serverPlan.acceptedLabels) {
        const ActualChunkId actual = ActualChunkForLabel(observerState, label);
        const auto& cohort = FindShuffledCohort(round.serverPlan, label);
        const std::size_t begin = actual.value * round.serverPlan.chunkSize;
        const std::size_t end = std::min(
            begin + round.serverPlan.chunkSize,
            config.modelDimension);
        for (std::size_t coordinate = begin; coordinate < end; ++coordinate) {
            result.mask[coordinate] = 1U;
        }
        for (const auto clientId : cohort.clients) {
            const auto& client = FindClient(clients, clientId);
            for (std::size_t coordinate = begin; coordinate < end; ++coordinate) {
                const long double value = client.update[coordinate];
                idealNumerator[coordinate] +=
                    static_cast<long double>(client.integerWeight) * value;
                const exact_integer::Integer term = client.integerWeight *
                    exact_integer::RoundProduct(client.update[coordinate], config.scaling.factor);
                result.integerNumerator[coordinate] += term;
            }
        }
    }

    const long double idealDenominator = config.weightDenominator;
    const long double integerDenominator =
        idealDenominator * scalingFactor;
    for (std::size_t index = 0; index < config.modelDimension; ++index) {
        if (result.mask[index] == 0U) {
            continue;
        }
        result.ideal[index] = static_cast<double>(
            idealNumerator[index] / idealDenominator);
        result.quantized[index] = static_cast<double>(
            result.integerNumerator[index].convert_to<long double>() / integerDenominator);
    }
    return result;
}

void ValidateAggregateReference(
    const ReferenceValues& reference,
    long double aggregateBound) {

    for (const double value : reference.ideal) {
        if (!std::isfinite(value) ||
            std::abs(static_cast<long double>(value)) > aggregateBound) {
            throw std::invalid_argument(
                "sparse aggregate exceeds its configured bound");
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

ProtocolNoiseBreakdownMeasurement PrivilegedMeasureNoiseBreakdown(
    const PackedAggregateRLWECiphertext& ciphertext,
    const SparseServerExecution::NoiseCheckpoints& checkpoints,
    const ProtocolSetupResult& setup,
    const SparsePathCompressionProfile& profile,
    const ServerSparseAdmissionPlan& plan,
    const ClientSparsePrivateRoundState& observerState,
    const ReferenceValues& reference) {

    DCRTPoly a = checkpoints.compressedA;
    DCRTPoly collectiveSecret(
        setup.common.rns.paramsQ,
        Format::EVALUATION,
        true);
    for (const auto clientId : ciphertext.survivors.clients) {
        const auto& keys = FindClientSecret(setup, clientId);
        DCRTPoly secret = (keys.modelKeyQ ? keys.modelKeyQ : keys.secretKeyQ)->GetPrivateElement();
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
    const auto totalValues =
        rns_polynomial::InterpolateCoefficients(std::move(phase));
    const auto encryptionValues = rns_polynomial::InterpolateCoefficients(
        checkpoints.encryptionPhase);
    const auto preKeySwitchValuesQP = rns_polynomial::InterpolateCoefficients(
        checkpoints.preKeySwitchModDownPhaseQP);
    const auto postKeySwitchValues = rns_polynomial::InterpolateCoefficients(
        checkpoints.postKeySwitchModDownPhase);
    const auto preM2RValuesQP = rns_polynomial::InterpolateCoefficients(
        checkpoints.preM2RModDownPhaseQP);
    const auto postM2RValues = rns_polynomial::InterpolateCoefficients(
        checkpoints.postM2RModDownPhase);
    const auto liftedEncryptionValuesQP =
        rns_polynomial::InterpolateCoefficients(
            LiftQPolynomialTimesPToQP(
                checkpoints.encryptionPhase,
                setup.common.rns));
    const auto liftedPostKeySwitchValuesQP =
        rns_polynomial::InterpolateCoefficients(
            LiftQPolynomialTimesPToQP(
                checkpoints.postKeySwitchModDownPhase,
                setup.common.rns));

    ProtocolNoiseBreakdownMeasurement result;
    for (std::size_t slot = 0;
         slot < ciphertext.shuffledLabels.labels.size();
         ++slot) {
        const ActualChunkId actual = ActualChunkForLabel(
            observerState,
            ciphertext.shuffledLabels.labels[slot]);
        ProtocolNoiseComponentResiduals residuals;
        residuals.encryption.reserve(plan.chunkSize);
        residuals.keySwitchCore.reserve(plan.chunkSize);
        residuals.keySwitchModDown.reserve(plan.chunkSize);
        residuals.m2rCore.reserve(plan.chunkSize);
        residuals.m2rModDown.reserve(plan.chunkSize);
        residuals.aggregateACompression.reserve(plan.chunkSize);
        residuals.total.reserve(plan.chunkSize);
        for (std::size_t offset = 0; offset < plan.chunkSize; ++offset) {
            const std::size_t coordinate =
                actual.value * plan.chunkSize + offset;
            const exact_integer::Integer expectedInteger =
                coordinate < reference.integerNumerator.size()
                    ? reference.integerNumerator[coordinate]
                    : exact_integer::Integer(0);
            const std::size_t coefficient =
                offset * plan.mlweRank + slot;
            const lbcrypto::BigInteger expected = SignedIntegerToResidue(
                expectedInteger,
                modulus);
            residuals.encryption.push_back(CenteredModularDifference(
                encryptionValues[coefficient], expected, modulus));
            const long double keySwitchCore = CenteredModularDifference(
                preKeySwitchValuesQP[coefficient],
                liftedEncryptionValuesQP[coefficient],
                modulusQP) / specialModulus;
            const long double keySwitchTotal = CenteredModularDifference(
                postKeySwitchValues[coefficient],
                encryptionValues[coefficient],
                modulus);
            residuals.keySwitchCore.push_back(keySwitchCore);
            residuals.keySwitchModDown.push_back(
                keySwitchTotal - keySwitchCore);
            const long double m2rCore = CenteredModularDifference(
                preM2RValuesQP[coefficient],
                liftedPostKeySwitchValuesQP[coefficient],
                modulusQP) / specialModulus;
            const long double m2rTotal = CenteredModularDifference(
                postM2RValues[coefficient],
                postKeySwitchValues[coefficient],
                modulus);
            residuals.m2rCore.push_back(m2rCore);
            residuals.m2rModDown.push_back(m2rTotal - m2rCore);
            residuals.aggregateACompression.push_back(
                CenteredModularDifference(
                    totalValues[coefficient],
                    postM2RValues[coefficient],
                    modulus));
            residuals.total.push_back(CenteredModularDifference(
                totalValues[coefficient], expected, modulus));
        }
        AppendProtocolNoiseBreakdownGroup(
            &result,
            residuals);
    }
    return result;
}

}  // namespace

SparsePathCompressionProfile MakeUncompressedSparseProfile() {
    return SparsePathCompressionProfile{};
}

SparsePathCompressionProfile MakeConfiguredSparseProfile(
    const ProtocolExperimentConfig& config) {

    SparsePathCompressionProfile result;
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

void PrewarmProtocolSparseTransforms(const ProtocolSetupResult& setup) {
    (void)RequireSparseSetup(setup);
    if (!RequiresSparseRingConversion(
            setup.mode,
            setup.common.mlweRank)) {
        return;
    }

    DCRTPoly qPolynomial(
        setup.common.rns.paramsQ,
        Format::COEFFICIENT,
        true);
    qPolynomial.SetFormat(Format::EVALUATION);
    qPolynomial.SetFormat(Format::COEFFICIENT);
    auto smallPolynomials = ring_conversion::Split(
        qPolynomial,
        setup.common.mlweRank);
    for (auto& polynomial : smallPolynomials) {
        polynomial.SetFormat(Format::EVALUATION);
        polynomial.SetFormat(Format::COEFFICIENT);
    }
    DCRTPoly combined = ring_conversion::Combine(smallPolynomials);
    combined.SetFormat(Format::EVALUATION);

    DCRTPoly qpPolynomial(
        setup.common.rns.paramsQP,
        Format::COEFFICIENT,
        true);
    qpPolynomial.SetFormat(Format::EVALUATION);
    qpPolynomial.SetFormat(Format::COEFFICIENT);
}

std::vector<double> ApplyLabeledAggregateChunksForClient(
    const ProtocolExperimentConfig& config,
    const ClientSparsePrivateRoundState& clientState,
    const std::vector<LabeledAggregateChunk>& chunks) {

    std::vector<double> result(config.modelDimension, 0.0);
    std::set<std::size_t> labels;
    for (const auto& chunk : chunks) {
        if (!labels.insert(chunk.label.value).second ||
            chunk.values.size() != clientState.chunkSize) {
            throw std::invalid_argument("invalid finalized labeled chunk");
        }
        const ActualChunkId actual = ActualChunkForLabel(
            clientState,
            chunk.label);
        const std::size_t begin = actual.value * clientState.chunkSize;
        const std::size_t end = std::min(
            begin + clientState.chunkSize,
            config.modelDimension);
        std::copy(
            chunk.values.begin(),
            chunk.values.begin() + (end - begin),
            result.begin() + begin);
    }
    return result;
}

void UpdateClientResidualFromFinalizedChunks(
    const ProtocolExperimentConfig& config,
    const ProtocolClientRoundInput& client,
    const ClientSparsePrivateRoundState& clientState,
    const std::vector<LabeledAggregateChunk>& finalizedChunks,
    bool errorFeedbackEnabled,
    std::vector<double>* residual) {

    if (residual == nullptr ||
        client.update.size() != config.modelDimension ||
        !(client.clientId == clientState.clientId)) {
        throw std::invalid_argument("invalid client residual update input");
    }
    if (!errorFeedbackEnabled) {
        residual->assign(config.modelDimension, 0.0);
        return;
    }
    *residual = client.update;
    std::set<std::size_t> seenLabels;
    for (const auto& finalized : finalizedChunks) {
        if (!seenLabels.insert(finalized.label.value).second ||
            finalized.values.size() != clientState.chunkSize) {
            throw std::invalid_argument("invalid finalized residual label set");
        }
        const bool contributed = std::any_of(
            clientState.selectedMappings.begin(),
            clientState.selectedMappings.end(),
            [&finalized](const ClientSparseChunkLabelMapping& mapping) {
                return mapping.label == finalized.label;
            });
        if (!contributed) {
            continue;
        }
        const ActualChunkId actual = ActualChunkForLabel(
            clientState,
            finalized.label);
        const std::size_t begin = actual.value * clientState.chunkSize;
        const std::size_t end = std::min(
            begin + clientState.chunkSize,
            config.modelDimension);
        std::fill(residual->begin() + begin, residual->begin() + end, 0.0);
    }
}

SparsePathUploadBatch EncryptProtocolSparseUploads(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& uploadRound,
    const ProtocolSetupResult& setup,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor) {

    const auto started = std::chrono::steady_clock::now();
    ValidateProtocolParameters(config).ThrowIfInvalid();
    (void)RequireSparseSetup(setup);
    if (!cc ||
        !(uploadRound.serverPlan.setupId == setup.common.setupId) ||
        uploadRound.clientPrivateStates.empty()) {
        throw std::invalid_argument("protocol sparse upload setup mismatch");
    }
    const long double scalingFactor = ScalingFactor(config.scaling);
    CryptoTaskExecutor sequentialExecutor(1);
    CryptoTaskExecutor& activeExecutor =
        executor == nullptr ? sequentialExecutor : *executor;

    SparsePathUploadBatch result;
    result.setupId = setup.common.setupId;
    result.roundId = uploadRound.serverPlan.roundId;
    result.supportHeaders.reserve(uploadRound.clientPrivateStates.size());
    for (const auto& state : uploadRound.clientPrivateStates) {
        ValidateSupportBitmapMsb0(
            state.chunkCount,
            state.uploadSupportBitmap);
        result.supportHeaders.push_back(SparseUploadSupportHeader{
            state.clientId,
            0,
            state.uploadSupportBitmap});
    }
    result.clientEncryptionSeconds.assign(config.clientCount, 0.0);
    result.clientStageTimings.resize(config.clientCount);
    result.packetPairs = EncryptClientPackets(
        cc,
        config,
        clients,
        uploadRound,
        setup,
        scalingFactor,
        compression,
        activeExecutor,
        &result.clientEncryptionSeconds,
        &result.clientStageTimings);
    result.sequentialSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

SparsePathResult FinalizeProtocolSparsePathAfterDropout(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const SparsePathUploadBatch& uploads,
    const ProtocolSparseRoundArtifacts& survivorRound,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    const auto finalizeStarted = std::chrono::steady_clock::now();

    ValidateProtocolParameters(config).ThrowIfInvalid();
    if (compression.quantizeFinalResult) {
        coefficient_compression::ValidateNoClippingDeliveryRange(
            compression.cd,
            compression.deliveryScale,
            config.updateBound);
    }
    (void)RequireSparseSetup(setup);
    if (!cc ||
        !(survivorRound.serverPlan.setupId == setup.common.setupId) ||
        !(uploads.setupId == setup.common.setupId) ||
        !(uploads.roundId == survivorRound.serverPlan.roundId) ||
        !(survivorKeys.setupId == setup.common.setupId) ||
        survivorRound.clientPrivateStates.empty() ||
        uploads.clientEncryptionSeconds.size() != config.clientCount ||
        uploads.clientStageTimings.size() != config.clientCount ||
        uploads.supportHeaders.empty()) {
        throw std::invalid_argument("protocol sparse path setup mismatch");
    }
    const long double scalingFactor = ScalingFactor(config.scaling);
    const auto& observerState = survivorRound.clientPrivateStates.front();
    CryptoTaskExecutor sequentialExecutor(1);
    CryptoTaskExecutor& activeExecutor =
        executor == nullptr ? sequentialExecutor : *executor;
    CryptoStageMetrics serverMetrics;
    std::vector<double> clientEncryptionSeconds =
        uploads.clientEncryptionSeconds;
    std::vector<double> clientKeySwitchSeconds(config.clientCount, 0.0);
    std::vector<double> clientPartialDecryptionSeconds(
        config.clientCount,
        0.0);
    std::vector<CryptoStageTimings> clientStageTimings =
        uploads.clientStageTimings;
    const auto& packetPairs = uploads.packetPairs;
    std::set<std::uint64_t> headerClients;
    for (const auto& header : uploads.supportHeaders) {
        ValidateSupportBitmapMsb0(
            survivorRound.serverPlan.chunkCount,
            header.supportBitmap);
        if (header.clientId.value >= config.clientCount ||
            !headerClients.insert(header.clientId.value).second) {
            throw std::invalid_argument(
                "duplicate or invalid sparse upload-support header");
        }
    }
    for (const auto& pair : packetPairs) {
        for (const auto* packet : {&pair.wire, &pair.bctShadow}) {
            const auto& header = FindUploadSupportHeader(
                uploads.supportHeaders,
                packet->clientId);
            if (packet->derivedLabels.labels != DeriveUploadGroupLabels(
                    header,
                    survivorRound.serverPlan.chunkCount,
                    survivorRound.serverPlan.mlweRank,
                    packet->groupIndex).labels) {
                throw std::invalid_argument(
                    "sparse packet order disagrees with upload-support bitmap");
            }
        }
    }
    /* Every accepted local value is now known to be encodable. The privileged
       observer validates the admitted aggregate, not a per-client FL bound. */
    ReferenceValues reference;
    if (compression.computeReference || compression.measurePreSmudgingNoise) {
        reference = ComputePrivilegedReference(config, clients, survivorRound,
            observerState, scalingFactor);
        ValidateAggregateReference(reference, config.updateBound);
    }
    const std::set<ClientId> survivorIds(
        survivorKeys.publicKey.survivors.clients.begin(),
        survivorKeys.publicKey.survivors.clients.end());
    const std::set<ShuffledChunkLabel> acceptedLabels(
        survivorRound.serverPlan.acceptedLabels.begin(),
        survivorRound.serverPlan.acceptedLabels.end());
    std::size_t survivorUploadedPacketCount = 0;
    const auto packetIsNeeded = [&](const SparseClientCiphertextPacket& packet) {
        if (survivorIds.count(packet.clientId) == 0) {
            return false;
        }
        ++survivorUploadedPacketCount;
        return std::any_of(
            packet.derivedLabels.labels.begin(),
            packet.derivedLabels.labels.end(),
            [&](ShuffledChunkLabel label) {
                return acceptedLabels.count(label) != 0;
            });
    };
    std::vector<SparseClientCiphertextPacket> packets;
    packets.reserve(packetPairs.size());
    for (const auto& pair : packetPairs) {
        if (packetIsNeeded(pair.wire)) {
            packets.push_back(pair.wire);
        }
    }
    SparseServerExecution execution = RunLabelOnlyServer(
        config,
        survivorRound.serverPlan,
        packets,
        setup,
        survivorKeys,
        scalingFactor,
        compression,
        activeExecutor,
        &serverMetrics,
        &clientKeySwitchSeconds,
        &clientPartialDecryptionSeconds,
        &clientStageTimings);
    execution.result.clientRLWECiphertextCount = packetPairs.size();
    execution.result.survivorClientRLWECiphertextCount =
        survivorUploadedPacketCount;
    execution.result.discardedClientRLWECiphertextCount =
        packetPairs.size() - survivorUploadedPacketCount;

    SparseServerExecution shadowExecution;
    if (compression.measurePreSmudgingNoise) {
        std::vector<SparseClientCiphertextPacket> shadowPackets;
        shadowPackets.reserve(packetPairs.size());
        for (const auto& pair : packetPairs) {
            if (survivorIds.count(pair.bctShadow.clientId) != 0 &&
                std::any_of(
                    pair.bctShadow.derivedLabels.labels.begin(),
                    pair.bctShadow.derivedLabels.labels.end(),
                    [&](ShuffledChunkLabel label) {
                        return acceptedLabels.count(label) != 0;
                    })) {
                shadowPackets.push_back(pair.bctShadow);
            }
        }
        SparsePathCompressionProfile shadowProfile = compression;
        shadowProfile.compressClientB = false;
        shadowProfile.compressPartialDecryption = false;
        shadowProfile.quantizeFinalResult = false;
        shadowProfile.measurePreSmudgingNoise = false;
        shadowExecution = RunLabelOnlyServer(
            config,
            survivorRound.serverPlan,
            shadowPackets,
            setup,
            survivorKeys,
            scalingFactor,
            shadowProfile,
            activeExecutor,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            true);
    }

    SparsePathResult result;
    result.server = std::move(execution.result);
    result.scalingFactor = scalingFactor;
    result.clientEncryptionSeconds = std::move(clientEncryptionSeconds);
    result.clientKeySwitchSeconds = std::move(clientKeySwitchSeconds);
    result.clientPartialDecryptionSeconds =
        std::move(clientPartialDecryptionSeconds);
    result.clientStageTimings = std::move(clientStageTimings);
    result.serverStageTimings = serverMetrics.Snapshot();
    result.cryptoStageTimings = result.serverStageTimings;
    for (const auto& timings : result.clientStageTimings) {
        AccumulateStageTimings(&result.cryptoStageTimings, timings);
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
    if (compression.measurePreSmudgingNoise) {
        result.observer.phaseAudit.phaseResidues.resize(config.modelDimension);
        result.observer.phaseAudit.integerReferences.resize(config.modelDimension);
        for (const auto& chunk : result.server.aggregateChunks) {
            const auto actual=ActualChunkForLabel(observerState,chunk.label);
            const auto begin=actual.value*observerState.chunkSize;
            const auto end=std::min(begin+observerState.chunkSize,config.modelDimension);
            for (std::size_t j=begin;j<end;++j) {
                const auto& ref=reference.integerNumerator[j];
                result.observer.phaseAudit.phaseResidues[j]=chunk.phaseResidues[j-begin];
                result.observer.phaseAudit.integerReferences[j]=ref.str();
            }
        }
    }
    result.observer.idealReference = reference.ideal;
    result.observer.quantizedReference = reference.quantized;
    result.observer.acceptedCoordinateMask = reference.mask;
    result.observer.aggregate = ApplyLabeledAggregateChunksForClient(
        config,
        observerState,
        result.server.aggregateChunks);
    if (compression.measurePreSmudgingNoise) {
        if (shadowExecution.finalCiphertexts.size() !=
                shadowExecution.noiseCheckpoints.size()) {
            throw std::runtime_error(
                "sparse noise checkpoint count is inconsistent");
        }
        for (std::size_t index = 0;
             index < shadowExecution.finalCiphertexts.size();
             ++index) {
            const ProtocolNoiseBreakdownMeasurement breakdown =
                PrivilegedMeasureNoiseBreakdown(
                    shadowExecution.finalCiphertexts[index],
                    shadowExecution.noiseCheckpoints[index],
                    setup,
                    compression,
                    survivorRound.serverPlan,
                    observerState,
                    reference);
            MergeProtocolNoiseBreakdownMeasurement(
                &result.preSmudgingNoiseBreakdown,
                breakdown);
            MergeProtocolNoiseMeasurement(
                &result.preSmudgingNoise,
                breakdown.total);
        }
    }
    result.sequentialSeconds = uploads.sequentialSeconds +
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - finalizeStarted).count();
    return result;
}

SparsePathResult RunProtocolSparsePathAfterDropout(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& uploadRound,
    const ProtocolSparseRoundArtifacts& survivorRound,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    const SparsePathUploadBatch uploads = EncryptProtocolSparseUploads(
        cc,
        config,
        clients,
        uploadRound,
        setup,
        compression,
        executor);
    return FinalizeProtocolSparsePathAfterDropout(
        cc,
        config,
        clients,
        uploads,
        survivorRound,
        setup,
        survivorKeys,
        compression,
        executor,
        aggregateMetrics);
}

SparsePathResult RunProtocolSparsePath(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<ProtocolClientRoundInput>& clients,
    const ProtocolSparseRoundArtifacts& round,
    const ProtocolSetupResult& setup,
    const SurvivorKeySet& survivorKeys,
    const SparsePathCompressionProfile& compression,
    CryptoTaskExecutor* executor,
    CryptoStageMetrics* aggregateMetrics) {

    return RunProtocolSparsePathAfterDropout(
        cc,
        config,
        clients,
        round,
        round,
        setup,
        survivorKeys,
        compression,
        executor,
        aggregateMetrics);
}

}  // namespace federated
