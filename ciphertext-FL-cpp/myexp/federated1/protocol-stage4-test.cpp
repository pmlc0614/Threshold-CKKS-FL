#include "protocol-stage4-test.h"

#include "collective-conversion-keygen.h"
#include "deterministic-randomness.h"
#include "federated-encrypted.h"
#include "protocol-parameters.h"
#include "protocol-setup.h"
#include "protocol-sparse-path.h"
#include "protocol-sparse-round.h"
#include "sparsification.h"
#include "survivor-keys.h"
#include "threshold-ckks.h"

#include "openfhe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

bool Check(bool condition, const std::string& description) {
    std::cout
        << (condition ? "[PASS] " : "[FAIL] ")
        << description
        << '\n';
    return condition;
}

template <typename T, typename = void>
struct HasPermutationMember : std::false_type {};

template <typename T>
struct HasPermutationMember<
    T,
    std::void_t<decltype(std::declval<T>().permutation)>> : std::true_type {};

template <typename T, typename = void>
struct HasInversePermutationMember : std::false_type {};

template <typename T>
struct HasInversePermutationMember<
    T,
    std::void_t<decltype(std::declval<T>().inversePermutation)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasActualChunksMember : std::false_type {};

template <typename T>
struct HasActualChunksMember<
    T,
    std::void_t<decltype(std::declval<T>().actualChunks)>> : std::true_type {};

template <typename T, typename = void>
struct HasClientOnlySetupMember : std::false_type {};

template <typename T>
struct HasClientOnlySetupMember<
    T,
    std::void_t<decltype(std::declval<T>().clientOnly)>> : std::true_type {};

lbcrypto::CryptoContext<DCRTPoly> MakeProtocolContext(
    const federated::ProtocolExperimentConfig& config) {

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(1);
    parameters.SetFirstModSize(config.modulus.qModulusBits.front());
    parameters.SetScalingModSize(config.modulus.qModulusBits[1] - 1);
    parameters.SetBatchSize(config.modulus.ringDimension / 2);
    parameters.SetRingDim(config.modulus.ringDimension);
    parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetScalingTechnique(lbcrypto::FIXEDMANUAL);
    parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(lbcrypto::BV);

    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    return cc;
}

lbcrypto::CryptoContext<DCRTPoly> MakeFullThresholdBaselineContext(
    std::size_t ringDimension) {

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(1);
    parameters.SetFirstModSize(55);
    parameters.SetScalingModSize(55);
    parameters.SetBatchSize(ringDimension / 2);
    parameters.SetRingDim(ringDimension);
    parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetScalingTechnique(lbcrypto::FIXEDMANUAL);
    parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
    parameters.SetNumLargeDigits(2);

    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::MULTIPARTY);
    return cc;
}

std::set<std::size_t> PreferredChunks(std::size_t clientId) {
    return {0, 1, 2 + clientId % 5, 7 + clientId % 5};
}

std::vector<federated::ProtocolClientRoundInput> GenerateInputs(
    const federated::ProtocolExperimentConfig& config) {

    using namespace federated;
    const DerivedDimensions dimensions = DeriveDimensions(config);
    std::vector<ProtocolClientRoundInput> clients;
    clients.reserve(config.clientCount);

    for (std::size_t clientId = 0;
         clientId < config.clientCount;
         ++clientId) {

        ProtocolClientRoundInput client;
        client.clientId = ClientId{clientId};
        client.integerWeight = config.integerWeights[clientId];
        client.update.assign(config.modelDimension, 0.0);
        const auto preferred = PreferredChunks(clientId);

        for (std::size_t chunk = 0;
             chunk < dimensions.chunkCount;
             ++chunk) {

            ByteVector context =
                deterministic_randomness::EncodeBigEndian64(clientId);
            const ByteVector encodedChunk =
                deterministic_randomness::EncodeBigEndian64(chunk);
            context.insert(
                context.end(),
                encodedChunk.begin(),
                encodedChunk.end());

            const bool selected = preferred.count(chunk) != 0;
            const long double sigma = selected
                ? config.syntheticUpdates.modelDelta.standardDeviation
                : config.syntheticUpdates.modelDelta.standardDeviation * 0.001L;
            const long double clip = selected ? 0.06L : 0.00006L;
            const auto values =
                deterministic_randomness::GenerateClippedGaussian(
                    config.masterSeed,
                    "STAGE4-CLIENT-CHUNK",
                    context,
                    dimensions.smallRingDimension,
                    0.0L,
                    sigma,
                    -clip,
                    clip);

            const std::size_t begin = chunk * dimensions.smallRingDimension;
            const std::size_t end = std::min(
                begin + dimensions.smallRingDimension,
                config.modelDimension);
            std::copy(
                values.begin(),
                values.begin() + (end - begin),
                client.update.begin() + begin);
        }
        clients.push_back(std::move(client));
    }
    return clients;
}

double MaximumError(
    const std::vector<double>& left,
    const std::vector<double>& right) {

    if (left.size() != right.size()) {
        throw std::invalid_argument("error vectors have different sizes");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

long double CompressionModelBound(
    const federated::ProtocolExperimentConfig& config,
    std::uint64_t q,
    long double scalingFactor,
    std::size_t survivorCount) {

    const long double denominator =
        static_cast<long double>(config.weightDenominator) * scalingFactor;
    const long double weightSum = config.weightDenominator;
    const long double c1 = weightSum * q /
        (denominator * std::ldexp(1.0L, config.compression.c1 + 1));
    const long double c2 = survivorCount * q /
        (denominator * std::ldexp(1.0L, config.compression.c2 + 1));

    /* Worst-case ternary convolution count N*|S| for full-a error. */
    const long double ca = static_cast<long double>(q) *
        config.modulus.ringDimension * survivorCount /
        (denominator * std::ldexp(1.0L, config.compression.ca + 1));
    const long double cd = config.updateBound /
        static_cast<long double>(
            (std::uint64_t{1} << (config.compression.cd - 1)) - 1U) /
        2.0L;

    const long double decodedSigmaSmg =
        std::sqrt(static_cast<long double>(survivorCount)) *
        config.smudging.sigmaSmg /
        denominator;

    /* Six-sigma smudging tail plus crypto/integer-rounding margin. */
    return c1 + c2 + ca + cd +
        6.0L * decodedSigmaSmg + 1.0e-4L;
}

std::vector<federated::ClientRoundInput> ToBaselineInputs(
    const std::vector<federated::ProtocolClientRoundInput>& clients) {

    std::vector<federated::ClientRoundInput> result;
    result.reserve(clients.size());
    for (const auto& client : clients) {
        federated::ClientRoundInput converted;
        converted.clientId = client.clientId.value;
        converted.alpha = client.integerWeight;
        converted.update = client.update;
        result.push_back(std::move(converted));
    }
    return result;
}

federated::FederatedEncryptedResult RunExistingThresholdBaseline(
    const federated::ProtocolExperimentConfig& protocolConfig,
    const std::vector<federated::ProtocolClientRoundInput>& protocolClients) {

    federated::FederatedConfig config;
    config.clientCount = protocolConfig.clientCount;
    config.modelDimension = protocolConfig.modelDimension;
    config.ringDimension = protocolConfig.modulus.ringDimension;
    config.mlweRank = protocolConfig.sparsification.mlweRank;
    config.selectedChunkCount =
        protocolConfig.sparsification.selectedChunksPerClient;
    config.minimumCohortSize =
        protocolConfig.sparsification.minimumCohort;
    config.weightDenominator = protocolConfig.weightDenominator;
    config.scaleBits = 44;
    config.sigmaShare = static_cast<double>(
        protocolConfig.smudging.sigmaSmg);

    const auto clients = ToBaselineInputs(protocolClients);
    const auto plan = federated::BuildSparseRoundPlan(config, clients);
    auto cc = MakeFullThresholdBaselineContext(config.ringDimension);
    threshold_ckks::ConfigureSmudgingSigma(cc, config.sigmaShare);
    const auto keys = threshold_ckks::GenerateCollectiveKeys(
        cc,
        config.clientCount);
    const auto conversionKeys =
        collective_conversion_keygen::GenerateCollectiveMLWEtoRLWEKeys(
            cc,
            keys.secretKeyShares,
            keys.collectivePublicKey,
            config.mlweRank);

    return federated::ComputeFederatedEncryptedAggregate(
        cc,
        keys.collectivePublicKey,
        keys.secretKeyShares,
        conversionKeys,
        config,
        clients,
        plan);
}

}  // namespace

int RunProtocolStage4Tests() {
    using namespace federated;
    bool passed = true;

    try {
        ProtocolExperimentConfig config = MakeSmokeConfig();
        auto cc = MakeProtocolContext(config);
        const ProtocolSetupResult setup = GenerateProtocolSetup(cc, config);
        const std::uint64_t q = setup.common.rns.paramsQ
            ->GetModulus().ConvertToInt();
        const long double sigmaSmg = std::ceil(SigmaSmgFromModulus(q, 16));
        config.smudging.sigmaSmg = sigmaSmg;
        config.smudging.auxSigmaSmg = sigmaSmg;

        SurvivorSet survivors;
        for (std::size_t client = 0; client < config.clientCount; ++client) {
            survivors.clients.push_back(ClientId{client});
        }
        const SurvivorKeySet keys = AggregateSurvivorKeys(setup, survivors);
        const auto clients = GenerateInputs(config);
        const RoundId roundId =
            deterministic_randomness::DeriveRoundId(config.masterSeed, 0);
        const ClientOnlySetupMaterial clientOnlySetup =
            GenerateClientOnlySetupMaterial(config, setup.common.setupId);
        const ProtocolSparseRoundArtifacts round =
            BuildProtocolSparseRoundArtifacts(
            config,
            clientOnlySetup,
            roundId,
            survivors,
            clients);
        const auto& plan = round.serverPlan;

        passed &= Check(
            !HasPermutationMember<ServerSparseAdmissionPlan>::value &&
                !HasInversePermutationMember<ServerSparseAdmissionPlan>::value &&
                !HasActualChunksMember<ServerSparseAdmissionPlan>::value &&
                !HasClientOnlySetupMember<ProtocolSetupResult>::value,
            "server setup/admission types expose no Kidx, actual chunk, or permutation state");

        bool clientsSharePermutation = !round.clientPrivateStates.empty();
        for (const auto& state : round.clientPrivateStates) {
            clientsSharePermutation &=
                state.kidx == round.clientPrivateStates.front().kidx &&
                state.permutation ==
                    round.clientPrivateStates.front().permutation &&
                state.inversePermutation ==
                    round.clientPrivateStates.front().inversePermutation;
        }
        passed &= Check(
            clientsSharePermutation,
            "clients holding the same Kidx derive the same round permutation");

        passed &= Check(
            plan.acceptedLabels.size() == 2,
            "minimum-cohort approval keeps exactly two shuffled labels");

        bool clientPacksAreLabelSorted = true;
        for (const auto& state : round.clientPrivateStates) {
            for (const auto& group : state.packingGroups) {
                clientPacksAreLabelSorted &= std::is_sorted(
                    group.shuffledLabels.labels.begin(),
                    group.shuffledLabels.labels.end());
            }
        }
        passed &= Check(
            clientPacksAreLabelSorted,
            "client packing order is shuffled-label order, not actual-index order");

        const SparsePathResult uncompressed = RunProtocolSparsePath(
            cc,
            config,
            clients,
            round,
            setup,
            keys,
            MakeUncompressedSparseProfile());

        const SparsePathResult compressed = RunProtocolSparsePath(
            cc,
            config,
            clients,
            round,
            setup,
            keys,
            MakeConfiguredSparseProfile(config));

        passed &= Check(
            uncompressed.server.clientRLWECiphertextCount == config.clientCount &&
                uncompressed.server.acceptedContributionCount == 16 &&
                uncompressed.server.aggregatedMLWECiphertextCount == 2 &&
                uncompressed.server.serverRLWECiphertextCount == 1,
            "packing, R2M aggregation, and M2R ciphertext counts are correct");

        bool serverResultIsLabelOnly =
            uncompressed.server.aggregateChunks.size() ==
            plan.acceptedLabels.size();
        for (std::size_t index = 0;
             index < uncompressed.server.aggregateChunks.size();
             ++index) {
            serverResultIsLabelOnly &=
                uncompressed.server.aggregateChunks[index].label ==
                    plan.acceptedLabels[index] &&
                uncompressed.server.aggregateChunks[index].values.size() ==
                    plan.chunkSize;
        }
        passed &= Check(
            serverResultIsLabelOnly,
            "server finalization returns labeled chunks instead of a dense model vector");

        const auto independentlyBuiltServerPlan =
            BuildServerSparseAdmissionPlan(
                config,
                plan.setupId,
                plan.roundId,
                survivors,
                plan.announcements);
        passed &= Check(
            independentlyBuiltServerPlan.acceptedLabels ==
                    plan.acceptedLabels &&
                independentlyBuiltServerPlan.publicPlanHash ==
                    plan.publicPlanHash,
            "server computes cohorts from shuffled-label announcements only");

        const auto clientApplied = ApplyLabeledAggregateChunksForClient(
            config,
            round.clientPrivateStates.front(),
            uncompressed.server.aggregateChunks);
        bool labeledChunksReachedActualCoordinates = true;
        for (const auto& chunk : uncompressed.server.aggregateChunks) {
            const auto& privateState = round.clientPrivateStates.front();
            const auto mapping = std::find_if(
                privateState.selectedMappings.begin(),
                privateState.selectedMappings.end(),
                [&chunk](const ClientSparseChunkLabelMapping& candidate) {
                    return candidate.label == chunk.label;
                });
            if (mapping == privateState.selectedMappings.end()) {
                labeledChunksReachedActualCoordinates = false;
                continue;
            }
            const std::size_t begin =
                mapping->actualChunk.value * privateState.chunkSize;
            const std::size_t end = std::min(
                begin + privateState.chunkSize,
                config.modelDimension);
            labeledChunksReachedActualCoordinates &= std::equal(
                clientApplied.begin() + begin,
                clientApplied.begin() + end,
                chunk.values.begin());
        }
        passed &= Check(
            labeledChunksReachedActualCoordinates &&
                clientApplied == uncompressed.observer.aggregate,
            "client inverse permutation applies labeled results to actual chunks");

        std::vector<double> residual;
        const auto& firstPrivate = round.clientPrivateStates.front();
        const auto& firstClient = clients[firstPrivate.clientId.value];
        UpdateClientResidualFromFinalizedChunks(
            config,
            firstClient,
            firstPrivate,
            uncompressed.server.aggregateChunks,
            true,
            &residual);
        bool finalizedLabelsWereCleared = true;
        bool rejectedLabelRemains = false;
        for (const auto& mapping : firstPrivate.selectedMappings) {
            const bool finalized = std::any_of(
                uncompressed.server.aggregateChunks.begin(),
                uncompressed.server.aggregateChunks.end(),
                [&mapping](const LabeledAggregateChunk& chunk) {
                    return chunk.label == mapping.label;
                });
            const std::size_t begin =
                mapping.actualChunk.value * firstPrivate.chunkSize;
            const std::size_t end = std::min(
                begin + firstPrivate.chunkSize,
                config.modelDimension);
            if (finalized) {
                finalizedLabelsWereCleared &= std::all_of(
                    residual.begin() + begin,
                    residual.begin() + end,
                    [](double value) { return value == 0.0; });
            }
            else {
                rejectedLabelRemains |= std::equal(
                    residual.begin() + begin,
                    residual.begin() + end,
                    firstClient.update.begin() + begin);
            }
        }
        passed &= Check(
            rejectedLabelRemains,
            "rejected labels remain in the client residual");
        passed &= Check(
            finalizedLabelsWereCleared,
            "only labels present in the finalized result clear residual chunks");

        const double uncompressedError = MaximumError(
            uncompressed.observer.aggregate,
            uncompressed.observer.quantizedReference);
        const double compressedError = MaximumError(
            compressed.observer.aggregate,
            compressed.observer.quantizedReference);
        const double compressedVsLossless = MaximumError(
            compressed.observer.aggregate,
            uncompressed.observer.aggregate);

        const long double compressionBound = CompressionModelBound(
            config,
            q,
            uncompressed.scalingFactor,
            survivors.clients.size());

        std::cout
            << "lossless sparse-path max error:     " << uncompressedError << '\n'
            << "compressed sparse-path max error:   " << compressedError << '\n'
            << "compressed-vs-lossless max error:   " << compressedVsLossless << '\n'
            << "configured compression model bound: "
            << static_cast<double>(compressionBound) << '\n';

        constexpr long double cryptoRoundingMargin = 1.0e-5L;
        const long double decodedProtocolSigmaSmg =
            std::sqrt(static_cast<long double>(survivors.clients.size())) *
            config.smudging.sigmaSmg /
            (static_cast<long double>(config.weightDenominator) *
             uncompressed.scalingFactor);
        passed &= Check(
            uncompressedError <=
                static_cast<double>(
                    6.0L * decodedProtocolSigmaSmg +
                    cryptoRoundingMargin),
            "lossless sparse path matches the integer reference");
        passed &= Check(
            compressedError <= static_cast<double>(compressionBound),
            "c1/ca/c2/cd sparse-path error stays inside the model bound");
        passed &= Check(
            compressed.server.planHash == uncompressed.server.planHash,
            "compressed and lossless paths use the identical fixed plan");

        std::cout << "[INFO] running existing full OpenFHE threshold baseline\n";
        const federated::FederatedEncryptedResult baseline =
            RunExistingThresholdBaseline(config, clients);
        const double baselineError = MaximumError(
            baseline.aggregate,
            uncompressed.observer.idealReference);
        const double baselineVsProtocol = MaximumError(
            baseline.aggregate,
            uncompressed.observer.aggregate);

        const long double decodedBaselineSigmaSmg =
            std::sqrt(static_cast<long double>(survivors.clients.size())) *
            config.smudging.sigmaSmg /
            (static_cast<long double>(config.weightDenominator) *
             static_cast<long double>(baseline.scalingFactor));

        /* The two paths draw independent threshold smudging errors. */
        const long double decodedCrossSigmaSmg = std::sqrt(
            decodedBaselineSigmaSmg * decodedBaselineSigmaSmg +
            decodedProtocolSigmaSmg * decodedProtocolSigmaSmg);
        const long double baselineTolerance =
            6.0L * decodedBaselineSigmaSmg + cryptoRoundingMargin;
        const long double baselineVsProtocolTolerance =
            6.0L * decodedCrossSigmaSmg +
            2.0L * cryptoRoundingMargin;

        std::cout
            << "full-threshold baseline max error:  " << baselineError << '\n'
            << "full-threshold baseline tolerance:  "
            << static_cast<double>(baselineTolerance) << '\n'
            << "baseline-vs-v17 lossless max error: "
            << baselineVsProtocol << '\n'
            << "baseline-vs-v17 tolerance:          "
            << static_cast<double>(baselineVsProtocolTolerance) << '\n';

        passed &= Check(
            baselineError <= static_cast<double>(baselineTolerance),
            "existing full threshold baseline matches the plaintext result");
        passed &= Check(
            baselineVsProtocol <=
                static_cast<double>(baselineVsProtocolTolerance),
            "v17 sparse path agrees with the existing full threshold baseline");
    }
    catch (const std::exception& error) {
        std::cout
            << "[FAIL] protocol stage-4 test threw: "
            << error.what()
            << '\n';
        passed = false;
    }

    std::cout << '\n';
    if (passed) {
        std::cout << "All protocol stage-4 tests passed.\n";
        return 0;
    }
    std::cout << "One or more protocol stage-4 tests failed.\n";
    return 1;
}
