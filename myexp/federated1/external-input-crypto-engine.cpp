#include "external-input-crypto-engine.h"

#include "protocol-auxiliary-path.h"
#include "protocol-full-model-path.h"
#include "protocol-sparse-path.h"
#include "protocol-sparse-round.h"
#include "protocol-setup.h"
#include "survivor-keys.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace federated {
namespace {

constexpr std::uint64_t kMaximumSizeT =
    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());

bool IsTerminal(RoundPhase phase) {
    return phase == RoundPhase::COMMITTED || phase == RoundPhase::ABORTED;
}

std::uint64_t LayoutNumel(const ProtocolTensorLayout& layout) {
    ValidateProtocolTensorLayout(layout);
    const auto& last = layout.entries.back();
    if (last.offset > std::numeric_limits<std::uint64_t>::max() - last.numel) {
        throw std::overflow_error("protocol layout size overflows uint64");
    }
    return last.offset + last.numel;
}

void RequireLayout(
    const ProtocolTensorLayout& layout,
    LayoutKind expectedKind,
    const std::string& expectedModelName) {

    ValidateProtocolTensorLayout(layout);
    if (layout.kind != expectedKind ||
        (!expectedModelName.empty() && layout.modelName != expectedModelName)) {
        throw std::invalid_argument("external engine layout metadata mismatch");
    }
}

std::vector<float> ToFloat32(
    const std::vector<double>& values,
    const char* name) {

    std::vector<float> result;
    result.reserve(values.size());
    for (const double value : values) {
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max()) {
            throw std::overflow_error(
                std::string(name) + " cannot be represented as float32");
        }
        result.push_back(static_cast<float>(value));
    }
    return result;
}

void ValidateDecryptedAggregate(
    const std::vector<double>& values,
    long double aggregateBound,
    const char* name) {

    for (const double value : values) {
        if (!std::isfinite(value) ||
            std::abs(static_cast<long double>(value)) > aggregateBound) {
            throw std::runtime_error(
                std::string(name) +
                " exceeds its configured aggregate bound");
        }
    }
}

}  // namespace

const ClientAdmissionResult& ExternalEngineAdmissionPlan::ForClient(
    ClientId clientId) const {

    const auto found = std::find_if(
        clientResults.begin(),
        clientResults.end(),
        [clientId](const ClientAdmissionResult& result) {
            return result.clientId == clientId;
        });
    if (found == clientResults.end()) {
        throw std::out_of_range("external admission result was not found");
    }
    return *found;
}

struct ExternalInputCryptoEngine::Impl {
    struct Registration {
        ClientId externalId;
        ClientId internalId;
        std::uint64_t exampleCount = 0;
        std::uint32_t alpha = 0;
    };

    explicit Impl(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context)
        : cc(std::move(context)) {}

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
    std::optional<ExternalEngineSetupRequest> request;
    std::optional<ExternalEngineSetupResult> manifest;
    ProtocolSetupResult setup;
    SurvivorKeySet survivorKeys;
    std::shared_ptr<SurvivorKeySet> roundSurvivorKeys;
    std::map<std::vector<std::size_t>, std::shared_ptr<SurvivorKeySet>>
        v17TargetKeyCache;
    std::optional<ClientOnlySetupMaterial> clientOnlySetup;
    std::vector<Registration> registrations;
    std::unique_ptr<CryptoTaskExecutor> cryptoExecutor;
    CryptoStageMetrics cryptoStageMetrics;

    std::optional<RoundPhase> phase;
    std::optional<RoundId> currentRoundId;
    std::optional<RoundId> lastRoundId;
    std::optional<ExternalEngineAbortedRoundTraffic> lastAbortedRoundTraffic;
    std::optional<std::uint64_t> currentLogicalRoundIndex;
    std::optional<std::uint64_t> lastCommittedLogicalRoundIndex;
    std::optional<std::uint64_t> lastAbortedLogicalRoundIndex;
    std::set<std::vector<std::size_t>> startedDropoutTargets;
    std::vector<std::uint64_t> finalContext;
    std::set<std::uint64_t> finalAcknowledgements;
    std::vector<std::optional<ClientSelectionRequest>> selections;
    std::optional<ProtocolSparseRoundArtifacts> sparseRound;
    std::optional<ExternalEngineAdmissionPlan> admission;
    std::optional<ProtocolSparseRoundArtifacts> survivorSparseRound;
    std::optional<ExternalEngineAdmissionPlan> survivorAdmission;
    std::optional<ExternalEngineSurvivorTransition> survivorTransition;
    std::vector<ExternalEngineSurvivorTransition> survivorTransitions;
    std::optional<SparsePathUploadBatch> sparseUploads;
    std::optional<FullModelUploadBatch> fullUploads;
    std::optional<AuxiliaryUploadBatch> auxiliaryUploads;
    double postUploadCachedEncryptionSequentialSeconds = 0.0;
    std::vector<std::optional<AcceptedChunkValues>> acceptedValues;
    std::vector<std::optional<ExternalFullParameterDeltaPayload>> fullValues;
    std::vector<std::optional<AuxiliaryDeltaPayload>> auxiliaryDeltas;

    const ProtocolExperimentConfig& Config() const {
        if (!request.has_value()) {
            throw std::logic_error("external crypto engine is not configured");
        }
        return request->protocol;
    }

    const ExternalEngineSetupResult& Manifest() const {
        if (!manifest.has_value()) {
            throw std::logic_error("external crypto engine is not configured");
        }
        return *manifest;
    }

    std::size_t PositionForExternalClient(ClientId clientId) const {
        const auto found = std::find_if(
            registrations.begin(),
            registrations.end(),
            [clientId](const Registration& registration) {
                return registration.externalId == clientId;
            });
        if (found == registrations.end()) {
            throw std::invalid_argument("external client is not registered");
        }
        return static_cast<std::size_t>(found - registrations.begin());
    }

    void RequirePhase(RoundPhase expected) const {
        if (!phase.has_value() || *phase != expected) {
            throw std::logic_error("external crypto engine round phase mismatch");
        }
    }

    void Transition(RoundPhase following) {
        if (!phase.has_value() ||
            !IsAllowedRoundTransition(*phase, following)) {
            throw std::logic_error("invalid external engine round transition");
        }
        phase = following;
    }

    void EnterValuesPhase() {
        if (phase == RoundPhase::ADMISSION_FINALIZED) {
            Transition(RoundPhase::VALUES_COLLECTING);
        }
        else {
            RequirePhase(RoundPhase::VALUES_COLLECTING);
        }
    }

    void ResetRoundBuffers() {
        finalContext.clear();
        finalAcknowledgements.clear();
        selections.assign(registrations.size(), std::nullopt);
        acceptedValues.assign(registrations.size(), std::nullopt);
        fullValues.assign(registrations.size(), std::nullopt);
        auxiliaryDeltas.assign(registrations.size(), std::nullopt);
        sparseRound.reset();
        admission.reset();
        roundSurvivorKeys.reset();
        survivorSparseRound.reset();
        survivorAdmission.reset();
        survivorTransition.reset();
        survivorTransitions.clear();
        sparseUploads.reset();
        fullUploads.reset();
        auxiliaryUploads.reset();
        postUploadCachedEncryptionSequentialSeconds = 0.0;
        lastAbortedRoundTraffic.reset();
    }

    const SurvivorKeySet& ActiveSurvivorKeys() const {
        return roundSurvivorKeys
            ? *roundSurvivorKeys
            : survivorKeys;
    }

    const ProtocolSparseRoundArtifacts& ActiveSparseRound() const {
        if (survivorSparseRound.has_value()) {
            return *survivorSparseRound;
        }
        if (!sparseRound.has_value()) {
            throw std::logic_error("external sparse round is unavailable");
        }
        return *sparseRound;
    }

    const ExternalEngineAdmissionPlan& ActiveAdmission() const {
        if (survivorAdmission.has_value()) {
            return *survivorAdmission;
        }
        if (!admission.has_value()) {
            throw std::logic_error("external admission is unavailable");
        }
        return *admission;
    }

    std::vector<ProtocolClientRoundInput> BuildSparseInputs() const {
        if (!admission.has_value()) {
            throw std::logic_error("external admission has not been finalized");
        }
        const auto& config = Config();
        const DerivedDimensions dimensions = DeriveDimensions(config);
        std::vector<ProtocolClientRoundInput> result;
        result.reserve(registrations.size());

        for (std::size_t position = 0;
             position < registrations.size();
             ++position) {

            if (!acceptedValues[position].has_value()) {
                throw std::logic_error("accepted client values are missing");
            }
            const auto& registration = registrations[position];
            const auto& payload = *acceptedValues[position];
            ProtocolClientRoundInput input;
            input.clientId = registration.internalId;
            input.integerWeight = registration.alpha;
            input.update.assign(config.modelDimension, 0.0);

            for (std::size_t row = 0;
                 row < payload.actualChunkIds.size();
                 ++row) {

                const std::size_t chunk = static_cast<std::size_t>(
                    payload.actualChunkIds[row].value);
                const std::size_t begin = chunk * dimensions.smallRingDimension;
                const std::size_t validCount = std::min(
                    dimensions.smallRingDimension,
                    config.modelDimension - begin);
                const std::size_t valueBegin =
                    row * dimensions.smallRingDimension;
                for (std::size_t offset = 0; offset < validCount; ++offset) {
                    input.update[begin + offset] =
                        payload.values[valueBegin + offset];
                }
            }
            result.push_back(std::move(input));
        }
        return result;
    }

    std::vector<AuxiliaryClientRoundInput> BuildAuxiliaryInputs() const {
        std::vector<AuxiliaryClientRoundInput> result;
        result.reserve(registrations.size());
        for (std::size_t position = 0;
             position < registrations.size();
             ++position) {

            if (!auxiliaryDeltas[position].has_value()) {
                throw std::logic_error("external auxiliary deltas are missing");
            }
            const auto& registration = registrations[position];
            const auto& payload = *auxiliaryDeltas[position];
            AuxiliaryClientRoundInput input;
            input.clientId = registration.internalId;
            input.integerWeight = registration.alpha;
            input.runningMeanDelta.assign(
                payload.runningMeanDelta.begin(),
                payload.runningMeanDelta.end());
            input.runningVarianceDelta.assign(
                payload.runningVarianceDelta.begin(),
                payload.runningVarianceDelta.end());
            result.push_back(std::move(input));
        }
        return result;
    }

    std::vector<FullModelClientRoundInput> BuildFullInputs() const {
        std::vector<FullModelClientRoundInput> result;
        result.reserve(registrations.size());
        for (std::size_t position = 0;
             position < registrations.size();
             ++position) {
            if (!fullValues[position].has_value()) {
                throw std::logic_error("external FULL parameter delta is missing");
            }
            const auto& registration = registrations[position];
            const auto& payload = *fullValues[position];
            FullModelClientRoundInput input;
            input.clientId = registration.internalId;
            input.integerWeight = registration.alpha;
            input.modelDelta.assign(
                payload.parameterDelta.begin(), payload.parameterDelta.end());
            result.push_back(std::move(input));
        }
        return result;
    }

    void PrepareUploads() {
        const auto& config = Config();
        if (config.mode == ProtocolMode::SP && !sparseUploads.has_value()) {
            SparsePathCompressionProfile profile =
                request->execution.useSparseCompression
                    ? MakeConfiguredSparseProfile(config)
                    : MakeUncompressedSparseProfile();
            profile.measurePreSmudgingNoise =
                request->execution.measurePreSmudgingNoise;
            profile.computeReference = profile.measurePreSmudgingNoise;
            auto uploads = EncryptProtocolSparseUploads(
                cc, config, BuildSparseInputs(), *sparseRound,
                setup, profile, cryptoExecutor.get());
            postUploadCachedEncryptionSequentialSeconds +=
                uploads.sequentialSeconds;
            sparseUploads = std::move(uploads);
        }
        if (config.mode == ProtocolMode::FULL && !fullUploads.has_value()) {
            FullModelCompressionProfile profile =
                request->execution.useSparseCompression
                    ? MakeConfiguredFullModelProfile(config)
                    : MakeUncompressedFullModelProfile();
            profile.measurePreSmudgingNoise =
                request->execution.measurePreSmudgingNoise;
            profile.computeReference = profile.measurePreSmudgingNoise;
            auto uploads = EncryptProtocolFullModelUploads(
                cc, config, *currentRoundId, BuildFullInputs(),
                setup, profile, cryptoExecutor.get());
            postUploadCachedEncryptionSequentialSeconds +=
                uploads.sequentialSeconds;
            fullUploads = std::move(uploads);
        }
        if (!auxiliaryUploads.has_value()) {
            AuxiliaryCompressionProfile profile =
                request->execution.useAuxiliaryCompression
                    ? MakeConfiguredAuxiliaryProfile(config)
                    : MakeUncompressedAuxiliaryProfile();
            profile.measurePreSmudgingNoise =
                request->execution.measurePreSmudgingNoise;
            profile.computeReference = profile.measurePreSmudgingNoise;
            auto uploads = EncryptProtocolAuxiliaryUploads(
                cc, config, *currentRoundId, BuildAuxiliaryInputs(),
                setup, profile, cryptoExecutor.get());
            postUploadCachedEncryptionSequentialSeconds +=
                uploads.sequentialSeconds;
            auxiliaryUploads = std::move(uploads);
        }
    }

    ExternalEngineAbortedRoundTraffic BuildAbortedRoundTraffic() const {
        ExternalEngineAbortedRoundTraffic traffic;
        traffic.roundId = *currentRoundId;
        if (Config().mode == ProtocolMode::SP) {
            traffic.sparseClientCiphertextCount =
                sparseUploads->packetPairs.size();
            traffic.uploadAcceptedChunkCount =
                admission->acceptedActualChunkIds.size();
            traffic.acceptedChunkCount =
                ActiveAdmission().acceptedActualChunkIds.size();
            traffic.sparseServerCiphertextCount =
                ActiveSparseRound().serverPlan.packingGroups.size();
            for (const auto& client : admission->clientResults) {
                traffic.uploadedSparseContributionCount +=
                    client.acceptedActualChunkIds.size();
            }
            for (const auto& client : ActiveAdmission().clientResults) {
                traffic.acceptedSparseContributionCount +=
                    client.acceptedActualChunkIds.size();
            }
            if (Config().protocolVersion == "v17") {
                traffic.selectionBitmapCount = registrations.size();
                traffic.admissionBitmapRecipientCount = registrations.size();
                for (const auto& transition : survivorTransitions) {
                    traffic.admissionBitmapRecipientCount +=
                        transition.survivorClientIds.size();
                }
                traffic.uploadSupportBitmapCount =
                    sparseUploads->supportHeaders.size();
            }
        }
        for (const auto& packets : auxiliaryUploads->wirePacketsByClient) {
            traffic.auxiliaryClientCiphertextCount += packets.size();
        }
        traffic.auxiliaryAggregateCiphertextCount =
            auxiliaryUploads->blockCount;
        if (survivorTransition.has_value()) {
            traffic.survivorClientIds =
                survivorTransition->survivorClientIds;
            traffic.droppedClientIds =
                survivorTransition->droppedClientIds;
        }
        else {
            for (const auto& client : registrations) {
                traffic.survivorClientIds.push_back(client.externalId);
            }
        }
        for (const auto& transition : survivorTransitions) {
            if (!transition.targetKeyCacheHit) {
                traffic.survivorNewTargetSizes.push_back(
                    transition.survivorClientIds.size());
            }
        }
        return traffic;
    }
};

ExternalInputCryptoEngine::ExternalInputCryptoEngine(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext)
    : impl_(std::make_unique<Impl>(std::move(cryptoContext))) {}

ExternalInputCryptoEngine::~ExternalInputCryptoEngine() = default;
ExternalInputCryptoEngine::ExternalInputCryptoEngine(
    ExternalInputCryptoEngine&&) noexcept = default;
ExternalInputCryptoEngine& ExternalInputCryptoEngine::operator=(
    ExternalInputCryptoEngine&&) noexcept = default;

ExternalEngineSetupResult ExternalInputCryptoEngine::Setup(
    const ExternalEngineSetupRequest& input) {

    if (!impl_->cc) {
        throw std::invalid_argument("external crypto engine context is null");
    }
    if (impl_->phase.has_value() && !IsTerminal(*impl_->phase)) {
        throw std::logic_error("cannot replace setup during an active round");
    }

    RequireLayout(input.parameterLayout, LayoutKind::PARAMETER, "");
    RequireLayout(
        input.runningMeanLayout,
        LayoutKind::RUNNING_MEAN,
        input.parameterLayout.modelName);
    RequireLayout(
        input.runningVarianceLayout,
        LayoutKind::RUNNING_VARIANCE,
        input.parameterLayout.modelName);

    const std::uint64_t parameterNumel = LayoutNumel(input.parameterLayout);
    const std::uint64_t runningMeanNumel = LayoutNumel(input.runningMeanLayout);
    const std::uint64_t runningVarianceNumel =
        LayoutNumel(input.runningVarianceLayout);
    if (parameterNumel > kMaximumSizeT ||
        runningMeanNumel > kMaximumSizeT ||
        runningVarianceNumel > kMaximumSizeT ||
        input.clients.empty()) {
        throw std::invalid_argument("external engine dimensions are invalid");
    }

    auto cryptoExecutor = std::make_unique<CryptoTaskExecutor>(
        input.execution.cryptoThreads);
    const auto alphas = QuantizeRegisteredClientWeights(
        input.clients,
        kWeightDenominator);
    ExternalEngineSetupRequest normalized = input;
    normalized.protocol.clientCount = input.clients.size();
    normalized.protocol.modelDimension =
        static_cast<std::size_t>(parameterNumel);
    normalized.protocol.syntheticUpdates.runningMeanDimension =
        static_cast<std::size_t>(runningMeanNumel);
    normalized.protocol.syntheticUpdates.runningVarianceDimension =
        static_cast<std::size_t>(runningVarianceNumel);
    normalized.protocol.weightDenominator = kWeightDenominator;
    normalized.protocol.integerWeights = alphas;
    normalized.protocol.dropoutEnabled = false;
    ValidateProtocolParameters(normalized.protocol).ThrowIfInvalid();

    std::set<std::uint64_t> externalIds;
    std::vector<Impl::Registration> registrations;
    registrations.reserve(input.clients.size());
    SurvivorSet survivors;
    for (std::size_t position = 0; position < input.clients.size(); ++position) {
        const auto& client = input.clients[position];
        if (!externalIds.insert(client.clientId.value).second) {
            throw std::invalid_argument("duplicate external client ID");
        }
        const ClientId internalId{position};
        registrations.push_back(Impl::Registration{
            client.clientId,
            internalId,
            client.exampleCount,
            alphas[position]});
        survivors.clients.push_back(internalId);
    }

    ProtocolSetupResult setup = GenerateProtocolSetup(
        impl_->cc,
        normalized.protocol);
    if (!setup.common.rns.paramsQ) {
        throw std::logic_error("generated setup is missing Q parameters");
    }
    const long double actualQ = std::stold(
        setup.common.rns.paramsQ->GetModulus().ToString());
    ValidateProtocolParameters(
        normalized.protocol,
        actualQ).ThrowIfInvalid();
    SurvivorKeySet keys = AggregateSurvivorKeys(setup, survivors);
    if (normalized.protocol.mode == ProtocolMode::SP) {
        PrewarmProtocolSparseTransforms(setup);
    }
    std::optional<ClientOnlySetupMaterial> clientOnly;
    if (normalized.protocol.mode == ProtocolMode::SP) {
        clientOnly = GenerateClientOnlySetupMaterial(
            normalized.protocol,
            setup.common.setupId);
    }
    const DerivedDimensions dimensions = DeriveDimensions(normalized.protocol);

    ExternalEngineSetupResult result;
    result.setupId = setup.common.setupId;
    result.crsSeed = setup.common.crsSeed;
    result.setupDescriptor = setup.common.descriptor.canonicalEncoding;
    result.setupDescriptorBytes =
        setup.common.descriptor.canonicalEncoding.size();
    for (const auto& tower : setup.common.rns.paramsQ->GetParams()) {
        result.qPrimes.push_back(
            tower->GetModulus().ConvertToInt());
    }
    for (const auto& tower : setup.common.rns.paramsP->GetParams()) {
        result.pPrimes.push_back(
            tower->GetModulus().ConvertToInt());
    }
    result.parameterLayoutHash = ProtocolLayoutSha256(input.parameterLayout);
    result.runningMeanLayoutHash =
        ProtocolLayoutSha256(input.runningMeanLayout);
    result.runningVarianceLayoutHash =
        ProtocolLayoutSha256(input.runningVarianceLayout);
    result.parameterNumel = parameterNumel;
    result.runningMeanNumel = runningMeanNumel;
    result.runningVarianceNumel = runningVarianceNumel;
    if (normalized.protocol.mode == ProtocolMode::SP) {
        result.chunkSize = dimensions.smallRingDimension;
        result.chunkCount = dimensions.chunkCount;
    }
    else {
        result.chunkSize = normalized.protocol.modulus.ringDimension;
        result.chunkCount =
            (parameterNumel + result.chunkSize - 1U) / result.chunkSize;
    }
    result.alphas = alphas;
    if (setup.timing.clientSecretKeyGenerationSeconds.size() !=
            input.clients.size() ||
        setup.timing.clientCachedShareGenerationSeconds.size() !=
            input.clients.size() ||
        keys.clientToCollectiveKeyGenerationSeconds.size() !=
            input.clients.size()) {
        throw std::runtime_error("native setup timing client count mismatch");
    }
    result.clientInitialSetupSeconds.resize(input.clients.size(), 0.0);
    for (std::size_t position = 0;
         position < input.clients.size();
         ++position) {
        result.clientInitialSetupSeconds[position] =
            setup.timing.clientSecretKeyGenerationSeconds[position] +
            setup.timing.clientCachedShareGenerationSeconds[position];
    }
    result.clientCollectiveKeyGenerationSeconds =
        keys.clientToCollectiveKeyGenerationSeconds;
    for (const auto& client : input.clients) {
        result.clientIds.push_back(client.clientId);
    }

    result.publicSetupBinding = SerializePublicSetupBinding(
        result.setupId, result.crsSeed, result.setupDescriptor, result.clientIds,
        result.parameterLayoutHash, result.runningMeanLayoutHash, result.runningVarianceLayoutHash);

    impl_->request = std::move(normalized);
    impl_->manifest = result;
    impl_->setup = std::move(setup);
    impl_->survivorKeys = std::move(keys);
    impl_->clientOnlySetup = std::move(clientOnly);
    impl_->registrations = std::move(registrations);
    impl_->cryptoExecutor = std::move(cryptoExecutor);
    impl_->cryptoStageMetrics.Reset();
    impl_->phase.reset();
    impl_->currentRoundId.reset();
    impl_->lastRoundId.reset();
    impl_->currentLogicalRoundIndex.reset();
    impl_->lastCommittedLogicalRoundIndex.reset();
    impl_->lastAbortedLogicalRoundIndex.reset();
    impl_->v17TargetKeyCache.clear();
    impl_->startedDropoutTargets.clear();
    impl_->ResetRoundBuffers();
    return result;
}

bool ExternalInputCryptoEngine::IsConfigured() const {
    return impl_->manifest.has_value();
}

const ExternalEngineSetupResult& ExternalInputCryptoEngine::SetupResult() const {
    return impl_->Manifest();
}

const ProtocolExperimentConfig&
ExternalInputCryptoEngine::EffectiveProtocolConfig() const {
    return impl_->Config();
}

std::optional<RoundPhase> ExternalInputCryptoEngine::Phase() const {
    return impl_->phase;
}

std::optional<ExternalEngineAbortedRoundTraffic>
ExternalInputCryptoEngine::LastAbortedRoundTraffic() const {
    return impl_->lastAbortedRoundTraffic;
}

void ExternalInputCryptoEngine::BeginRound(
    RoundId roundId,
    std::optional<std::uint64_t> logicalRoundIndex) {
    (void)impl_->Manifest();
    if (!impl_->Config().experimentalDynamicCohort &&
        impl_->phase == RoundPhase::ABORTED) {
        throw std::logic_error("fixed-cohort execution ended: retries are forbidden");
    }
    if (impl_->phase.has_value() && !IsTerminal(*impl_->phase)) {
        throw std::logic_error("another external crypto round is active");
    }
    if (impl_->lastRoundId.has_value() &&
        roundId.value <= impl_->lastRoundId->value) {
        throw std::invalid_argument("external round IDs must increase");
    }
    if (federated::UsesV17WireProtocol(impl_->Config().protocolVersion)) {
        const std::uint64_t logical = logicalRoundIndex.value_or(
            roundId.value);
        if (impl_->lastAbortedLogicalRoundIndex.has_value() &&
            (logical <= *impl_->lastAbortedLogicalRoundIndex ||
             logical - *impl_->lastAbortedLogicalRoundIndex != 1)) {
            throw std::invalid_argument(
                "v17 attempt cap is one; aborted logical round is closed");
        }
        if (!impl_->lastAbortedLogicalRoundIndex.has_value() &&
            impl_->lastCommittedLogicalRoundIndex.has_value() &&
            (logical <= *impl_->lastCommittedLogicalRoundIndex ||
             logical - *impl_->lastCommittedLogicalRoundIndex != 1)) {
            throw std::invalid_argument(
                "v17 next logical round must follow the last commit");
        }
        impl_->currentLogicalRoundIndex = logical;
    }
    else {
        impl_->currentLogicalRoundIndex.reset();
    }
    impl_->ResetRoundBuffers();
    impl_->cryptoStageMetrics.Reset();
    impl_->currentRoundId = roundId;
    impl_->phase = RoundPhase::CREATED;
    impl_->Transition(
        impl_->Config().mode == ProtocolMode::SP
            ? RoundPhase::SELECTION_COLLECTING
            : RoundPhase::VALUES_COLLECTING);
}

void ExternalInputCryptoEngine::SubmitSelection(
    const ClientSelectionRequest& input) {

    if (impl_->Config().mode != ProtocolMode::SP) {
        throw std::logic_error("FULL rounds do not accept sparse selections");
    }
    impl_->RequirePhase(RoundPhase::SELECTION_COLLECTING);
    ValidateClientSelectionRequest(input);
    const auto& manifest = impl_->Manifest();
    if (!(input.setupId == manifest.setupId) ||
        !impl_->currentRoundId.has_value() ||
        !(input.roundId == *impl_->currentRoundId) ||
        input.parameterLayoutHash != manifest.parameterLayoutHash) {
        throw std::invalid_argument("external client selection metadata mismatch");
    }
    const std::size_t position = impl_->PositionForExternalClient(
        input.clientId);
    if (impl_->selections[position].has_value()) {
        throw std::invalid_argument("duplicate external client selection");
    }
    const DerivedDimensions dimensions = DeriveDimensions(impl_->Config());
    if (input.selectedActualChunkIds.size() !=
        dimensions.selectedChunksPerClient) {
        throw std::invalid_argument("external selection count mismatch");
    }
    for (const auto chunk : input.selectedActualChunkIds) {
        if (chunk.value >= dimensions.chunkCount) {
            throw std::invalid_argument("external selected chunk is out of range");
        }
    }
    impl_->selections[position] = input;
}

ExternalEngineAdmissionPlan ExternalInputCryptoEngine::FinalizeAdmission() {
    if (impl_->Config().mode != ProtocolMode::SP) {
        throw std::logic_error("FULL rounds do not perform sparse admission");
    }
    impl_->RequirePhase(RoundPhase::SELECTION_COLLECTING);
    if (std::any_of(
            impl_->selections.begin(),
            impl_->selections.end(),
            [](const auto& selection) { return !selection.has_value(); })) {
        if (!impl_->Config().experimentalDynamicCohort) AbortRound();
        throw std::logic_error("selection is required from every client");
    }

    std::vector<ClientSelectionRequest> internalSelections;
    internalSelections.reserve(impl_->registrations.size());
    for (std::size_t position = 0;
         position < impl_->registrations.size();
         ++position) {
        ClientSelectionRequest translated = *impl_->selections[position];
        translated.clientId = impl_->registrations[position].internalId;
        internalSelections.push_back(std::move(translated));
    }

    SurvivorSet survivors;
    for (const auto& registration : impl_->registrations) {
        survivors.clients.push_back(registration.internalId);
    }
    ProtocolSparseRoundArtifacts round =
        BuildProtocolSparseRoundArtifactsFromSelections(
            impl_->Config(),
            *impl_->clientOnlySetup,
            *impl_->currentRoundId,
            survivors,
            internalSelections);

    const DerivedDimensions dimensions = DeriveDimensions(impl_->Config());
    ExternalEngineAdmissionPlan result;
    result.setupId = impl_->Manifest().setupId;
    result.roundId = *impl_->currentRoundId;
    result.cohortSizes.assign(dimensions.chunkCount, 0U);
    for (const auto& selection : impl_->selections) {
        for (const auto chunk : selection->selectedActualChunkIds) {
            ++result.cohortSizes[chunk.value];
        }
    }
    for (std::size_t chunk = 0; chunk < dimensions.chunkCount; ++chunk) {
        if (result.cohortSizes[chunk] >=
            impl_->Config().sparsification.minimumCohort) {
            result.acceptedActualChunkIds.push_back(ActualChunkId{chunk});
        }
    }

    const std::set<ActualChunkId> accepted(
        result.acceptedActualChunkIds.begin(),
        result.acceptedActualChunkIds.end());
    for (std::size_t position = 0;
         position < impl_->registrations.size();
         ++position) {
        ClientAdmissionResult client;
        client.setupId = result.setupId;
        client.roundId = result.roundId;
        client.clientId = impl_->registrations[position].externalId;
        for (const auto chunk :
             impl_->selections[position]->selectedActualChunkIds) {
            if (accepted.count(chunk) != 0) {
                client.acceptedActualChunkIds.push_back(chunk);
            }
        }
        result.clientResults.push_back(std::move(client));
    }

    std::vector<ActualChunkId> serverAccepted;
    const auto& observer = round.clientPrivateStates.front();
    for (const auto label : round.serverPlan.acceptedLabels) {
        serverAccepted.push_back(ActualChunkForLabel(observer, label));
    }
    std::sort(serverAccepted.begin(), serverAccepted.end());
    if (serverAccepted != result.acceptedActualChunkIds) {
        throw std::runtime_error(
            "actual-ID admission disagrees with shuffled-label server plan");
    }

    impl_->sparseRound = std::move(round);
    impl_->admission = result;
    impl_->Transition(RoundPhase::ADMISSION_FINALIZED);
    return result;
}

void ExternalInputCryptoEngine::SubmitAcceptedValues(
    const AcceptedChunkValues& input) {

    if (impl_->Config().mode != ProtocolMode::SP) {
        throw std::logic_error("FULL rounds require a dense parameter delta");
    }
    impl_->EnterValuesPhase();
    ValidateAcceptedChunkValues(input);
    const auto& manifest = impl_->Manifest();
    if (!(input.setupId == manifest.setupId) ||
        !impl_->currentRoundId.has_value() ||
        !(input.roundId == *impl_->currentRoundId) ||
        input.chunkSize != manifest.chunkSize) {
        throw std::invalid_argument("accepted external value metadata mismatch");
    }
    const std::size_t position = impl_->PositionForExternalClient(
        input.clientId);
    if (impl_->acceptedValues[position].has_value()) {
        throw std::invalid_argument("duplicate accepted external values");
    }
    const auto& expected = impl_->admission->ForClient(
        input.clientId).acceptedActualChunkIds;
    if (input.actualChunkIds != expected) {
        throw std::invalid_argument(
            "external value IDs do not match cohort admission");
    }

    const auto& config = impl_->Config();
    for (std::size_t row = 0; row < input.actualChunkIds.size(); ++row) {
        const std::size_t begin = static_cast<std::size_t>(
            input.actualChunkIds[row].value * input.chunkSize);
        const std::size_t validCount = std::min<std::size_t>(
            input.chunkSize,
            config.modelDimension - begin);
        for (std::size_t offset = 0; offset < input.chunkSize; ++offset) {
            const float value = input.values[row * input.chunkSize + offset];
            if (offset >= validCount && value != 0.0F) {
                throw std::invalid_argument(
                    "external final sparse chunk has nonzero padding");
            }
        }
    }
    impl_->acceptedValues[position] = input;
}

void ExternalInputCryptoEngine::SubmitFullParameterDelta(
    const ExternalFullParameterDeltaPayload& input) {

    if (impl_->Config().mode != ProtocolMode::FULL) {
        throw std::logic_error("SP rounds do not accept FULL parameter deltas");
    }
    impl_->RequirePhase(RoundPhase::VALUES_COLLECTING);
    const auto& manifest = impl_->Manifest();
    if (!(input.setupId == manifest.setupId) ||
        !impl_->currentRoundId.has_value() ||
        !(input.roundId == *impl_->currentRoundId) ||
        input.parameterDelta.size() != manifest.parameterNumel) {
        throw std::invalid_argument("external FULL parameter metadata mismatch");
    }
    if (std::any_of(
            input.parameterDelta.begin(),
            input.parameterDelta.end(),
            [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("FULL parameter delta must be finite");
    }
    const std::size_t position = impl_->PositionForExternalClient(
        input.clientId);
    if (impl_->fullValues[position].has_value()) {
        throw std::invalid_argument("duplicate external FULL parameter delta");
    }
    impl_->fullValues[position] = input;
}

void ExternalInputCryptoEngine::SubmitAuxiliaryDeltas(
    const AuxiliaryDeltaPayload& input) {

    impl_->EnterValuesPhase();
    ValidateAuxiliaryDeltaPayload(input);
    const auto& manifest = impl_->Manifest();
    if (!(input.setupId == manifest.setupId) ||
        !impl_->currentRoundId.has_value() ||
        !(input.roundId == *impl_->currentRoundId) ||
        input.runningMeanDelta.size() != manifest.runningMeanNumel ||
        input.runningVarianceDelta.size() != manifest.runningVarianceNumel) {
        throw std::invalid_argument("external auxiliary metadata mismatch");
    }
    const std::size_t position = impl_->PositionForExternalClient(
        input.clientId);
    if (impl_->auxiliaryDeltas[position].has_value()) {
        throw std::invalid_argument("duplicate external auxiliary deltas");
    }
    impl_->auxiliaryDeltas[position] = input;
}

ExternalEngineSurvivorTransition
ExternalInputCryptoEngine::SetRoundSurvivors(
    const std::vector<ClientId>& survivorClientIds) {

    const auto& config = impl_->Config();
    if (!config.experimentalDynamicCohort) {
        if (impl_->phase.has_value() && !IsTerminal(*impl_->phase)) AbortRound();
        throw std::logic_error("fixed-cohort execution forbids survivor reconfiguration");
    }
    const bool v17 = federated::UsesV17WireProtocol(config.protocolVersion);
    if (!v17 && config.mode != ProtocolMode::SP) {
        throw std::logic_error(
            "post-upload survivor transition currently requires SP mode");
    }
    impl_->RequirePhase(RoundPhase::VALUES_COLLECTING);
    if (!v17 && impl_->survivorTransition.has_value()) {
        throw std::logic_error("round survivors have already been fixed");
    }
    if (!impl_->currentRoundId.has_value() || survivorClientIds.empty()) {
        throw std::invalid_argument("round survivor set is empty");
    }
    const bool missingModelUploads = config.mode == ProtocolMode::SP
        ? std::any_of(
              impl_->acceptedValues.begin(),
              impl_->acceptedValues.end(),
              [](const auto& values) { return !values.has_value(); })
        : std::any_of(
              impl_->fullValues.begin(),
              impl_->fullValues.end(),
              [](const auto& values) { return !values.has_value(); });
    if (missingModelUploads ||
        std::any_of(
            impl_->auxiliaryDeltas.begin(),
            impl_->auxiliaryDeltas.end(),
            [](const auto& values) { return !values.has_value(); })) {
        throw std::logic_error(
            "survivors can be fixed only after every model/aux upload");
    }

    std::set<std::uint64_t> externalSeen;
    SurvivorSet internalSurvivors;
    internalSurvivors.clients.reserve(survivorClientIds.size());
    for (const ClientId externalId : survivorClientIds) {
        if (!externalSeen.insert(externalId.value).second) {
            throw std::invalid_argument("round survivor set contains duplicates");
        }
        const std::size_t position =
            impl_->PositionForExternalClient(externalId);
        internalSurvivors.clients.push_back(
            impl_->registrations[position].internalId);
    }
    std::sort(
        internalSurvivors.clients.begin(),
        internalSurvivors.clients.end());
    if (v17 && impl_->survivorTransition.has_value()) {
        const auto& previous =
            impl_->ActiveSurvivorKeys().publicKey.survivors.clients;
        if (internalSurvivors.clients.size() >= previous.size() ||
            !std::includes(
                previous.begin(), previous.end(),
                internalSurvivors.clients.begin(),
                internalSurvivors.clients.end())) {
            throw std::invalid_argument(
                "v17 pre-decryption survivors may only decrease");
        }
    }
    if (internalSurvivors.clients.size() <
            config.sparsification.minimumCohort ||
        internalSurvivors.clients.size() <=
            config.sparsification.collusionBound + 1) {
        throw std::invalid_argument(
            "round survivor set violates cohort/collusion requirements");
    }

    std::optional<ProtocolSparseRoundArtifacts> survivorRound;
    ExternalEngineAdmissionPlan survivorAdmission;
    survivorAdmission.setupId = impl_->Manifest().setupId;
    survivorAdmission.roundId = *impl_->currentRoundId;
    if (config.mode == ProtocolMode::SP) {
        std::vector<ClientSelectionRequest> internalSelections;
        internalSelections.reserve(internalSurvivors.clients.size());
        for (const ClientId internalId : internalSurvivors.clients) {
            const std::size_t position = internalId.value;
            if (position >= impl_->selections.size() ||
                !impl_->selections[position].has_value()) {
                throw std::logic_error("round survivor selection is unavailable");
            }
            ClientSelectionRequest translated = *impl_->selections[position];
            translated.clientId = internalId;
            internalSelections.push_back(std::move(translated));
        }

        survivorRound = BuildProtocolSparseRoundArtifactsFromSelections(
            config,
            *impl_->clientOnlySetup,
            *impl_->currentRoundId,
            internalSurvivors,
            internalSelections);
        const DerivedDimensions dimensions = DeriveDimensions(config);
        survivorAdmission.cohortSizes.assign(dimensions.chunkCount, 0U);
    for (const auto& selection : internalSelections) {
        for (const auto chunk : selection.selectedActualChunkIds) {
            ++survivorAdmission.cohortSizes[chunk.value];
        }
    }
    for (std::size_t chunk = 0; chunk < dimensions.chunkCount; ++chunk) {
        if (survivorAdmission.cohortSizes[chunk] >=
            config.sparsification.minimumCohort) {
            survivorAdmission.acceptedActualChunkIds.push_back(
                ActualChunkId{chunk});
        }
    }
    const std::set<ActualChunkId> accepted(
        survivorAdmission.acceptedActualChunkIds.begin(),
        survivorAdmission.acceptedActualChunkIds.end());
    for (std::size_t position = 0;
         position < impl_->registrations.size();
         ++position) {
        ClientAdmissionResult client;
        client.setupId = survivorAdmission.setupId;
        client.roundId = survivorAdmission.roundId;
        client.clientId = impl_->registrations[position].externalId;
        if (std::binary_search(
                internalSurvivors.clients.begin(),
                internalSurvivors.clients.end(),
                impl_->registrations[position].internalId)) {
            for (const auto chunk :
                 impl_->selections[position]->selectedActualChunkIds) {
                if (accepted.count(chunk) != 0) {
                    client.acceptedActualChunkIds.push_back(chunk);
                }
            }
        }
        survivorAdmission.clientResults.push_back(std::move(client));
    }
    std::vector<ActualChunkId> serverAccepted;
    const auto& observer = survivorRound->clientPrivateStates.front();
    for (const auto label : survivorRound->serverPlan.acceptedLabels) {
        serverAccepted.push_back(ActualChunkForLabel(observer, label));
    }
    std::sort(serverAccepted.begin(), serverAccepted.end());
    if (serverAccepted != survivorAdmission.acceptedActualChunkIds) {
        throw std::runtime_error(
            "survivor admission disagrees with shuffled-label server plan");
    }
    }

    /* Local ciphertexts are materialized once, before changing the target. */
    impl_->PrepareUploads();

    std::shared_ptr<SurvivorKeySet> roundKeys;
    const SurvivorKeySet* keyView = nullptr;
    bool reusedTargetKeys = false;
    if (v17 && internalSurvivors.clients ==
            impl_->survivorKeys.publicKey.survivors.clients) {
        keyView = &impl_->survivorKeys;
        reusedTargetKeys = true;
    }
    else if (v17) {
        std::vector<std::size_t> cacheKey;
        cacheKey.reserve(internalSurvivors.clients.size());
        for (const auto clientId : internalSurvivors.clients) {
            cacheKey.push_back(clientId.value);
        }
        const auto found = impl_->v17TargetKeyCache.find(cacheKey);
        if (found != impl_->v17TargetKeyCache.end()) {
            roundKeys = found->second;
            reusedTargetKeys = true;
        }
        else {
            if (impl_->startedDropoutTargets.count(cacheKey) == 0) {
                if (impl_->startedDropoutTargets.size() >= config.dropoutTargetCap) {
                    AbortRound();
                    throw std::logic_error("v17 dropout target cap exhausted");
                }
                // Count before construction, including failed key generation.
                impl_->startedDropoutTargets.insert(cacheKey);
            }
            roundKeys = std::make_shared<SurvivorKeySet>(
                AggregateSurvivorKeys(impl_->setup, internalSurvivors));
            impl_->v17TargetKeyCache.emplace(
                std::move(cacheKey), roundKeys);
        }
        keyView = roundKeys.get();
    }
    else {
        roundKeys = std::make_shared<SurvivorKeySet>(
            AggregateSurvivorKeys(impl_->setup, internalSurvivors));
        keyView = roundKeys.get();
    }
    ExternalEngineSurvivorTransition transition;
    transition.setupId = impl_->Manifest().setupId;
    transition.roundId = *impl_->currentRoundId;
    transition.admission = survivorAdmission;
    transition.targetKeyCacheHit = reusedTargetKeys;
    transition.serverTargetReconstructionSeconds = reusedTargetKeys
        ? 0.0 : keyView->serverTargetReconstructionSeconds;
    transition.clientCollectiveKeyGenerationSeconds.assign(
        impl_->registrations.size(),
        0.0);
    for (std::size_t index = 0;
         index < internalSurvivors.clients.size();
         ++index) {
        const std::size_t position = internalSurvivors.clients[index].value;
        transition.clientCollectiveKeyGenerationSeconds[position] =
            reusedTargetKeys
                ? 0.0
                : keyView->clientToCollectiveKeyGenerationSeconds[index];
    }
    transition.maximumClientCollectiveKeyGenerationSeconds =
        *std::max_element(
            transition.clientCollectiveKeyGenerationSeconds.begin(),
            transition.clientCollectiveKeyGenerationSeconds.end());
    for (const auto& registration : impl_->registrations) {
        if (std::binary_search(
                internalSurvivors.clients.begin(),
                internalSurvivors.clients.end(),
                registration.internalId)) {
            transition.survivorClientIds.push_back(registration.externalId);
        }
        else {
            transition.droppedClientIds.push_back(registration.externalId);
        }
    }

    impl_->roundSurvivorKeys = std::move(roundKeys);
    impl_->survivorSparseRound = std::move(survivorRound);
    impl_->survivorAdmission = std::move(survivorAdmission);
    impl_->survivorTransition = transition;
    impl_->survivorTransitions.push_back(transition);
    // A shrink invalidates every acknowledgement of the old revision.
    impl_->finalContext.clear();
    impl_->finalAcknowledgements.clear();
    return transition;
}

std::vector<std::uint64_t> ExternalInputCryptoEngine::PrepareFinalization() {
    impl_->EnterValuesPhase();
    const auto missing = [](const auto& values) {
        return std::any_of(values.begin(), values.end(),
            [](const auto& value) { return !value.has_value(); });
    };
    if (missing(impl_->auxiliaryDeltas) ||
        (impl_->Config().mode == ProtocolMode::SP
            ? missing(impl_->acceptedValues) : missing(impl_->fullValues)))
        throw std::logic_error("finalization notice requires complete model/aux uploads");
    if (!impl_->finalContext.empty()) return impl_->finalContext;
    auto& context = impl_->finalContext;
    context = {17, 1}; // encoding version, FinalizeNotice phase
    for (const auto byte : impl_->Manifest().setupId.bytes) context.push_back(byte);
    context.push_back(impl_->currentLogicalRoundIndex.value_or(impl_->currentRoundId->value));
    context.push_back(impl_->currentRoundId->value);
    context.push_back(impl_->survivorTransitions.size());
    const auto& survivors = impl_->ActiveSurvivorKeys().publicKey.survivors.clients;
    context.push_back(survivors.size());
    for (const auto id : survivors)
        context.push_back(impl_->registrations.at(id.value).externalId.value);
    context.push_back(impl_->Config().mode == ProtocolMode::SP ? 1 : 0);
    if (impl_->Config().mode == ProtocolMode::SP) {
        const auto& plan = impl_->ActiveSparseRound().serverPlan;
        context.push_back(plan.acceptedSupportBitmap.size());
        for (const auto byte : plan.acceptedSupportBitmap) context.push_back(byte);
        context.push_back(plan.packingGroups.size());
        for (const auto& group : plan.packingGroups) {
            context.push_back(group.labels.size());
            for (const auto label : group.labels) context.push_back(label.value);
        }
    }
    return context;
}

void ExternalInputCryptoEngine::AcknowledgeFinalization(
    ClientId clientId, const std::vector<std::uint64_t>& context) {
    impl_->RequirePhase(RoundPhase::VALUES_COLLECTING);
    if (impl_->finalContext.empty() || context != impl_->finalContext)
        throw std::invalid_argument("stale or mismatched finalization context");
    const auto internal = impl_->registrations.at(
        impl_->PositionForExternalClient(clientId)).internalId;
    const auto& survivors = impl_->ActiveSurvivorKeys().publicKey.survivors.clients;
    if (std::find(survivors.begin(), survivors.end(), internal) == survivors.end())
        throw std::invalid_argument("finalization acknowledgement is not from a survivor");
    impl_->finalAcknowledgements.insert(clientId.value);
}

ExternalEngineRoundResult ExternalInputCryptoEngine::FinalizeRound(
    const std::vector<ClientId>& postRequestDropouts) {
    const auto finalizeStarted = std::chrono::steady_clock::now();
    // Only uploads prepared before this call lie outside finalize's wall time.
    // PrepareUploads below is already covered by finalizeStarted.
    const double previouslyPreparedUploadSeconds =
        impl_->postUploadCachedEncryptionSequentialSeconds;
    try {
    impl_->EnterValuesPhase();
    if (impl_->Config().experimentalDynamicCohort &&
        (impl_->finalContext.empty() || impl_->finalAcknowledgements.size() !=
         impl_->ActiveSurvivorKeys().publicKey.survivors.clients.size())) {
        throw std::logic_error("all final survivors must acknowledge the finalization context");
    }
    const bool missingModelValues = impl_->Config().mode == ProtocolMode::SP
        ? std::any_of(
              impl_->acceptedValues.begin(),
              impl_->acceptedValues.end(),
              [](const auto& values) { return !values.has_value(); })
        : std::any_of(
              impl_->fullValues.begin(),
              impl_->fullValues.end(),
              [](const auto& values) { return !values.has_value(); });
    if (missingModelValues ||
        std::any_of(
            impl_->auxiliaryDeltas.begin(),
            impl_->auxiliaryDeltas.end(),
            [](const auto& values) { return !values.has_value(); })) {
        if (!impl_->Config().experimentalDynamicCohort) AbortRound();
        throw std::logic_error(
            "model values and auxiliary deltas are required from every client");
    }
    if (!postRequestDropouts.empty()) {
        if (!federated::UsesV17WireProtocol(impl_->Config().protocolVersion) ||
            !impl_->request->execution.useSparseCompression ||
            !impl_->request->execution.useAuxiliaryCompression) {
            throw std::logic_error(
                "post-request dropout requires compressed v17 model and auxiliary paths");
        }
        std::set<std::uint64_t> seen;
        const auto& finalSurvivors =
            impl_->ActiveSurvivorKeys().publicKey.survivors.clients;
        for (const ClientId externalId : postRequestDropouts) {
            if (!seen.insert(externalId.value).second) {
                throw std::invalid_argument(
                    "duplicate post-request dropout client");
            }
            const auto internalId = impl_->registrations.at(
                impl_->PositionForExternalClient(externalId)).internalId;
            if (!std::binary_search(
                    finalSurvivors.begin(), finalSurvivors.end(),
                    internalId)) {
                throw std::invalid_argument(
                    "post-request dropout is not a final survivor");
            }
        }
    }

    if (federated::UsesV17WireProtocol(impl_->Config().protocolVersion)) {
        impl_->PrepareUploads();
    }
    impl_->Transition(RoundPhase::ENCRYPTED);
    const SurvivorKeySet& activeSurvivorKeys =
        impl_->ActiveSurvivorKeys();
    AuxiliaryCompressionProfile auxiliaryProfile =
        impl_->request->execution.useAuxiliaryCompression
            ? MakeConfiguredAuxiliaryProfile(impl_->Config())
            : MakeUncompressedAuxiliaryProfile();
    auxiliaryProfile.measurePreSmudgingNoise =
        impl_->request->execution.measurePreSmudgingNoise;
    auxiliaryProfile.computeReference = auxiliaryProfile.measurePreSmudgingNoise;

    const bool v17Combined = federated::UsesV17WireProtocol(impl_->Config().protocolVersion) &&
        auxiliaryProfile.compressPartialDecryption &&
        impl_->request->execution.useSparseCompression;
    auto v17SeedGate = v17Combined
        ? std::make_shared<V17DecryptionSeedGate>(
              2, !postRequestDropouts.empty())
        : std::shared_ptr<V17DecryptionSeedGate>{};
    auxiliaryProfile.v17SeedGate = v17SeedGate;

    const auto auxiliaryInputs = impl_->BuildAuxiliaryInputs();
    const auto runAuxiliary = [&]() -> AuxiliaryPathResult {
        return impl_->auxiliaryUploads.has_value()
            ? FinalizeProtocolAuxiliaryPathAfterDropout(
              impl_->cc,
              impl_->Config(),
              *impl_->currentRoundId,
              auxiliaryInputs,
              *impl_->auxiliaryUploads,
              impl_->setup,
              activeSurvivorKeys,
              auxiliaryProfile,
              impl_->cryptoExecutor.get(),
              &impl_->cryptoStageMetrics)
            : RunProtocolAuxiliaryPath(
              impl_->cc,
              impl_->Config(),
              *impl_->currentRoundId,
              auxiliaryInputs,
              impl_->setup,
              activeSurvivorKeys,
              auxiliaryProfile,
              impl_->cryptoExecutor.get(),
              &impl_->cryptoStageMetrics);
    };
    std::future<AuxiliaryPathResult> auxiliaryFuture;
    std::optional<AuxiliaryPathResult> auxiliaryValue;
    if (v17Combined) {
        auxiliaryFuture = std::async(std::launch::async, [&]() {
            try {
                return runAuxiliary();
            }
            catch (...) {
                v17SeedGate->Abort(std::current_exception());
                throw;
            }
        });
    }
    else {
        auxiliaryValue = runAuxiliary();
    }

    ExternalEngineRoundResult result;
    result.setupId = impl_->Manifest().setupId;
    result.roundId = *impl_->currentRoundId;
    std::vector<double> parameterAggregate;
    std::vector<double> modelEncryptionSeconds;
    std::vector<double> modelKeySwitchSeconds;
    std::vector<double> modelPartialDecryptionSeconds;
    try {
    if (impl_->Config().mode == ProtocolMode::SP) {
        SparsePathCompressionProfile sparseProfile =
            impl_->request->execution.useSparseCompression
                ? MakeConfiguredSparseProfile(impl_->Config())
                : MakeUncompressedSparseProfile();
        sparseProfile.measurePreSmudgingNoise =
            impl_->request->execution.measurePreSmudgingNoise;
        sparseProfile.computeReference = sparseProfile.measurePreSmudgingNoise;
        sparseProfile.v17SeedGate = v17SeedGate;
        const auto sparseInputs = impl_->BuildSparseInputs();
        const SparsePathResult sparse = impl_->sparseUploads.has_value()
            ? FinalizeProtocolSparsePathAfterDropout(
                  impl_->cc,
                  impl_->Config(),
                  sparseInputs,
                  *impl_->sparseUploads,
                  impl_->ActiveSparseRound(),
                  impl_->setup,
                  activeSurvivorKeys,
                  sparseProfile,
                  impl_->cryptoExecutor.get(),
                  &impl_->cryptoStageMetrics)
            : RunProtocolSparsePathAfterDropout(
                  impl_->cc,
                  impl_->Config(),
                  sparseInputs,
                  *impl_->sparseRound,
                  impl_->ActiveSparseRound(),
                  impl_->setup,
                  activeSurvivorKeys,
                  sparseProfile,
                  impl_->cryptoExecutor.get(),
                  &impl_->cryptoStageMetrics);
        result.acceptedActualChunkIds =
            impl_->ActiveAdmission().acceptedActualChunkIds;
        result.cohortSizes = impl_->ActiveAdmission().cohortSizes;
        result.modelPhaseAudit = sparse.observer.phaseAudit;
        parameterAggregate = sparse.observer.aggregate;
        result.sparsePlanHash = sparse.server.planHash;
        result.sparseClientCiphertextCount =
            sparse.server.clientRLWECiphertextCount;
        result.sparseSurvivorClientCiphertextCount =
            sparse.server.survivorClientRLWECiphertextCount;
        result.sparseDiscardedClientCiphertextCount =
            sparse.server.discardedClientRLWECiphertextCount;
        result.acceptedSparseContributionCount =
            sparse.server.acceptedContributionCount;
        result.aggregatedMLWECiphertextCount =
            sparse.server.aggregatedMLWECiphertextCount;
        result.sparseServerCiphertextCount =
            sparse.server.serverRLWECiphertextCount;
        const bool usesRingConversion = RequiresSparseRingConversion(
            impl_->Config());
        result.r2mCiphertextCount = usesRingConversion
            ? sparse.server.aggregatedMLWECiphertextCount
            : 0;
        result.m2rCiphertextCount = usesRingConversion
            ? sparse.server.serverRLWECiphertextCount
            : 0;
        modelEncryptionSeconds = sparse.clientEncryptionSeconds;
        modelKeySwitchSeconds = sparse.clientKeySwitchSeconds;
        modelPartialDecryptionSeconds = sparse.clientPartialDecryptionSeconds;
        result.sparseModelClientStageTimings = sparse.clientStageTimings;
        result.sparseModelServerStageTimings = sparse.serverStageTimings;
    }
    else {
        FullModelCompressionProfile fullProfile =
            impl_->request->execution.useSparseCompression
                ? MakeConfiguredFullModelProfile(impl_->Config())
                : MakeUncompressedFullModelProfile();
        fullProfile.measurePreSmudgingNoise =
            impl_->request->execution.measurePreSmudgingNoise;
        fullProfile.computeReference = fullProfile.measurePreSmudgingNoise;
        fullProfile.v17SeedGate = v17SeedGate;
        const auto fullInputs = impl_->BuildFullInputs();
        const FullModelPathResult full = impl_->fullUploads.has_value()
            ? FinalizeProtocolFullModelPathAfterDropout(
                  impl_->cc, impl_->Config(), *impl_->currentRoundId,
                  fullInputs, *impl_->fullUploads, impl_->setup,
                  activeSurvivorKeys, fullProfile,
                  impl_->cryptoExecutor.get(),
                  &impl_->cryptoStageMetrics)
            : RunProtocolFullModelPath(
                  impl_->cc, impl_->Config(), *impl_->currentRoundId,
                  fullInputs, impl_->setup, activeSurvivorKeys,
                  fullProfile, impl_->cryptoExecutor.get(),
                  &impl_->cryptoStageMetrics);
        result.modelPhaseAudit = full.phaseAudit;
        parameterAggregate = full.aggregate;
        result.fullModelClientCiphertextCount = full.clientCiphertextCount;
        result.fullModelAggregateCiphertextCount = full.aggregateCiphertextCount;
        result.r2mCiphertextCount = full.r2mCiphertextCount;
        result.m2rCiphertextCount = full.m2rCiphertextCount;
        modelEncryptionSeconds = full.clientEncryptionSeconds;
        modelKeySwitchSeconds = full.clientKeySwitchSeconds;
        modelPartialDecryptionSeconds = full.clientPartialDecryptionSeconds;
        result.fullModelClientStageTimings = full.clientStageTimings;
        result.fullModelServerStageTimings = full.serverStageTimings;
    }
    }
    catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (v17SeedGate) {
            v17SeedGate->Abort(failure);
        }
        if (auxiliaryFuture.valid()) {
            auxiliaryFuture.wait();
        }
        if (!postRequestDropouts.empty()) {
            try {
                std::rethrow_exception(failure);
            }
            catch (const V17PostRequestDropout&) {
                impl_->lastAbortedRoundTraffic =
                    impl_->BuildAbortedRoundTraffic();
            }
            catch (...) {
                // Ordinary crypto failures are not a measured late dropout.
            }
            AbortRound();
        }
        std::rethrow_exception(failure);
    }
    if (auxiliaryFuture.valid()) {
        try {
            auxiliaryValue = auxiliaryFuture.get();
        }
        catch (...) {
            if (!postRequestDropouts.empty()) {
                AbortRound();
            }
            throw;
        }
    }
    const AuxiliaryPathResult& auxiliary = *auxiliaryValue;
    result.auxPhaseAudit = auxiliary.phaseAudit;
    ValidateDecryptedAggregate(
        auxiliary.runningMeanDelta,
        impl_->Config().auxUpdateBound,
        "decrypted running-mean aggregate");
    ValidateDecryptedAggregate(
        auxiliary.runningVarianceDelta,
        impl_->Config().auxUpdateBound,
        "decrypted running-variance aggregate");
    ValidateDecryptedAggregate(
        parameterAggregate,
        impl_->Config().updateBound,
        "decrypted parameter aggregate");
    result.parameterDelta = ToFloat32(parameterAggregate, "parameter delta");
    result.runningMeanDelta = ToFloat32(
        auxiliary.runningMeanDelta,
        "running mean delta");
    result.runningVarianceDelta = ToFloat32(
        auxiliary.runningVarianceDelta,
        "running variance delta");
    result.auxiliaryClientCiphertextCount =
        auxiliary.clientCiphertextCount;
    result.auxiliarySurvivorClientCiphertextCount =
        auxiliary.survivorClientCiphertextCount;
    result.auxiliaryDiscardedClientCiphertextCount =
        auxiliary.discardedClientCiphertextCount;
    result.auxiliaryAggregateCiphertextCount =
        auxiliary.aggregateCiphertextCount;
    result.auxiliaryClientStageTimings = auxiliary.clientStageTimings;
    result.auxiliaryServerStageTimings = auxiliary.serverStageTimings;
    const std::size_t clientCount = impl_->registrations.size();
    if (modelEncryptionSeconds.size() != clientCount ||
        modelKeySwitchSeconds.size() != clientCount ||
        modelPartialDecryptionSeconds.size() != clientCount ||
        auxiliary.clientEncryptionSeconds.size() != clientCount ||
        auxiliary.clientKeySwitchSeconds.size() != clientCount ||
        auxiliary.clientPartialDecryptionSeconds.size() != clientCount) {
        throw std::runtime_error("native timing client count mismatch");
    }
    result.clientEncryptionSeconds.resize(clientCount, 0.0);
    result.clientKeySwitchSeconds.resize(clientCount, 0.0);
    result.clientPartialDecryptionSeconds.resize(clientCount, 0.0);
    result.clientTotalCryptoSeconds.resize(clientCount, 0.0);
    std::vector<double> collectiveKeyGenerationSeconds(clientCount, 0.0);
    if (impl_->survivorTransition.has_value()) {
        const auto& transition = *impl_->survivorTransition;
        result.survivorReconfigurationApplied =
            !transition.droppedClientIds.empty();
        result.survivorTargetKeyCacheHit = transition.targetKeyCacheHit;
        result.survivorClientIds = transition.survivorClientIds;
        result.droppedClientIds = transition.droppedClientIds;
        result.survivorCollectiveKeyGenerationSeconds.assign(
            clientCount, 0.0);
        for (const auto& attempt : impl_->survivorTransitions) {
            result.survivorTargetReconstructionSeconds +=
                attempt.serverTargetReconstructionSeconds;
            if (!attempt.targetKeyCacheHit) {
                result.survivorNewTargetSizes.push_back(
                    attempt.survivorClientIds.size());
            }
            for (std::size_t client = 0; client < clientCount; ++client) {
                result.survivorCollectiveKeyGenerationSeconds[client] +=
                    attempt.clientCollectiveKeyGenerationSeconds[client];
            }
        }
        result.maximumSurvivorCollectiveKeyGenerationSeconds =
            *std::max_element(
                result.survivorCollectiveKeyGenerationSeconds.begin(),
                result.survivorCollectiveKeyGenerationSeconds.end());
        collectiveKeyGenerationSeconds =
            result.survivorCollectiveKeyGenerationSeconds;
        result.survivorReconfigurationSequentialSeconds =
            result.survivorTargetReconstructionSeconds +
            std::accumulate(
                collectiveKeyGenerationSeconds.begin(),
                collectiveKeyGenerationSeconds.end(), 0.0);
    }
    else {
        for (const auto& registration : impl_->registrations) {
            result.survivorClientIds.push_back(registration.externalId);
        }
        result.survivorCollectiveKeyGenerationSeconds.assign(
            clientCount,
            0.0);
    }
    if (impl_->Config().mode == ProtocolMode::SP &&
        impl_->Config().protocolVersion == "v17") {
        result.selectionBitmapCount = clientCount;
        result.admissionBitmapRecipientCount = clientCount;
        for (const auto& transition : impl_->survivorTransitions) {
            result.admissionBitmapRecipientCount +=
                transition.survivorClientIds.size();
        }
        result.uploadSupportBitmapCount =
            impl_->sparseUploads->supportHeaders.size();
        result.resultRecipientCount = clientCount;
    }
    for (std::size_t client = 0; client < clientCount; ++client) {
        result.clientEncryptionSeconds[client] =
            modelEncryptionSeconds[client] +
            auxiliary.clientEncryptionSeconds[client];
        result.clientKeySwitchSeconds[client] =
            modelKeySwitchSeconds[client] +
            auxiliary.clientKeySwitchSeconds[client];
        result.clientPartialDecryptionSeconds[client] =
            modelPartialDecryptionSeconds[client] +
            auxiliary.clientPartialDecryptionSeconds[client];
        result.clientTotalCryptoSeconds[client] =
            result.clientEncryptionSeconds[client] +
            result.clientKeySwitchSeconds[client] +
            result.clientPartialDecryptionSeconds[client] +
            collectiveKeyGenerationSeconds[client];
    }
    result.cryptoSequentialSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - finalizeStarted).count() +
        result.survivorReconfigurationSequentialSeconds +
        previouslyPreparedUploadSeconds;
    double actualClientPhaseSeconds = std::accumulate(
        result.clientTotalCryptoSeconds.begin(),
        result.clientTotalCryptoSeconds.end(),
        0.0);
    // Client vectors contain batch service only. Executor waits overlap another
    // batch's service and must not be subtracted again or added to server time.
    result.cryptoServerSeconds = std::max(
        0.0,
        result.cryptoSequentialSeconds - actualClientPhaseSeconds);
    std::vector<double> clientPreAggregationSeconds(clientCount, 0.0);
    for (std::size_t client = 0; client < clientCount; ++client) {
        clientPreAggregationSeconds[client] =
            result.clientEncryptionSeconds[client] +
            collectiveKeyGenerationSeconds[client] +
            result.clientKeySwitchSeconds[client];
    }
    result.maximumClientPreAggregationSeconds = *std::max_element(
        clientPreAggregationSeconds.begin(),
        clientPreAggregationSeconds.end());
    result.maximumClientPartialDecryptionSeconds = *std::max_element(
        result.clientPartialDecryptionSeconds.begin(),
        result.clientPartialDecryptionSeconds.end());
    result.maximumClientTotalCryptoSeconds = *std::max_element(
        result.clientTotalCryptoSeconds.begin(),
        result.clientTotalCryptoSeconds.end());
    result.estimatedParallelCryptoSeconds =
        result.cryptoServerSeconds +
        result.maximumClientPreAggregationSeconds +
        result.maximumClientPartialDecryptionSeconds;
    if (!impl_->cryptoExecutor) {
        throw std::logic_error("external crypto executor is not configured");
    }
    result.cryptoThreadCount = impl_->cryptoExecutor->ThreadCount();
    result.cryptoStageTimings = impl_->cryptoStageMetrics.Snapshot();
    impl_->Transition(RoundPhase::DECRYPTING);
    return result;
    }
    catch (...) {
        // Every failed attempt closes its logical round, including preflight,
        // crypto, decoding and result-validation failures.
        if (impl_->phase.has_value() && !IsTerminal(*impl_->phase)) {
            AbortRound();
        }
        throw;
    }
}

void ExternalInputCryptoEngine::CommitRound() {
    impl_->RequirePhase(RoundPhase::DECRYPTING);
    impl_->Transition(RoundPhase::COMMITTED);
    impl_->lastRoundId = impl_->currentRoundId;
    if (federated::UsesV17WireProtocol(impl_->Config().protocolVersion)) {
        impl_->lastCommittedLogicalRoundIndex =
            impl_->currentLogicalRoundIndex;
        impl_->lastAbortedLogicalRoundIndex.reset();
    }
}

void ExternalInputCryptoEngine::AbortRound() {
    if (!impl_->phase.has_value() || IsTerminal(*impl_->phase)) {
        throw std::logic_error("no abortable external crypto round is active");
    }
    impl_->Transition(RoundPhase::ABORTED);
    impl_->lastRoundId = impl_->currentRoundId;
    if (federated::UsesV17WireProtocol(impl_->Config().protocolVersion)) {
        impl_->lastAbortedLogicalRoundIndex =
            impl_->currentLogicalRoundIndex;
    }
}

}  // namespace federated
