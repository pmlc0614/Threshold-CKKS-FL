#include "protocol-sparse-round.h"

#include "deterministic-randomness.h"
#include "support-bitmap.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace federated {
namespace {

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
        throw std::invalid_argument("survivor client input is missing");
    }
    return *found;
}

const ClientSelectionRequest& FindSelection(
    const std::vector<ClientSelectionRequest>& selections,
    ClientId clientId) {

    const auto found = std::find_if(
        selections.begin(),
        selections.end(),
        [clientId](const ClientSelectionRequest& selection) {
            return selection.clientId == clientId;
        });
    if (found == selections.end()) {
        throw std::invalid_argument("survivor client selection is missing");
    }
    return *found;
}

void ValidateClientInput(
    const ProtocolExperimentConfig& config,
    const ProtocolClientRoundInput& client) {

    if (config.mode != ProtocolMode::SP ||
        client.clientId.value >= config.clientCount ||
        client.update.size() != config.modelDimension ||
        client.integerWeight != config.integerWeights[client.clientId.value]) {
        throw std::invalid_argument("invalid sparse client input");
    }
}

std::vector<ActualChunkId> SelectTopChunks(
    const ProtocolClientRoundInput& client,
    std::size_t chunkSize,
    std::size_t chunkCount,
    std::size_t selectedCount) {

    std::vector<std::pair<long double, std::size_t>> scored;
    scored.reserve(chunkCount);
    for (std::size_t chunk = 0; chunk < chunkCount; ++chunk) {
        const std::size_t begin = chunk * chunkSize;
        const std::size_t end = std::min(begin + chunkSize, client.update.size());
        long double score = 0.0L;
        for (std::size_t index = begin; index < end; ++index) {
            if (!std::isfinite(client.update[index])) {
                throw std::invalid_argument("client update contains non-finite data");
            }
            const long double value = client.update[index];
            score += value * value;
        }
        scored.emplace_back(score, chunk);
    }
    std::sort(
        scored.begin(),
        scored.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return left.second < right.second;
        });

    std::vector<ActualChunkId> result;
    result.reserve(selectedCount);
    for (std::size_t index = 0; index < selectedCount; ++index) {
        result.push_back(ActualChunkId{scored[index].second});
    }
    return result;
}

ClientSparsePrivateRoundState BuildPrivateStateForSelectedChunks(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    SetupId setupId,
    RoundId roundId,
    ClientId clientId,
    const std::vector<ActualChunkId>& selectedChunks) {

    const DerivedDimensions dimensions = DeriveDimensions(config);
    if (!(setupId == clientOnlySetup.setupId) ||
        clientId.value >= config.clientCount ||
        selectedChunks.size() != dimensions.selectedChunksPerClient) {
        throw std::invalid_argument("invalid explicit sparse selection metadata");
    }
    std::set<std::uint64_t> seen;
    for (const auto chunk : selectedChunks) {
        if (chunk.value >= dimensions.chunkCount ||
            !seen.insert(chunk.value).second) {
            throw std::invalid_argument(
                "explicit sparse selection contains an invalid chunk ID");
        }
    }

    ClientSparsePrivateRoundState state;
    state.setupId = setupId;
    state.roundId = roundId;
    state.clientId = clientId;
    state.chunkSize = dimensions.smallRingDimension;
    state.chunkCount = dimensions.chunkCount;
    state.mlweRank = config.sparsification.mlweRank;
    state.kidx = clientOnlySetup.clientGroupIndexKey;
    state.permutation = deterministic_randomness::GenerateIndexPermutation(
        state.kidx,
        state.setupId,
        state.roundId,
        state.chunkCount);
    state.inversePermutation =
        deterministic_randomness::InvertPermutation(state.permutation);
    state.actualSelectedChunks = selectedChunks;
    state.selectedMappings.reserve(state.actualSelectedChunks.size());
    for (const auto actual : state.actualSelectedChunks) {
        state.selectedMappings.push_back(ClientSparseChunkLabelMapping{
            actual,
            ShuffledChunkLabel{state.permutation[actual.value]}});
    }
    return state;
}

std::uint64_t PublicPlanHash(const ServerSparseAdmissionPlan& plan) {
    ByteVector serialized(plan.setupId.bytes.begin(), plan.setupId.bytes.end());
    const auto append = [&serialized](std::uint64_t value) {
        const ByteVector encoded =
            deterministic_randomness::EncodeBigEndian64(value);
        serialized.insert(serialized.end(), encoded.begin(), encoded.end());
    };
    append(plan.roundId.value);
    append(plan.chunkCount);
    append(plan.mlweRank);
    for (const auto& announcement : plan.announcements) {
        append(announcement.clientId.value);
        append(announcement.supportBitmap.size());
        for (const auto byte : announcement.supportBitmap) {
            serialized.push_back(byte);
        }
    }
    append(plan.acceptedSupportBitmap.size());
    serialized.insert(
        serialized.end(),
        plan.acceptedSupportBitmap.begin(),
        plan.acceptedSupportBitmap.end());
    for (const auto& cohort : plan.cohorts) {
        append(cohort.label.value);
        append(cohort.accepted ? 1U : 0U);
        append(cohort.clients.size());
        for (const auto client : cohort.clients) {
            append(client.value);
        }
    }
    const auto digest = deterministic_randomness::Sha256(serialized);
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        result = (result << 8U) | digest[index];
    }
    return result;
}

}  // namespace

ClientSparsePrivateRoundState BuildClientSparsePrivateRoundState(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    RoundId roundId,
    const ProtocolClientRoundInput& client) {

    ValidateProtocolParameters(config).ThrowIfInvalid();
    ValidateClientInput(config, client);
    const DerivedDimensions dimensions = DeriveDimensions(config);
    return BuildPrivateStateForSelectedChunks(
        config,
        clientOnlySetup,
        clientOnlySetup.setupId,
        roundId,
        client.clientId,
        SelectTopChunks(
            client,
            dimensions.smallRingDimension,
            dimensions.chunkCount,
            dimensions.selectedChunksPerClient));
}

ClientSparsePrivateRoundState BuildClientSparsePrivateRoundStateFromSelection(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    const ClientSelectionRequest& selection) {

    ValidateProtocolParameters(config).ThrowIfInvalid();
    ValidateClientSelectionRequest(selection);
    return BuildPrivateStateForSelectedChunks(
        config,
        clientOnlySetup,
        selection.setupId,
        selection.roundId,
        selection.clientId,
        selection.selectedActualChunkIds);
}

SparseIndexAnnouncement BuildSparseIndexAnnouncement(
    const ClientSparsePrivateRoundState& state) {

    SparseIndexAnnouncement result;
    result.clientId = state.clientId;
    std::vector<ShuffledChunkLabel> selectedLabels;
    selectedLabels.reserve(state.selectedMappings.size());
    for (const auto& mapping : state.selectedMappings) {
        selectedLabels.push_back(mapping.label);
    }
    std::sort(selectedLabels.begin(), selectedLabels.end());
    if (std::adjacent_find(
            selectedLabels.begin(),
            selectedLabels.end()) != selectedLabels.end()) {
        throw std::invalid_argument("client announcement contains duplicate labels");
    }
    result.supportBitmap = EncodeSupportBitmapMsb0(
        state.chunkCount,
        selectedLabels);
    return result;
}

ServerSparseAdmissionPlan BuildServerSparseAdmissionPlan(
    const ProtocolExperimentConfig& config,
    SetupId setupId,
    RoundId roundId,
    const SurvivorSet& survivors,
    const std::vector<SparseIndexAnnouncement>& announcements) {

    ValidateProtocolParameters(config).ThrowIfInvalid();
    if (config.mode != ProtocolMode::SP) {
        throw std::invalid_argument("server sparse admission requires SP mode");
    }
    const DerivedDimensions dimensions = DeriveDimensions(config);
    ServerSparseAdmissionPlan plan;
    plan.setupId = setupId;
    plan.roundId = roundId;
    plan.chunkSize = dimensions.smallRingDimension;
    plan.chunkCount = dimensions.chunkCount;
    plan.mlweRank = config.sparsification.mlweRank;
    plan.announcements = announcements;

    for (std::size_t value = 0; value < plan.chunkCount; ++value) {
        ShuffledChunkCohort cohort;
        cohort.label = ShuffledChunkLabel{value};
        for (const auto& announcement : plan.announcements) {
            if (SupportBitmapContains(
                    announcement.supportBitmap,
                    plan.chunkCount,
                    cohort.label)) {
                cohort.clients.push_back(announcement.clientId);
            }
        }
        cohort.accepted = cohort.clients.size() >=
            config.sparsification.minimumCohort;
        if (cohort.accepted) {
            plan.acceptedLabels.push_back(cohort.label);
        }
        plan.cohorts.push_back(std::move(cohort));
    }
    plan.acceptedSupportBitmap = EncodeSupportBitmapMsb0(
        plan.chunkCount,
        plan.acceptedLabels);

    for (std::size_t offset = 0;
         offset < plan.acceptedLabels.size();
         offset += plan.mlweRank) {
        PackedShuffledLabels group;
        const std::size_t end = std::min(
            offset + plan.mlweRank,
            plan.acceptedLabels.size());
        group.labels.insert(
            group.labels.end(),
            plan.acceptedLabels.begin() + offset,
            plan.acceptedLabels.begin() + end);
        plan.packingGroups.push_back(std::move(group));
    }
    plan.publicPlanHash = PublicPlanHash(plan);
    ValidateServerSparseAdmissionPlan(config, survivors, plan);
    return plan;
}

void FinalizeClientSparsePrivateRoundState(
    const ServerSparseAdmissionPlan& serverPlan,
    ClientSparsePrivateRoundState* state) {

    if (state == nullptr ||
        !(state->setupId == serverPlan.setupId) ||
        !(state->roundId == serverPlan.roundId) ||
        state->chunkSize != serverPlan.chunkSize ||
        state->chunkCount != serverPlan.chunkCount ||
        state->mlweRank != serverPlan.mlweRank) {
        throw std::invalid_argument("client/server sparse-round metadata mismatch");
    }

    std::vector<ClientSparseChunkLabelMapping> accepted;
    for (const auto& mapping : state->selectedMappings) {
        if (FindShuffledCohort(serverPlan, mapping.label).accepted) {
            accepted.push_back(mapping);
        }
    }
    std::sort(
        accepted.begin(),
        accepted.end(),
        [](const auto& left, const auto& right) {
            return left.label < right.label;
        });

    std::vector<ShuffledChunkLabel> uploadLabels;
    uploadLabels.reserve(accepted.size());
    for (const auto& mapping : accepted) {
        uploadLabels.push_back(mapping.label);
    }
    state->uploadSupportBitmap = EncodeSupportBitmapMsb0(
        state->chunkCount,
        uploadLabels);

    state->packingGroups.clear();
    for (std::size_t offset = 0;
         offset < accepted.size();
         offset += state->mlweRank) {
        ProtocolClientPackingGroup group;
        group.clientId = state->clientId;
        group.groupIndex = offset / state->mlweRank;
        const std::size_t end = std::min(
            offset + state->mlweRank,
            accepted.size());
        for (std::size_t index = offset; index < end; ++index) {
            group.actualChunks.push_back(accepted[index].actualChunk);
            group.shuffledLabels.labels.push_back(accepted[index].label);
        }
        state->packingGroups.push_back(std::move(group));
    }
}

ProtocolSparseRoundArtifacts BuildProtocolSparseRoundArtifacts(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    RoundId roundId,
    const SurvivorSet& survivors,
    const std::vector<ProtocolClientRoundInput>& clients) {

    ProtocolSparseRoundArtifacts result;
    std::vector<SparseIndexAnnouncement> announcements;
    std::set<std::size_t> seen;
    for (const auto clientId : survivors.clients) {
        if (!seen.insert(clientId.value).second) {
            throw std::invalid_argument("survivor set contains duplicates");
        }
        const auto& client = FindClient(clients, clientId);
        auto state = BuildClientSparsePrivateRoundState(
            config,
            clientOnlySetup,
            roundId,
            client);
        announcements.push_back(BuildSparseIndexAnnouncement(state));
        result.clientPrivateStates.push_back(std::move(state));
    }
    result.serverPlan = BuildServerSparseAdmissionPlan(
        config,
        clientOnlySetup.setupId,
        roundId,
        survivors,
        announcements);
    for (auto& state : result.clientPrivateStates) {
        FinalizeClientSparsePrivateRoundState(result.serverPlan, &state);
        const auto& client = FindClient(clients, state.clientId);
        ValidateClientSparsePrivateRoundState(
            config,
            client,
            BuildSparseIndexAnnouncement(state),
            result.serverPlan,
            state);
    }
    return result;
}

ProtocolSparseRoundArtifacts BuildProtocolSparseRoundArtifactsFromSelections(
    const ProtocolExperimentConfig& config,
    const ClientOnlySetupMaterial& clientOnlySetup,
    RoundId roundId,
    const SurvivorSet& survivors,
    const std::vector<ClientSelectionRequest>& selections) {

    if (selections.size() != survivors.clients.size()) {
        throw std::invalid_argument(
            "external selection count must match survivor count");
    }

    ProtocolSparseRoundArtifacts result;
    std::vector<SparseIndexAnnouncement> announcements;
    std::set<std::size_t> seen;
    for (const auto clientId : survivors.clients) {
        if (!seen.insert(clientId.value).second) {
            throw std::invalid_argument("survivor set contains duplicates");
        }
        const auto& selection = FindSelection(selections, clientId);
        if (!(selection.roundId == roundId)) {
            throw std::invalid_argument("external selection round mismatch");
        }
        auto state = BuildClientSparsePrivateRoundStateFromSelection(
            config,
            clientOnlySetup,
            selection);
        announcements.push_back(BuildSparseIndexAnnouncement(state));
        result.clientPrivateStates.push_back(std::move(state));
    }
    result.serverPlan = BuildServerSparseAdmissionPlan(
        config,
        clientOnlySetup.setupId,
        roundId,
        survivors,
        announcements);
    for (auto& state : result.clientPrivateStates) {
        FinalizeClientSparsePrivateRoundState(result.serverPlan, &state);

        ProtocolClientRoundInput validationInput;
        validationInput.clientId = state.clientId;
        validationInput.integerWeight =
            config.integerWeights[state.clientId.value];
        validationInput.update.assign(config.modelDimension, 0.0);
        ValidateClientSparsePrivateRoundState(
            config,
            validationInput,
            BuildSparseIndexAnnouncement(state),
            result.serverPlan,
            state);
    }
    return result;
}

void ValidateClientSparsePrivateRoundState(
    const ProtocolExperimentConfig& config,
    const ProtocolClientRoundInput& client,
    const SparseIndexAnnouncement& announcement,
    const ServerSparseAdmissionPlan& serverPlan,
    const ClientSparsePrivateRoundState& state) {

    ValidateClientInput(config, client);
    const DerivedDimensions dimensions = DeriveDimensions(config);
    if (!(state.setupId == serverPlan.setupId) ||
        !(state.roundId == serverPlan.roundId) ||
        !(state.clientId == client.clientId) ||
        !(announcement.clientId == client.clientId) ||
        state.chunkSize != dimensions.smallRingDimension ||
        state.chunkCount != dimensions.chunkCount ||
        state.mlweRank != config.sparsification.mlweRank ||
        state.permutation.size() != state.chunkCount ||
        state.inversePermutation.size() != state.chunkCount ||
        state.actualSelectedChunks.size() != dimensions.selectedChunksPerClient ||
        state.selectedMappings.size() != state.actualSelectedChunks.size()) {
        throw std::invalid_argument("client-private sparse state shape is invalid");
    }
    for (std::size_t actual = 0; actual < state.chunkCount; ++actual) {
        if (state.permutation[actual] >= state.chunkCount ||
            state.inversePermutation[state.permutation[actual]] != actual) {
            throw std::invalid_argument("client-private permutation is invalid");
        }
    }
    for (std::size_t index = 0; index < state.selectedMappings.size(); ++index) {
        const auto& mapping = state.selectedMappings[index];
        if (!(mapping.actualChunk == state.actualSelectedChunks[index]) ||
            mapping.actualChunk.value >= state.chunkCount ||
            state.permutation[mapping.actualChunk.value] != mapping.label.value ||
            !(ActualChunkForLabel(state, mapping.label) == mapping.actualChunk)) {
            throw std::invalid_argument("client-private label mapping is invalid");
        }
    }
    if (BuildSparseIndexAnnouncement(state).supportBitmap !=
        announcement.supportBitmap) {
        throw std::invalid_argument("client announcement is inconsistent");
    }

    std::vector<ClientSparseChunkLabelMapping> expected;
    for (const auto& mapping : state.selectedMappings) {
        if (FindShuffledCohort(serverPlan, mapping.label).accepted) {
            expected.push_back(mapping);
        }
    }
    const auto byLabel = [](const auto& left, const auto& right) {
        return left.label < right.label;
    };
    std::sort(expected.begin(), expected.end(), byLabel);
    std::vector<ClientSparseChunkLabelMapping> packed;
    std::size_t expectedGroup = 0;
    for (const auto& group : state.packingGroups) {
        if (!(group.clientId == state.clientId) ||
            group.groupIndex != expectedGroup++ ||
            group.actualChunks.empty() ||
            group.actualChunks.size() > state.mlweRank ||
            group.actualChunks.size() != group.shuffledLabels.labels.size() ||
            !std::is_sorted(
                group.shuffledLabels.labels.begin(),
                group.shuffledLabels.labels.end())) {
            throw std::invalid_argument("client-private packing group is invalid");
        }
        for (std::size_t slot = 0; slot < group.actualChunks.size(); ++slot) {
            packed.push_back(ClientSparseChunkLabelMapping{
                group.actualChunks[slot],
                group.shuffledLabels.labels[slot]});
        }
    }
    if (packed.size() != expected.size()) {
        throw std::invalid_argument("client-private packing omits a contribution");
    }
    for (std::size_t index = 0; index < packed.size(); ++index) {
        if (!(packed[index].actualChunk == expected[index].actualChunk) ||
            !(packed[index].label == expected[index].label)) {
            throw std::invalid_argument("client-private packing mapping mismatch");
        }
    }
    std::vector<ShuffledChunkLabel> expectedUploadLabels;
    expectedUploadLabels.reserve(expected.size());
    for (const auto& mapping : expected) {
        expectedUploadLabels.push_back(mapping.label);
    }
    if (state.uploadSupportBitmap != EncodeSupportBitmapMsb0(
            state.chunkCount,
            expectedUploadLabels)) {
        throw std::invalid_argument("client upload-support bitmap is inconsistent");
    }
}

void ValidateServerSparseAdmissionPlan(
    const ProtocolExperimentConfig& config,
    const SurvivorSet& survivors,
    const ServerSparseAdmissionPlan& plan) {

    const DerivedDimensions dimensions = DeriveDimensions(config);
    if (plan.chunkSize != dimensions.smallRingDimension ||
        plan.chunkCount != dimensions.chunkCount ||
        plan.mlweRank != config.sparsification.mlweRank ||
        plan.announcements.size() != survivors.clients.size() ||
        plan.cohorts.size() != plan.chunkCount) {
        throw std::invalid_argument("server sparse admission plan shape is invalid");
    }
    std::set<std::size_t> seen;
    for (std::size_t index = 0; index < plan.announcements.size(); ++index) {
        const auto& announcement = plan.announcements[index];
        if (!(announcement.clientId == survivors.clients[index]) ||
            announcement.clientId.value >= config.clientCount ||
            !seen.insert(announcement.clientId.value).second ||
            DecodeSupportBitmapMsb0(
                plan.chunkCount,
                announcement.supportBitmap,
                dimensions.selectedChunksPerClient).size() !=
                    dimensions.selectedChunksPerClient) {
            throw std::invalid_argument("server sparse announcement is invalid");
        }
    }

    std::vector<ShuffledChunkLabel> expectedAccepted;
    for (std::size_t value = 0; value < plan.chunkCount; ++value) {
        const ShuffledChunkLabel label{value};
        const auto& cohort = plan.cohorts[value];
        std::vector<ClientId> expectedClients;
        for (const auto& announcement : plan.announcements) {
            if (SupportBitmapContains(
                    announcement.supportBitmap,
                    plan.chunkCount,
                    label)) {
                expectedClients.push_back(announcement.clientId);
            }
        }
        const bool accepted = expectedClients.size() >=
            config.sparsification.minimumCohort;
        if (!(cohort.label == label) ||
            cohort.clients != expectedClients ||
            cohort.accepted != accepted) {
            throw std::invalid_argument("server shuffled-label cohort is inconsistent");
        }
        if (accepted) {
            expectedAccepted.push_back(label);
        }
    }
    if (expectedAccepted != plan.acceptedLabels) {
        throw std::invalid_argument("server accepted-label list is inconsistent");
    }
    if (plan.acceptedSupportBitmap != EncodeSupportBitmapMsb0(
            plan.chunkCount,
            plan.acceptedLabels)) {
        throw std::invalid_argument("server admission bitmap is inconsistent");
    }
    std::vector<ShuffledChunkLabel> packed;
    for (const auto& group : plan.packingGroups) {
        if (group.labels.empty() || group.labels.size() > plan.mlweRank) {
            throw std::invalid_argument("server label packing group is invalid");
        }
        packed.insert(packed.end(), group.labels.begin(), group.labels.end());
    }
    if (packed != plan.acceptedLabels ||
        plan.publicPlanHash != PublicPlanHash(plan)) {
        throw std::invalid_argument("server packing or public plan hash is invalid");
    }
}

ActualChunkId ActualChunkForLabel(
    const ClientSparsePrivateRoundState& state,
    ShuffledChunkLabel label) {

    if (label.value >= state.inversePermutation.size()) {
        throw std::out_of_range("shuffled label is outside client permutation");
    }
    return ActualChunkId{state.inversePermutation[label.value]};
}

const ShuffledChunkCohort& FindShuffledCohort(
    const ServerSparseAdmissionPlan& plan,
    ShuffledChunkLabel label) {

    if (label.value >= plan.cohorts.size() ||
        !(plan.cohorts[label.value].label == label)) {
        throw std::out_of_range("shuffled cohort was not found");
    }
    return plan.cohorts[label.value];
}

const ClientSparsePrivateRoundState& FindClientSparsePrivateState(
    const std::vector<ClientSparsePrivateRoundState>& states,
    ClientId clientId) {

    const auto found = std::find_if(
        states.begin(),
        states.end(),
        [clientId](const ClientSparsePrivateRoundState& state) {
            return state.clientId == clientId;
        });
    if (found == states.end()) {
        throw std::invalid_argument("client-private sparse state is missing");
    }
    return *found;
}

}  // namespace federated
