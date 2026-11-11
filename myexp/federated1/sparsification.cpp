#include "sparsification.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace federated {
namespace {

bool IsPowerOfTwo(size_t value) {
    return value != 0 &&
           (value & (value - 1)) == 0;
}

size_t CeilDivide(
    size_t numerator,
    size_t denominator) {

    if (denominator == 0) {
        throw std::invalid_argument(
            "CeilDivide: denominator is zero");
    }

    return numerator / denominator +
           (numerator % denominator != 0 ? 1 : 0);
}

size_t GetChunkSize(
    const FederatedConfig& config) {

    return config.ringDimension /
           config.mlweRank;
}

size_t GetChunkCount(
    const FederatedConfig& config) {

    return CeilDivide(
        config.modelDimension,
        GetChunkSize(config));
}

/* Feed each uint64 to FNV-1a in little-endian order. */
void HashUint64(
    uint64_t value,
    uint64_t& hash) {

    constexpr uint64_t prime =
        1099511628211ULL;

    for (size_t byteIndex = 0;
         byteIndex < 8;
         ++byteIndex) {

        const uint8_t byte =
            static_cast<uint8_t>(
                value & 0xffULL);

        hash ^= byte;
        hash *= prime;

        value >>= 8;
    }
}

uint64_t HashChunkId(size_t chunkId) {
    if (chunkId == kInvalidChunkId) {
        return std::numeric_limits<
            uint64_t>::max();
    }

    return static_cast<uint64_t>(
        chunkId);
}

uint64_t CalculatePlanHash(
    const SparseRoundPlan& plan) {

    uint64_t hash =
        14695981039346656037ULL;

    HashUint64(
        static_cast<uint64_t>(
            plan.chunkSize),
        hash);

    HashUint64(
        static_cast<uint64_t>(
            plan.chunkCount),
        hash);

    HashUint64(
        static_cast<uint64_t>(
            plan.clientPlans.size()),
        hash);

    for (const auto& clientPlan :
         plan.clientPlans) {

        HashUint64(
            static_cast<uint64_t>(
                clientPlan.clientId),
            hash);

        HashUint64(
            static_cast<uint64_t>(
                clientPlan
                    .selectedChunkIds
                    .size()),
            hash);

        for (const size_t chunkId :
             clientPlan.selectedChunkIds) {

            HashUint64(
                HashChunkId(chunkId),
                hash);
        }

        HashUint64(
            static_cast<uint64_t>(
                clientPlan
                    .packingGroups
                    .size()),
            hash);

        for (const auto& group :
             clientPlan.packingGroups) {

            HashUint64(
                static_cast<uint64_t>(
                    group.chunkIds.size()),
                hash);

            for (const size_t chunkId :
                 group.chunkIds) {

                HashUint64(
                    HashChunkId(chunkId),
                    hash);
            }
        }
    }

    HashUint64(
        static_cast<uint64_t>(
            plan.cohorts.size()),
        hash);

    for (const auto& cohort :
         plan.cohorts) {

        HashUint64(
            static_cast<uint64_t>(
                cohort.chunkId),
            hash);

        HashUint64(
            cohort.accepted ? 1ULL : 0ULL,
            hash);

        HashUint64(
            static_cast<uint64_t>(
                cohort.clientIds.size()),
            hash);

        for (const size_t clientId :
             cohort.clientIds) {

            HashUint64(
                static_cast<uint64_t>(
                    clientId),
                hash);
        }
    }

    HashUint64(
        static_cast<uint64_t>(
            plan.acceptedChunkIds.size()),
        hash);

    for (const size_t chunkId :
         plan.acceptedChunkIds) {

        HashUint64(
            static_cast<uint64_t>(
                chunkId),
            hash);
    }

    HashUint64(
        static_cast<uint64_t>(
            plan.serverPackingGroups.size()),
        hash);

    for (const auto& group :
         plan.serverPackingGroups) {

        HashUint64(
            static_cast<uint64_t>(
                group.chunkIds.size()),
            hash);

        for (const size_t chunkId :
             group.chunkIds) {

            HashUint64(
                HashChunkId(chunkId),
                hash);
        }
    }

    return hash;
}

std::vector<ChunkPackingGroup>
MakePackingGroups(
    const std::vector<size_t>& chunkIds,
    size_t rank) {

    if (rank == 0) {
        throw std::invalid_argument(
            "MakePackingGroups: rank is zero");
    }

    std::vector<ChunkPackingGroup> groups;

    const size_t groupCount =
        CeilDivide(
            chunkIds.size(),
            rank);

    groups.reserve(groupCount);

    for (size_t groupIndex = 0;
         groupIndex < groupCount;
         ++groupIndex) {

        ChunkPackingGroup group;

        group.chunkIds.assign(
            rank,
            kInvalidChunkId);

        for (size_t slot = 0;
             slot < rank;
             ++slot) {

            const size_t sourceIndex =
                groupIndex * rank + slot;

            if (sourceIndex <
                chunkIds.size()) {

                group.chunkIds[slot] =
                    chunkIds[sourceIndex];
            }
        }

        groups.emplace_back(
            std::move(group));
    }

    return groups;
}

void ValidatePackingGroups(
    const std::vector<ChunkPackingGroup>& groups,
    const std::vector<size_t>& expectedChunkIds,
    size_t rank,
    size_t chunkCount,
    const char* functionName) {

    const size_t expectedGroupCount =
        expectedChunkIds.empty()
            ? 0
            : CeilDivide(
                  expectedChunkIds.size(),
                  rank);

    if (groups.size() !=
        expectedGroupCount) {

        throw std::invalid_argument(
            std::string(functionName) +
            ": unexpected number of packing groups");
    }

    std::vector<size_t> flattened;
    flattened.reserve(
        expectedChunkIds.size());

    bool paddingSeen = false;

    for (const auto& group : groups) {
        if (group.chunkIds.size() != rank) {
            throw std::invalid_argument(
                std::string(functionName) +
                ": packing group size must equal rank");
        }

        for (const size_t chunkId :
             group.chunkIds) {

            if (chunkId ==
                kInvalidChunkId) {

                paddingSeen = true;
                continue;
            }

            if (paddingSeen) {
                throw std::invalid_argument(
                    std::string(functionName) +
                    ": valid chunk appears after padding");
            }

            if (chunkId >= chunkCount) {
                throw std::invalid_argument(
                    std::string(functionName) +
                    ": packing group contains an "
                    "invalid chunk ID");
            }

            flattened.push_back(
                chunkId);
        }
    }

    if (flattened != expectedChunkIds) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": packing groups do not match "
            "the expected chunk ordering");
    }
}

}  // namespace

void ValidateFederatedRoundInputs(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients) {

    if (config.clientCount < 2) {
        throw std::invalid_argument(
            "clientCount must be at least two");
    }

    if (clients.size() !=
        config.clientCount) {

        throw std::invalid_argument(
            "number of clients does not match "
            "config.clientCount");
    }

    if (config.modelDimension == 0) {
        throw std::invalid_argument(
            "modelDimension must be greater than zero");
    }

    if (!IsPowerOfTwo(
            config.ringDimension)) {

        throw std::invalid_argument(
            "ringDimension must be a power of two");
    }

    if (!IsPowerOfTwo(
            config.mlweRank)) {

        throw std::invalid_argument(
            "mlweRank must be a power of two");
    }

    if (config.ringDimension %
            config.mlweRank !=
        0) {

        throw std::invalid_argument(
            "ringDimension must be divisible "
            "by mlweRank");
    }

    const size_t chunkSize =
        GetChunkSize(config);

    if (!IsPowerOfTwo(chunkSize)) {
        throw std::invalid_argument(
            "small-ring chunk size must be "
            "a power of two");
    }

    const size_t chunkCount =
        GetChunkCount(config);

    if (config.selectedChunkCount == 0 ||
        config.selectedChunkCount >
            chunkCount) {

        throw std::invalid_argument(
            "selectedChunkCount must be between "
            "one and the number of chunks");
    }

    if (config.minimumCohortSize == 0 ||
        config.minimumCohortSize >
            config.clientCount) {

        throw std::invalid_argument(
            "minimumCohortSize must be between "
            "one and clientCount");
    }

    if (config.weightDenominator == 0) {
        throw std::invalid_argument(
            "weightDenominator must be nonzero");
    }

    if (config.scaleBits == 0 ||
        config.scaleBits >= 63) {

        throw std::invalid_argument(
            "scaleBits must be between 1 and 62");
    }

    if (!std::isfinite(config.sigmaShare) ||
        config.sigmaShare <= 0.0) {

        throw std::invalid_argument(
            "sigmaShare must be finite and positive");
    }

    if (!std::isfinite(
            config.updateStandardDeviation) ||
        config.updateStandardDeviation <= 0.0) {

        throw std::invalid_argument(
            "updateStandardDeviation must be "
            "finite and positive");
    }

    std::set<size_t> clientIds;

    uint64_t totalAlpha = 0;

    for (const auto& client : clients) {
        if (!clientIds
                 .insert(client.clientId)
                 .second) {

            throw std::invalid_argument(
                "duplicate clientId");
        }

        if (client.alpha == 0 ||
            client.alpha >
                config.weightDenominator) {

            throw std::invalid_argument(
                "each alpha must be between one "
                "and weightDenominator");
        }

        totalAlpha += client.alpha;

        if (client.update.size() !=
            config.modelDimension) {

            throw std::invalid_argument(
                "client update size does not match "
                "modelDimension");
        }

        for (const double value :
             client.update) {

            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "client update contains a "
                    "non-finite value");
            }
        }
    }

    /* This baseline helper permits a subset weight sum <=128. Production setup separately requires total registered weight 128. */
    if (totalAlpha >
        config.weightDenominator) {

        throw std::invalid_argument(
            "sum of alpha values exceeds "
            "weightDenominator");
    }
}

std::vector<double> ComputeChunkScores(
    const std::vector<double>& update,
    size_t chunkSize) {

    if (update.empty()) {
        throw std::invalid_argument(
            "ComputeChunkScores: update is empty");
    }

    if (chunkSize == 0) {
        throw std::invalid_argument(
            "ComputeChunkScores: chunkSize is zero");
    }

    const size_t chunkCount =
        CeilDivide(
            update.size(),
            chunkSize);

    std::vector<double> scores(
        chunkCount,
        0.0);

    for (size_t chunkId = 0;
         chunkId < chunkCount;
         ++chunkId) {

        const size_t begin =
            chunkId * chunkSize;

        const size_t end =
            std::min(
                begin + chunkSize,
                update.size());

        long double score = 0.0L;

        for (size_t index = begin;
             index < end;
             ++index) {

            const double value =
                update[index];

            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "ComputeChunkScores: update "
                    "contains a non-finite value");
            }

            const long double extendedValue =
                static_cast<long double>(
                    value);

            score +=
                extendedValue *
                extendedValue;
        }

        if (!std::isfinite(score) ||
            score >
                static_cast<long double>(
                    std::numeric_limits<
                        double>::max())) {

            throw std::overflow_error(
                "ComputeChunkScores: chunk score "
                "overflow");
        }

        scores[chunkId] =
            static_cast<double>(score);
    }

    return scores;
}

std::vector<size_t> SelectTopChunks(
    const std::vector<double>& chunkScores,
    size_t selectedChunkCount) {

    if (chunkScores.empty()) {
        throw std::invalid_argument(
            "SelectTopChunks: scores are empty");
    }

    if (selectedChunkCount == 0 ||
        selectedChunkCount >
            chunkScores.size()) {

        throw std::invalid_argument(
            "SelectTopChunks: invalid selected "
            "chunk count");
    }

    for (const double score : chunkScores) {
        if (!std::isfinite(score) ||
            score < 0.0) {

            throw std::invalid_argument(
                "SelectTopChunks: invalid "
                "chunk score");
        }
    }

    std::vector<size_t> chunkIds(
        chunkScores.size());

    std::iota(
        chunkIds.begin(),
        chunkIds.end(),
        size_t{0});

    std::sort(
        chunkIds.begin(),
        chunkIds.end(),
        [&chunkScores](
            size_t left,
            size_t right) {

            if (chunkScores[left] !=
                chunkScores[right]) {

                return chunkScores[left] >
                       chunkScores[right];
            }

            return left < right;
        });

    chunkIds.resize(
        selectedChunkCount);

    return chunkIds;
}

SparseRoundPlan BuildSparseRoundPlan(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients) {

    ValidateFederatedRoundInputs(
        config,
        clients);

    const size_t chunkSize =
        GetChunkSize(config);

    const size_t chunkCount =
        GetChunkCount(config);

    /* Sort by client ID so the plan does not depend on input ordering. */
    std::vector<const ClientRoundInput*>
        sortedClients;

    sortedClients.reserve(
        clients.size());

    for (const auto& client : clients) {
        sortedClients.push_back(
            &client);
    }

    std::sort(
        sortedClients.begin(),
        sortedClients.end(),
        [](
            const ClientRoundInput* left,
            const ClientRoundInput* right) {

            return left->clientId <
                   right->clientId;
        });

    std::vector<std::vector<size_t>>
        cohortMembers(
            chunkCount);

    SparseRoundPlan plan;

    plan.chunkSize = chunkSize;
    plan.chunkCount = chunkCount;

    plan.clientPlans.reserve(
        sortedClients.size());

    for (const auto* client :
         sortedClients) {

        const auto scores =
            ComputeChunkScores(
                client->update,
                chunkSize);

        auto selectedChunkIds =
            SelectTopChunks(
                scores,
                config.selectedChunkCount);

        ClientSparsePlan clientPlan;

        clientPlan.clientId =
            client->clientId;

        clientPlan.selectedChunkIds =
            selectedChunkIds;

        clientPlan.packingGroups =
            MakePackingGroups(
                selectedChunkIds,
                config.mlweRank);

        for (const size_t chunkId :
             selectedChunkIds) {

            cohortMembers[chunkId]
                .push_back(
                    client->clientId);
        }

        plan.clientPlans.emplace_back(
            std::move(clientPlan));
    }

    plan.cohorts.reserve(
        chunkCount);

    for (size_t chunkId = 0;
         chunkId < chunkCount;
         ++chunkId) {

        ChunkCohort cohort;

        cohort.chunkId = chunkId;
        cohort.clientIds =
            cohortMembers[chunkId];

        cohort.accepted =
            cohort.clientIds.size() >=
            config.minimumCohortSize;

        if (cohort.accepted) {
            plan.acceptedChunkIds
                .push_back(
                    chunkId);
        }

        plan.cohorts.emplace_back(
            std::move(cohort));
    }

    plan.serverPackingGroups =
        MakePackingGroups(
            plan.acceptedChunkIds,
            config.mlweRank);

    plan.planHash =
        CalculatePlanHash(plan);

    ValidateSparseRoundPlan(
        config,
        clients,
        plan);

    return plan;
}

void ValidateSparseRoundPlan(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients,
    const SparseRoundPlan& plan) {

    ValidateFederatedRoundInputs(
        config,
        clients);

    const size_t expectedChunkSize =
        GetChunkSize(config);

    const size_t expectedChunkCount =
        GetChunkCount(config);

    if (plan.chunkSize !=
            expectedChunkSize ||
        plan.chunkCount !=
            expectedChunkCount) {

        throw std::invalid_argument(
            "ValidateSparseRoundPlan: chunk "
            "metadata mismatch");
    }

    if (plan.clientPlans.size() !=
        clients.size()) {

        throw std::invalid_argument(
            "ValidateSparseRoundPlan: unexpected "
            "number of client plans");
    }

    std::set<size_t> expectedClientIds;

    for (const auto& client : clients) {
        expectedClientIds.insert(
            client.clientId);
    }

    std::set<size_t> planClientIds;

    std::vector<std::vector<size_t>>
        expectedCohortMembers(
            expectedChunkCount);

    size_t previousClientId = 0;
    bool firstClient = true;

    for (const auto& clientPlan :
         plan.clientPlans) {

        if (!firstClient &&
            clientPlan.clientId <=
                previousClientId) {

            throw std::invalid_argument(
                "ValidateSparseRoundPlan: client "
                "plans are not strictly ordered");
        }

        firstClient = false;
        previousClientId =
            clientPlan.clientId;

        if (expectedClientIds.count(
                clientPlan.clientId) == 0) {

            throw std::invalid_argument(
                "ValidateSparseRoundPlan: unknown "
                "clientId");
        }

        if (!planClientIds
                 .insert(clientPlan.clientId)
                 .second) {

            throw std::invalid_argument(
                "ValidateSparseRoundPlan: duplicate "
                "client plan");
        }

        if (clientPlan
                .selectedChunkIds
                .size() !=
            config.selectedChunkCount) {

            throw std::invalid_argument(
                "ValidateSparseRoundPlan: selected "
                "chunk count mismatch");
        }

        std::set<size_t> selectedSet;

        for (const size_t chunkId :
             clientPlan.selectedChunkIds) {

            if (chunkId >=
                expectedChunkCount) {

                throw std::invalid_argument(
                    "ValidateSparseRoundPlan: invalid "
                    "selected chunk ID");
            }

            if (!selectedSet
                     .insert(chunkId)
                     .second) {

                throw std::invalid_argument(
                    "ValidateSparseRoundPlan: duplicate "
                    "selected chunk ID");
            }

            expectedCohortMembers[chunkId]
                .push_back(
                    clientPlan.clientId);
        }

        ValidatePackingGroups(
            clientPlan.packingGroups,
            clientPlan.selectedChunkIds,
            config.mlweRank,
            expectedChunkCount,
            "ValidateSparseRoundPlan");
    }

    if (planClientIds !=
        expectedClientIds) {

        throw std::invalid_argument(
            "ValidateSparseRoundPlan: client-plan "
            "ID set mismatch");
    }

    if (plan.cohorts.size() !=
        expectedChunkCount) {

        throw std::invalid_argument(
            "ValidateSparseRoundPlan: unexpected "
            "number of cohorts");
    }

    std::vector<size_t>
        expectedAcceptedChunkIds;

    for (size_t chunkId = 0;
         chunkId < expectedChunkCount;
         ++chunkId) {

        const auto& cohort =
            plan.cohorts[chunkId];

        if (cohort.chunkId != chunkId) {
            throw std::invalid_argument(
                "ValidateSparseRoundPlan: cohorts "
                "are not ordered by chunk ID");
        }

        if (cohort.clientIds !=
            expectedCohortMembers[chunkId]) {

            throw std::invalid_argument(
                "ValidateSparseRoundPlan: cohort "
                "membership mismatch");
        }

        const bool expectedAccepted =
            cohort.clientIds.size() >=
            config.minimumCohortSize;

        if (cohort.accepted !=
            expectedAccepted) {

            throw std::invalid_argument(
                "ValidateSparseRoundPlan: incorrect "
                "minimum-cohort decision");
        }

        if (expectedAccepted) {
            expectedAcceptedChunkIds
                .push_back(
                    chunkId);
        }
    }

    if (plan.acceptedChunkIds !=
        expectedAcceptedChunkIds) {

        throw std::invalid_argument(
            "ValidateSparseRoundPlan: accepted "
            "chunk list mismatch");
    }

    ValidatePackingGroups(
        plan.serverPackingGroups,
        plan.acceptedChunkIds,
        config.mlweRank,
        expectedChunkCount,
        "ValidateSparseRoundPlan");

    const uint64_t expectedHash =
        CalculatePlanHash(plan);

    if (plan.planHash != expectedHash) {
        throw std::invalid_argument(
            "ValidateSparseRoundPlan: plan hash "
            "mismatch");
    }
}

const ClientSparsePlan& FindClientSparsePlan(
    const SparseRoundPlan& plan,
    size_t clientId) {

    const auto iterator =
        std::lower_bound(
            plan.clientPlans.begin(),
            plan.clientPlans.end(),
            clientId,
            [](
                const ClientSparsePlan& clientPlan,
                size_t targetClientId) {

                return clientPlan.clientId <
                       targetClientId;
            });

    if (iterator ==
            plan.clientPlans.end() ||
        iterator->clientId != clientId) {

        throw std::out_of_range(
            "FindClientSparsePlan: clientId "
            "not found");
    }

    return *iterator;
}

const ChunkCohort& FindChunkCohort(
    const SparseRoundPlan& plan,
    size_t chunkId) {

    const auto iterator =
        std::lower_bound(
            plan.cohorts.begin(),
            plan.cohorts.end(),
            chunkId,
            [](
                const ChunkCohort& cohort,
                size_t targetChunkId) {

                return cohort.chunkId <
                       targetChunkId;
            });

    if (iterator ==
            plan.cohorts.end() ||
        iterator->chunkId != chunkId) {

        throw std::out_of_range(
            "FindChunkCohort: chunkId not found");
    }

    return *iterator;
}

}  // namespace federated