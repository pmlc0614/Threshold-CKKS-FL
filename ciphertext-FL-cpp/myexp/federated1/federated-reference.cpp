#include "federated-reference.h"

#include "coefficient-codec.h"
#include "sparsification.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace federated {
namespace {

/* An int64 quantized value times uint32 total weight needs at most about 95 bits; signed __int128 suffices for this small test oracle. */
using WideInteger = __int128_t;

/* Match the codec's ScaleAndRound rule. */
std::int64_t ScaleAndRoundLikeCodec(
    double value,
    double scalingFactor) {

    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "client update contains a non-finite value");
    }

    if (!std::isfinite(scalingFactor) ||
        scalingFactor <= 0.0) {

        throw std::invalid_argument(
            "scalingFactor must be finite and positive");
    }

    const long double scaled =
        static_cast<long double>(value) *
        static_cast<long double>(scalingFactor);

    const long double minimum =
        static_cast<long double>(
            std::numeric_limits<std::int64_t>::min());

    const long double maximum =
        static_cast<long double>(
            std::numeric_limits<std::int64_t>::max());

    if (scaled < minimum ||
        scaled > maximum) {

        throw std::overflow_error(
            "scaled client update does not fit in int64_t");
    }

    return static_cast<std::int64_t>(
        std::llround(scaled));
}

std::unordered_map<std::size_t, const ClientRoundInput*>
BuildClientLookup(
    const std::vector<ClientRoundInput>& clients) {

    std::unordered_map<
        std::size_t,
        const ClientRoundInput*> lookup;

    lookup.reserve(clients.size());

    for (const auto& client : clients) {
        const auto inserted =
            lookup.emplace(
                client.clientId,
                &client);

        if (!inserted.second) {
            throw std::invalid_argument(
                "duplicate clientId in federated inputs");
        }
    }

    return lookup;
}

const ClientRoundInput& FindClient(
    const std::unordered_map<
        std::size_t,
        const ClientRoundInput*>& lookup,
    std::size_t clientId) {

    const auto iterator =
        lookup.find(clientId);

    if (iterator == lookup.end() ||
        iterator->second == nullptr) {

        throw std::invalid_argument(
            "sparse plan references an unknown client");
    }

    return *iterator->second;
}

}  // namespace

FederatedReferenceResult ComputeFederatedReference(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients,
    const SparseRoundPlan& plan) {

    ValidateFederatedRoundInputs(
        config,
        clients);

    ValidateSparseRoundPlan(
        config,
        clients,
        plan);

    const double scalingFactor =
        coefficient_codec::ScalingFactorFromBits(
            config.scaleBits);

    if (config.weightDenominator == 0) {
        throw std::invalid_argument(
            "weightDenominator must be positive");
    }

    const auto clientLookup =
        BuildClientLookup(clients);

    /* Accumulate sum_i alpha_i*update_i. */
    std::vector<long double> idealNumerators(
        config.modelDimension,
        0.0L);

    /* Accumulate weighted encoded integers in 128 bits to avoid int64 overflow. */
    std::vector<WideInteger> quantizedNumerators(
        config.modelDimension,
        static_cast<WideInteger>(0));

    std::vector<std::uint8_t> acceptedMask(
        config.modelDimension,
        static_cast<std::uint8_t>(0));

    for (const auto& cohort : plan.cohorts) {
        if (!cohort.accepted) {
            continue;
        }

        if (cohort.chunkId >= plan.chunkCount) {
            throw std::invalid_argument(
                "accepted cohort has an invalid chunkId");
        }

        const std::size_t chunkBegin =
            cohort.chunkId * plan.chunkSize;

        const std::size_t chunkEnd =
            std::min(
                chunkBegin + plan.chunkSize,
                config.modelDimension);

        for (std::size_t coordinate = chunkBegin;
             coordinate < chunkEnd;
             ++coordinate) {

            acceptedMask[coordinate] =
                static_cast<std::uint8_t>(1);
        }

        /* The cohort includes only clients selecting this chunk. */
        for (const std::size_t clientId :
             cohort.clientIds) {

            const ClientRoundInput& client =
                FindClient(
                    clientLookup,
                    clientId);

            const long double alpha =
                static_cast<long double>(
                    client.alpha);

            const WideInteger integerAlpha =
                static_cast<WideInteger>(
                    client.alpha);

            for (std::size_t coordinate = chunkBegin;
                 coordinate < chunkEnd;
                 ++coordinate) {

                const double value =
                    client.update[coordinate];

                idealNumerators[coordinate] +=
                    alpha *
                    static_cast<long double>(value);

                /* Match encryption order: round Delta*value, multiply by alpha_i, then sum client contributions. */
                const std::int64_t quantized =
                    ScaleAndRoundLikeCodec(
                        value,
                        scalingFactor);

                quantizedNumerators[coordinate] +=
                    static_cast<WideInteger>(
                        quantized) *
                    integerAlpha;
            }
        }
    }

    FederatedReferenceResult result;

    result.idealAggregate.assign(
        config.modelDimension,
        0.0);

    result.quantizedAggregate.assign(
        config.modelDimension,
        0.0);

    result.acceptedCoordinateMask =
        std::move(acceptedMask);

    result.planHash =
        plan.planHash;

    result.scalingFactor =
        scalingFactor;

    const long double denominator =
        static_cast<long double>(
            config.weightDenominator);

    const long double quantizedDenominator =
        denominator *
        static_cast<long double>(
            scalingFactor);

    for (std::size_t coordinate = 0;
         coordinate < config.modelDimension;
         ++coordinate) {

        if (result.acceptedCoordinateMask[coordinate] == 0) {
            continue;
        }

        result.idealAggregate[coordinate] =
            static_cast<double>(
                idealNumerators[coordinate] /
                denominator);

        /* GCC supports direct conversion from __int128_t to long double. */
        const long double integerNumerator =
            static_cast<long double>(
                quantizedNumerators[coordinate]);

        result.quantizedAggregate[coordinate] =
            static_cast<double>(
                integerNumerator /
                quantizedDenominator);
    }

    return result;
}

}  // namespace federated