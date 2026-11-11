#include "federated-simulation-test.h"

#include "chunk-packing.h"
#include "coefficient-codec.h"
#include "collective-conversion-keygen.h"
#include "federated-encrypted.h"
#include "federated-reference.h"
#include "sparsification.h"
#include "threshold-ckks.h"
#include "modulus-diagnostics.h"

#include "openfhe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TestState {
    std::size_t failureCount = 0;

    void Check(
        bool condition,
        const std::string& description) {

        std::cout
            << (condition ? "[PASS] " : "[FAIL] ")
            << description
            << '\n';

        if (!condition) {
            ++failureCount;
        }
    }
};

federated::FederatedConfig MakeTestConfig() {
    federated::FederatedConfig config;

    config.clientCount = 8;

    config.ringDimension =
        static_cast<std::size_t>(1) << 13;

    config.mlweRank = 4;

    /* Fixture N=8192, rank=4 gives 2048 entries per chunk and 12 chunks for d=24576. */
    config.modelDimension =
        3 * config.ringDimension;

    config.selectedChunkCount = 4;
    config.minimumCohortSize = 3;

    config.weightDenominator = 128;
    config.scaleBits = 60;

    config.sigmaShare =
        std::ldexp(1.0, 48);

    config.updateStandardDeviation =
        0.01;

    config.randomSeed =
        20250817;

    return config;
}

/* All clients choose chunks 0 and 1; distribute other choices so their cohorts have fewer than three clients. */
std::set<std::size_t> PreferredChunks(
    std::size_t clientId) {

    return {
        0,
        1,
        2 + (clientId % 5),
        7 + (clientId % 5)
    };
}

void NormalizeChunk(
    std::vector<double>& update,
    std::size_t begin,
    std::size_t end,
    double targetStandardDeviation) {

    long double squaredNorm = 0.0L;

    for (std::size_t index = begin;
         index < end;
         ++index) {

        const long double value =
            static_cast<long double>(
                update[index]);

        squaredNorm +=
            value * value;
    }

    if (squaredNorm <= 0.0L) {
        throw std::runtime_error(
            "generated Gaussian chunk has zero norm");
    }

    const long double length =
        static_cast<long double>(
            end - begin);

    const long double targetNorm =
        static_cast<long double>(
            targetStandardDeviation) *
        std::sqrt(length);

    const long double multiplier =
        targetNorm /
        std::sqrt(squaredNorm);

    for (std::size_t index = begin;
         index < end;
         ++index) {

        update[index] =
            static_cast<double>(
                static_cast<long double>(
                    update[index]) *
                multiplier);
    }
}

std::vector<federated::ClientRoundInput>
GenerateClientInputs(
    const federated::FederatedConfig& config) {

    if (config.weightDenominator %
            config.clientCount !=
        0) {

        throw std::invalid_argument(
            "the test requires equal integer weights");
    }

    const std::size_t chunkSize =
        config.ringDimension /
        config.mlweRank;

    const std::size_t chunkCount =
        (config.modelDimension +
         chunkSize - 1) /
        chunkSize;

    const std::uint32_t alpha =
        static_cast<std::uint32_t>(
            config.weightDenominator /
            config.clientCount);

    std::mt19937_64 generator(
        config.randomSeed);

    std::normal_distribution<double>
        standardGaussian(
            0.0,
            1.0);

    std::vector<federated::ClientRoundInput>
        clients;

    clients.reserve(
        config.clientCount);

    for (std::size_t clientId = 0;
         clientId < config.clientCount;
         ++clientId) {

        federated::ClientRoundInput client;

        client.clientId =
            clientId;

        client.alpha =
            alpha;

        client.update.resize(
            config.modelDimension);

        for (double& value : client.update) {
            value =
                standardGaussian(generator);
        }

        const std::set<std::size_t> preferred =
            PreferredChunks(clientId);

        for (std::size_t chunkId = 0;
             chunkId < chunkCount;
             ++chunkId) {

            const std::size_t begin =
                chunkId * chunkSize;

            const std::size_t end =
                std::min(
                    begin + chunkSize,
                    config.modelDimension);

            const bool isPreferred =
                preferred.find(chunkId) !=
                preferred.end();

            NormalizeChunk(
                client.update,
                begin,
                end,
                isPreferred
                    ? config.updateStandardDeviation
                    : config.updateStandardDeviation *
                          0.001);
        }

        clients.emplace_back(
            std::move(client));
    }

    return clients;
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly>
MakeCryptoContext(
    const federated::FederatedConfig& config) {

    lbcrypto::CCParams<
        lbcrypto::CryptoContextCKKSRNS>
        parameters;

    /* In this CKKS fixture depth=1 creates two Q towers; no ciphertext multiplication is needed. */
    parameters.SetMultiplicativeDepth(1);

    /* Fixture Q=q0*q1, approximately 55 bits per tower and 110 bits total. */
    parameters.SetFirstModSize(55);
    parameters.SetScalingModSize(55);

    parameters.SetBatchSize(
        static_cast<std::uint32_t>(
            config.ringDimension / 2));

    parameters.SetRingDim(
        static_cast<std::uint32_t>(
            config.ringDimension));

    /* Ask OpenFHE to check N=8192 and QP for UNIFORM_TERNARY at the 128-bit setting. */
    parameters.SetSecurityLevel(
        lbcrypto::HEStd_128_classic);

    parameters.SetScalingTechnique(
        lbcrypto::FIXEDMANUAL);

    parameters.SetSecretKeyDist(
        lbcrypto::UNIFORM_TERNARY);

    parameters.SetKeySwitchTechnique(
        lbcrypto::HYBRID);

    /* Use two HYBRID digits with one Q tower each; expected bit sizes are Q~110, P~60, QP~170. */
    parameters.SetNumLargeDigits(2);

    auto cc =
        lbcrypto::GenCryptoContext(
            parameters);

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::MULTIPARTY);

    threshold_ckks::ConfigureSmudgingSigma(
        cc,
        config.sigmaShare);

    return cc;
}

bool CheckPackingRoundTrip(
    const federated::FederatedConfig& config,
    const std::vector<
        federated::ClientRoundInput>& clients,
    const federated::SparseRoundPlan& plan) {

    for (const auto& clientPlan :
         plan.clientPlans) {

        const federated::ClientRoundInput*
            client = nullptr;

        for (const auto& candidate : clients) {
            if (candidate.clientId ==
                clientPlan.clientId) {

                client =
                    &candidate;

                break;
            }
        }

        if (client == nullptr) {
            return false;
        }

        for (const auto& group :
             clientPlan.packingGroups) {

            const auto packed =
                federated::PackInterleavedChunks(
                    client->update,
                    group,
                    config);

            const auto unpacked =
                federated::UnpackInterleavedChunks(
                    packed,
                    config);

            if (unpacked.size() !=
                config.mlweRank) {

                return false;
            }

            for (std::size_t slot = 0;
                 slot < config.mlweRank;
                 ++slot) {

                const std::size_t chunkId =
                    group.chunkIds[slot];

                if (chunkId ==
                    federated::kInvalidChunkId) {

                    for (const double value :
                         unpacked[slot]) {

                        if (value != 0.0) {
                            return false;
                        }
                    }

                    continue;
                }

                const auto expected =
                    federated::ExtractPaddedChunk(
                        client->update,
                        chunkId,
                        config);

                if (unpacked[slot] != expected) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool SelectedChunksAreCorrect(
    const federated::SparseRoundPlan& plan) {

    for (const auto& clientPlan :
         plan.clientPlans) {

        const std::set<std::size_t> actual(
            clientPlan.selectedChunkIds.begin(),
            clientPlan.selectedChunkIds.end());

        if (actual !=
            PreferredChunks(
                clientPlan.clientId)) {

            return false;
        }
    }

    return true;
}

struct ComparisonResult {
    bool passed = false;
    std::size_t failureCount = 0;
    double maximumError = 0.0;
    std::size_t maximumErrorIndex = 0;
};

ComparisonResult CompareValues(
    const std::string& label,
    const std::vector<double>& expected,
    const std::vector<double>& actual,
    double tolerance) {

    ComparisonResult result;

    if (expected.size() != actual.size()) {
        std::cerr
            << label
            << ": size mismatch, expected="
            << expected.size()
            << ", actual="
            << actual.size()
            << '\n';

        return result;
    }

    constexpr std::size_t maxFailuresToPrint =
        10;

    for (std::size_t index = 0;
         index < expected.size();
         ++index) {

        double error;

        if (!std::isfinite(actual[index])) {
            error =
                std::numeric_limits<double>::
                    infinity();
        }
        else {
            error =
                std::abs(
                    expected[index] -
                    actual[index]);
        }

        if (error >
            result.maximumError) {

            result.maximumError =
                error;

            result.maximumErrorIndex =
                index;
        }

        if (error > tolerance) {
            if (result.failureCount <
                maxFailuresToPrint) {

                std::cerr
                    << "  index="
                    << index
                    << ", expected="
                    << expected[index]
                    << ", actual="
                    << actual[index]
                    << ", error="
                    << error
                    << '\n';
            }

            ++result.failureCount;
        }
    }

    result.passed =
        result.failureCount == 0;

    std::cout
        << label
        << ": "
        << (result.passed ? "PASS" : "FAIL")
        << ", checked="
        << expected.size()
        << ", failures="
        << result.failureCount
        << ", maxError="
        << result.maximumError
        << ", maxErrorIndex="
        << result.maximumErrorIndex
        << '\n';

    return result;
}

bool RejectedCoordinatesAreExactlyZero(
    const federated::FederatedEncryptedResult&
        encrypted,
    const federated::FederatedReferenceResult&
        reference) {

    if (encrypted.aggregate.size() !=
            reference.acceptedCoordinateMask.size() ||
        reference.quantizedAggregate.size() !=
            reference.acceptedCoordinateMask.size()) {

        return false;
    }

    for (std::size_t coordinate = 0;
         coordinate <
             reference.acceptedCoordinateMask.size();
         ++coordinate) {

        if (reference.acceptedCoordinateMask[
                coordinate] != 0) {

            continue;
        }

        if (encrypted.aggregate[coordinate] != 0.0 ||
            reference.quantizedAggregate[
                coordinate] != 0.0) {

            return false;
        }
    }

    return true;
}

}  // namespace

int RunFederatedSimulationTests() {
    try {
        TestState state;

        const federated::FederatedConfig config =
            MakeTestConfig();

        const auto clients =
            GenerateClientInputs(
                config);

        /* Compute chunk selection and cohorts once in the plaintext fixture. */
        const federated::SparseRoundPlan plan =
            federated::BuildSparseRoundPlan(
                config,
                clients);

        federated::ValidateSparseRoundPlan(
            config,
            clients,
            plan);

        std::cout
            << std::fixed
            << std::setprecision(10);

        std::cout
            << "client count: "
            << config.clientCount
            << '\n'
            << "model dimension: "
            << config.modelDimension
            << '\n'
            << "ring dimension: "
            << config.ringDimension
            << '\n'
            << "MLWE rank: "
            << config.mlweRank
            << '\n'
            << "chunk size: "
            << plan.chunkSize
            << '\n'
            << "chunk count: "
            << plan.chunkCount
            << '\n'
            << "selected chunks/client: "
            << config.selectedChunkCount
            << '\n'
            << "minimum cohort tau: "
            << config.minimumCohortSize
            << '\n'
            << "accepted chunks: "
            << plan.acceptedChunkIds.size()
            << '\n'
            << "plan hash: "
            << plan.planHash
            << '\n';

        state.Check(
            config.clientCount == 8,
            "simulation uses eight clients");

        state.Check(
            config.minimumCohortSize == 3,
            "minimum cohort size is tau=3");

        state.Check(
            plan.chunkSize == 2048,
            "small-ring chunk size is 2048");

        state.Check(
            plan.chunkCount == 12,
            "model consists of twelve chunks");

        state.Check(
            SelectedChunksAreCorrect(plan),
            "each client selects the intended chunks");

        const std::vector<std::size_t>
            expectedAcceptedChunks = {0, 1};

        state.Check(
            plan.acceptedChunkIds ==
                expectedAcceptedChunks,
            "only chunks 0 and 1 satisfy tau=3");

        state.Check(
            CheckPackingRoundTrip(
                config,
                clients,
                plan),
            "client chunk packing round-trip");

        /* Compute the plaintext reference. */
        const auto reference =
            federated::ComputeFederatedReference(
                config,
                clients,
                plan);

        const double quantizationTolerance =
            std::max(
                1.0 /
                    reference.scalingFactor,
                16.0 *
                    std::numeric_limits<double>::
                        epsilon());

        const auto quantizationComparison =
            CompareValues(
                "ideal plaintext -> quantized plaintext",
                reference.idealAggregate,
                reference.quantizedAggregate,
                quantizationTolerance);

        state.Check(
            quantizationComparison.passed,
            "coefficient quantization error is bounded");

        /* Create the context and eight-party collective key. */
        std::cout
            << "\n[INFO] generating crypto context\n";

        const auto cc =
            MakeCryptoContext(config);

        const auto modulusInformation =
            modulus_diagnostics::Inspect(cc);

        modulus_diagnostics::Print(
            modulusInformation);

        modulus_diagnostics::
            ValidateTwoModulusProfile(
                modulusInformation);

        state.Check(
            modulusInformation.qTowerBits.size() == 2,
            "crypto context has exactly two Q towers");

        state.Check(
            modulusInformation.pTowerBits.size() == 1,
            "HYBRID key switching has one P tower");

        state.Check(
            modulusInformation.qpBits <= 218,
            "QP satisfies the N=8192 "
            "128-bit security bound");

        state.Check(
            cc->GetRingDimension() ==
                config.ringDimension,
            "crypto-context ring dimension matches config");

        state.Check(
            threshold_ckks::GetSmudgingSigma(cc) ==
                config.sigmaShare,
            "configured smudging sigma matches config");

        std::cout
            << "[INFO] generating "
            << config.clientCount
            << "-party collective keys\n";

        const auto keySet =
            threshold_ckks::
                GenerateCollectiveKeys(
                    cc,
                    config.clientCount);

        state.Check(
            keySet.collectivePublicKey != nullptr,
            "collective public key was generated");

        state.Check(
            keySet.secretKeyShares.size() ==
                config.clientCount,
            "one secret-key share per client was generated");

        /* Combine conversion-key shares without reconstructing the collective secret. */
        std::cout
            << "[INFO] generating collective "
            << "MLWE-to-RLWE conversion keys\n";

        const auto collectiveConversionKeys =
            collective_conversion_keygen::
                GenerateCollectiveMLWEtoRLWEKeys(
                    cc,
                    keySet.secretKeyShares,
                    keySet.collectivePublicKey,
                    config.mlweRank);

        state.Check(
            collectiveConversionKeys.size() ==
                config.mlweRank,
            "one collective conversion key per MLWE column");

        /* Pass the same fixed plan to the encrypted path. */
        std::cout
            << "[INFO] running encrypted "
            << "federated aggregation\n";

        const auto encrypted =
            federated::
                ComputeFederatedEncryptedAggregate(
                    cc,
                    keySet.collectivePublicKey,
                    keySet.secretKeyShares,
                    collectiveConversionKeys,
                    config,
                    clients,
                    plan);

        state.Check(
            encrypted.planHash ==
                reference.planHash &&
            encrypted.planHash ==
                plan.planHash,
            "plaintext and encrypted paths use "
            "the same fixed sparse plan");

        const std::size_t expectedClientCiphertexts =
            config.clientCount *
            ((config.selectedChunkCount +
              config.mlweRank - 1) /
             config.mlweRank);

        state.Check(
            encrypted.clientCiphertextCount ==
                expectedClientCiphertexts,
            "expected number of client RLWE "
            "ciphertexts was generated");

        std::size_t expectedContributions = 0;

        for (const auto& cohort : plan.cohorts) {
            if (cohort.accepted) {
                expectedContributions +=
                    cohort.clientIds.size();
            }
        }

        state.Check(
            encrypted
                .acceptedClientContributionCount ==
                expectedContributions,
            "all accepted client-chunk contributions "
            "were aggregated");

        state.Check(
            encrypted
                .aggregatedMLWECiphertextCount ==
                plan.acceptedChunkIds.size(),
            "one aggregated MLWE ciphertext exists "
            "per accepted chunk");

        state.Check(
            encrypted.serverCiphertextCount ==
                plan.serverPackingGroups.size(),
            "one restored RLWE ciphertext exists "
            "per server packing group");

        /* Compare against quantizedAggregate, which weights round(Delta*update_i), rather than the unquantized idealAggregate. */
        constexpr double encryptedTolerance =
            1e-3;

        const auto encryptedComparison =
            CompareValues(
                "quantized plaintext -> encrypted "
                "federated aggregation",
                reference.quantizedAggregate,
                encrypted.aggregate,
                encryptedTolerance);

        state.Check(
            encryptedComparison.passed,
            "encrypted aggregation matches the "
            "quantized plaintext reference");

        state.Check(
            RejectedCoordinatesAreExactlyZero(
                encrypted,
                reference),
            "rejected chunks remain zero in the "
            "final model vector");

        std::cout
            << "\nscaling factor: 2^"
            << config.scaleBits
            << " = "
            << reference.scalingFactor
            << '\n'
            << "sigma share: "
            << config.sigmaShare
            << '\n'
            << "quantization max error: "
            << quantizationComparison.maximumError
            << '\n'
            << "encrypted max error: "
            << encryptedComparison.maximumError
            << '\n'
            << "encrypted tolerance: "
            << encryptedTolerance
            << '\n'
            << "client RLWE ciphertexts: "
            << encrypted.clientCiphertextCount
            << '\n'
            << "accepted MLWE ciphertexts: "
            << encrypted
                   .aggregatedMLWECiphertextCount
            << '\n'
            << "server RLWE ciphertexts: "
            << encrypted.serverCiphertextCount
            << '\n';

        if (state.failureCount != 0) {
            std::cerr
                << "\nOne or more federated "
                << "simulation tests failed.\n";

            return 1;
        }

        std::cout
            << "\nAll federated simulation "
            << "tests passed.\n";

        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr
            << "[ERROR] "
            << exception.what()
            << '\n';

        return 1;
    }
}