#include "protocol-stage5-test.h"
#include "coefficient-compression.h"

#include "deterministic-randomness.h"
#include "protocol-auxiliary-path.h"
#include "protocol-full-model-path.h"
#include "protocol-parameters.h"
#include "protocol-setup.h"
#include "survivor-keys.h"

#include "openfhe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
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

lbcrypto::CryptoContext<DCRTPoly> MakeContext(
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

std::vector<federated::FullModelClientRoundInput> GenerateModelInputs(
    const federated::ProtocolExperimentConfig& config) {

    using namespace federated;
    std::vector<FullModelClientRoundInput> result;
    result.reserve(config.clientCount);
    for (std::size_t client = 0; client < config.clientCount; ++client) {
        FullModelClientRoundInput input;
        input.clientId = ClientId{client};
        input.integerWeight = config.integerWeights[client];
        input.modelDelta =
            deterministic_randomness::GenerateClippedGaussian(
                config.masterSeed,
                "STAGE5-FULL-MODEL-DELTA",
                deterministic_randomness::EncodeBigEndian64(client),
                config.modelDimension,
                config.syntheticUpdates.modelDelta.mean,
                config.syntheticUpdates.modelDelta.standardDeviation,
                config.syntheticUpdates.modelDelta.minimum,
                config.syntheticUpdates.modelDelta.maximum);
        result.push_back(std::move(input));
    }
    return result;
}

std::vector<federated::AuxiliaryClientRoundInput> GenerateAuxiliaryInputs(
    const federated::ProtocolExperimentConfig& config) {

    using namespace federated;
    std::vector<AuxiliaryClientRoundInput> result;
    result.reserve(config.clientCount);

    for (std::size_t client = 0; client < config.clientCount; ++client) {
        AuxiliaryClientRoundInput input;
        input.clientId = ClientId{client};
        input.integerWeight = config.integerWeights[client];
        const ByteVector context =
            deterministic_randomness::EncodeBigEndian64(client);

        input.runningMeanDelta =
            deterministic_randomness::GenerateClippedGaussian(
                config.masterSeed,
                "STAGE5-RUNNING-MEAN-DELTA",
                context,
                config.syntheticUpdates.runningMeanDimension,
                config.syntheticUpdates.runningMeanDelta.mean,
                config.syntheticUpdates.runningMeanDelta.standardDeviation,
                config.syntheticUpdates.runningMeanDelta.minimum,
                config.syntheticUpdates.runningMeanDelta.maximum);
        input.runningVarianceDelta =
            deterministic_randomness::GenerateClippedGaussian(
                config.masterSeed,
                "STAGE5-RUNNING-VARIANCE-DELTA",
                context,
                config.syntheticUpdates.runningVarianceDimension,
                config.syntheticUpdates.runningVarianceDelta.mean,
                config.syntheticUpdates.runningVarianceDelta.standardDeviation,
                config.syntheticUpdates.runningVarianceDelta.minimum,
                config.syntheticUpdates.runningVarianceDelta.maximum);
        result.push_back(std::move(input));
    }
    return result;
}

double MaximumError(
    const std::vector<double>& left,
    const std::vector<double>& right) {

    if (left.size() != right.size()) {
        throw std::invalid_argument("direct-RLWE error vector sizes differ");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

double MaximumAuxiliaryError(
    const federated::AuxiliaryPathResult& result,
    bool useIdealReference) {

    const auto& meanReference = useIdealReference
        ? result.idealRunningMeanDelta
        : result.quantizedRunningMeanDelta;
    const auto& varianceReference = useIdealReference
        ? result.idealRunningVarianceDelta
        : result.quantizedRunningVarianceDelta;
    return std::max(
        MaximumError(result.runningMeanDelta, meanReference),
        MaximumError(result.runningVarianceDelta, varianceReference));
}

}  // namespace

int RunProtocolStage5Tests() {
    using namespace federated;
    bool passed = true;

    try {
        using namespace coefficient_compression;
        bool exactMean = true;
        for (std::uint64_t q = 3; q < 40; q += 2) {
            for (std::uint32_t bits = 1; (std::uint64_t{1} << bits) <= q; ++bits) {
                const auto k = std::uint64_t{1} << bits;
                auto grid = [=](std::uint64_t j) { return (q*j + k/2)/k; };
                for (std::uint64_t x = 0; x < q; ++x) {
                    std::uint64_t j = 0;
                    while (grid(j+1) <= x) ++j;
                    const auto width = grid(j+1)-grid(j);
                    std::uint64_t sum = 0;
                    for (std::uint64_t coin = 0; coin < width; ++coin) {
                        const auto code = CompressAggregateCoefficientWithCoin(
                            lbcrypto::BigInteger(x), lbcrypto::BigInteger(q), bits,
                            lbcrypto::BigInteger(coin));
                        const auto decoded = DecompressCoefficient(code, lbcrypto::BigInteger(q), bits).ConvertToInt();
                        const auto lift = j == k-1 && code == 0 ? q : decoded;
                        exactMean &= code == j || code == (j+1)%k;
                        sum += lift;
                    }
                    exactMean &= sum == x*width;
                }
            }
        }
        passed &= Check(exactMean, "aggregate-a exhaustive grid coins have exact zero mean including wrap and width-one cells");
        bool rejected = false;
        try { (void)CompressAggregateCoefficient(lbcrypto::BigInteger(0), lbcrypto::BigInteger(7), 3); }
        catch (const std::invalid_argument&) { rejected = true; }
        passed &= Check(rejected, "aggregate-a rejects 2^c greater than Q");
        ProtocolExperimentConfig config = MakeSmokeConfig();
        config.mode = ProtocolMode::FULL;
        config.modelDimension = 2U * config.modulus.ringDimension + 137U;

        auto cc = MakeContext(config);
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
        const RoundId roundId =
            deterministic_randomness::DeriveRoundId(config.masterSeed, 0);
        const auto modelInputs = GenerateModelInputs(config);
        const auto auxiliaryInputs = GenerateAuxiliaryInputs(config);

        const FullModelPathResult fullLossless = RunProtocolFullModelPath(
            cc,
            config,
            roundId,
            modelInputs,
            setup,
            keys,
            MakeUncompressedFullModelProfile());
        const FullModelPathResult fullCompressed = RunProtocolFullModelPath(
            cc,
            config,
            roundId,
            modelInputs,
            setup,
            keys,
            MakeConfiguredFullModelProfile(config));
        const AuxiliaryPathResult auxLossless = RunProtocolAuxiliaryPath(
            cc,
            config,
            roundId,
            auxiliaryInputs,
            setup,
            keys,
            MakeUncompressedAuxiliaryProfile());
        const AuxiliaryPathResult auxCompressed = RunProtocolAuxiliaryPath(
            cc,
            config,
            roundId,
            auxiliaryInputs,
            setup,
            keys,
            MakeConfiguredAuxiliaryProfile(config));

        const std::size_t ringDimension = config.modulus.ringDimension;
        const std::size_t expectedModelBlocks =
            (config.modelDimension + ringDimension - 1U) / ringDimension;
        const std::size_t auxiliaryValueCount =
            config.syntheticUpdates.runningMeanDimension +
            config.syntheticUpdates.runningVarianceDimension;
        const std::size_t expectedAuxiliaryBlocks =
            (auxiliaryValueCount + ringDimension - 1U) / ringDimension;

        passed &= Check(
            setup.mode == ProtocolMode::FULL &&
                !setup.sparse.has_value() &&
                !keys.m2rKey.has_value(),
            "FULL setup and survivor keys contain no M2R material");
        passed &= Check(
            fullLossless.valueCount == config.modelDimension &&
                fullLossless.blockCount == expectedModelBlocks &&
                config.modelDimension % ringDimension != 0,
            "FULL model uses fixed-order ceil(d/N) blocks with final padding");
        passed &= Check(
            auxLossless.auxiliaryValueCount == auxiliaryValueCount &&
                auxLossless.blockCount == expectedAuxiliaryBlocks,
            "auxiliary running statistics use an independent block sequence");
        passed &= Check(
            fullLossless.clientCiphertextCount ==
                    survivors.clients.size() * expectedModelBlocks &&
                fullLossless.aggregateCiphertextCount == expectedModelBlocks &&
                fullCompressed.clientCiphertextCount ==
                    survivors.clients.size() * expectedModelBlocks &&
                fullCompressed.aggregateCiphertextCount == expectedModelBlocks &&
                auxLossless.clientCiphertextCount ==
                    survivors.clients.size() * expectedAuxiliaryBlocks &&
                auxLossless.aggregateCiphertextCount == expectedAuxiliaryBlocks &&
                auxCompressed.clientCiphertextCount ==
                    survivors.clients.size() * expectedAuxiliaryBlocks &&
                auxCompressed.aggregateCiphertextCount == expectedAuxiliaryBlocks,
            "model and auxiliary ciphertext counts match their block counts");
        passed &= Check(
            fullLossless.r2mCiphertextCount == 0 &&
                fullLossless.m2rCiphertextCount == 0 &&
                fullLossless.residualValueCount == 0 &&
                auxLossless.r2mCiphertextCount == 0 &&
                auxLossless.m2rCiphertextCount == 0,
            "direct-RLWE paths create no routing, R2M/M2R, or residual state");

        auto normalProfile = MakeUncompressedFullModelProfile();
        normalProfile.computeReference = false;
        const auto normalFull = RunProtocolFullModelPath(
            cc, config, roundId, modelInputs, setup, keys, normalProfile);
        const auto normalAux = RunProtocolAuxiliaryPath(
            cc, config, roundId, auxiliaryInputs, setup, keys, normalProfile);
        passed &= Check(normalFull.idealReference.empty() &&
            normalFull.quantizedReference.empty() &&
            normalAux.idealRunningMeanDelta.empty() &&
            normalAux.quantizedRunningVarianceDelta.empty() &&
            MaximumError(normalFull.aggregate, fullLossless.idealReference) < 0.01 &&
            MaximumError(normalAux.runningMeanDelta, auxLossless.idealRunningMeanDelta) < 0.01,
            "normal Full/aux execution omits references and preserves decryption");

        const double fullLosslessError = MaximumError(
            fullLossless.aggregate,
            fullLossless.quantizedReference);
        const double fullPlaintextError = MaximumError(
            fullLossless.aggregate,
            fullLossless.idealReference);
        const double fullQuantizationError = MaximumError(
            fullLossless.quantizedReference,
            fullLossless.idealReference);
        const double fullCompressedError = MaximumError(
            fullCompressed.aggregate,
            fullCompressed.quantizedReference);
        const double auxLosslessError = MaximumAuxiliaryError(
            auxLossless,
            false);
        const double auxPlaintextError = MaximumAuxiliaryError(
            auxLossless,
            true);
        const double auxQuantizationError = std::max(
            MaximumError(
                auxLossless.quantizedRunningMeanDelta,
                auxLossless.idealRunningMeanDelta),
            MaximumError(
                auxLossless.quantizedRunningVarianceDelta,
                auxLossless.idealRunningVarianceDelta));
        const double auxCompressedError = MaximumAuxiliaryError(
            auxCompressed,
            false);
        const long double fullCompressionBound =
            FullModelCompressionModelBound(
                config,
                q,
                fullLossless.scalingFactor,
                survivors.clients.size());
        const long double auxCompressionBound =
            AuxiliaryCompressionModelBound(
                config,
                q,
                auxLossless.scalingFactor,
                survivors.clients.size());
        const long double fullDecodedSigma =
            std::sqrt(static_cast<long double>(survivors.clients.size())) *
            config.smudging.sigmaSmg /
            (static_cast<long double>(config.weightDenominator) *
             fullLossless.scalingFactor);
        const long double auxDecodedSigma =
            std::sqrt(static_cast<long double>(survivors.clients.size())) *
            config.smudging.auxSigmaSmg /
            (static_cast<long double>(config.weightDenominator) *
             auxLossless.scalingFactor);

        std::cout
            << "FULL lossless max error:              "
            << fullLosslessError << '\n'
            << "FULL plaintext-weighted max error:    "
            << fullPlaintextError << '\n'
            << "FULL compressed max error:            "
            << fullCompressedError << '\n'
            << "FULL compression model bound:         "
            << static_cast<double>(fullCompressionBound) << '\n'
            << "auxiliary lossless max error:          "
            << auxLosslessError << '\n'
            << "auxiliary plaintext-weighted max error: "
            << auxPlaintextError << '\n'
            << "auxiliary compressed max error:        "
            << auxCompressedError << '\n'
            << "auxiliary compression model bound:     "
            << static_cast<double>(auxCompressionBound) << '\n';

        passed &= Check(
            fullLosslessError <=
                static_cast<double>(6.0L * fullDecodedSigma + 1.0e-5L),
            "FULL encrypted aggregate matches its integer-weighted reference");
        passed &= Check(
            fullPlaintextError <=
                static_cast<double>(6.0L * fullDecodedSigma + 1.0e-5L) +
                    fullQuantizationError,
            "FULL encrypted aggregate matches the plaintext weighted aggregate");
        passed &= Check(
            fullCompressedError <= static_cast<double>(fullCompressionBound),
            "FULL c1/ca/c2/cd error stays inside the model bound");
        passed &= Check(
            auxLosslessError <=
                static_cast<double>(6.0L * auxDecodedSigma + 1.0e-5L),
            "auxiliary aggregate matches its integer-weighted reference");
        passed &= Check(
            auxPlaintextError <=
                static_cast<double>(6.0L * auxDecodedSigma + 1.0e-5L) +
                    auxQuantizationError,
            "auxiliary aggregate separately matches its plaintext reference");
        passed &= Check(
            auxCompressedError <= static_cast<double>(auxCompressionBound),
            "auxiliary c1_aux/ca_aux/c2_aux/cd_aux error stays inside its bound");

        bool varianceRemainsPositive = true;
        for (const double delta : auxCompressed.runningVarianceDelta) {
            varianceRemainsPositive &=
                config.syntheticUpdates.initialRunningVariance + delta > 0.0;
        }
        passed &= Check(
            varianceRemainsPositive,
            "applying the auxiliary variance delta keeps running variance positive");
    }
    catch (const std::exception& error) {
        std::cout
            << "[FAIL] protocol stage-5 test threw: "
            << error.what()
            << '\n';
        passed = false;
    }

    std::cout << '\n';
    if (passed) {
        std::cout << "All protocol stage-5 tests passed.\n";
        return 0;
    }
    std::cout << "One or more protocol stage-5 tests failed.\n";
    return 1;
}
