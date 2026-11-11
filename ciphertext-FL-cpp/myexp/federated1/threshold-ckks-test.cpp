#include "threshold-ckks-test.h"

#include "coefficient-codec.h"
#include "modulus-diagnostics.h"
#include "threshold-ckks.h"

#include "openfhe.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool CheckValues(
    const std::string& label,
    const std::vector<double>& expected,
    const std::vector<double>& actual,
    double tolerance) {

    if (expected.size() != actual.size()) {
        std::cerr
            << label
            << ": size mismatch, expected="
            << expected.size()
            << ", actual="
            << actual.size()
            << '\n';

        return false;
    }

    bool passed = true;
    size_t failureCount = 0;

    constexpr size_t maxFailuresToPrint = 10;

    double maxError = 0.0;
    size_t maxErrorIndex = 0;

    for (size_t index = 0;
         index < expected.size();
         ++index) {

        double error;

        if (!std::isfinite(actual[index])) {
            error =
                std::numeric_limits<double>::infinity();
        }
        else {
            error =
                std::abs(
                    expected[index] -
                    actual[index]);
        }

        if (error > maxError) {
            maxError = error;
            maxErrorIndex = index;
        }

        if (error > tolerance) {
            passed = false;

            if (failureCount <
                maxFailuresToPrint) {

                std::cerr
                    << "  index=" << index
                    << ", expected="
                    << expected[index]
                    << ", actual="
                    << actual[index]
                    << ", error="
                    << error
                    << '\n';
            }

            ++failureCount;
        }
    }

    std::cout
        << label
        << ": "
        << (passed ? "PASS" : "FAIL")
        << ", checked="
        << expected.size()
        << ", failures="
        << failureCount
        << ", maxError="
        << maxError
        << ", maxErrorIndex="
        << maxErrorIndex
        << '\n';

    return passed;
}

std::vector<std::vector<double>>
MakeClientUpdates(
    size_t partyCount,
    size_t ringDimension) {

    if (partyCount != 3) {
        throw std::invalid_argument(
            "this test expects exactly three parties");
    }

    std::vector<std::vector<double>> updates(
        partyCount,
        std::vector<double>(
            ringDimension,
            0.0));

    for (size_t index = 0;
         index < ringDimension;
         ++index) {

        const int centered0 =
            static_cast<int>(index % 17) - 8;

        const int centered1 =
            static_cast<int>((index * 3) % 19) - 9;

        const int centered2 =
            static_cast<int>((index * 5) % 13) - 6;

        updates[0][index] =
            static_cast<double>(centered0) /
            64.0;

        updates[1][index] =
            static_cast<double>(centered1) /
            128.0;

        updates[2][index] =
            static_cast<double>(centered2) /
            96.0;
    }

    return updates;
}

std::vector<double> SumUpdates(
    const std::vector<
        std::vector<double>>& updates) {

    if (updates.empty()) {
        throw std::invalid_argument(
            "updates must not be empty");
    }

    const size_t valueCount =
        updates[0].size();

    std::vector<double> result(
        valueCount,
        0.0);

    for (size_t partyIndex = 0;
         partyIndex < updates.size();
         ++partyIndex) {

        if (updates[partyIndex].size() !=
            valueCount) {

            throw std::invalid_argument(
                "all updates must have the same size");
        }

        for (size_t index = 0;
             index < valueCount;
             ++index) {

            result[index] +=
                updates[partyIndex][index];
        }
    }

    return result;
}

}  // namespace

int RunThresholdCKKSTests() {
    try {
        constexpr uint32_t ringDimension =
            1u << 13;

        constexpr size_t partyCount = 3;

        /* Coefficient-codec Delta=2^60, independent of OpenFHE ScalingModSize. */
        constexpr uint32_t scaleBits = 60;

        /* Per-party smudging standard deviation in the baseline threshold helper. */
        constexpr uint32_t sigmaShareBits = 48;

        const double sigmaShare =
            std::ldexp(
                1.0,
                static_cast<int>(
                    sigmaShareBits));

        const double scalingFactor =
            coefficient_codec::
                ScalingFactorFromBits(
                    scaleBits);

        /* Independent shares give decoded sigma=sqrt(partyCount)*sigmaShare/Delta. This functional test uses ten sigma to tolerate extreme samples across 8192 coefficients, not to certify security. */
        const double expectedNoiseStandardDeviation =
            std::sqrt(
                static_cast<double>(
                    partyCount)) *
            sigmaShare /
            scalingFactor;

        const double tolerance =
            10.0 *
            expectedNoiseStandardDeviation;

        lbcrypto::CCParams<
            lbcrypto::CryptoContextCKKSRNS>
            parameters;

        /* Depth=1 produces two Q towers in this CKKS fixture. */
        parameters.SetMultiplicativeDepth(1);

        /* Q=q0*q1 with approximately 55-bit towers, about 110 bits total. */
        parameters.SetFirstModSize(55);
        parameters.SetScalingModSize(55);

        parameters.SetBatchSize(
            ringDimension / 2);

        parameters.SetRingDim(
            ringDimension);

        /* Let OpenFHE check N=8192 and QP at its 128-bit security setting. */
        parameters.SetSecurityLevel(
            lbcrypto::HEStd_128_classic);

        parameters.SetScalingTechnique(
            lbcrypto::FIXEDMANUAL);

        parameters.SetSecretKeyDist(
            lbcrypto::UNIFORM_TERNARY);

        /* HYBRID switching uses a special modulus P. */
        parameters.SetKeySwitchTechnique(
            lbcrypto::HYBRID);

        /* Split the two Q towers into two HYBRID digits; this fixture produces one approximately 60-bit P tower. */
        parameters.SetNumLargeDigits(2);

        auto cc =
            lbcrypto::GenCryptoContext(
                parameters);

        cc->Enable(lbcrypto::PKE);
        cc->Enable(lbcrypto::KEYSWITCH);
        cc->Enable(lbcrypto::LEVELEDSHE);
        cc->Enable(lbcrypto::MULTIPARTY);

        /* Configure noise before partial decryption, preferably before key generation. */
        threshold_ckks::ConfigureSmudgingSigma(
            cc,
            sigmaShare);

        const auto modulusInformation =
            modulus_diagnostics::Inspect(cc);

        modulus_diagnostics::Print(
            modulusInformation);

        modulus_diagnostics::
            ValidateTwoModulusProfile(
                modulusInformation);


        std::cout
            << std::fixed
            << std::setprecision(10);

        std::cout
            << "ring dimension: "
            << cc->GetRingDimension()
            << '\n';

        std::cout
            << "number of parties: "
            << partyCount
            << '\n';

        std::cout
            << "scaling factor: 2^"
            << scaleBits
            << " = "
            << scalingFactor
            << '\n';

        std::cout
            << "sigma share: 2^"
            << sigmaShareBits
            << " = "
            << threshold_ckks::GetSmudgingSigma(cc)
            << '\n';

        std::cout
            << "tolerance: "
            << tolerance
            << '\n';

        /* This single-process test stores all simulated party shares together. */
        const auto keySet =
            threshold_ckks::
                GenerateCollectiveKeys(
                    cc,
                    partyCount);

        if (!keySet.collectivePublicKey) {
            throw std::runtime_error(
                "collective public key is null");
        }

        if (keySet.secretKeyShares.size() !=
            partyCount) {

            throw std::runtime_error(
                "unexpected number of secret "
                "key shares");
        }

        const auto clientUpdates =
            MakeClientUpdates(
                partyCount,
                ringDimension);

        bool allPassed = true;

        /* Test one update encrypted under the collective key and decrypted using three shares. */
        const auto singleCiphertext =
            threshold_ckks::
                EncryptCoefficients(
                    cc,
                    keySet.collectivePublicKey,
                    clientUpdates[0],
                    scaleBits);

        const auto singleDecoded =
            threshold_ckks::
                ThresholdDecryptCoefficients(
                    cc,
                    singleCiphertext,
                    keySet.secretKeyShares,
                    scalingFactor);

        allPassed &=
            CheckValues(
                "collective encrypt -> "
                "threshold decrypt -> decode",
                clientUpdates[0],
                singleDecoded,
                tolerance);

        /* Each client encrypts its update under the same collective key in this baseline. */
        std::vector<
            lbcrypto::Ciphertext<
                lbcrypto::DCRTPoly>>
            encryptedUpdates;

        encryptedUpdates.reserve(
            partyCount);

        for (size_t partyIndex = 0;
             partyIndex < partyCount;
             ++partyIndex) {

            encryptedUpdates.push_back(
                threshold_ckks::
                    EncryptCoefficients(
                        cc,
                        keySet.collectivePublicKey,
                        clientUpdates[partyIndex],
                        scaleBits));
        }

        /* Add the three ciphertexts without exposing plaintext updates. */
        const auto aggregateCiphertext =
            threshold_ckks::
                AggregateCiphertexts(
                    cc,
                    encryptedUpdates);

        /* Generate partials, combine them, then decode polynomial coefficients. */
        const auto aggregateDecoded =
            threshold_ckks::
                ThresholdDecryptCoefficients(
                    cc,
                    aggregateCiphertext,
                    keySet.secretKeyShares,
                    scalingFactor);

        const auto expectedAggregate =
            SumUpdates(
                clientUpdates);

        allPassed &=
            CheckValues(
                "encrypt each -> aggregate -> "
                "threshold decrypt -> decode",
                expectedAggregate,
                aggregateDecoded,
                tolerance);

        if (!allPassed) {
            std::cerr
                << "\nOne or more threshold CKKS "
                << "tests failed.\n";

            return 1;
        }

        std::cout
            << "\nAll threshold CKKS tests passed.\n";

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