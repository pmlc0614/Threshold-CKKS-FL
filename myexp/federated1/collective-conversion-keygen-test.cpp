#include "collective-conversion-keygen-test.h"

#include "coefficient-codec.h"
#include "collective-conversion-keygen.h"
#include "modulus-diagnostics.h"
#include "ring-conversion.h"
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

bool EqualDCRTPoly(
    const lbcrypto::DCRTPoly& left,
    const lbcrypto::DCRTPoly& right) {

    if (left.GetFormat() != right.GetFormat()) {
        return false;
    }

    const auto& leftTowers =
        left.GetAllElements();

    const auto& rightTowers =
        right.GetAllElements();

    if (leftTowers.size() !=
        rightTowers.size()) {

        return false;
    }

    for (size_t towerIndex = 0;
         towerIndex < leftTowers.size();
         ++towerIndex) {

        if (leftTowers[towerIndex]
                .GetModulus() !=
            rightTowers[towerIndex]
                .GetModulus()) {

            return false;
        }

        const auto& leftValues =
            leftTowers[towerIndex]
                .GetValues();

        const auto& rightValues =
            rightTowers[towerIndex]
                .GetValues();

        if (leftValues.GetLength() !=
            rightValues.GetLength()) {

            return false;
        }

        for (size_t coefficientIndex = 0;
             coefficientIndex <
                 leftValues.GetLength();
             ++coefficientIndex) {

            if (leftValues[coefficientIndex] !=
                rightValues[coefficientIndex]) {

                return false;
            }
        }
    }

    return true;
}

bool CheckCommonAVectors(
    const std::vector<
        collective_conversion_keygen::
            MLWEtoRLWEKeyShare>& partyShares) {

    if (partyShares.size() < 2 ||
        partyShares[0].keys.empty()) {

        std::cerr
            << "common AVector check: FAIL, "
            << "invalid party shares\n";

        return false;
    }

    const size_t rank =
        partyShares[0].rank;

    for (const auto& partyShare :
         partyShares) {

        if (partyShare.rank != rank ||
            partyShare.keys.size() != rank) {

            std::cerr
                << "common AVector check: FAIL, "
                << "rank mismatch\n";

            return false;
        }
    }

    for (size_t splitIndex = 0;
         splitIndex < rank;
         ++splitIndex) {

        const auto& referenceAVector =
            partyShares[0]
                .keys[splitIndex]
                ->GetAVector();

        for (size_t partyIndex = 1;
             partyIndex < partyShares.size();
             ++partyIndex) {

            const auto& candidateAVector =
                partyShares[partyIndex]
                    .keys[splitIndex]
                    ->GetAVector();

            if (referenceAVector.size() !=
                candidateAVector.size()) {

                std::cerr
                    << "common AVector check: FAIL, "
                    << "splitIndex="
                    << splitIndex
                    << ", partyIndex="
                    << partyIndex
                    << ", vector-size mismatch\n";

                return false;
            }

            for (size_t partIndex = 0;
                 partIndex <
                     referenceAVector.size();
                 ++partIndex) {

                if (!EqualDCRTPoly(
                        referenceAVector[partIndex],
                        candidateAVector[partIndex])) {

                    std::cerr
                        << "common AVector check: FAIL, "
                        << "splitIndex="
                        << splitIndex
                        << ", partyIndex="
                        << partyIndex
                        << ", decompositionPart="
                        << partIndex
                        << '\n';

                    return false;
                }
            }
        }
    }

    std::cout
        << "common AVector check: PASS"
        << ", parties="
        << partyShares.size()
        << ", rank="
        << rank
        << '\n';

    return true;
}

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
                std::numeric_limits<double>::
                    infinity();
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

std::vector<double> MakeInput(
    size_t ringDimension) {

    std::vector<double> input(
        ringDimension);

    for (size_t index = 0;
         index < ringDimension;
         ++index) {

        const int centered =
            static_cast<int>(
                index % 17) -
            8;

        /* Power-of-two denominators yield negligible coefficient-rounding error at scale 2^40. */
        input[index] =
            static_cast<double>(centered) /
            128.0;
    }

    return input;
}

}  // namespace

int RunCollectiveConversionKeyTests() {
    try {
        constexpr uint32_t ringDimension =
            1u << 13;

        constexpr size_t partyCount = 3;
        constexpr size_t mlweRank = 4;

        constexpr uint32_t scaleBits = 60;
        constexpr uint32_t sigmaShareBits = 48;

        const double scalingFactor =
            coefficient_codec::
                ScalingFactorFromBits(
                    scaleBits);

        const double sigmaShare =
            std::ldexp(
                1.0,
                static_cast<int>(
                    sigmaShareBits));

        /* Expected decoded-domain standard deviation of threshold smudging. */
        const double expectedNoiseStandardDeviation =
            std::sqrt(
                static_cast<double>(
                    partyCount)) *
            sigmaShare /
            scalingFactor;

        /* Tolerance for direct threshold decryption. */
        const double directTolerance =
            10.0 *
            expectedNoiseStandardDeviation;

        /* Use twice the direct tolerance for the additional conversion-key noise in the R2M/M2R baseline. */
        const double conversionTolerance =
            2.0 *
            directTolerance;

        lbcrypto::CCParams<
            lbcrypto::CryptoContextCKKSRNS>
            parameters;

        /* Generate two Q towers. */
        parameters.SetMultiplicativeDepth(1);

        /* Q=q0*q1 with approximately 55 bits per tower. */
        parameters.SetFirstModSize(55);
        parameters.SetScalingModSize(55);

        parameters.SetBatchSize(
            ringDimension / 2);

        parameters.SetRingDim(
            ringDimension);

        /* Ask OpenFHE to check the QP size at N=8192 against its 128-bit security setting. */
        parameters.SetSecurityLevel(
            lbcrypto::HEStd_128_classic);

        parameters.SetScalingTechnique(
            lbcrypto::FIXEDMANUAL);

        parameters.SetSecretKeyDist(
            lbcrypto::UNIFORM_TERNARY);

        parameters.SetKeySwitchTechnique(
            lbcrypto::HYBRID);

        /* Use one Q tower per HYBRID digit to retain a single special-P tower. */
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
            << '\n'
            << "number of parties: "
            << partyCount
            << '\n'
            << "MLWE rank: "
            << mlweRank
            << '\n'
            << "small ring dimension: "
            << cc->GetRingDimension() /
                   mlweRank
            << '\n'
            << "scaling factor: 2^"
            << scaleBits
            << " = "
            << scalingFactor
            << '\n'
            << "sigma share: 2^"
            << sigmaShareBits
            << " = "
            << threshold_ckks::
                   GetSmudgingSigma(cc)
            << '\n'
            << "expected noise std: "
            << expectedNoiseStandardDeviation
            << '\n'
            << "direct tolerance: "
            << directTolerance
            << '\n'
            << "conversion tolerance: "
            << conversionTolerance
            << '\n';

        /* Generate party shares and a collective public key. The test co-locates the shares but never reconstructs their sum. */
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
                "unexpected number of secret-key shares");
        }

        const auto input =
            MakeInput(
                ringDimension);

        /* Encrypt the original RLWE ciphertext under the collective public key. */
        const auto originalCiphertext =
            threshold_ckks::
                EncryptCoefficients(
                    cc,
                    keySet.collectivePublicKey,
                    input,
                    scaleBits);

        /* First verify the collective key and coefficient codec by direct threshold decryption. */
        const auto directlyDecrypted =
            threshold_ckks::
                ThresholdDecryptCoefficients(
                    cc,
                    originalCiphertext,
                    keySet.secretKeyShares,
                    scalingFactor);

        bool allPassed = true;

        allPassed =
            CheckValues(
                "collective encrypt -> "
                "threshold decrypt",
                input,
                directlyDecrypted,
                directTolerance) &&
            allPassed;

        /* Convert the original RLWE ciphertext to k MLWE ciphertexts. */
        const auto mlweBundle =
            ring_conversion::RLWEtoMLWE(
                originalCiphertext,
                mlweRank);

        if (mlweBundle.ciphertexts.size() !=
            mlweRank) {

            throw std::runtime_error(
                "unexpected number of MLWE ciphertexts");
        }

        /* Party zero generates the lead switching share and common AVector template. */
        const auto leadKeyShare =
            collective_conversion_keygen::
                GenerateLeadMLWEtoRLWEKeyShare(
                    cc,
                    keySet.secretKeyShares[0],
                    mlweRank);

        std::vector<
            collective_conversion_keygen::
                MLWEtoRLWEKeyShare>
            partyKeyShares;

        partyKeyShares.reserve(
            partyCount);

        partyKeyShares.push_back(
            leadKeyShare);

        /* Other parties reuse the common AVector and generate their BVector contributions. */
        for (size_t partyIndex = 1;
             partyIndex < partyCount;
             ++partyIndex) {

            partyKeyShares.emplace_back(
                collective_conversion_keygen::
                    GenerateMainMLWEtoRLWEKeyShare(
                        cc,
                        keySet.secretKeyShares[
                            partyIndex],
                        leadKeyShare,
                        partyIndex));
        }

        const bool commonAVectorsPassed =
            CheckCommonAVectors(
                partyKeyShares);

        allPassed =
            commonAVectorsPassed &&
            allPassed;

        if (!commonAVectorsPassed) {
            throw std::runtime_error(
                "party switching-key shares do "
                "not use common AVectors");
        }

        /* Combine the shares without reconstructing the collective secret. */
        const auto collectiveConversionKeys =
            collective_conversion_keygen::
                AggregateMLWEtoRLWEKeyShares(
                    cc,
                    partyKeyShares,
                    keySet.collectivePublicKey);

        if (collectiveConversionKeys.size() !=
            mlweRank) {

            throw std::runtime_error(
                "unexpected number of collective "
                "conversion keys");
        }

        std::cout
            << "number of collective conversion keys: "
            << collectiveConversionKeys.size()
            << '\n';

        if (!collectiveConversionKeys.empty()) {
            std::cout
                << "HYBRID decomposition parts per key: "
                << collectiveConversionKeys[0]
                       ->GetAVector()
                       .size()
                << '\n';
        }

        /* Use the baseline collective switching key to recover RLWE under the original collective secret. */
        const auto restoredCiphertext =
            ring_conversion::MLWEtoRLWE(
                cc,
                mlweBundle,
                collectiveConversionKeys,
                originalCiphertext);

        /* Threshold-decrypt the recovered RLWE ciphertext with individual party shares. */
        const auto restoredValues =
            threshold_ckks::
                ThresholdDecryptCoefficients(
                    cc,
                    restoredCiphertext,
                    keySet.secretKeyShares,
                    scalingFactor);

        allPassed =
            CheckValues(
                "collective encrypt -> RLWE to MLWE "
                "-> collective MLWE to RLWE "
                "-> threshold decrypt",
                input,
                restoredValues,
                conversionTolerance) &&
            allPassed;

        if (!allPassed) {
            std::cerr
                << "\nOne or more collective "
                << "conversion-key tests failed.\n";

            return 1;
        }

        std::cout
            << "\nAll collective conversion-key "
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