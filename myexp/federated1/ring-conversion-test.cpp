#include "ring-conversion-test.h"
#include "coefficient-codec.h"
#include "ring-conversion.h"

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

std::vector<double> ExpectedSplit(
    const std::vector<double>& input,
    size_t rank,
    size_t part) {

    if (rank == 0 ||
        part >= rank ||
        input.size() % rank != 0) {

        throw std::invalid_argument(
            "ExpectedSplit: invalid rank or part");
    }

    std::vector<double> result(
        input.size() / rank);

    for (size_t index = 0;
         index < result.size();
         ++index) {

        result[index] =
            input[index * rank + part];
    }

    return result;
}

std::vector<double> ExpectedMultX(
    const std::vector<double>& input) {

    if (input.empty()) {
        throw std::invalid_argument(
            "ExpectedMultX: input is empty");
    }

    std::vector<double> result(
        input.size());

    result[0] = -input.back();

    for (size_t index = 1;
         index < input.size();
         ++index) {

        result[index] =
            input[index - 1];
    }

    return result;
}

std::vector<double> ExpectedEmbedding(
    const std::vector<double>& smallInput,
    size_t rank) {

    std::vector<double> result(
        smallInput.size() * rank,
        0.0);

    for (size_t index = 0;
         index < smallInput.size();
         ++index) {

        result[index * rank] =
            smallInput[index];
    }

    return result;
}

std::vector<double> DecryptMLWECiphertext(
    const ring_conversion::MLWECiphertext& ciphertext,
    const std::vector<lbcrypto::DCRTPoly>& splitSecret,
    double scalingFactor) {

    if (ciphertext.rank == 0 ||
        ciphertext.a.size() != ciphertext.rank) {

        throw std::invalid_argument(
            "DecryptMLWECiphertext: invalid MLWE ciphertext");
    }

    if (splitSecret.size() != ciphertext.rank) {
        throw std::invalid_argument(
            "DecryptMLWECiphertext: secret rank mismatch");
    }

    lbcrypto::DCRTPoly result =
        ciphertext.b;

    result.SetFormat(
        Format::EVALUATION);

    for (size_t index = 0;
         index < ciphertext.rank;
         ++index) {

        lbcrypto::DCRTPoly a =
            ciphertext.a[index];

        lbcrypto::DCRTPoly secret =
            splitSecret[index];

        a.SetFormat(
            Format::EVALUATION);

        secret.SetFormat(
            Format::EVALUATION);

        result += a * secret;
    }

    return coefficient_codec::Decode(
        result,
        scalingFactor);
}

}  // namespace

int RunRingConversionTests() {
    try {
        constexpr uint32_t ringDimension =
            1u << 13;  // 8192

        constexpr size_t mlweRank = 4;
        constexpr uint32_t scaleBits = 25;
        constexpr uint32_t multiplicativeDepth = 0;
        constexpr uint32_t firstModSize = 50;
        constexpr uint32_t scalingModSize = 40;
        constexpr uint32_t batchSize =
            ringDimension / 2;

        lbcrypto::CCParams<
            lbcrypto::CryptoContextCKKSRNS>
            parameters;

        parameters.SetMultiplicativeDepth(
            multiplicativeDepth);

        parameters.SetFirstModSize(
            firstModSize);

        parameters.SetScalingModSize(
            scalingModSize);

        parameters.SetBatchSize(
            batchSize);

        parameters.SetRingDim(
            ringDimension);

        parameters.SetSecurityLevel(
            lbcrypto::HEStd_NotSet);

        parameters.SetScalingTechnique(
            lbcrypto::FIXEDMANUAL);

        auto cc =
            lbcrypto::GenCryptoContext(
                parameters);

        cc->Enable(lbcrypto::PKE);
        cc->Enable(lbcrypto::KEYSWITCH);

        const double scalingFactor =
            coefficient_codec::ScalingFactorFromBits(
                scaleBits);

        std::vector<double> input(
            ringDimension);

        for (size_t index = 0;
             index < input.size();
             ++index) {

            const int centered =
                static_cast<int>(index % 17) - 8;

            input[index] =
                static_cast<double>(centered) /
                16.0;
        }

        std::cout
            << std::fixed
            << std::setprecision(10);

        std::cout
            << "ring dimension: "
            << cc->GetRingDimension()
            << '\n';

        std::cout
            << "MLWE rank: "
            << mlweRank
            << '\n';

        std::cout
            << "small ring dimension: "
            << cc->GetRingDimension() / mlweRank
            << '\n';

        std::cout
            << "number of input values: "
            << input.size()
            << '\n';

        std::cout
            << "scaling factor: 2^"
            << scaleBits
            << " = "
            << scalingFactor
            << '\n';

        const double exactCodecTolerance =
            0.5 / scalingFactor + 1e-12;

        const double encryptionTolerance =
            1e-4;

        const double keySwitchTolerance =
            1e-3;

        bool allPassed = true;

        // ---------------------------------------------------------
        // 1. coefficient Encode -> Decode
        // ---------------------------------------------------------
        const auto encoded =
            coefficient_codec::Encode(
                cc,
                input,
                scalingFactor);

        const auto directlyDecoded =
            coefficient_codec::Decode(
                encoded,
                scalingFactor);

        allPassed =
            CheckValues(
                "coefficient encode -> decode",
                input,
                directlyDecoded,
                exactCodecTolerance) &&
            allPassed;

        // ---------------------------------------------------------
        // 2. Split -> Combine
        // ---------------------------------------------------------
        const auto splitEncoded =
            ring_conversion::Split(
                encoded,
                mlweRank);

        const auto combinedEncoded =
            ring_conversion::Combine(
                splitEncoded);

        const auto combinedDecoded =
            coefficient_codec::Decode(
                combinedEncoded,
                scalingFactor);

        allPassed =
            CheckValues(
                "Split -> Combine",
                input,
                combinedDecoded,
                exactCodecTolerance) &&
            allPassed;

        // ---------------------------------------------------------
        // 3. MultX
        // ---------------------------------------------------------
        const auto splitZeroExpected =
            ExpectedSplit(
                input,
                mlweRank,
                0);

        const auto multipliedByX =
            ring_conversion::MultX(
                splitEncoded[0]);

        const auto multipliedByXDecoded =
            coefficient_codec::Decode(
                multipliedByX,
                scalingFactor);

        const auto multipliedByXExpected =
            ExpectedMultX(
                splitZeroExpected);

        allPassed =
            CheckValues(
                "MultX",
                multipliedByXExpected,
                multipliedByXDecoded,
                exactCodecTolerance) &&
            allPassed;

        // ---------------------------------------------------------
        // 4. RingEmbedding
        // ---------------------------------------------------------
        constexpr size_t embeddingPart = 1;

        const auto embeddingInputExpected =
            ExpectedSplit(
                input,
                mlweRank,
                embeddingPart);

        const auto embedded =
            ring_conversion::RingEmbedding(
                splitEncoded[embeddingPart],
                mlweRank);

        const auto embeddedDecoded =
            coefficient_codec::Decode(
                embedded,
                scalingFactor);

        const auto embeddedExpected =
            ExpectedEmbedding(
                embeddingInputExpected,
                mlweRank);

        allPassed =
            CheckValues(
                "RingEmbedding",
                embeddedExpected,
                embeddedDecoded,
                exactCodecTolerance) &&
            allPassed;

        // ---------------------------------------------------------
        // 5. Standard encryption and decryption.
        // ---------------------------------------------------------
        const auto keyPair =
            cc->KeyGen();

        if (!keyPair.good()) {
            throw std::runtime_error(
                "KeyGen failed");
        }

        const auto plaintext =
            coefficient_codec::MakePlaintext(
                cc,
                input,
                scaleBits);

        const auto ciphertext =
            cc->Encrypt(
                keyPair.publicKey,
                plaintext);

        const auto normallyDecrypted =
            coefficient_codec::Decrypt(
                cc,
                ciphertext,
                keyPair.secretKey,
                scalingFactor);

        allPassed =
            CheckValues(
                "encode -> encrypt -> decrypt -> decode",
                input,
                normallyDecrypted,
                encryptionTolerance) &&
            allPassed;

        // ---------------------------------------------------------
        // 6. RLWE -> MLWE
        // ---------------------------------------------------------
        const auto mlweBundle =
            ring_conversion::RLWEtoMLWE(
                ciphertext,
                mlweRank);

        std::cout
            << "number of converted MLWE ciphertexts: "
            << mlweBundle.ciphertexts.size()
            << '\n';

        const auto splitSecret =
            ring_conversion::Split(
                keyPair.secretKey
                    ->GetPrivateElement(),
                mlweRank);

        for (size_t index = 0;
             index < mlweBundle.ciphertexts.size();
             ++index) {

            const auto expectedPart =
                ExpectedSplit(
                    input,
                    mlweRank,
                    index);

            const auto decryptedPart =
                DecryptMLWECiphertext(
                    mlweBundle.ciphertexts[index],
                    splitSecret,
                    scalingFactor);

            const std::string label =
                "RLWE -> MLWE ciphertext " +
                std::to_string(index);

            allPassed =
                CheckValues(
                    label,
                    expectedPart,
                    decryptedPart,
                    encryptionTolerance) &&
                allPassed;
        }

        // ---------------------------------------------------------
        // 7. Generate baseline MLWE-to-RLWE conversion keys.
        // ---------------------------------------------------------
        std::cout
            << "generating "
            << mlweRank
            << " MLWE -> RLWE key-switching keys..."
            << '\n';

        const auto conversionKeys =
            ring_conversion::GenerateMLWEtoRLWEKeys(
                cc,
                keyPair.secretKey,
                mlweRank);

        // ---------------------------------------------------------
        // 8. RLWE -> MLWE -> RLWE
        // ---------------------------------------------------------
        const auto restoredCiphertext =
            ring_conversion::MLWEtoRLWE(
                cc,
                mlweBundle,
                conversionKeys,
                ciphertext);

        const auto restoredValues =
            coefficient_codec::Decrypt(
                cc,
                restoredCiphertext,
                keyPair.secretKey,
                scalingFactor);

        allPassed =
            CheckValues(
                "RLWE -> MLWE -> RLWE",
                input,
                restoredValues,
                keySwitchTolerance) &&
            allPassed;

        if (allPassed) {
            std::cout
                << "All coefficient codec and ring conversion tests passed."
                << '\n';

            return 0;
        }

        std::cerr
            << "One or more tests failed."
            << '\n';

        return 1;
    }
    catch (const std::exception& exception) {
        std::cerr
            << "Error: "
            << exception.what()
            << '\n';

        return 1;
    }
}