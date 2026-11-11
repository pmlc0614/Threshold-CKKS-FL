#include "coefficient-codec-test.h"
#include "coefficient-codec.h"

#include "openfhe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
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
        std::cerr << label
                  << ": size mismatch: expected=" << expected.size()
                  << ", actual=" << actual.size()
                  << '\n';
        return false;
    }

    bool passed = true;
    size_t failureCount = 0;
    constexpr size_t maxFailuresToPrint = 10;

    double maxError = 0.0;
    size_t maxErrorIndex = 0;

    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(expected[i] - actual[i]);

        if (error > maxError) {
            maxError      = error;
            maxErrorIndex = i;
        }

        if (error > tolerance) {
            passed = false;

            if (failureCount < maxFailuresToPrint) {
                std::cerr
                    << "  index=" << i
                    << ", expected=" << expected[i]
                    << ", actual=" << actual[i]
                    << ", error=" << error
                    << '\n';
            }

            ++failureCount;
        }
    }

    std::cout
        << label << ": "
        << (passed ? "PASS" : "FAIL")
        << ", checked=" << expected.size()
        << ", failures=" << failureCount
        << ", maxError=" << maxError
        << ", maxErrorIndex=" << maxErrorIndex
        << '\n';

    return passed;
}

}  // namespace

int RunCoefficientCodecTests() {
    try {
        // constexpr uint32_t ringDimension = 16;
        constexpr uint32_t ringDimension = 1u << 13;
        constexpr uint32_t scaleBits = 25;

        const double scalingFactor =
            coefficient_codec::
                ScalingFactorFromBits(
                    scaleBits);

        /* Small ring dimension is a codec test fixture, not a security parameter set. */
        lbcrypto::CCParams<
            lbcrypto::CryptoContextCKKSRNS>
            parameters;

        parameters.SetMultiplicativeDepth(0);
        parameters.SetFirstModSize(50);
        parameters.SetScalingModSize(40);

        parameters.SetBatchSize(
            ringDimension / 2);

        parameters.SetRingDim(
            ringDimension);

        parameters.SetSecurityLevel(
            lbcrypto::HEStd_NotSet);

        parameters.SetScalingTechnique(
            lbcrypto::FIXEDMANUAL);

        auto cc =
            lbcrypto::GenCryptoContext(
                parameters);

        cc->Enable(
            lbcrypto::PKE);

        std::vector<double> input(ringDimension);

        for (size_t i = 0; i < input.size(); ++i) {
            const int centered = static_cast<int>(i % 17) - 8;
            input[i] = static_cast<double>(centered) / 16.0;
        }

        /* The encoder zero-pads short inputs; expect N decoded coefficients. */
        const std::vector<double> expected = input;

        std::cout
            << std::fixed
            << std::setprecision(10);

        std::cout
            << "ring dimension: "
            << cc->GetRingDimension()
            << '\n';

        std::cout
            << "number of input values: "
            << input.size()
            << '\n';

        std::cout
            << "number of decoded values: "
            << expected.size()
            << '\n';

        std::cout
            << "scaling factor: 2^"
            << scaleBits
            << " = "
            << scalingFactor
            << '\n';

        bool allPassed = true;

        /* Test coefficient encode/decode without encryption. */
        const auto encoded =
            coefficient_codec::Encode(
                cc,
                input,
                scalingFactor);

        const auto decoded =
            coefficient_codec::Decode(
                encoded,
                scalingFactor);

        allPassed &=
            CheckValues(
                "encode -> decode",
                expected,
                decoded,
                1.0 / scalingFactor);

        bool rejectedUnencodableValue = false;
        try {
            (void)coefficient_codec::Encode(
                cc,
                {std::ldexp(1.0, 30)},
                scalingFactor);
        }
        catch (const std::overflow_error&) {
            rejectedUnencodableValue = true;
        }
        std::cout
            << (rejectedUnencodableValue ? "[PASS] " : "[FAIL] ")
            << "local input outside the scaled centered-Q range is rejected\n";
        allPassed &= rejectedUnencodableValue;

        /* Test actual public-key encryption and secret-key decryption. */
        const auto keyPair =
            cc->KeyGen();

        if (!keyPair.good()) {
            throw std::runtime_error(
                "key generation failed");
        }

        auto plaintext =
            coefficient_codec::MakePlaintext(
                cc,
                input,
                scaleBits);

        const auto ciphertext =
            cc->Encrypt(
                keyPair.publicKey,
                plaintext);

        if (!ciphertext) {
            throw std::runtime_error(
                "encryption failed");
        }

        const auto decrypted =
            coefficient_codec::Decrypt(
                cc,
                ciphertext,
                keyPair.secretKey,
                scalingFactor);

        /* The result includes encoding-rounding error and RLWE noise. */
        allPassed &=
            CheckValues(
                "encode -> encrypt -> decrypt -> decode",
                expected,
                decrypted,
                1e-4);

        if (!allPassed) {
            std::cerr
                << "\nOne or more coefficient "
                << "codec tests failed.\n";

            return 1;
        }

        std::cout
            << "\nAll coefficient codec "
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
