#include "threshold-ckks.h"

#include "coefficient-codec.h"
#include "openfhe.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace threshold_ckks {
namespace {

void ValidateCryptoContext(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc) {

    if (!cc) {
        throw std::invalid_argument(
            "crypto context is null");
    }
}

void ValidateCiphertext(
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& ciphertext,
    const char* argumentName) {

    if (!ciphertext) {
        throw std::invalid_argument(
            std::string(argumentName) +
            " is null");
    }
}

void ValidateScalingFactor(
    double scalingFactor) {

    if (!std::isfinite(scalingFactor) ||
        scalingFactor <= 0.0) {

        throw std::invalid_argument(
            "scalingFactor must be finite and positive");
    }
}

std::shared_ptr<
    lbcrypto::CryptoParametersCKKSRNS>
GetCKKSCryptoParameters(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc) {

    ValidateCryptoContext(cc);

    auto cryptoParameters =
        std::dynamic_pointer_cast<
            lbcrypto::CryptoParametersCKKSRNS>(
                cc->GetCryptoParameters());

    if (!cryptoParameters) {
        throw std::runtime_error(
            "the crypto context does not use "
            "CKKS-RNS parameters");
    }

    return cryptoParameters;
}

/* OpenFHE multiparty decryption returns a vector; these single-ciphertext helpers require exactly one partial. */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
ExtractSinglePartialDecryption(
    std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>
        partialDecryptions,
    const char* operationName) {

    if (partialDecryptions.size() != 1) {
        throw std::runtime_error(
            std::string(operationName) +
            " returned an unexpected number "
            "of partial decryptions");
    }

    if (!partialDecryptions[0]) {
        throw std::runtime_error(
            std::string(operationName) +
            " returned a null partial decryption");
    }

    if (partialDecryptions[0]
            ->GetElements()
            .size() != 1) {

        throw std::runtime_error(
            std::string(operationName) +
            " did not return a one-element "
            "partial ciphertext");
    }

    return std::move(
        partialDecryptions[0]);
}

}  // namespace

void ConfigureSmudgingSigma(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    double sigmaShare) {

    ValidateCryptoContext(cc);

    if (!std::isfinite(sigmaShare) ||
        sigmaShare <= 0.0) {

        throw std::invalid_argument(
            "sigmaShare must be finite and positive");
    }

    /* This OpenFHE baseline Gaussian generator supports sigmas up to approximately 59 bits; the v17 sampler is separate. */
    if (std::log2(sigmaShare) > 59.0) {
        throw std::invalid_argument(
            "sigmaShare exceeds the supported "
            "59-bit Gaussian sampler range");
    }

    auto cryptoParameters =
        GetCKKSCryptoParameters(cc);

    /* Use FIXED_NOISE_MULTIPARTY: NOISE_FLOODING_MULTIPARTY selects a different branch in rns-multiparty.cpp. */
    cryptoParameters->SetMultipartyMode(
        lbcrypto::FIXED_NOISE_MULTIPARTY);

    /* Select GetFloodingDiscreteGaussianGenerator rather than the fixed MP_SD constant. */
    cryptoParameters->SetDecryptionNoiseMode(
        lbcrypto::NOISE_FLOODING_DECRYPT);

    cryptoParameters->SetExecutionMode(
        lbcrypto::EXEC_EVALUATION);

    /* Set the Gaussian generator's standard deviation. */
    cryptoParameters
        ->SetFloodingDistributionParameter(
            sigmaShare);
}

double GetSmudgingSigma(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc) {

    const auto cryptoParameters =
        GetCKKSCryptoParameters(cc);

    return cryptoParameters
        ->GetFloodingDistributionParameter();
}

CollectiveKeySet GenerateCollectiveKeys(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    size_t partyCount) {

    ValidateCryptoContext(cc);

    if (partyCount < 2) {
        throw std::invalid_argument(
            "partyCount must be at least 2");
    }

    CollectiveKeySet result;

    result.secretKeyShares.reserve(
        partyCount);

    /* The first client creates the initial public key and its secret share s_0. */
    auto currentKeyPair =
        cc->KeyGen();

    if (!currentKeyPair.good()) {
        throw std::runtime_error(
            "initial key generation failed");
    }

    result.secretKeyShares.push_back(
        currentKeyPair.secretKey);

    auto currentPublicKey =
        currentKeyPair.publicKey;

    /* Each MultipartyKeyGen adds a new share; the final public key corresponds to s=sum_i s_i. */
    for (size_t partyIndex = 1;
         partyIndex < partyCount;
         ++partyIndex) {

        currentKeyPair =
            cc->MultipartyKeyGen(
                currentPublicKey);

        if (!currentKeyPair.good()) {
            throw std::runtime_error(
                "multiparty key generation failed "
                "for party " +
                std::to_string(partyIndex));
        }

        result.secretKeyShares.push_back(
            currentKeyPair.secretKey);

        currentPublicKey =
            currentKeyPair.publicKey;
    }

    if (!currentPublicKey) {
        throw std::runtime_error(
            "collective public key is null");
    }

    result.collectivePublicKey =
        std::move(currentPublicKey);

    return result;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
EncryptCoefficients(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PublicKey<
        lbcrypto::DCRTPoly>& collectivePublicKey,
    const std::vector<double>& values,
    uint32_t scaleBits) {

    ValidateCryptoContext(cc);

    if (!collectivePublicKey) {
        throw std::invalid_argument(
            "collective public key is null");
    }

    const auto plaintext =
        coefficient_codec::MakePlaintext(
            cc,
            values,
            scaleBits);

    if (!plaintext) {
        throw std::runtime_error(
            "coefficient plaintext creation failed");
    }

    auto ciphertext =
        cc->Encrypt(
            collectivePublicKey,
            plaintext);

    if (!ciphertext) {
        throw std::runtime_error(
            "encryption with the collective "
            "public key failed");
    }

    return ciphertext;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
AggregateCiphertexts(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>&
        ciphertexts) {

    ValidateCryptoContext(cc);

    if (ciphertexts.empty()) {
        throw std::invalid_argument(
            "ciphertexts must not be empty");
    }

    for (size_t index = 0;
         index < ciphertexts.size();
         ++index) {

        if (!ciphertexts[index]) {
            throw std::invalid_argument(
                "ciphertexts contains a null "
                "ciphertext at index " +
                std::to_string(index));
        }
    }

    /* EvalAdd returns a new ciphertext without modifying its inputs. */
    auto aggregate =
        ciphertexts[0];

    for (size_t index = 1;
         index < ciphertexts.size();
         ++index) {

        aggregate =
            cc->EvalAdd(
                aggregate,
                ciphertexts[index]);

        if (!aggregate) {
            throw std::runtime_error(
                "ciphertext aggregation failed "
                "at index " +
                std::to_string(index));
        }
    }

    return aggregate;
}

std::vector<
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>
CreatePartialDecryptions(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& ciphertext,
    const std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>&
        secretKeyShares) {

    ValidateCryptoContext(cc);
    ValidateCiphertext(
        ciphertext,
        "ciphertext");

    if (secretKeyShares.size() < 2) {
        throw std::invalid_argument(
            "at least two secret key shares "
            "are required");
    }

    for (size_t partyIndex = 0;
         partyIndex < secretKeyShares.size();
         ++partyIndex) {

        if (!secretKeyShares[partyIndex]) {
            throw std::invalid_argument(
                "secretKeyShares contains a null "
                "key at index " +
                std::to_string(partyIndex));
        }
    }

    std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>
        partialDecryptions;

    partialDecryptions.reserve(
        secretKeyShares.size());

    /* Only one client generates the lead share, which includes c_0. */
    auto leadResult =
        cc->MultipartyDecryptLead(
            {ciphertext},
            secretKeyShares[0]);

    partialDecryptions.push_back(
        ExtractSinglePartialDecryption(
            std::move(leadResult),
            "MultipartyDecryptLead"));

    /* Other clients generate main shares containing c_1*s_i. */
    for (size_t partyIndex = 1;
         partyIndex < secretKeyShares.size();
         ++partyIndex) {

        auto mainResult =
            cc->MultipartyDecryptMain(
                {ciphertext},
                secretKeyShares[partyIndex]);

        partialDecryptions.push_back(
            ExtractSinglePartialDecryption(
                std::move(mainResult),
                "MultipartyDecryptMain"));
    }

    return partialDecryptions;
}

lbcrypto::Poly FusePartialDecryptions(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>&
        partialDecryptions) {

    ValidateCryptoContext(cc);

    if (partialDecryptions.size() < 2) {
        throw std::invalid_argument(
            "at least two partial decryptions "
            "are required");
    }

    for (size_t index = 0;
         index < partialDecryptions.size();
         ++index) {

        ValidateCiphertext(
            partialDecryptions[index],
            "partial decryption");

        if (partialDecryptions[index]
                ->GetElements()
                .size() != 1) {

            throw std::invalid_argument(
                "each partial decryption must "
                "contain exactly one polynomial");
        }
    }

    lbcrypto::Poly fusedPolynomial;

    /* Use the Poly fusion overload to add DCRT shares and CRT-interpolate, bypassing the standard CKKS slot decoder. */
    const lbcrypto::DecryptResult result =
        cc->GetScheme()
            ->MultipartyDecryptFusion(
                partialDecryptions,
                &fusedPolynomial);

    if (!result.isValid) {
        throw std::runtime_error(
            "multiparty decryption fusion failed");
    }

    return fusedPolynomial;
}

std::vector<double> FuseAndDecodeCoefficients(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>&
        partialDecryptions,
    double scalingFactor) {

    ValidateScalingFactor(
        scalingFactor);

    auto fusedPolynomial =
        FusePartialDecryptions(
            cc,
            partialDecryptions);

    return coefficient_codec::Decode(
        std::move(fusedPolynomial),
        scalingFactor);
}

std::vector<double> ThresholdDecryptCoefficients(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& ciphertext,
    const std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>&
        secretKeyShares,
    double scalingFactor) {

    ValidateScalingFactor(
        scalingFactor);

    auto partialDecryptions =
        CreatePartialDecryptions(
            cc,
            ciphertext,
            secretKeyShares);

    return FuseAndDecodeCoefficients(
        cc,
        partialDecryptions,
        scalingFactor);
}

}  // namespace threshold_ckks