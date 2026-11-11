#include "seeded-dense-rlwe.h"

#include "protocol-randomness.h"

#include "deterministic-randomness.h"
#include "rns-polynomial.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

ByteVector DenseAContext(const SetupId& setupId, RoundId roundId,
    ClientId clientId, std::size_t blockIndex) {
    return deterministic_randomness::V17LocalCiphertextAContext(
        setupId, roundId, clientId, blockIndex);
}

void ValidatePublicMaterial(const CommonSetupMaterial& material) {
    if (!material.rns.paramsQ ||
        material.rns.qTowerCount == 0 ||
        material.rns.paramsQ->GetParams().size() !=
            material.rns.qTowerCount) {

        throw std::invalid_argument(
            "seeded dense RLWE requires a nonempty protocol Q basis");
    }
}

void ValidateQPolynomial(
    const DCRTPoly& polynomial,
    const CommonSetupMaterial& material,
    const char* name) {

    rns_polynomial::ValidateBasis(
        polynomial,
        material.rns.paramsQ,
        name);
}

DCRTPoly SampleError(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    long double standardDeviation, bool certifiedV17) {

    if (!parameters ||
        !std::isfinite(standardDeviation) ||
        standardDeviation <= 0.0L ||
        standardDeviation > std::ldexp(1.0L, 59)) {

        throw std::invalid_argument("invalid dense encryption error sigma");
    }
    return SampleProtocolGaussianPolynomial(parameters, standardDeviation, certifiedV17);
}

SeededDenseRLWEPacket EncryptImpl(
    DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t blockIndex,
    std::size_t validValueCount,
    const Seed128& aSeed,
    bool compressB,
    std::uint32_t compressionBits,
    long double encryptionErrorStandardDeviation,
    const std::string& v17Path) {

    ValidatePublicMaterial(publicMaterial);
    if (!clientSecret.encryptionKeyQ) {
        throw std::invalid_argument("dense client secret key is null");
    }
    ValidateQPolynomial(messageQ, publicMaterial, "dense message");

    DCRTPoly secret = clientSecret.encryptionKeyQ->GetPrivateElement();
    ValidateQPolynomial(secret, publicMaterial, "dense client secret");

    const std::size_t ringDimension =
        publicMaterial.rns.paramsQ->GetRingDimension();
    if (validValueCount == 0 || validValueCount > ringDimension) {
        throw std::invalid_argument("dense block valid-value count is invalid");
    }

    DCRTPoly coefficientMessage = messageQ;
    coefficientMessage.SetFormat(Format::COEFFICIENT);
    for (const auto& tower : coefficientMessage.GetAllElements()) {
        const auto& messageValues = tower.GetValues();

        for (std::size_t index = validValueCount;
             index < ringDimension;
             ++index) {

            if (messageValues[index] != lbcrypto::NativeInteger(0)) {
                throw std::invalid_argument(
                    "dense plaintext has nonzero padding coefficients");
            }
        }
    }

    if (!v17Path.empty() && v17Path != "FULL-MODEL" &&
        v17Path != "AUX") {
        throw std::invalid_argument("invalid v17 dense ciphertext path");
    }
    const std::string aDomain = v17Path == "FULL-MODEL"
        ? "ENC-FULL-A" : v17Path == "AUX" ? "ENC-AUX-A" : "ENC-DENSE-A";
    DCRTPoly a = GenerateSeededUniformDCRT(
        publicMaterial.rns.paramsQ,
        aSeed,
        aDomain,
        DenseAContext(
            publicMaterial.setupId,
            roundId,
            clientSecret.clientId,
            blockIndex));

    messageQ.SetFormat(Format::EVALUATION);
    secret.SetFormat(Format::EVALUATION);
    DCRTPoly b = messageQ +
        SampleError(
            publicMaterial.rns.paramsQ,
            encryptionErrorStandardDeviation, publicMaterial.certifiedV17Gaussian) -
        a * secret;

    SeededDenseRLWEPacket packet;
    packet.setupId = publicMaterial.setupId;
    packet.roundId = roundId;
    packet.clientId = clientSecret.clientId;
    packet.blockIndex = blockIndex;
    packet.validValueCount = validValueCount;
    packet.ringDimension = ringDimension;
    packet.aSeed = aSeed;
    packet.v17Path = v17Path;
    packet.bIsCompressed = compressB;

    if (compressB) {
        packet.compressedB =
            coefficient_compression::CompressAllCoefficients(
                b,
                compressionBits);
    }
    else {
        packet.uncompressedB =
            rns_polynomial::InterpolateCoefficients(b);
    }
    return packet;
}

}  // namespace

SeededDenseRLWEPacket EncryptSeededDenseRLWE(
    DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t blockIndex,
    std::size_t validValueCount,
    const Seed128& aSeed,
    std::uint32_t compressionBits,
    long double encryptionErrorStandardDeviation) {

    return EncryptImpl(
        std::move(messageQ),
        clientSecret,
        publicMaterial,
        roundId,
        blockIndex,
        validValueCount,
        aSeed,
        true,
        compressionBits,
        encryptionErrorStandardDeviation,
        {});
}

SeededDenseRLWEPacket EncryptSeededDenseRLWEUncompressed(
    DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t blockIndex,
    std::size_t validValueCount,
    const Seed128& aSeed,
    long double encryptionErrorStandardDeviation,
    const std::string& v17Path) {

    return EncryptImpl(
        std::move(messageQ),
        clientSecret,
        publicMaterial,
        roundId,
        blockIndex,
        validValueCount,
        aSeed,
        false,
        0,
        encryptionErrorStandardDeviation,
        v17Path);
}

SeededDenseRLWEPacket CompressSeededDenseRLWEPacket(
    const SeededDenseRLWEPacket& uncompressedPacket,
    const CommonSetupMaterial& publicMaterial,
    std::uint32_t compressionBits,
    bool stochastic) {

    if (uncompressedPacket.bIsCompressed) {
        throw std::invalid_argument(
            "c1 derivation requires an uncompressed dense packet");
    }
    const RestoredDenseRLWECiphertext restored =
        RestoreSeededDenseRLWE(uncompressedPacket, publicMaterial);
    SeededDenseRLWEPacket result = uncompressedPacket;
    result.bIsCompressed = true;
    result.compressedB = stochastic
        ? coefficient_compression::CompressAllCoefficientsStochastic(
              restored.b, compressionBits)
        : coefficient_compression::CompressAllCoefficients(
              restored.b, compressionBits);
    result.uncompressedB.clear();
    return result;
}

RestoredDenseRLWECiphertext RestoreSeededDenseRLWE(
    const SeededDenseRLWEPacket& packet,
    const CommonSetupMaterial& publicMaterial) {

    ValidatePublicMaterial(publicMaterial);
    const std::size_t ringDimension =
        publicMaterial.rns.paramsQ->GetRingDimension();
    if (!(packet.setupId == publicMaterial.setupId) ||
        packet.ringDimension != ringDimension ||
        packet.validValueCount == 0 ||
        packet.validValueCount > ringDimension) {

        throw std::invalid_argument("dense packet metadata does not match setup");
    }

    RestoredDenseRLWECiphertext result;
    result.setupId = packet.setupId;
    result.roundId = packet.roundId;
    result.clientId = packet.clientId;
    result.blockIndex = packet.blockIndex;
    result.validValueCount = packet.validValueCount;
    if (!packet.v17Path.empty() && packet.v17Path != "FULL-MODEL" &&
        packet.v17Path != "AUX") {
        throw std::invalid_argument("invalid v17 dense ciphertext path");
    }
    const std::string aDomain = packet.v17Path == "FULL-MODEL"
        ? "ENC-FULL-A" : packet.v17Path == "AUX"
            ? "ENC-AUX-A" : "ENC-DENSE-A";
    result.a = GenerateSeededUniformDCRT(
        publicMaterial.rns.paramsQ,
        packet.aSeed,
        aDomain,
        DenseAContext(
            packet.setupId,
            packet.roundId,
            packet.clientId,
            packet.blockIndex));

    if (packet.bIsCompressed) {
        if (!packet.uncompressedB.empty() ||
            packet.compressedB.coefficientCount != ringDimension) {

            throw std::invalid_argument("compressed dense b payload is invalid");
        }
        result.b = coefficient_compression::DecompressAllCoefficients(
            publicMaterial.rns.paramsQ,
            packet.compressedB);
    }
    else {
        if (packet.uncompressedB.size() != ringDimension ||
            packet.compressedB.coefficientCount != 0 ||
            !packet.compressedB.bytes.empty()) {

            throw std::invalid_argument("uncompressed dense b payload is invalid");
        }

        const lbcrypto::BigInteger modulus =
            publicMaterial.rns.paramsQ->GetModulus();
        for (const auto& coefficient : packet.uncompressedB) {
            if (coefficient >= modulus) {
                throw std::invalid_argument(
                    "dense b coefficient is outside composite Q");
            }
        }
        result.b = rns_polynomial::FromCompositeCoefficients(
            publicMaterial.rns.paramsQ,
            packet.uncompressedB);
    }

    result.b.SetFormat(Format::EVALUATION);
    return result;
}

}  // namespace federated
