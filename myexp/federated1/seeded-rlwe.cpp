#include "seeded-rlwe.h"

#include "protocol-randomness.h"

#include "deterministic-randomness.h"
#include "rns-polynomial.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

ByteVector EncryptionAContext(const SetupId& setupId, RoundId roundId,
    ClientId clientId, std::size_t blockIndex) {
    return deterministic_randomness::V17LocalCiphertextAContext(
        setupId, roundId, clientId, blockIndex);
}

void ValidatePublicMaterial(const CommonSetupMaterial& publicMaterial) {
    const auto& rns = publicMaterial.rns;

    if (!rns.paramsQ ||
        rns.qTowerCount == 0 ||
        rns.paramsQ->GetParams().size() != rns.qTowerCount ||
        publicMaterial.descriptor.mode != ProtocolMode::SP ||
        publicMaterial.mlweRank == 0) {

        throw std::invalid_argument(
            "seeded sparse RLWE requires an SP setup and nonempty Q basis");
    }

    const std::size_t ringDimension = rns.paramsQ->GetRingDimension();
    if (ringDimension == 0 ||
        ringDimension % publicMaterial.mlweRank != 0) {

        throw std::invalid_argument(
            "ring dimension must be divisible by the MLWE rank");
    }
}

void ValidateQPolynomial(
    DCRTPoly polynomial,
    const CommonSetupMaterial& publicMaterial,
    const char* name) {

    rns_polynomial::ValidateBasis(
        polynomial,
        publicMaterial.rns.paramsQ,
        name);
}

void ValidateInactiveMessageCoefficients(
    DCRTPoly messageQ,
    std::size_t mlweRank,
    std::size_t activeStreamCount) {

    messageQ.SetFormat(Format::COEFFICIENT);
    for (const auto& tower : messageQ.GetAllElements()) {
        const auto& values = tower.GetValues();

        for (std::size_t index = 0;
             index < values.GetLength();
             ++index) {

            if (index % mlweRank >= activeStreamCount &&
                values[index] != lbcrypto::NativeInteger(0)) {

                throw std::invalid_argument(
                    "message has a nonzero coefficient in an omitted stream");
            }
        }
    }
}

DCRTPoly SampleEncryptionError(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    long double standardDeviation, bool certifiedV17) {

    if (!parameters ||
        !std::isfinite(standardDeviation) ||
        standardDeviation <= 0.0L ||
        standardDeviation > std::ldexp(1.0L, 59)) {

        throw std::invalid_argument(
            "invalid seeded RLWE error standard deviation");
    }

    return SampleProtocolGaussianPolynomial(parameters, standardDeviation, certifiedV17);
}

SeededSparseRLWEPacket EncryptSeededSparseRLWEImpl(
    DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t packGroupIndex,
    std::size_t activeStreamCount,
    const Seed128& aSeed,
    bool compressB,
    std::uint32_t compressionBits,
    long double encryptionErrorStandardDeviation,
    bool v17Domain) {

    ValidatePublicMaterial(publicMaterial);

    if (!clientSecret.encryptionKeyQ) {
        throw std::invalid_argument("client Q secret key is null");
    }

    ValidateQPolynomial(messageQ, publicMaterial, "message");

    DCRTPoly secretQ = clientSecret.encryptionKeyQ->GetPrivateElement();
    ValidateQPolynomial(secretQ, publicMaterial, "client secret");

    const std::size_t ringDimension =
        publicMaterial.rns.paramsQ->GetRingDimension();

    const std::vector<std::size_t> selectedIndices =
        SparseStreamCoefficientIndices(
            ringDimension,
            publicMaterial.mlweRank,
            activeStreamCount);

    ValidateInactiveMessageCoefficients(
        messageQ,
        publicMaterial.mlweRank,
        activeStreamCount);

    const ByteVector context = EncryptionAContext(
        publicMaterial.setupId,
        roundId,
        clientSecret.clientId,
        packGroupIndex);

    DCRTPoly a = GenerateSeededUniformDCRT(
        publicMaterial.rns.paramsQ,
        aSeed,
        v17Domain ? "ENC-SP-A" : "ENC-A",
        context);

    messageQ.SetFormat(Format::EVALUATION);
    secretQ.SetFormat(Format::EVALUATION);

    const DCRTPoly error = SampleEncryptionError(
        publicMaterial.rns.paramsQ,
        encryptionErrorStandardDeviation, publicMaterial.certifiedV17Gaussian);

    DCRTPoly b = messageQ + error - a * secretQ;

    SeededSparseRLWEPacket packet;
    packet.setupId = publicMaterial.setupId;
    packet.roundId = roundId;
    packet.clientId = clientSecret.clientId;
    packet.packGroupIndex = packGroupIndex;
    packet.ringDimension = ringDimension;
    packet.mlweRank = publicMaterial.mlweRank;
    packet.activeStreamCount = activeStreamCount;
    packet.aSeed = aSeed;
    packet.v17Domain = v17Domain;
    packet.bIsCompressed = compressB;

    if (compressB) {
        packet.compressedB =
            coefficient_compression::CompressSelectedCoefficients(
                b,
                compressionBits,
                selectedIndices);
    }
    else {
        packet.uncompressedB =
            rns_polynomial::InterpolateCoefficientsAtIndices(
                b,
                selectedIndices);
    }

    return packet;
}

}  // namespace

std::vector<std::size_t> SparseStreamCoefficientIndices(
    std::size_t ringDimension,
    std::size_t mlweRank,
    std::size_t activeStreamCount) {

    if (ringDimension == 0 ||
        mlweRank == 0 ||
        ringDimension % mlweRank != 0) {

        throw std::invalid_argument(
            "invalid ring dimension or MLWE rank for sparse streams");
    }

    if (activeStreamCount == 0 ||
        activeStreamCount > mlweRank) {

        throw std::invalid_argument(
            "active stream count must be in [1, mlweRank]");
    }

    std::vector<std::size_t> indices;
    indices.reserve(
        ringDimension / mlweRank * activeStreamCount);

    for (std::size_t base = 0;
         base < ringDimension;
         base += mlweRank) {

        for (std::size_t stream = 0;
             stream < activeStreamCount;
             ++stream) {

            indices.push_back(base + stream);
        }
    }

    return indices;
}

SeededSparseRLWEPacket EncryptSeededSparseRLWE(
    DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t packGroupIndex,
    std::size_t activeStreamCount,
    const Seed128& aSeed,
    std::uint32_t compressionBits,
    long double encryptionErrorStandardDeviation) {

    return EncryptSeededSparseRLWEImpl(
        std::move(messageQ),
        clientSecret,
        publicMaterial,
        roundId,
        packGroupIndex,
        activeStreamCount,
        aSeed,
        true,
        compressionBits,
        encryptionErrorStandardDeviation,
        false);
}

SeededSparseRLWEPacket EncryptSeededSparseRLWEUncompressed(
    DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t packGroupIndex,
    std::size_t activeStreamCount,
    const Seed128& aSeed,
    long double encryptionErrorStandardDeviation,
    bool v17Domain) {

    return EncryptSeededSparseRLWEImpl(
        std::move(messageQ),
        clientSecret,
        publicMaterial,
        roundId,
        packGroupIndex,
        activeStreamCount,
        aSeed,
        false,
        0,
        encryptionErrorStandardDeviation,
        v17Domain);
}

SeededSparseRLWEPacket CompressSeededSparseRLWEPacket(
    const SeededSparseRLWEPacket& uncompressedPacket,
    const CommonSetupMaterial& publicMaterial,
    std::uint32_t compressionBits,
    bool stochastic) {

    if (uncompressedPacket.bIsCompressed) {
        throw std::invalid_argument(
            "c1 derivation requires an uncompressed sparse packet");
    }
    const RestoredSparseRLWECiphertext restored =
        RestoreSeededSparseRLWE(uncompressedPacket, publicMaterial);
    SeededSparseRLWEPacket result = uncompressedPacket;
    result.bIsCompressed = true;
    result.compressedB = stochastic
        ? coefficient_compression::CompressSelectedCoefficientsStochastic(
              restored.b, compressionBits, restored.transmittedBIndices)
        : coefficient_compression::CompressSelectedCoefficients(
              restored.b, compressionBits, restored.transmittedBIndices);
    result.uncompressedB.clear();
    return result;
}

RestoredSparseRLWECiphertext RestoreSeededSparseRLWE(
    const SeededSparseRLWEPacket& packet,
    const CommonSetupMaterial& publicMaterial) {

    ValidatePublicMaterial(publicMaterial);

    const std::size_t expectedRingDimension =
        publicMaterial.rns.paramsQ->GetRingDimension();

    if (!(packet.setupId == publicMaterial.setupId)) {
        throw std::invalid_argument(
            "seeded RLWE packet belongs to a different setup");
    }

    if (packet.ringDimension != expectedRingDimension ||
        packet.mlweRank != publicMaterial.mlweRank) {

        throw std::invalid_argument(
            "seeded RLWE packet dimensions do not match setup");
    }

    const std::vector<std::size_t> selectedIndices =
        SparseStreamCoefficientIndices(
            packet.ringDimension,
            packet.mlweRank,
            packet.activeStreamCount);

    if (packet.bIsCompressed) {
        if (packet.compressedB.coefficientCount !=
                selectedIndices.size() ||
            !packet.uncompressedB.empty()) {

            throw std::invalid_argument(
                "compressed sparse b payload is inconsistent");
        }
    }
    else if (packet.uncompressedB.size() != selectedIndices.size() ||
             packet.compressedB.coefficientCount != 0 ||
             !packet.compressedB.bytes.empty()) {

        throw std::invalid_argument(
            "uncompressed sparse b payload is inconsistent");
    }

    const ByteVector context = EncryptionAContext(
        packet.setupId,
        packet.roundId,
        packet.clientId,
        packet.packGroupIndex);

    RestoredSparseRLWECiphertext restored;
    restored.setupId = packet.setupId;
    restored.roundId = packet.roundId;
    restored.clientId = packet.clientId;
    restored.packGroupIndex = packet.packGroupIndex;
    restored.activeStreamCount = packet.activeStreamCount;
    restored.transmittedBIndices = selectedIndices;
    restored.a = GenerateSeededUniformDCRT(
        publicMaterial.rns.paramsQ,
        packet.aSeed,
        packet.v17Domain ? "ENC-SP-A" : "ENC-A",
        context);
    if (packet.bIsCompressed) {
        restored.b =
            coefficient_compression::DecompressSelectedCoefficients(
                publicMaterial.rns.paramsQ,
                packet.compressedB,
                selectedIndices);
    }
    else {
        const lbcrypto::BigInteger modulus =
            publicMaterial.rns.paramsQ->GetModulus();

        for (const auto& coefficient : packet.uncompressedB) {
            if (coefficient >= modulus) {
                throw std::invalid_argument(
                    "uncompressed sparse b coefficient is outside composite Q");
            }
        }

        restored.b = rns_polynomial::FromCompositeCoefficientsAtIndices(
            publicMaterial.rns.paramsQ,
            packet.uncompressedB,
            selectedIndices);
    }
    restored.b.SetFormat(Format::EVALUATION);

    return restored;
}

}  // namespace federated
