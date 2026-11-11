#ifndef FEDERATED1_SEEDED_DENSE_RLWE_H
#define FEDERATED1_SEEDED_DENSE_RLWE_H

#include "coefficient-compression.h"
#include "protocol-setup.h"
#include "protocol-types.h"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace federated {

struct SeededDenseRLWEPacket {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::size_t blockIndex = 0;
    std::size_t validValueCount = 0;
    std::size_t ringDimension = 0;
    Seed128 aSeed{};
    /* Empty for legacy packets; otherwise FULL-MODEL or AUX. */
    std::string v17Path;
    bool bIsCompressed = true;
    coefficient_compression::CompressedCoefficientPayload compressedB;
    std::vector<lbcrypto::BigInteger> uncompressedB;
};

struct RestoredDenseRLWECiphertext {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::size_t blockIndex = 0;
    std::size_t validValueCount = 0;
    lbcrypto::DCRTPoly b;
    lbcrypto::DCRTPoly a;
};

SeededDenseRLWEPacket EncryptSeededDenseRLWE(
    lbcrypto::DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t blockIndex,
    std::size_t validValueCount,
    const Seed128& aSeed,
    std::uint32_t compressionBits,
    long double encryptionErrorStandardDeviation = 3.2L);

SeededDenseRLWEPacket EncryptSeededDenseRLWEUncompressed(
    lbcrypto::DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t blockIndex,
    std::size_t validValueCount,
    const Seed128& aSeed,
    long double encryptionErrorStandardDeviation = 3.2L,
    const std::string& v17Path = {});

/* Derive the c1 wire packet from the exact same lossless b sample. */
SeededDenseRLWEPacket CompressSeededDenseRLWEPacket(
    const SeededDenseRLWEPacket& uncompressedPacket,
    const CommonSetupMaterial& publicMaterial,
    std::uint32_t compressionBits,
    bool stochastic = false);

RestoredDenseRLWECiphertext RestoreSeededDenseRLWE(
    const SeededDenseRLWEPacket& packet,
    const CommonSetupMaterial& publicMaterial);

}  // namespace federated

#endif  // FEDERATED1_SEEDED_DENSE_RLWE_H
