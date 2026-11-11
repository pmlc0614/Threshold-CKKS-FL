#ifndef FEDERATED1_SEEDED_RLWE_H
#define FEDERATED1_SEEDED_RLWE_H

#include "coefficient-compression.h"
#include "protocol-setup.h"
#include "protocol-types.h"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace federated {

/* Sparse upload packet: regenerate a from aSeed and send only interleaved b coefficients of active streams as c1-bit codes. */
struct SeededSparseRLWEPacket {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::size_t packGroupIndex = 0;
    std::size_t ringDimension = 0;
    std::size_t mlweRank = 0;
    std::size_t activeStreamCount = 0;
    Seed128 aSeed{};
    bool v17Domain = false;
    bool bIsCompressed = true;
    coefficient_compression::CompressedCoefficientPayload compressedB;

    /* Lossless test packets retain selected composite-Q coefficients. */
    std::vector<lbcrypto::BigInteger> uncompressedB;
};

/* Key-switch input reconstructed from a received packet. */
struct RestoredSparseRLWECiphertext {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::size_t packGroupIndex = 0;
    std::size_t activeStreamCount = 0;
    std::vector<std::size_t> transmittedBIndices;
    lbcrypto::DCRTPoly b;
    lbcrypto::DCRTPoly a;
};

/* Return the sorted union of active I_h = {mlweRank*x+h | 0 <= x < N/mlweRank}. */
std::vector<std::size_t> SparseStreamCoefficientIndices(
    std::size_t ringDimension,
    std::size_t mlweRank,
    std::size_t activeStreamCount);

/* Form b = message + e - a*x_i modulo Q and transmit only active b slots. Inactive messageQ stream coefficients must be zero. */
SeededSparseRLWEPacket EncryptSeededSparseRLWE(
    lbcrypto::DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t packGroupIndex,
    std::size_t activeStreamCount,
    const Seed128& aSeed,
    std::uint32_t compressionBits,
    long double encryptionErrorStandardDeviation = 3.2L);

SeededSparseRLWEPacket EncryptSeededSparseRLWEUncompressed(
    lbcrypto::DCRTPoly messageQ,
    const ClientSecretSetupMaterial& clientSecret,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    std::size_t packGroupIndex,
    std::size_t activeStreamCount,
    const Seed128& aSeed,
    long double encryptionErrorStandardDeviation = 3.2L,
    bool v17Domain = false);

/* Derive c1 codes from the same lossless b sample, allowing a test shadow and wire path to share encryption noise. */
SeededSparseRLWEPacket CompressSeededSparseRLWEPacket(
    const SeededSparseRLWEPacket& uncompressedPacket,
    const CommonSetupMaterial& publicMaterial,
    std::uint32_t compressionBits,
    bool stochastic = false);

/* Regenerate full a with the path-specific encryption tag, restore received b slots, and fill omitted coefficients with zero. */
RestoredSparseRLWECiphertext RestoreSeededSparseRLWE(
    const SeededSparseRLWEPacket& packet,
    const CommonSetupMaterial& publicMaterial);

}  // namespace federated

#endif  // FEDERATED1_SEEDED_RLWE_H
