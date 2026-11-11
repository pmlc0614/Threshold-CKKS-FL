#ifndef FEDERATED1_DETERMINISTIC_RANDOMNESS_H
#define FEDERATED1_DETERMINISTIC_RANDOMNESS_H

#include "protocol-types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace federated {
namespace deterministic_randomness {

using Digest256 = std::array<std::uint8_t, 32>;

ByteVector EncodeBigEndian64(std::uint64_t value);

ByteVector Shake128(
    const ByteVector& input,
    std::size_t outputLength);

Digest256 Sha256(const ByteVector& input);

Digest256 HmacSha256(
    const ByteVector& key,
    const ByteVector& message);

Seed128 DeriveSeed128(
    const MasterSeed256& masterSeed,
    const std::string& domain,
    const ByteVector& context = {});

SetupId DeriveSetupId(const MasterSeed256& masterSeed);

SetupId DeriveSetupId(
    const MasterSeed256& masterSeed,
    const ByteVector& canonicalSetupDescriptor);

Seed128 DeriveClientGroupIndexKey(
    const MasterSeed256& masterSeed);

Seed128 DeriveClientGroupIndexKey(
    const MasterSeed256& masterSeed,
    const SetupId& setupId);

RoundId DeriveRoundId(
    const MasterSeed256& masterSeed,
    std::uint64_t roundNumber);

Seed128 DeriveIndexPermutationSeed(
    const Seed128& clientGroupIndexKey,
    const SetupId& setupId,
    RoundId roundId,
    std::size_t chunkCount);

std::vector<std::size_t> GenerateIndexPermutation(
    const Seed128& clientGroupIndexKey,
    const SetupId& setupId,
    RoundId roundId,
    std::size_t chunkCount);

std::vector<std::size_t> InvertPermutation(
    const std::vector<std::size_t>& permutation);

ByteVector V17LocalCiphertextAContext(
    const SetupId& setupId, RoundId roundId, ClientId clientId,
    std::uint64_t blockIndex);

ByteVector V17LocalCiphertextATag(
    const SetupId& setupId,
    RoundId roundId,
    ClientId clientId,
    std::uint64_t blockIndex,
    const std::string& path);

/* v17 finalized decryption instance tags for public offset/dither. */
ByteVector V17DecryptionContext(
    const SetupId& setupId,
    RoundId roundId,
    const std::string& path,
    std::uint64_t instanceIndex);

ByteVector V17PublicOffsetTag(
    const SetupId& setupId,
    RoundId roundId,
    const std::string& path,
    std::uint64_t instanceIndex);

ByteVector V17PublicDitherTag(
    const SetupId& setupId,
    RoundId roundId,
    const std::string& path,
    std::uint64_t instanceIndex,
    ClientId clientId);

/* Expand SHAKE128(UPOLY || seed || tag || BE64(N) || BE64(modulus)); use big-endian 64-bit words and high-tail rejection for uniform coefficients. */
std::vector<std::uint64_t> GenerateUniformPolynomial(
    const Seed128& seed,
    const ByteVector& tag,
    std::size_t ringDimension,
    std::uint64_t modulus);

/* Convenience overload for an ASCII tag. */
std::vector<std::uint64_t> GenerateUniformPolynomial(
    const Seed128& seed,
    const std::string& domain,
    std::size_t ringDimension,
    std::uint64_t modulus);

/* Clipped Box-Muller generator for synthetic simulations only, never protocol noise. */
std::vector<double> GenerateClippedGaussian(
    const MasterSeed256& masterSeed,
    const std::string& domain,
    const ByteVector& context,
    std::size_t valueCount,
    long double mean,
    long double standardDeviation,
    long double minimum,
    long double maximum);

}  // namespace deterministic_randomness
}  // namespace federated

#endif  // FEDERATED1_DETERMINISTIC_RANDOMNESS_H
