#ifndef COEFFICIENT_CODEC_H
#define COEFFICIENT_CODEC_H

#include "openfhe.h"
#include "exact-integer.h"

#include <cstdint>
#include <vector>

namespace coefficient_codec {

void ValidateEncodableExact(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values, const exact_integer::Integer& scale);
lbcrypto::DCRTPoly EncodeExact(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values, const exact_integer::Integer& scale);

// Compute 2^scaleBits.
double ScalingFactorFromBits(uint32_t scaleBits);

// Check that local coefficients can be encoded safely for the actual Q and scale.
// This representability check is independent of the empirical FL bound.
void ValidateEncodable(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values,
    double scalingFactor);

// Encode real values directly in polynomial coefficients.
// Zero-pad to ring dimension N when values.size() < N.
lbcrypto::DCRTPoly Encode(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values,
    double scalingFactor);

// Wrap a coefficient-encoded polynomial in a CKKS Plaintext object.
lbcrypto::Plaintext MakePlaintext(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values,
    uint32_t scaleBits);

// Decode all N coefficients of a DCRTPoly.
std::vector<double> Decode(
    lbcrypto::DCRTPoly polynomial,
    double scalingFactor);

// Decode all N coefficients after CRT interpolation.
std::vector<double> Decode(
    lbcrypto::Poly polynomial,
    double scalingFactor);

// Decrypt and decode all N coefficients.
std::vector<double> Decrypt(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ciphertext,
    const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
    double scalingFactor);

}  // namespace coefficient_codec

#endif  // COEFFICIENT_CODEC_H
