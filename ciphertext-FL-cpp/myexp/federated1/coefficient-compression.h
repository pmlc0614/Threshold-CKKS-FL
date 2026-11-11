#ifndef FEDERATED1_COEFFICIENT_COMPRESSION_H
#define FEDERATED1_COEFFICIENT_COMPRESSION_H

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace federated {
namespace coefficient_compression {

struct CompressedCoefficientPayload {
    std::uint32_t bitsPerCoefficient = 0;
    std::size_t coefficientCount = 0;
    std::vector<std::uint8_t> bytes;
};

std::uint64_t CompressCoefficient(
    const lbcrypto::BigInteger& coefficient,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits);

lbcrypto::BigInteger DecompressCoefficient(
    std::uint64_t compressed,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits);

/* Unbiased reconstruction-grid codec for Sparse aggregate a, distinct from c1/c2. */
std::uint64_t CompressAggregateCoefficientWithCoin(
    const lbcrypto::BigInteger& coefficient, const lbcrypto::BigInteger& modulus,
    std::uint32_t bits, const lbcrypto::BigInteger& coin);
std::uint64_t CompressAggregateCoefficient(
    const lbcrypto::BigInteger& coefficient, const lbcrypto::BigInteger& modulus,
    std::uint32_t bits);
CompressedCoefficientPayload CompressAggregateCoefficients(
    lbcrypto::DCRTPoly polynomial, std::uint32_t bits);

/* v17 normalized stochastic code. The explicit-coin form is for vectors/tests;
 * the ordinary form samples an independent private uniform coin in Z_Q. */
std::uint64_t CompressStochasticCoefficientWithCoin(
    const lbcrypto::BigInteger& coefficient,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits,
    const lbcrypto::BigInteger& coin);

std::uint64_t CompressStochasticCoefficient(
    const lbcrypto::BigInteger& coefficient,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits);

/* d_K is the same integer reconstruction as deterministic Decomp, but c2
 * must first add the transmitted codes modulo 2^bits. */
lbcrypto::BigInteger ReconstructStochasticCodeSum(
    const std::vector<std::uint64_t>& codes,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits);

std::vector<std::uint8_t> PackUnsignedValues(
    const std::vector<std::uint64_t>& values,
    std::uint32_t bitsPerValue);

std::vector<std::uint64_t> UnpackUnsignedValues(
    const std::vector<std::uint8_t>& bytes,
    std::size_t valueCount,
    std::uint32_t bitsPerValue);

CompressedCoefficientPayload CompressAllCoefficients(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits);

CompressedCoefficientPayload CompressSelectedCoefficients(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits,
    const std::vector<std::size_t>& coefficientIndices);

CompressedCoefficientPayload CompressAllCoefficientsStochastic(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits);

CompressedCoefficientPayload CompressSelectedCoefficientsStochastic(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits,
    const std::vector<std::size_t>& coefficientIndices);

lbcrypto::DCRTPoly DecompressSummedStochasticCodes(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<CompressedCoefficientPayload>& clientPayloads);

lbcrypto::DCRTPoly DecompressSelectedSummedStochasticCodes(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<CompressedCoefficientPayload>& clientPayloads,
    const std::vector<std::size_t>& coefficientIndices);

lbcrypto::DCRTPoly DecompressAllCoefficients(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const CompressedCoefficientPayload& payload);

/* Restore unselected coefficients as zero. */
lbcrypto::DCRTPoly DecompressSelectedCoefficients(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const CompressedCoefficientPayload& payload,
    const std::vector<std::size_t>& coefficientIndices);

/* Signed cd-bit symmetric quantization maps representableBound to the largest positive code; clamp=false rejects out-of-range input. */
std::int64_t QuantizeSigned(
    long double value,
    std::uint32_t bits,
    long double representableBound,
    bool clamp);

long double DequantizeSigned(
    std::int64_t quantized,
    std::uint32_t bits,
    long double representableBound);

/*
 * v17 final-delivery quantization with an explicit Delta_d. No clipping is
 * permitted: |value|*deliveryScale exceeding the signed bit range throws.
 */
std::int64_t QuantizeSignedWithScale(
    long double value,
    std::uint32_t bits,
    long double deliveryScale);

long double DequantizeSignedWithScale(
    std::int64_t quantized,
    std::uint32_t bits,
    long double deliveryScale);

/*
 * Validate Delta_d*U_max <= 2^(c_d-1)-1 before a protocol path starts.
 * This is deliberately separate from per-value quantization so a malformed
 * delivery profile fails before any encryption work is performed.
 */
void ValidateNoClippingDeliveryRange(
    std::uint32_t bits,
    long double deliveryScale,
    long double absoluteBound);

std::vector<std::uint8_t> PackSignedValues(
    const std::vector<std::int64_t>& values,
    std::uint32_t bitsPerValue);

std::vector<std::int64_t> UnpackSignedValues(
    const std::vector<std::uint8_t>& bytes,
    std::size_t valueCount,
    std::uint32_t bitsPerValue);

}  // namespace coefficient_compression
}  // namespace federated

#endif  // FEDERATED1_COEFFICIENT_COMPRESSION_H
