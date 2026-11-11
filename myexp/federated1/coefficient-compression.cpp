#include "protocol-rounding.h"
#include "coefficient-compression.h"
#include "codec-timing.h"

#include "protocol-randomness.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace federated {
namespace coefficient_compression {
namespace {

void ValidateBits(std::uint32_t bits) {
    if (bits == 0 || bits > 63) {
        throw std::invalid_argument("compression bits must be in [1, 63]");
    }
}

lbcrypto::BigInteger PowerOfTwo(std::uint32_t bits) {
    lbcrypto::BigInteger value(1);
    value <<= bits;
    return value;
}

std::uint64_t ValueMask(std::uint32_t bits) {
    ValidateBits(bits);
    return (std::uint64_t{1} << bits) - 1U;
}

std::size_t PackedByteCount(
    std::size_t valueCount,
    std::uint32_t bitsPerValue) {

    ValidateBits(bitsPerValue);

    if (valueCount >
        (std::numeric_limits<std::size_t>::max() - 7U) /
            bitsPerValue) {

        throw std::overflow_error("packed coefficient payload is too large");
    }

    return (valueCount * bitsPerValue + 7U) / 8U;
}

void ValidateModulus(const lbcrypto::BigInteger& modulus) {
    if (modulus < lbcrypto::BigInteger(2)) {
        throw std::invalid_argument("coefficient modulus must be at least two");
    }
}

void ValidateStochasticParameters(
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits) {

    ValidateBits(bits);
    ValidateModulus(modulus);
    if (bits < 2 || (PowerOfTwo(bits) << 1) > modulus) {
        throw std::invalid_argument(
            "stochastic compression requires 4 <= 2^bits <= Q/2");
    }
}

void ValidateIndices(
    const std::vector<std::size_t>& indices,
    std::size_t ringDimension) {

    std::unordered_set<std::size_t> seen;
    seen.reserve(indices.size());

    for (const std::size_t index : indices) {
        if (index >= ringDimension) {
            throw std::out_of_range("coefficient index exceeds ring dimension");
        }

        if (!seen.insert(index).second) {
            throw std::invalid_argument("coefficient indices contain duplicates");
        }
    }
}

// Immutable per-payload constants. Private coins still come from the same
// sampler, in the same order, including exact rejection sampling.
class Quantizer {
 public:
    Quantizer(const lbcrypto::BigInteger& modulus, std::uint32_t bits, bool stochastic)
        : modulus_(modulus), bits_(bits) {
        if (stochastic) ValidateStochasticParameters(modulus, bits);
        else { ValidateBits(bits); ValidateModulus(modulus); }
        levels_ = PowerOfTwo(bits);
        mask_ = ValueMask(bits);
#if defined(__SIZEOF_INT128__)
        if (modulus.GetMSB() <= 64) nativeModulus_ = modulus.ConvertToInt();
#endif
    }

    std::uint64_t Deterministic(const lbcrypto::BigInteger& coefficient) const {
#if defined(__SIZEOF_INT128__)
        if (nativeModulus_) {
            const Wide scaled = Wide(Normalize(coefficient)) << bits_;
            return static_cast<std::uint64_t>(
                (scaled + nativeModulus_ / 2) / nativeModulus_) & mask_;
        }
#endif
        return ((coefficient.Mod(modulus_) * levels_ + (modulus_ >> 1)) /
                modulus_).Mod(levels_).ConvertToInt();
    }

    std::uint64_t Stochastic(const lbcrypto::BigInteger& coefficient,
                             const lbcrypto::BigInteger& coin) const {
        if (coin >= modulus_)
            throw std::invalid_argument("stochastic coin is outside Z_Q");
#if defined(__SIZEOF_INT128__)
        if (nativeModulus_) {
            const Wide scaled = Wide(Normalize(coefficient)) << bits_;
            const auto quotient = static_cast<std::uint64_t>(scaled / nativeModulus_);
            const auto remainder = static_cast<std::uint64_t>(scaled % nativeModulus_);
            return (quotient + (coin.ConvertToInt() < remainder ? 1U : 0U)) & mask_;
        }
#endif
        const auto scaled = coefficient.Mod(modulus_) * levels_;
        auto code = scaled / modulus_;
        if (coin < scaled.Mod(modulus_)) code += lbcrypto::BigInteger(1);
        return code.Mod(levels_).ConvertToInt();
    }

 private:
#if defined(__SIZEOF_INT128__)
    using Wide = unsigned __int128;
    std::uint64_t Normalize(const lbcrypto::BigInteger& coefficient) const {
        if (coefficient.GetMSB() <= 64) return coefficient.ConvertToInt() % nativeModulus_;
        return coefficient.Mod(modulus_).ConvertToInt();
    }
    std::uint64_t nativeModulus_ = 0;
#endif
    lbcrypto::BigInteger modulus_, levels_;
    std::uint32_t bits_;
    std::uint64_t mask_;
};

std::vector<std::uint64_t> CompressPolyValues(
    const lbcrypto::Poly& polynomial,
    std::uint32_t bits,
    const std::vector<std::size_t>& indices,
    bool stochastic) {
    codec_timing::Scope timer(codec_timing::Quantize);

    const auto& coefficients = polynomial.GetValues();
    const lbcrypto::BigInteger modulus = polynomial.GetModulus();
    const Quantizer quantizer(modulus, bits, stochastic);

    std::vector<std::uint64_t> compressed;
    compressed.reserve(indices.size());

    for (const std::size_t index : indices) {
        compressed.push_back(stochastic
            ? quantizer.Stochastic(coefficients[index], SampleProtocolUniformBelow(modulus))
            : quantizer.Deterministic(coefficients[index]));
    }

    return compressed;
}

std::vector<std::uint64_t> SumStochasticPayloads(
    const std::vector<CompressedCoefficientPayload>& clientPayloads,
    std::size_t coefficientCount,
    const lbcrypto::BigInteger& modulus) {

    codec_timing::Scope timer(codec_timing::CodeSum);
    if (clientPayloads.empty()) {
        throw std::invalid_argument("stochastic code sum needs clients");
    }
    const std::uint32_t bits = clientPayloads.front().bitsPerCoefficient;
    ValidateStochasticParameters(modulus, bits);
    const std::uint64_t mask = ValueMask(bits);
    std::vector<std::uint64_t> sums(coefficientCount, 0);
    for (const auto& payload : clientPayloads) {
        if (payload.bitsPerCoefficient != bits ||
            payload.coefficientCount != coefficientCount) {
            throw std::invalid_argument(
                "stochastic code payload shapes differ");
        }
        const auto codes = UnpackUnsignedValues(
            payload.bytes, coefficientCount, bits);
        for (std::size_t index = 0; index < coefficientCount; ++index) {
            sums[index] = (sums[index] + codes[index]) & mask;
        }
    }
    return sums;
}

std::vector<std::size_t> AllIndices(std::size_t count) {
    std::vector<std::size_t> indices(count);

    for (std::size_t index = 0; index < count; ++index) {
        indices[index] = index;
    }

    return indices;
}

std::size_t RingDimension(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters) {

    if (!parameters) {
        throw std::invalid_argument("DCRT parameters are null");
    }

    const std::size_t cyclotomicOrder = parameters->GetCyclotomicOrder();
    if (cyclotomicOrder == 0 || cyclotomicOrder % 2 != 0) {
        throw std::invalid_argument("unsupported cyclotomic order");
    }

    return cyclotomicOrder / 2;
}

lbcrypto::DCRTPoly DecompressValuesAtIndices(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<std::uint64_t>& packedValues,
    std::uint32_t bits,
    const std::vector<std::size_t>& indices) {

    codec_timing::Scope timer(codec_timing::Reconstruct);
    const std::size_t ringDimension = RingDimension(parameters);
    ValidateBits(bits);
    ValidateIndices(indices, ringDimension);
    if (packedValues.size() != indices.size()) {
        throw std::invalid_argument("payload coefficient count does not match the index count");
    }

    const lbcrypto::BigInteger compositeModulus = parameters->GetModulus();
    std::vector<lbcrypto::BigInteger> coefficients(
        ringDimension,
        lbcrypto::BigInteger(0));

    for (std::size_t position = 0; position < indices.size(); ++position) {
        coefficients[indices[position]] =
            DecompressCoefficient(
                packedValues[position],
                compositeModulus,
                bits);
    }

    lbcrypto::DCRTPoly result(
        parameters,
        Format::COEFFICIENT,
        true);

    auto& towers = result.GetAllElements();
    const auto& towerParameters = parameters->GetParams();

    if (towers.size() != towerParameters.size()) {
        throw std::runtime_error("DCRT tower count mismatch");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < towers.size();
         ++towerIndex) {

        const lbcrypto::NativeInteger nativeModulus =
            towerParameters[towerIndex]->GetModulus();

        const lbcrypto::BigInteger bigModulus(nativeModulus);
        lbcrypto::NativeVector towerValues(ringDimension, nativeModulus);

        for (std::size_t coefficientIndex = 0;
             coefficientIndex < ringDimension;
             ++coefficientIndex) {

            const lbcrypto::BigInteger residue =
                coefficients[coefficientIndex].Mod(bigModulus);

            towerValues[coefficientIndex] =
                lbcrypto::NativeInteger(residue.ConvertToInt());
        }

        towers[towerIndex].SetValues(
            std::move(towerValues),
            Format::COEFFICIENT);
    }

    return result;
}

lbcrypto::DCRTPoly DecompressAtIndices(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const CompressedCoefficientPayload& payload,
    const std::vector<std::size_t>& indices) {
    if (payload.coefficientCount != indices.size())
        throw std::invalid_argument("payload coefficient count does not match the index count");
    const auto values = UnpackUnsignedValues(
        payload.bytes, payload.coefficientCount, payload.bitsPerCoefficient);
    return DecompressValuesAtIndices(parameters, values, payload.bitsPerCoefficient, indices);
}

}  // namespace

std::uint64_t CompressCoefficient(
    const lbcrypto::BigInteger& coefficient,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits) {

    return Quantizer(modulus, bits, false).Deterministic(coefficient);
}

lbcrypto::BigInteger DecompressCoefficient(
    std::uint64_t compressed,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits) {

    ValidateBits(bits);
    ValidateModulus(modulus);

    const std::uint64_t mask = ValueMask(bits);
    if ((compressed & ~mask) != 0) {
        throw std::invalid_argument(
            "compressed coefficient exceeds its declared bit width");
    }

    const lbcrypto::BigInteger levels = PowerOfTwo(bits);
    const lbcrypto::BigInteger value(compressed);

    return ((value * modulus + (levels >> 1)) / levels).Mod(modulus);
}

namespace {
struct AggregateCell {
    lbcrypto::BigInteger levels, index, lower, width, remainder;
};
AggregateCell FindAggregateCell(const lbcrypto::BigInteger& coefficient,
                               const lbcrypto::BigInteger& modulus, std::uint32_t bits) {
    ValidateBits(bits);
    ValidateModulus(modulus);
    const auto levels = PowerOfTwo(bits);
    if (levels > modulus) throw std::invalid_argument("aggregate-a requires 2^bits <= Q");
    const auto x = coefficient.Mod(modulus);
    auto grid = [&](const lbcrypto::BigInteger& j) {
        return (modulus * j + (levels >> 1)) / levels;
    };
    auto j = (x * levels) / modulus;
    const lbcrypto::BigInteger one(1);
    if (grid(j) > x) j -= one;
    else if (grid(j + one) <= x) j += one;
    const auto lower = grid(j);
    return {levels, j, lower, grid(j + one) - lower, x - lower};
}
std::uint64_t EncodeAggregateCell(const AggregateCell& cell,
                                const lbcrypto::BigInteger& coin) {
    if (coin >= cell.width) throw std::invalid_argument("aggregate-a coin outside grid cell");
    auto code = cell.index;
    if (coin < cell.remainder) code += lbcrypto::BigInteger(1);
    return code.Mod(cell.levels).ConvertToInt();
}
}  // namespace

std::uint64_t CompressAggregateCoefficientWithCoin(
    const lbcrypto::BigInteger& coefficient, const lbcrypto::BigInteger& modulus,
    std::uint32_t bits, const lbcrypto::BigInteger& coin) {
    return EncodeAggregateCell(FindAggregateCell(coefficient, modulus, bits), coin);
}
std::uint64_t CompressAggregateCoefficient(
    const lbcrypto::BigInteger& coefficient, const lbcrypto::BigInteger& modulus,
    std::uint32_t bits) {
    const auto cell = FindAggregateCell(coefficient, modulus, bits);
    const auto coin = cell.width == lbcrypto::BigInteger(1)
        ? lbcrypto::BigInteger(0) : SampleProtocolUniformBelow(cell.width);
    return EncodeAggregateCell(cell, coin);
}
CompressedCoefficientPayload CompressAggregateCoefficients(
    lbcrypto::DCRTPoly polynomial, std::uint32_t bits) {
    lbcrypto::Poly interpolated;
    {
        codec_timing::Scope timer(codec_timing::FormatAndCRT);
        polynomial.SetFormat(Format::COEFFICIENT);
        interpolated = polynomial.CRTInterpolate();
        interpolated.SetFormat(Format::COEFFICIENT);
    }
    const auto modulus = interpolated.GetModulus();
    std::vector<std::uint64_t> codes;
    const auto& values = interpolated.GetValues();
    codes.reserve(values.GetLength());
    {
        codec_timing::Scope timer(codec_timing::Quantize);
        for (std::size_t i = 0; i < values.GetLength(); ++i)
            codes.push_back(CompressAggregateCoefficient(values[i], modulus, bits));
    }
    return {bits, codes.size(), PackUnsignedValues(codes, bits)};
}

std::uint64_t CompressStochasticCoefficientWithCoin(
    const lbcrypto::BigInteger& coefficient,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits,
    const lbcrypto::BigInteger& coin) {

    return Quantizer(modulus, bits, true).Stochastic(coefficient, coin);
}

std::uint64_t CompressStochasticCoefficient(
    const lbcrypto::BigInteger& coefficient,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits) {

    const Quantizer quantizer(modulus, bits, true);
    return quantizer.Stochastic(coefficient, SampleProtocolUniformBelow(modulus));
}

lbcrypto::BigInteger ReconstructStochasticCodeSum(
    const std::vector<std::uint64_t>& codes,
    const lbcrypto::BigInteger& modulus,
    std::uint32_t bits) {

    ValidateStochasticParameters(modulus, bits);
    const std::uint64_t mask = ValueMask(bits);
    std::uint64_t sum = 0;
    for (const std::uint64_t code : codes) {
        if ((code & ~mask) != 0) {
            throw std::invalid_argument("stochastic code exceeds bit width");
        }
        sum = (sum + code) & mask;
    }
    return DecompressCoefficient(sum, modulus, bits);
}

std::vector<std::uint8_t> PackUnsignedValues(
    const std::vector<std::uint64_t>& values,
    std::uint32_t bitsPerValue) {
    codec_timing::Scope timer(codec_timing::Pack);

    const std::uint64_t mask = ValueMask(bitsPerValue);
    std::vector<std::uint8_t> bytes(
        PackedByteCount(values.size(), bitsPerValue),
        0U);

    std::size_t bitOffset = 0;

    for (const std::uint64_t value : values) {
        if ((value & ~mask) != 0) {
            throw std::invalid_argument("value exceeds packed bit width");
        }

        std::uint32_t remaining = bitsPerValue;
        while (remaining != 0) {
            const auto available = static_cast<std::uint32_t>(8U - bitOffset % 8U);
            const auto take = std::min(available, remaining);
            remaining -= take;
            const auto fragment = (value >> remaining) & ((1U << take) - 1U);
            bytes[bitOffset / 8U] |= static_cast<std::uint8_t>(fragment << (available - take));
            bitOffset += take;
        }
    }

    return bytes;
}

std::vector<std::uint64_t> UnpackUnsignedValues(
    const std::vector<std::uint8_t>& bytes,
    std::size_t valueCount,
    std::uint32_t bitsPerValue) {
    codec_timing::Scope timer(codec_timing::Unpack);

    const std::size_t expectedBytes =
        PackedByteCount(valueCount, bitsPerValue);

    if (bytes.size() != expectedBytes) {
        throw std::invalid_argument("packed byte count is not canonical");
    }

    std::vector<std::uint64_t> values(valueCount, 0U);
    std::size_t bitOffset = 0;

    for (std::uint64_t& value : values) {
        std::uint32_t remaining = bitsPerValue;
        while (remaining != 0) {
            const auto available = static_cast<std::uint32_t>(8U - bitOffset % 8U);
            const auto take = std::min(available, remaining);
            const auto fragment = (bytes[bitOffset / 8U] >> (available - take)) & ((1U << take) - 1U);
            value = (value << take) | fragment;
            bitOffset += take;
            remaining -= take;
        }
    }

    const std::size_t unusedBits = expectedBytes * 8U - bitOffset;
    if (unusedBits > 0 && !bytes.empty()) {
        const std::uint8_t unusedMask =
            static_cast<std::uint8_t>((1U << unusedBits) - 1U);

        if ((bytes.back() & unusedMask) != 0) {
            throw std::invalid_argument("nonzero padding bits in packed payload");
        }
    }

    return values;
}

CompressedCoefficientPayload CompressAllCoefficients(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits) {

    lbcrypto::Poly interpolated;
    {
        codec_timing::Scope timer(codec_timing::FormatAndCRT);
        polynomial.SetFormat(Format::COEFFICIENT);
        interpolated = polynomial.CRTInterpolate();
        interpolated.SetFormat(Format::COEFFICIENT);
    }

    const std::vector<std::size_t> indices =
        AllIndices(interpolated.GetValues().GetLength());

    const std::vector<std::uint64_t> compressed =
        CompressPolyValues(interpolated, bits, indices, false);

    return CompressedCoefficientPayload{
        bits,
        compressed.size(),
        PackUnsignedValues(compressed, bits)};
}

CompressedCoefficientPayload CompressSelectedCoefficients(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits,
    const std::vector<std::size_t>& coefficientIndices) {

    lbcrypto::Poly interpolated;
    {
        codec_timing::Scope timer(codec_timing::FormatAndCRT);
        polynomial.SetFormat(Format::COEFFICIENT);
        interpolated = polynomial.CRTInterpolate();
        interpolated.SetFormat(Format::COEFFICIENT);
    }

    ValidateIndices(
        coefficientIndices,
        interpolated.GetValues().GetLength());

    const std::vector<std::uint64_t> compressed =
        CompressPolyValues(interpolated, bits, coefficientIndices, false);

    return CompressedCoefficientPayload{
        bits,
        compressed.size(),
        PackUnsignedValues(compressed, bits)};
}

CompressedCoefficientPayload CompressAllCoefficientsStochastic(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits) {

    lbcrypto::Poly interpolated;
    {
        codec_timing::Scope timer(codec_timing::FormatAndCRT);
        polynomial.SetFormat(Format::COEFFICIENT);
        interpolated = polynomial.CRTInterpolate();
        interpolated.SetFormat(Format::COEFFICIENT);
    }
    ValidateStochasticParameters(interpolated.GetModulus(), bits);
    const auto indices = AllIndices(interpolated.GetValues().GetLength());
    const auto codes = CompressPolyValues(
        interpolated, bits, indices, true);
    return {bits, codes.size(), PackUnsignedValues(codes, bits)};
}

CompressedCoefficientPayload CompressSelectedCoefficientsStochastic(
    lbcrypto::DCRTPoly polynomial,
    std::uint32_t bits,
    const std::vector<std::size_t>& coefficientIndices) {

    lbcrypto::Poly interpolated;
    {
        codec_timing::Scope timer(codec_timing::FormatAndCRT);
        polynomial.SetFormat(Format::COEFFICIENT);
        interpolated = polynomial.CRTInterpolate();
        interpolated.SetFormat(Format::COEFFICIENT);
    }
    ValidateStochasticParameters(interpolated.GetModulus(), bits);
    ValidateIndices(coefficientIndices, interpolated.GetValues().GetLength());
    const auto codes = CompressPolyValues(
        interpolated, bits, coefficientIndices, true);
    return {bits, codes.size(), PackUnsignedValues(codes, bits)};
}

lbcrypto::DCRTPoly DecompressSummedStochasticCodes(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<CompressedCoefficientPayload>& clientPayloads) {

    const std::size_t count = RingDimension(parameters);
    const auto summed = SumStochasticPayloads(
        clientPayloads, count, parameters->GetModulus());
    return DecompressValuesAtIndices(parameters, summed,
        clientPayloads.front().bitsPerCoefficient, AllIndices(count));
}

lbcrypto::DCRTPoly DecompressSelectedSummedStochasticCodes(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<CompressedCoefficientPayload>& clientPayloads,
    const std::vector<std::size_t>& coefficientIndices) {

    ValidateIndices(coefficientIndices, RingDimension(parameters));
    const auto summed = SumStochasticPayloads(
        clientPayloads,
        coefficientIndices.size(),
        parameters->GetModulus());
    return DecompressValuesAtIndices(parameters, summed,
        clientPayloads.front().bitsPerCoefficient, coefficientIndices);
}

lbcrypto::DCRTPoly DecompressAllCoefficients(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const CompressedCoefficientPayload& payload) {

    const std::vector<std::size_t> indices =
        AllIndices(RingDimension(parameters));

    return DecompressAtIndices(parameters, payload, indices);
}

lbcrypto::DCRTPoly DecompressSelectedCoefficients(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const CompressedCoefficientPayload& payload,
    const std::vector<std::size_t>& coefficientIndices) {

    return DecompressAtIndices(parameters, payload, coefficientIndices);
}

std::int64_t QuantizeSigned(
    long double value,
    std::uint32_t bits,
    long double representableBound,
    bool clamp) {

    ValidateBits(bits);

    if (bits < 2) {
        throw std::invalid_argument("signed quantization requires at least two bits");
    }

    if (!std::isfinite(value) ||
        !std::isfinite(representableBound) ||
        representableBound <= 0.0L) {

        throw std::invalid_argument("invalid signed quantization input");
    }

    if (!clamp &&
        (value < -representableBound || value > representableBound)) {

        throw std::out_of_range("signed value exceeds representable bound");
    }

    const long double bounded =
        std::max(-representableBound, std::min(value, representableBound));

    const std::int64_t maximumMagnitude =
        (std::int64_t{1} << (bits - 1U)) - 1;

    const long double scale =
        static_cast<long double>(maximumMagnitude) / representableBound;

    return static_cast<std::int64_t>(protocol_rounding::NearestInteger(bounded * scale));
}

long double DequantizeSigned(
    std::int64_t quantized,
    std::uint32_t bits,
    long double representableBound) {

    ValidateBits(bits);

    if (bits < 2 ||
        !std::isfinite(representableBound) ||
        representableBound <= 0.0L) {

        throw std::invalid_argument("invalid signed dequantization parameters");
    }

    const std::int64_t maximumMagnitude =
        (std::int64_t{1} << (bits - 1U)) - 1;

    if (quantized < -maximumMagnitude || quantized > maximumMagnitude) {
        throw std::out_of_range("signed quantized value exceeds bit width");
    }

    return static_cast<long double>(quantized) * representableBound /
           static_cast<long double>(maximumMagnitude);
}

std::int64_t QuantizeSignedWithScale(
    long double value,
    std::uint32_t bits,
    long double deliveryScale) {

    ValidateBits(bits);
    if (bits < 2 || !std::isfinite(value) ||
        !std::isfinite(deliveryScale) || deliveryScale <= 0.0L) {
        throw std::invalid_argument(
            "invalid no-clipping delivery quantization input");
    }
    const std::int64_t maximumMagnitude =
        (std::int64_t{1} << (bits - 1U)) - 1;
    const long double scaled = value * deliveryScale;
    if (!std::isfinite(scaled) ||
        scaled < -static_cast<long double>(maximumMagnitude) ||
        scaled > static_cast<long double>(maximumMagnitude)) {
        throw std::out_of_range(
            "final-delivery value exceeds the configured signed range");
    }
    return static_cast<std::int64_t>(protocol_rounding::NearestInteger(scaled));
}

void ValidateNoClippingDeliveryRange(
    std::uint32_t bits,
    long double deliveryScale,
    long double absoluteBound) {

    ValidateBits(bits);
    if (bits < 2 || !std::isfinite(deliveryScale) ||
        deliveryScale <= 0.0L || !std::isfinite(absoluteBound) ||
        absoluteBound <= 0.0L) {
        throw std::invalid_argument(
            "invalid no-clipping final-delivery profile");
    }
    const long double maximumMagnitude = std::ldexp(
        1.0L,
        static_cast<int>(bits - 1U)) - 1.0L;
    if (deliveryScale * absoluteBound > maximumMagnitude) {
        throw std::out_of_range(
            "deliveryScale*updateBound exceeds the signed final-delivery range; clipping is forbidden");
    }
}

long double DequantizeSignedWithScale(
    std::int64_t quantized,
    std::uint32_t bits,
    long double deliveryScale) {

    ValidateBits(bits);
    if (bits < 2 || !std::isfinite(deliveryScale) ||
        deliveryScale <= 0.0L) {
        throw std::invalid_argument(
            "invalid no-clipping delivery dequantization input");
    }
    const std::int64_t maximumMagnitude =
        (std::int64_t{1} << (bits - 1U)) - 1;
    if (quantized < -maximumMagnitude || quantized > maximumMagnitude) {
        throw std::out_of_range(
            "final-delivery integer exceeds the configured bit width");
    }
    return static_cast<long double>(quantized) / deliveryScale;
}

std::vector<std::uint8_t> PackSignedValues(
    const std::vector<std::int64_t>& values,
    std::uint32_t bitsPerValue) {

    ValidateBits(bitsPerValue);

    if (bitsPerValue < 2) {
        throw std::invalid_argument("signed packing requires at least two bits");
    }

    const std::int64_t minimum =
        -(std::int64_t{1} << (bitsPerValue - 1U));
    const std::int64_t maximum =
        (std::int64_t{1} << (bitsPerValue - 1U)) - 1;
    const std::uint64_t mask = ValueMask(bitsPerValue);

    std::vector<std::uint64_t> unsignedValues;
    unsignedValues.reserve(values.size());

    for (const std::int64_t value : values) {
        if (value < minimum || value > maximum) {
            throw std::out_of_range("signed value exceeds packed bit width");
        }

        unsignedValues.push_back(static_cast<std::uint64_t>(value) & mask);
    }

    return PackUnsignedValues(unsignedValues, bitsPerValue);
}

std::vector<std::int64_t> UnpackSignedValues(
    const std::vector<std::uint8_t>& bytes,
    std::size_t valueCount,
    std::uint32_t bitsPerValue) {

    ValidateBits(bitsPerValue);

    if (bitsPerValue < 2) {
        throw std::invalid_argument("signed unpacking requires at least two bits");
    }

    const std::vector<std::uint64_t> unsignedValues =
        UnpackUnsignedValues(bytes, valueCount, bitsPerValue);

    const std::uint64_t signBit =
        std::uint64_t{1} << (bitsPerValue - 1U);
    const std::uint64_t modulus = std::uint64_t{1} << bitsPerValue;

    std::vector<std::int64_t> signedValues;
    signedValues.reserve(unsignedValues.size());

    for (const std::uint64_t value : unsignedValues) {
        if ((value & signBit) == 0) {
            signedValues.push_back(static_cast<std::int64_t>(value));
        }
        else {
            signedValues.push_back(
                -static_cast<std::int64_t>(modulus - value));
        }
    }

    return signedValues;
}

}  // namespace coefficient_compression
}  // namespace federated
