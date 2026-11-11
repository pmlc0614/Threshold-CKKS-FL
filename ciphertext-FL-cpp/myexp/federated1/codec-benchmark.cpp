#include "coefficient-compression.h"
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace federated::coefficient_compression;
using Values = std::vector<std::uint64_t>;
using Bytes = std::vector<std::uint8_t>;

Bytes ReferencePack(const Values& values, unsigned bits) {
    Bytes bytes((values.size() * bits + 7) / 8, 0);
    std::size_t offset = 0;
    for (auto value : values)
        for (unsigned bit = 0; bit < bits; ++bit, ++offset)
            bytes[offset / 8] |= ((value >> (bits - 1 - bit)) & 1U) << (7 - offset % 8);
    return bytes;
}
Values ReferenceUnpack(const Bytes& bytes, std::size_t count, unsigned bits) {
    Values values(count, 0);
    std::size_t offset = 0;
    for (auto& value : values)
        for (unsigned bit = 0; bit < bits; ++bit, ++offset)
            value = (value << 1U) | ((bytes[offset / 8] >> (7 - offset % 8)) & 1U);
    return values;
}
volatile std::uint64_t checksum = 0;
template<class F> double Measure(F operation) {
    for (int i = 0; i < 5; ++i) operation();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) operation();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count() / 100;
}
int main() {
    // A serialization-only comparison, not an encrypted-round speedup claim.
    for (unsigned bits : {15U, 21U, 48U, 54U, 63U}) {
        Values values(8192);
        const auto mask = (std::uint64_t{1} << bits) - 1;
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = (i * 0x9e3779b97f4a7c15ULL) & mask;
        const auto bytes = ReferencePack(values, bits);
        if (PackUnsignedValues(values, bits) != bytes ||
            UnpackUnsignedValues(bytes, values.size(), bits) != values)
            throw std::runtime_error("codec mismatch");
        const auto oldPack = Measure([&] { checksum = ReferencePack(values, bits).back(); });
        const auto newPack = Measure([&] { checksum = PackUnsignedValues(values, bits).back(); });
        const auto oldUnpack = Measure([&] { checksum = ReferenceUnpack(bytes, values.size(), bits).back(); });
        const auto newUnpack = Measure([&] { checksum = UnpackUnsignedValues(bytes, values.size(), bits).back(); });
        std::cout << "bits=" << bits << " coefficients=8192 repeats=100 ms/call"
                  << " pack_old=" << oldPack << " pack_new=" << newPack
                  << " unpack_old=" << oldUnpack << " unpack_new=" << newUnpack << '\n';
    }
    using I = lbcrypto::BigInteger;
    const I q = (I(1) << 63) - I(25);
    std::vector<I> coefficients, coins;
    for (std::uint64_t i = 0; i < 8192; ++i) {
        coefficients.push_back(I(i * 0x9e3779b97f4a7c15ULL).Mod(q));
        coins.push_back(I(i * 0xd1342543de82ef95ULL).Mod(q));
    }
    for (unsigned bits : {15U, 21U, 25U, 48U, 54U}) {
        const I levels = I(1) << bits;
        auto reference = [&](std::size_t i) {
            const I scaled = coefficients[i].Mod(q) * levels;
            I code = scaled / q;
            if (coins[i] < scaled.Mod(q)) code += I(1);
            return code.Mod(levels).ConvertToInt();
        };
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            if (reference(i) != CompressStochasticCoefficientWithCoin(coefficients[i], q, bits, coins[i]))
                throw std::runtime_error("quantizer mismatch");
        const auto oldTime = Measure([&] {
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < coefficients.size(); ++i) sum ^= reference(i);
            checksum = sum;
        });
        const auto newTime = Measure([&] {
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < coefficients.size(); ++i)
                sum ^= CompressStochasticCoefficientWithCoin(coefficients[i], q, bits, coins[i]);
            checksum = sum;
        });
        std::cout << "quantize bits=" << bits << " coefficients=8192 fixed-coins ms/batch"
                  << " big_integer=" << oldTime << " exact_native=" << newTime << '\n';
    }

}
