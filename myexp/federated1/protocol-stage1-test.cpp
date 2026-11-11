#include "protocol-stage1-test.h"

#include "coefficient-compression.h"
#include "deterministic-randomness.h"
#include "protocol-parameters.h"
#include "protocol-randomness.h"
#include "protocol-v17-decryption.h"
#include "rns-polynomial.h"

#include "openfhe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using federated::ByteVector;
template <typename Container>
std::string Hex(const Container& bytes) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');

    for (const std::uint8_t byte : bytes) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }

    return output.str();
}

ByteVector Bytes(const std::string& value) {
    return ByteVector(value.begin(), value.end());
}

bool Check(bool condition, const std::string& description) {
    std::cout
        << (condition ? "[PASS] " : "[FAIL] ")
        << description
        << '\n';

    return condition;
}

lbcrypto::BigInteger ModularDistance(
    const lbcrypto::BigInteger& left,
    const lbcrypto::BigInteger& right,
    const lbcrypto::BigInteger& modulus) {

    const lbcrypto::BigInteger normalizedLeft = left.Mod(modulus);
    const lbcrypto::BigInteger normalizedRight = right.Mod(modulus);

    const lbcrypto::BigInteger direct =
        normalizedLeft >= normalizedRight
            ? normalizedLeft - normalizedRight
            : normalizedRight - normalizedLeft;

    const lbcrypto::BigInteger wrapped = modulus - direct;
    return direct < wrapped ? direct : wrapped;
}

bool TestKnownHashVectors() {
    using namespace federated::deterministic_randomness;

    bool passed = true;

    const auto sha256 = Sha256(Bytes("abc"));
    passed &= Check(
        Hex(sha256) ==
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        "SHA-256 known-answer vector");

    const ByteVector hmacKey(20, 0x0bU);
    const auto hmac = HmacSha256(hmacKey, Bytes("Hi There"));
    passed &= Check(
        Hex(hmac) ==
            "b0344c61d8db38535ca8afceaf0bf12b"
            "881dc200c9833da726e9376c2e32cff7",
        "HMAC-SHA256 RFC 4231 vector");

    const ByteVector shake = Shake128({}, 32);
    passed &= Check(
        Hex(shake) ==
            "7f9c2ba4e88f827d616045507605853e"
            "d73b8093f6efbc88eb1a6eacfa66ef26",
        "SHAKE128 NIST empty-input vector");

    return passed;
}

bool TestDeterministicProtocolRandomness() {
    using namespace federated;
    using namespace federated::deterministic_randomness;

    bool passed = true;
    const auto config = MakeSmokeConfig();
    const SetupId setupId = DeriveSetupId(config.masterSeed);
    const Seed128 indexKey =
        DeriveClientGroupIndexKey(config.masterSeed);

    const RoundId firstRound = DeriveRoundId(config.masterSeed, 0);
    const RoundId secondRound = DeriveRoundId(config.masterSeed, 1);

    const auto permutation = GenerateIndexPermutation(
        indexKey,
        setupId,
        firstRound,
        24);

    const auto repeated = GenerateIndexPermutation(
        indexKey,
        setupId,
        firstRound,
        24);

    const auto nextRound = GenerateIndexPermutation(
        indexKey,
        setupId,
        secondRound,
        24);

    const auto inverse = InvertPermutation(permutation);
    bool inverseWorks = true;

    for (std::size_t index = 0; index < permutation.size(); ++index) {
        inverseWorks &= inverse[permutation[index]] == index;
    }

    passed &= Check(permutation == repeated, "round permutation is deterministic");
    passed &= Check(permutation != nextRound, "round ID separates permutations");
    passed &= Check(inverseWorks, "permutation inverse round-trip");

    const std::uint64_t modulus =
        lbcrypto::LastPrime<lbcrypto::NativeInteger>(
            52,
            2 * config.modulus.ringDimension)
            .ConvertToInt();

    const Seed128 polynomialSeed =
        DeriveSeed128(config.masterSeed, "CRS-SEED");

    const auto polynomial = GenerateUniformPolynomial(
        polynomialSeed,
        "PK-P1",
        config.modulus.ringDimension,
        modulus);

    const auto repeatedPolynomial = GenerateUniformPolynomial(
        polynomialSeed,
        "PK-P1",
        config.modulus.ringDimension,
        modulus);

    const auto otherDomainPolynomial = GenerateUniformPolynomial(
        polynomialSeed,
        "ENC-A",
        config.modulus.ringDimension,
        modulus);

    const bool inRange = std::all_of(
        polynomial.begin(),
        polynomial.end(),
        [modulus](std::uint64_t value) {
            return value < modulus;
        });

    passed &= Check(
        polynomial == repeatedPolynomial,
        "uniform polynomial generation is deterministic");

    passed &= Check(
        polynomial != otherDomainPolynomial,
        "polynomial domain separation changes output");

    passed &= Check(inRange, "uniform polynomial coefficients are in range");

    const ByteVector gaussianContext = EncodeBigEndian64(0);
    const auto gaussian = GenerateClippedGaussian(
        config.masterSeed,
        "SIM-UPDATE",
        gaussianContext,
        24576,
        0.0L,
        0.01L,
        -0.06L,
        0.06L);

    const auto repeatedGaussian = GenerateClippedGaussian(
        config.masterSeed,
        "SIM-UPDATE",
        gaussianContext,
        24576,
        0.0L,
        0.01L,
        -0.06L,
        0.06L);

    const bool gaussianInRange = std::all_of(
        gaussian.begin(),
        gaussian.end(),
        [](double value) {
            return value >= -0.06 && value <= 0.06;
        });

    passed &= Check(
        gaussian == repeatedGaussian,
        "clipped Gaussian generation is deterministic");

    passed &= Check(
        gaussianInRange,
        "clipped Gaussian values respect configured bounds");

    return passed;
}

bool TestV17RandomnessVectors() {
    using namespace federated;
    using namespace federated::deterministic_randomness;

    const Seed128 seed{
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    const SetupId setupId{seed};
    bool passed = true;
    passed &= Check(
        GenerateUniformPolynomial(seed, "TEST", 8, 17) ==
            std::vector<std::uint64_t>{9, 16, 6, 16, 13, 6, 9, 15},
        "v17 UPoly uses big-endian SHAKE words");
    passed &= Check(
        GenerateUniformPolynomial(
            seed, "TEST", 8, (std::uint64_t{1} << 63U) + 1U) ==
            std::vector<std::uint64_t>{
                6682066746797445437ULL, 5506372382094518211ULL,
                7704359256357121651ULL, 999280074432178412ULL,
                2156058280058096826ULL, 6852514248703363859ULL,
                3951679014926302421ULL, 8658367813260540874ULL},
        "v17 UPoly rejects the high tail and continues the SHAKE stream");
    passed &= Check(
        GenerateIndexPermutation(seed, setupId, RoundId{7}, 8) ==
            std::vector<std::size_t>{7, 4, 1, 5, 0, 3, 2, 6},
        "v17 IDX-FY Fisher-Yates known-answer vector");

    ByteVector expectedLocalTag = Bytes("ENC-SP-A");
    expectedLocalTag.insert(
        expectedLocalTag.end(), seed.begin(), seed.end());
    const auto round = EncodeBigEndian64(7);
    expectedLocalTag.insert(
        expectedLocalTag.end(), round.begin(), round.end());
    const auto client = EncodeBigEndian64(2);
    expectedLocalTag.insert(
        expectedLocalTag.end(), client.begin(), client.end());
    const auto block = EncodeBigEndian64(3);
    expectedLocalTag.insert(
        expectedLocalTag.end(), block.begin(), block.end());
    passed &= Check(
        V17LocalCiphertextATag(
            setupId, RoundId{7}, ClientId{2}, 3, "SP-MODEL") ==
            expectedLocalTag &&
        V17LocalCiphertextATag(
            setupId, RoundId{7}, ClientId{2}, 3, "AUX") !=
            expectedLocalTag,
        "v17 model and auxiliary local a domains are distinct");

    ByteVector expectedOffset = Bytes("C2-OFFSET");
    expectedOffset.insert(
        expectedOffset.end(), seed.begin(), seed.end());
    expectedOffset.insert(
        expectedOffset.end(), round.begin(), round.end());
    const auto path = Bytes("AUX");
    expectedOffset.insert(expectedOffset.end(), path.begin(), path.end());
    const auto instance = EncodeBigEndian64(3);
    expectedOffset.insert(
        expectedOffset.end(), instance.begin(), instance.end());
    passed &= Check(
        V17PublicOffsetTag(setupId, RoundId{7}, "AUX", 3) ==
            expectedOffset,
        "v17 public offset tag has the exact byte layout");
    ByteVector expectedDither = Bytes("C2-DITH");
    expectedDither.insert(
        expectedDither.end(), seed.begin(), seed.end());
    expectedDither.insert(
        expectedDither.end(), round.begin(), round.end());
    expectedDither.insert(
        expectedDither.end(), path.begin(), path.end());
    expectedDither.insert(
        expectedDither.end(), instance.begin(), instance.end());
    const auto ditherClient = EncodeBigEndian64(1);
    expectedDither.insert(
        expectedDither.end(), ditherClient.begin(), ditherClient.end());
    passed &= Check(
        V17PublicDitherTag(
            setupId, RoundId{7}, "AUX", 3, ClientId{1}) ==
            expectedDither &&
        expectedDither !=
            V17PublicDitherTag(
                setupId, RoundId{7}, "AUX", 3, ClientId{2}),
        "v17 public dither tag is byte-exact and client-addressed");

    bool privateSamplesValid = true;
    for (std::size_t index = 0; index < 128; ++index) {
        const auto sample = SampleProtocolUniformBelow(
            lbcrypto::BigInteger(17));
        privateSamplesValid &= sample < lbcrypto::BigInteger(17);
    }
    passed &= Check(
        privateSamplesValid &&
            SampleFreshProtocolSeed128() != SampleFreshProtocolSeed128(),
        "private stochastic coins and fresh 128-bit seeds use OpenFHE PRNG");
    return passed;
}

bool TestV17StochasticCompression() {
    using namespace federated::coefficient_compression;
    bool passed = true;
    const lbcrypto::BigInteger q(17);

    passed &= Check(
        CompressStochasticCoefficientWithCoin(
            lbcrypto::BigInteger(2), q, 2, lbcrypto::BigInteger(0)) == 1 &&
        CompressStochasticCoefficientWithCoin(
            lbcrypto::BigInteger(2), q, 2, lbcrypto::BigInteger(8)) == 0 &&
        CompressStochasticCoefficientWithCoin(
            lbcrypto::BigInteger(16), q, 2, lbcrypto::BigInteger(0)) == 0,
        "v17 SQ uses the exact coin threshold and modulo-K wrap");
    passed &= Check(
        ReconstructStochasticCodeSum({1, 1}, q, 2) ==
            lbcrypto::BigInteger(9) &&
        DecompressCoefficient(1, q, 2) * lbcrypto::BigInteger(2) ==
            lbcrypto::BigInteger(8),
        "v17 c2 sums codes before its single integer reconstruction");

    bool invalidCoinRejected = false;
    try {
        (void)CompressStochasticCoefficientWithCoin(
            lbcrypto::BigInteger(2), q, 2, q);
    }
    catch (const std::invalid_argument&) {
        invalidCoinRejected = true;
    }
    passed &= Check(invalidCoinRejected, "stochastic coin must be in Z_Q");

    auto parameters =
        std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
            64, static_cast<std::uint32_t>(2),
            static_cast<std::uint32_t>(30));
    const lbcrypto::BigInteger modulus = parameters->GetModulus();
    bool compositeCoinsValid = true;
    for (std::size_t index = 0; index < 32; ++index) {
        compositeCoinsValid &=
            federated::SampleProtocolUniformBelow(modulus) < modulus;
    }
    passed &= Check(
        compositeCoinsValid,
        "private stochastic coin sampler supports composite two-tower Q");
    const std::vector<std::size_t> selected{0, 5, 17, 31};
    std::vector<lbcrypto::BigInteger> values(
        32, lbcrypto::BigInteger(0));
    for (std::size_t position = 0; position < selected.size(); ++position) {
        values[selected[position]] =
            (modulus * lbcrypto::BigInteger(
                static_cast<std::uint64_t>(position + 1))) /
            lbcrypto::BigInteger(8);
    }
    const auto polynomial =
        federated::rns_polynomial::FromCompositeCoefficients(
            parameters, values);
    const auto first = CompressSelectedCoefficientsStochastic(
        polynomial, 5, selected);
    const auto second = CompressSelectedCoefficientsStochastic(
        polynomial, 5, selected);
    const auto combined = DecompressSelectedSummedStochasticCodes(
        parameters, {first, second}, selected);
    const auto combinedValues = combined.CRTInterpolate().GetValues();
    const auto firstCodes = UnpackUnsignedValues(
        first.bytes, selected.size(), 5);
    const auto secondCodes = UnpackUnsignedValues(
        second.bytes, selected.size(), 5);
    bool codeSumMatches = true;
    for (std::size_t index = 0; index < combinedValues.GetLength(); ++index) {
        const auto it = std::find(selected.begin(), selected.end(), index);
        if (it == selected.end()) {
            codeSumMatches &= combinedValues[index] == lbcrypto::BigInteger(0);
        }
        else {
            const std::size_t position =
                static_cast<std::size_t>(it - selected.begin());
            codeSumMatches &= combinedValues[index] ==
                ReconstructStochasticCodeSum(
                    {firstCodes[position], secondCodes[position]},
                    modulus, 5);
        }
    }
    passed &= Check(
        codeSumMatches,
        "two-tower selected c2 payloads use code-sum then one reconstruction");

    const auto fullFirst = CompressAllCoefficientsStochastic(
        polynomial, 5);
    const auto fullSecond = CompressAllCoefficientsStochastic(
        polynomial, 5);
    const auto fullCombined = DecompressSummedStochasticCodes(
        parameters, {fullFirst, fullSecond});
    const auto fullValues = fullCombined.CRTInterpolate().GetValues();
    const auto fullFirstCodes = UnpackUnsignedValues(
        fullFirst.bytes, 32, 5);
    const auto fullSecondCodes = UnpackUnsignedValues(
        fullSecond.bytes, 32, 5);
    bool fullSumMatches = true;
    for (std::size_t index = 0; index < 32; ++index) {
        fullSumMatches &= fullValues[index] ==
            ReconstructStochasticCodeSum(
                {fullFirstCodes[index], fullSecondCodes[index]},
                modulus, 5);
    }
    passed &= Check(
        fullSumMatches,
        "two-tower full c2 payloads use code-sum then one reconstruction");

    const federated::Seed128 seedD{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const federated::SetupId setupId{seedD};
    std::vector<lbcrypto::BigInteger> smallValues(32);
    for (std::size_t index = 0; index < smallValues.size(); ++index) {
        smallValues[index] = lbcrypto::BigInteger(
            static_cast<std::uint64_t>(100 * index));
    }
    auto smallPolynomial =
        federated::rns_polynomial::FromCompositeCoefficients(
            parameters, smallValues);
    smallPolynomial.SetFormat(Format::EVALUATION);
    const std::vector<federated::ClientId> survivorIds{
        federated::ClientId{0}, federated::ClientId{1}};
    const auto response0 = federated::MakeV17PartialResponse(
        smallPolynomial, parameters, seedD, setupId,
        federated::RoundId{7}, "FULL-MODEL", 0,
        survivorIds[0], survivorIds[0], 50);
    const auto response1 = federated::MakeV17PartialResponse(
        smallPolynomial, parameters, seedD, setupId,
        federated::RoundId{7}, "FULL-MODEL", 0,
        survivorIds[1], survivorIds[0], 50);
    const lbcrypto::DCRTPoly zero(
        parameters, Format::EVALUATION, true);
    const auto recovered = federated::RecoverV17Phase(
        zero, parameters, {response0, response1}, survivorIds,
        seedD, setupId, federated::RoundId{7}, "FULL-MODEL", 0);
    auto recoveredCoefficients = recovered;
    recoveredCoefficients.SetFormat(Format::COEFFICIENT);
    const auto recoveredValues =
        recoveredCoefficients.CRTInterpolate().GetValues();
    const lbcrypto::BigInteger responseBound =
        lbcrypto::BigInteger(8) *
        (modulus / lbcrypto::BigInteger(std::uint64_t{1} << 50U) +
            lbcrypto::BigInteger(1));
    bool ditherCancelled = true;
    for (std::size_t index = 0; index < smallValues.size(); ++index) {
        ditherCancelled &= ModularDistance(
            recoveredValues[index],
            (smallValues[index] + smallValues[index]).Mod(modulus),
            modulus) <= responseBound;
    }
    passed &= Check(ditherCancelled,
        "v17 FULL code-sum cancels public offset and both client dithers");
    return passed;
}

bool TestProtocolParameters() {
    using namespace federated;

    bool passed = true;
    const ProtocolExperimentConfig config = MakeSmokeConfig();
    const ValidationReport staticReport = ValidateProtocolParameters(config);
    const DerivedDimensions dimensions = DeriveDimensions(config);

    passed &= Check(!staticReport.HasErrors(), "smoke fixture parameters validate");
    passed &= Check(
        dimensions.smallRingDimension == 2048,
        "N=8192 and k=4 give small-ring dimension 2048");

    passed &= Check(dimensions.chunkCount == 12, "d=24576 gives 12 chunks");
    passed &= Check(
        dimensions.clientCiphertextsPerRound == 1,
        "four selected chunks fit in one client RLWE ciphertext");

    const ProtocolExperimentConfig resnetConfig =
        MakeResNet20Cifar10FixtureConfig();
    const DerivedDimensions resnetDimensions =
        DeriveDimensions(resnetConfig);
    passed &= Check(
        !ValidateProtocolParameters(resnetConfig).HasErrors() &&
            resnetDimensions.chunkCount == 134 &&
            resnetDimensions.selectedChunksPerClient == 40 &&
            resnetDimensions.clientCiphertextsPerRound == 10,
        "ResNet-20 rho=0.30 derives 40 selected chunks and ten pack groups");

    const std::uint64_t modulus =
        lbcrypto::LastPrime<lbcrypto::NativeInteger>(
            52,
            2 * config.modulus.ringDimension)
            .ConvertToInt();

    const NumericDiagnostics diagnostics =
        ComputeNumericDiagnostics(config, modulus);

    const ValidationReport numericReport =
        ValidateProtocolParameters(config, modulus);

    passed &= Check(!numericReport.HasErrors(), "actual-q parameters validate");
    passed &= Check(
        numericReport.HasWarnings(),
        "aggressive c1/c2 profile produces precision warnings");

    passed &= Check(
        diagnostics.centeredCorrectnessMargin > 0.0L,
        "the explicit model factor leaves positive centered plaintext margin");
    passed &= Check(
        diagnostics.auxCenteredCorrectnessMargin > 0.0L,
        "the explicit auxiliary factor leaves positive centered plaintext margin");

    ProtocolExperimentConfig invalid = config;
    invalid.scaling.factor = std::uint64_t{1} << 51U;
    passed &= Check(
        ValidateProtocolParameters(invalid, modulus).HasErrors(),
        "an explicit model factor too large for actual Q is rejected");

    invalid = config;
    invalid.auxScaling.factor = std::uint64_t{1} << 51U;
    passed &= Check(
        ValidateProtocolParameters(invalid, modulus).HasErrors(),
        "an explicit auxiliary factor too large for actual Q is rejected");

    invalid = config;
    invalid.sparsification.minimumCohort = 1;
    passed &= Check(
        ValidateProtocolParameters(invalid).HasErrors(),
        "tau < collusionBound + 2 is rejected");

    invalid = config;
    invalid.sparsification.selectionRatio = 1.01L;
    passed &= Check(
        ValidateProtocolParameters(invalid).HasErrors(),
        "a sparsification ratio above one is rejected");

    invalid = config;
    invalid.modelDeliveryScale = 4096.0L;
    passed &= Check(
        ValidateProtocolParameters(invalid).HasErrors(),
        "Delta_d*U_max above 2^(cd-1)-1 is rejected before execution");

    invalid = config;
    invalid.auxDeliveryScale = 4096.0L;
    passed &= Check(
        ValidateProtocolParameters(invalid).HasErrors(),
        "Delta_d_aux*U_max_aux above 2^(cd_aux-1)-1 is rejected");

    return passed;
}

bool TestExactQuantizerCompatibility() {
    using I = lbcrypto::BigInteger;
    using namespace federated::coefficient_compression;
    bool passed = true;
    std::uint64_t random = 123456789;
    auto next = [&]() { random ^= random << 13; random ^= random >> 7;
                       random ^= random << 17; return random; };
    const std::vector<I> moduli{I(8), I(17), (I(1) << 63) - I(25),
        (I(1) << 64) - I(59), (I(1) << 64) - I(1),
        I(1) << 64, (I(1) << 95) - I(19)};
    for (const auto& q : moduli) {
        for (std::uint32_t bits = 1; bits <= 63; ++bits) {
            const I levels = I(1) << bits;
            std::vector<I> inputs{I(0), I(1), q / I(2), q - I(1), q, q + I(1),
                                  (q << 4) + I(7)};
            for (int i = 0; i < 24; ++i)
                inputs.push_back(((I(next()) << 64) + I(next())).Mod(q));
            for (const auto& x : inputs) {
                const I scaled = x.Mod(q) * levels;
                const auto expected = ((scaled + (q >> 1)) / q).Mod(levels).ConvertToInt();
                passed &= CompressCoefficient(x, q, bits) == expected;
                if (bits < 2 || (levels << 1) > q) continue;
                const I remainder = scaled.Mod(q);
                std::vector<I> coins{I(0), q - I(1), remainder};
                if (remainder > I(0)) coins.push_back(remainder - I(1));
                for (const auto& coin : coins) {
                    I code = scaled / q;
                    if (coin < remainder) code += I(1);
                    passed &= CompressStochasticCoefficientWithCoin(x, q, bits, coin) ==
                              code.Mod(levels).ConvertToInt();
                }
                bool rejected = false;
                try { (void)CompressStochasticCoefficientWithCoin(x, q, bits, q); }
                catch (const std::invalid_argument&) { rejected = true; }
                passed &= rejected;
            }
        }
    }
    return Check(passed, "quantizer matches BigInteger oracle at rounding, wrap, 64-bit and fallback boundaries");
}

bool TestCodecByteCompatibility() {
    using namespace federated::coefficient_compression;
    bool passed = true;
    std::uint64_t state = 1234567;
    for (std::uint32_t bits = 1; bits <= 63; ++bits) {
        for (const std::size_t count : {0U, 1U, 2U, 7U, 8U, 9U, 257U}) {
            const auto mask = (std::uint64_t{1} << bits) - 1;
            std::vector<std::uint64_t> values(count);
            for (auto& value : values) {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                value = state & mask;
            }
            if (count) values.front() = mask;
            if (count > 1) values.back() = 0;
            // Independent legacy bit-by-bit oracle, including MSB-first padding.
            std::vector<std::uint8_t> expected((count * bits + 7) / 8, 0);
            std::size_t offset = 0;
            for (const auto value : values)
                for (std::uint32_t bit = 0; bit < bits; ++bit, ++offset)
                    expected[offset / 8] |= ((value >> (bits - 1 - bit)) & 1U) << (7 - offset % 8);
            passed &= PackUnsignedValues(values, bits) == expected;
            passed &= UnpackUnsignedValues(expected, count, bits) == values;
            if (count * bits % 8) {
                auto bad = expected; bad.back() |= 1;
                bool rejected = false;
                try { (void)UnpackUnsignedValues(bad, count, bits); }
                catch (const std::invalid_argument&) { rejected = true; }
                passed &= rejected;
            }
            auto bad = expected; bad.push_back(0);
            bool rejected = false;
            try { (void)UnpackUnsignedValues(bad, count, bits); }
            catch (const std::invalid_argument&) { rejected = true; }
            passed &= rejected;
        }
        bool rejected = false;
        try { (void)PackUnsignedValues({std::uint64_t{1} << bits}, bits); }
        catch (const std::invalid_argument&) { rejected = true; }
        passed &= rejected;
    }
    return Check(passed, "codec matches legacy bytes for every width 1..63 and rejects malformed payloads");
}

bool TestBitPackingAndSignedQuantization() {
    using namespace federated::coefficient_compression;

    bool passed = true;
    const std::vector<std::uint64_t> values{
        0, 1, 7, 0x12345, (std::uint64_t{1} << 19U) - 1U};

    const auto packed = PackUnsignedValues(values, 19);
    const auto unpacked = UnpackUnsignedValues(packed, values.size(), 19);

    passed &= Check(values == unpacked, "19-bit unsigned packing round-trip");
    passed &= Check(
        packed.size() == (values.size() * 19U + 7U) / 8U,
        "packed unsigned payload has canonical byte length");

    const std::vector<long double> realValues{-1.0L, -0.25L, 0.0L, 0.5L, 1.0L};
    std::vector<std::int64_t> quantized;
    bool signedAccuracy = true;

    for (const long double value : realValues) {
        const std::int64_t encoded = QuantizeSigned(value, 13, 1.0L, false);
        const long double decoded = DequantizeSigned(encoded, 13, 1.0L);
        quantized.push_back(encoded);
        signedAccuracy &= std::fabs(decoded - value) <= 1.0L / 4095.0L;
    }

    const auto packedSigned = PackSignedValues(quantized, 13);
    const auto unpackedSigned =
        UnpackSignedValues(packedSigned, quantized.size(), 13);

    passed &= Check(signedAccuracy, "13-bit signed quantization error is bounded");
    passed &= Check(
        quantized == unpackedSigned,
        "13-bit signed packing round-trip");

    bool noClippingRejected = false;
    try {
        (void)QuantizeSignedWithScale(1.0001L, 13, 4095.0L);
    }
    catch (const std::out_of_range&) {
        noClippingRejected = true;
    }
    passed &= Check(
        noClippingRejected,
        "explicit delivery-scale quantization rejects instead of clipping");
    const std::int64_t scaled = QuantizeSignedWithScale(
        0.25L,
        13,
        4095.0L);
    passed &= Check(
        std::abs(
            DequantizeSignedWithScale(scaled, 13, 4095.0L) - 0.25L) <=
                0.5L / 4095.0L,
        "explicit delivery-scale quantization round-trips within half a step");

    return passed;
}

bool TestCoefficientCompression() {
    using namespace federated::coefficient_compression;

    bool passed = true;
    constexpr std::uint32_t bits = 19;
    const auto config = federated::MakeSmokeConfig();
    const std::uint32_t cyclotomicOrder = static_cast<std::uint32_t>(
        2 * config.modulus.ringDimension);

    auto parameters =
        std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
            cyclotomicOrder,
            static_cast<std::uint32_t>(1),
            static_cast<std::uint32_t>(52));

    const lbcrypto::BigInteger modulus = parameters->GetModulus();
    const std::vector<lbcrypto::BigInteger> samples{
        lbcrypto::BigInteger(0),
        lbcrypto::BigInteger(1),
        modulus >> 1,
        modulus - lbcrypto::BigInteger(1)};

    const lbcrypto::BigInteger levels =
        lbcrypto::BigInteger(std::uint64_t{1} << bits);

    const lbcrypto::BigInteger errorBound =
        (modulus + levels - lbcrypto::BigInteger(1)) / levels +
        lbcrypto::BigInteger(1);

    bool scalarBounded = true;
    for (const auto& sample : samples) {
        const std::uint64_t compressed =
            CompressCoefficient(sample, modulus, bits);

        const lbcrypto::BigInteger decompressed =
            DecompressCoefficient(compressed, modulus, bits);

        scalarBounded &=
            ModularDistance(sample, decompressed, modulus) <= errorBound;
    }

    passed &= Check(scalarBounded, "scalar Comp/Decomp modular error is bounded");

    lbcrypto::DCRTPoly polynomial(
        parameters,
        Format::COEFFICIENT,
        true);

    auto& tower = polynomial.GetAllElements().front();
    const lbcrypto::NativeInteger nativeModulus = tower.GetModulus();
    lbcrypto::NativeVector towerValues(
        config.modulus.ringDimension,
        nativeModulus);

    for (std::size_t index = 0; index < towerValues.GetLength(); ++index) {
        towerValues[index] = lbcrypto::NativeInteger(
            (static_cast<std::uint64_t>(index) * 104729U + 17U) %
            nativeModulus.ConvertToInt());
    }

    tower.SetValues(std::move(towerValues), Format::COEFFICIENT);

    const auto allPayload = CompressAllCoefficients(polynomial, bits);
    const lbcrypto::DCRTPoly allRestored =
        DecompressAllCoefficients(parameters, allPayload);

    const lbcrypto::Poly originalPoly = polynomial.CRTInterpolate();
    const lbcrypto::Poly restoredPoly = allRestored.CRTInterpolate();
    bool allBounded = true;

    for (std::size_t index = 0;
         index < originalPoly.GetValues().GetLength();
         ++index) {

        allBounded &=
            ModularDistance(
                originalPoly.GetValues()[index],
                restoredPoly.GetValues()[index],
                modulus) <= errorBound;
    }

    passed &= Check(allBounded, "full DCRT Comp/Decomp error is bounded");

    const std::vector<std::size_t> selected{0, 7, 1023, 2048, 4095};
    const auto selectedPayload =
        CompressSelectedCoefficients(polynomial, 15, selected);

    const lbcrypto::DCRTPoly selectedRestored =
        DecompressSelectedCoefficients(
            parameters,
            selectedPayload,
            selected);

    const lbcrypto::Poly selectedPoly = selectedRestored.CRTInterpolate();
    const std::set<std::size_t> selectedSet(selected.begin(), selected.end());
    bool unselectedAreZero = true;
    bool selectedAreBounded = true;

    lbcrypto::BigInteger selectedLevels(1);
    selectedLevels <<= 15;

    const lbcrypto::BigInteger selectedErrorBound =
        (modulus + selectedLevels - lbcrypto::BigInteger(1)) /
            selectedLevels +
        lbcrypto::BigInteger(1);

    for (std::size_t index = 0;
         index < selectedPoly.GetValues().GetLength();
         ++index) {

        if (selectedSet.count(index) == 0) {
            unselectedAreZero &=
                selectedPoly.GetValues()[index] == lbcrypto::BigInteger(0);
        }
        else {
            selectedAreBounded &=
                ModularDistance(
                    originalPoly.GetValues()[index],
                    selectedPoly.GetValues()[index],
                    modulus) <= selectedErrorBound;
        }
    }

    passed &= Check(
        selectedPayload.coefficientCount == selected.size(),
        "selected compression transmits only requested coefficients");

    passed &= Check(
        unselectedAreZero,
        "selected decompression zero-fills omitted coefficients");

    passed &= Check(
        selectedAreBounded,
        "selected DCRT Comp/Decomp error is bounded");

    return passed;
}

bool TestMultiTowerCoefficientCompression() {
    using namespace federated::coefficient_compression;
    using namespace federated::rns_polynomial;

    bool passed = true;
    constexpr std::uint32_t bits = 19;
    constexpr std::uint32_t cyclotomicOrder = 64;
    constexpr std::size_t ringDimension = cyclotomicOrder / 2;

    for (std::uint32_t towerCount = 1;
         towerCount <= 3;
         ++towerCount) {

        auto parameters =
            std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
                cyclotomicOrder,
                towerCount,
                static_cast<std::uint32_t>(50));

        const lbcrypto::BigInteger modulus = parameters->GetModulus();
        const lbcrypto::BigInteger levels(
            std::uint64_t{1} << bits);
        const lbcrypto::BigInteger errorBound =
            (modulus + levels - lbcrypto::BigInteger(1)) / levels +
            lbcrypto::BigInteger(1);

        std::vector<lbcrypto::BigInteger> coefficients(
            ringDimension,
            lbcrypto::BigInteger(0));

        for (std::size_t index = 0;
             index < ringDimension;
             ++index) {

            coefficients[index] =
                (lbcrypto::BigInteger(
                     static_cast<std::uint64_t>(index + 1)) *
                     (modulus / lbcrypto::BigInteger(
                         static_cast<std::uint64_t>(ringDimension + 1))) +
                 lbcrypto::BigInteger(
                     static_cast<std::uint64_t>(
                         index * 104729U + 17U)))
                    .Mod(modulus);
        }

        const lbcrypto::DCRTPoly polynomial =
            FromCompositeCoefficients(parameters, coefficients);
        const auto payload = CompressAllCoefficients(polynomial, bits);
        const lbcrypto::DCRTPoly restored =
            DecompressAllCoefficients(parameters, payload);
        const auto restoredCoefficients =
            InterpolateCoefficients(restored);

        bool bounded = restoredCoefficients.size() == coefficients.size();
        for (std::size_t index = 0;
             bounded && index < coefficients.size();
             ++index) {

            bounded &= ModularDistance(
                coefficients[index],
                restoredCoefficients[index],
                modulus) <= errorBound;
        }

        passed &= Check(
            bounded,
            std::to_string(towerCount) +
                "-tower composite-Q compression round-trip is bounded");

        const std::vector<std::size_t> selected{0, 5, 17, 31};
        const auto selectedPayload =
            CompressSelectedCoefficients(polynomial, bits, selected);
        const auto selectedRestored = InterpolateCoefficients(
            DecompressSelectedCoefficients(
                parameters,
                selectedPayload,
                selected));

        bool selectedCorrect = true;
        const std::set<std::size_t> selectedSet(
            selected.begin(),
            selected.end());

        for (std::size_t index = 0;
             index < ringDimension;
             ++index) {

            if (selectedSet.count(index) == 0) {
                selectedCorrect &=
                    selectedRestored[index] == lbcrypto::BigInteger(0);
            }
            else {
                selectedCorrect &= ModularDistance(
                    coefficients[index],
                    selectedRestored[index],
                    modulus) <= errorBound;
            }
        }

        passed &= Check(
            selectedCorrect,
            std::to_string(towerCount) +
                "-tower selected compression preserves composite-Q semantics");
    }

    return passed;
}

}  // namespace

int RunProtocolStage1Tests() {
    bool passed = true;

    auto v17 = federated::MakeSmokeConfig();
    v17.protocolVersion = "v17";
    v17.modulus.qModulusBits = {52};
    v17.modulus.hybridLargeDigits = 1;
    bool rejected = false;
    try { federated::ValidateProtocolParameters(v17).ThrowIfInvalid(); }
    catch (const std::exception&) { rejected = true; }
    passed &= Check(rejected, "v17 rejects the legacy one-limb configuration");
    v17.modulus.qModulusBits = {28, 28};
    v17.modulus.hybridLargeDigits = 2;
    v17.sparsification.collusionBound = 1;
    passed &= TestKnownHashVectors();
    passed &= TestDeterministicProtocolRandomness();
    passed &= TestV17RandomnessVectors();
    passed &= TestV17StochasticCompression();
    passed &= TestProtocolParameters();
    passed &= TestExactQuantizerCompatibility();
    passed &= TestCodecByteCompatibility();
    passed &= TestBitPackingAndSignedQuantization();
    passed &= TestCoefficientCompression();
    passed &= TestMultiTowerCoefficientCompression();

    std::cout
        << '\n'
        << (passed
                ? "All protocol stage-1 tests passed."
                : "One or more protocol stage-1 tests failed.")
        << '\n';

    return passed ? 0 : 1;
}
