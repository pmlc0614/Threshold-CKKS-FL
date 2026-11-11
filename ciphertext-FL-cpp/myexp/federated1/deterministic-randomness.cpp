#include "deterministic-randomness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace federated {
namespace deterministic_randomness {
namespace {

std::uint32_t RotateRight32(std::uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32U - shift));
}

std::uint64_t RotateLeft64(std::uint64_t value, unsigned int shift) {
    if (shift == 0) {
        return value;
    }

    return (value << shift) | (value >> (64U - shift));
}

std::uint32_t LoadBigEndian32(const std::uint8_t* input) {
    return (static_cast<std::uint32_t>(input[0]) << 24U) |
           (static_cast<std::uint32_t>(input[1]) << 16U) |
           (static_cast<std::uint32_t>(input[2]) << 8U) |
           static_cast<std::uint32_t>(input[3]);
}

std::uint64_t LoadBigEndian64(const std::uint8_t* input) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value = (value << 8U) | input[index];
    }
    return value;
}

void StoreBigEndian32(std::uint32_t value, std::uint8_t* output) {
    output[0] = static_cast<std::uint8_t>(value >> 24U);
    output[1] = static_cast<std::uint8_t>(value >> 16U);
    output[2] = static_cast<std::uint8_t>(value >> 8U);
    output[3] = static_cast<std::uint8_t>(value);
}

std::uint64_t LoadLittleEndian64(const std::uint8_t* input) {
    std::uint64_t value = 0;

    for (std::size_t index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(input[index]) << (8U * index);
    }

    return value;
}

void StoreLittleEndian64(std::uint64_t value, std::uint8_t* output) {
    for (std::size_t index = 0; index < 8; ++index) {
        output[index] = static_cast<std::uint8_t>(value >> (8U * index));
    }
}

class Sha256State {
public:
    Sha256State()
        : state_{
              0x6a09e667U,
              0xbb67ae85U,
              0x3c6ef372U,
              0xa54ff53aU,
              0x510e527fU,
              0x9b05688cU,
              0x1f83d9abU,
              0x5be0cd19U} {}

    void Update(const std::uint8_t* data, std::size_t length) {
        if (length > 0 && data == nullptr) {
            throw std::invalid_argument("SHA-256 input pointer is null");
        }

        totalBytes_ += length;

        while (length > 0) {
            const std::size_t copied =
                std::min(length, block_.size() - blockBytes_);

            std::memcpy(
                block_.data() + blockBytes_,
                data,
                copied);

            blockBytes_ += copied;
            data += copied;
            length -= copied;

            if (blockBytes_ == block_.size()) {
                Transform(block_.data());
                blockBytes_ = 0;
            }
        }
    }

    Digest256 Finalize() {
        if (totalBytes_ >
            std::numeric_limits<std::uint64_t>::max() / 8U) {

            throw std::overflow_error("SHA-256 input is too long");
        }

        const std::uint64_t bitLength =
            static_cast<std::uint64_t>(totalBytes_) * 8U;

        block_[blockBytes_++] = 0x80U;

        if (blockBytes_ > 56) {
            std::fill(block_.begin() + blockBytes_, block_.end(), 0U);
            Transform(block_.data());
            blockBytes_ = 0;
        }

        std::fill(block_.begin() + blockBytes_, block_.begin() + 56, 0U);

        for (std::size_t index = 0; index < 8; ++index) {
            block_[63 - index] =
                static_cast<std::uint8_t>(bitLength >> (8U * index));
        }

        Transform(block_.data());

        Digest256 digest{};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            StoreBigEndian32(state_[index], digest.data() + 4 * index);
        }

        return digest;
    }

private:
    void Transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> roundConstants{
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

        std::array<std::uint32_t, 64> words{};

        for (std::size_t index = 0; index < 16; ++index) {
            words[index] = LoadBigEndian32(block + 4 * index);
        }

        for (std::size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t s0 =
                RotateRight32(words[index - 15], 7) ^
                RotateRight32(words[index - 15], 18) ^
                (words[index - 15] >> 3U);

            const std::uint32_t s1 =
                RotateRight32(words[index - 2], 17) ^
                RotateRight32(words[index - 2], 19) ^
                (words[index - 2] >> 10U);

            words[index] =
                words[index - 16] + s0 +
                words[index - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sum1 =
                RotateRight32(e, 6) ^
                RotateRight32(e, 11) ^
                RotateRight32(e, 25);

            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 =
                h + sum1 + choice + roundConstants[index] + words[index];

            const std::uint32_t sum0 =
                RotateRight32(a, 2) ^
                RotateRight32(a, 13) ^
                RotateRight32(a, 22);

            const std::uint32_t majority =
                (a & b) ^ (a & c) ^ (b & c);

            const std::uint32_t temporary2 = sum0 + majority;

            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> block_{};
    std::size_t blockBytes_ = 0;
    std::size_t totalBytes_ = 0;
};

void KeccakF1600(std::array<std::uint64_t, 25>* state) {
    static constexpr std::array<std::uint64_t, 24> roundConstants{
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL};

    static constexpr std::array<unsigned int, 25> rotationOffsets{
        0, 1, 62, 28, 27,
        36, 44, 6, 55, 20,
        3, 10, 43, 25, 39,
        41, 45, 15, 21, 8,
        18, 2, 61, 56, 14};

    for (const std::uint64_t roundConstant : roundConstants) {
        std::array<std::uint64_t, 5> columnParity{};
        std::array<std::uint64_t, 5> theta{};

        for (std::size_t x = 0; x < 5; ++x) {
            columnParity[x] =
                (*state)[x] ^ (*state)[x + 5] ^
                (*state)[x + 10] ^ (*state)[x + 15] ^
                (*state)[x + 20];
        }

        for (std::size_t x = 0; x < 5; ++x) {
            theta[x] =
                columnParity[(x + 4) % 5] ^
                RotateLeft64(columnParity[(x + 1) % 5], 1);
        }

        for (std::size_t y = 0; y < 5; ++y) {
            for (std::size_t x = 0; x < 5; ++x) {
                (*state)[x + 5 * y] ^= theta[x];
            }
        }

        std::array<std::uint64_t, 25> rhoPi{};
        for (std::size_t y = 0; y < 5; ++y) {
            for (std::size_t x = 0; x < 5; ++x) {
                const std::size_t destinationX = y;
                const std::size_t destinationY = (2 * x + 3 * y) % 5;

                rhoPi[destinationX + 5 * destinationY] =
                    RotateLeft64(
                        (*state)[x + 5 * y],
                        rotationOffsets[x + 5 * y]);
            }
        }

        for (std::size_t y = 0; y < 5; ++y) {
            for (std::size_t x = 0; x < 5; ++x) {
                (*state)[x + 5 * y] =
                    rhoPi[x + 5 * y] ^
                    ((~rhoPi[(x + 1) % 5 + 5 * y]) &
                     rhoPi[(x + 2) % 5 + 5 * y]);
            }
        }

        (*state)[0] ^= roundConstant;
    }
}

class Shake128Xof {
public:
    explicit Shake128Xof(const ByteVector& input) {
        constexpr std::size_t rateBytes = 168;
        std::size_t inputOffset = 0;

        while (input.size() - inputOffset >= rateBytes) {
            for (std::size_t lane = 0; lane < rateBytes / 8; ++lane) {
                state_[lane] ^=
                    LoadLittleEndian64(input.data() + inputOffset + 8 * lane);
            }

            KeccakF1600(&state_);
            inputOffset += rateBytes;
        }

        std::array<std::uint8_t, rateBytes> finalBlock{};
        const std::size_t remaining = input.size() - inputOffset;

        if (remaining > 0) {
            std::copy(
                input.begin() + static_cast<std::ptrdiff_t>(inputOffset),
                input.end(),
                finalBlock.begin());
        }

        finalBlock[remaining] ^= 0x1fU;
        finalBlock[rateBytes - 1] ^= 0x80U;

        for (std::size_t lane = 0; lane < rateBytes / 8; ++lane) {
            state_[lane] ^= LoadLittleEndian64(finalBlock.data() + 8 * lane);
        }

        KeccakF1600(&state_);
        RefreshOutputBlock();
    }

    void Squeeze(std::uint8_t* output, std::size_t length) {
        if (length > 0 && output == nullptr) {
            throw std::invalid_argument("SHAKE128 output pointer is null");
        }

        while (length > 0) {
            if (outputOffset_ == outputBlock_.size()) {
                KeccakF1600(&state_);
                RefreshOutputBlock();
            }

            const std::size_t copied =
                std::min(length, outputBlock_.size() - outputOffset_);

            std::memcpy(output, outputBlock_.data() + outputOffset_, copied);
            output += copied;
            outputOffset_ += copied;
            length -= copied;
        }
    }

    std::uint64_t SqueezeUint64() {
        std::array<std::uint8_t, 8> bytes{};
        Squeeze(bytes.data(), bytes.size());
        return LoadLittleEndian64(bytes.data());
    }

private:
    void RefreshOutputBlock() {
        for (std::size_t lane = 0; lane < outputBlock_.size() / 8; ++lane) {
            StoreLittleEndian64(state_[lane], outputBlock_.data() + 8 * lane);
        }

        outputOffset_ = 0;
    }

    std::array<std::uint64_t, 25> state_{};
    std::array<std::uint8_t, 168> outputBlock_{};
    std::size_t outputOffset_ = 0;
};

void AppendString(ByteVector* output, const std::string& value) {
    output->insert(output->end(), value.begin(), value.end());
}

template <std::size_t Size>
void AppendArray(
    ByteVector* output,
    const std::array<std::uint8_t, Size>& value) {

    output->insert(output->end(), value.begin(), value.end());
}

void AppendBigEndian64(ByteVector* output, std::uint64_t value) {
    const ByteVector encoded = EncodeBigEndian64(value);
    output->insert(output->end(), encoded.begin(), encoded.end());
}

std::uint64_t SampleUniformBelow(
    Shake128Xof* xof,
    std::uint64_t bound) {

    if (bound == 0) {
        throw std::invalid_argument("uniform sampling bound must be positive");
    }

    // 2^64 mod bound, evaluated without representing 2^64 in uint64_t.
    // If this is zero, L = 2^64 and every word is accepted.
    const std::uint64_t rejectedTail =
        static_cast<std::uint64_t>(-bound) % bound;
    const std::uint64_t lastAccepted =
        std::numeric_limits<std::uint64_t>::max() - rejectedTail;

    while (true) {
        std::array<std::uint8_t, 8> word{};
        xof->Squeeze(word.data(), word.size());
        const std::uint64_t candidate = LoadBigEndian64(word.data());
        if (candidate <= lastAccepted) {
            return candidate % bound;
        }
    }
}

long double SampleOpenUnitInterval(Shake128Xof* xof) {
    /* Use the high 53 bits and a half-unit offset, keeping the Box-Muller logarithm input strictly between zero and one. */
    const std::uint64_t mantissa = xof->SqueezeUint64() >> 11U;
    return std::ldexp(
        static_cast<long double>(mantissa) + 0.5L,
        -53);
}

}  // namespace

ByteVector EncodeBigEndian64(std::uint64_t value) {
    ByteVector encoded(8);

    for (std::size_t index = 0; index < encoded.size(); ++index) {
        encoded[index] =
            static_cast<std::uint8_t>(value >> (56U - 8U * index));
    }

    return encoded;
}

ByteVector Shake128(
    const ByteVector& input,
    std::size_t outputLength) {

    ByteVector output(outputLength);
    Shake128Xof xof(input);
    xof.Squeeze(output.data(), output.size());
    return output;
}

Digest256 Sha256(const ByteVector& input) {
    Sha256State state;
    state.Update(input.data(), input.size());
    return state.Finalize();
}

Digest256 HmacSha256(
    const ByteVector& key,
    const ByteVector& message) {

    constexpr std::size_t blockSize = 64;
    std::array<std::uint8_t, blockSize> normalizedKey{};

    if (key.size() > blockSize) {
        const Digest256 keyDigest = Sha256(key);
        std::copy(keyDigest.begin(), keyDigest.end(), normalizedKey.begin());
    }
    else {
        std::copy(key.begin(), key.end(), normalizedKey.begin());
    }

    std::array<std::uint8_t, blockSize> innerPad{};
    std::array<std::uint8_t, blockSize> outerPad{};

    for (std::size_t index = 0; index < blockSize; ++index) {
        innerPad[index] = normalizedKey[index] ^ 0x36U;
        outerPad[index] = normalizedKey[index] ^ 0x5cU;
    }

    ByteVector innerInput(innerPad.begin(), innerPad.end());
    innerInput.insert(innerInput.end(), message.begin(), message.end());
    const Digest256 innerDigest = Sha256(innerInput);

    ByteVector outerInput(outerPad.begin(), outerPad.end());
    outerInput.insert(outerInput.end(), innerDigest.begin(), innerDigest.end());
    return Sha256(outerInput);
}

Seed128 DeriveSeed128(
    const MasterSeed256& masterSeed,
    const std::string& domain,
    const ByteVector& context) {

    if (domain.empty()) {
        throw std::invalid_argument("seed derivation domain must not be empty");
    }

    ByteVector input;
    AppendString(&input, "FEDERATED-V17-SEED");
    AppendBigEndian64(&input, domain.size());
    AppendString(&input, domain);
    AppendArray(&input, masterSeed);
    AppendBigEndian64(&input, context.size());
    input.insert(input.end(), context.begin(), context.end());

    const ByteVector output = Shake128(input, Seed128{}.size());
    Seed128 seed{};
    std::copy(output.begin(), output.end(), seed.begin());
    return seed;
}

SetupId DeriveSetupId(const MasterSeed256& masterSeed) {
    return SetupId{DeriveSeed128(masterSeed, "SETUP-ID")};
}

SetupId DeriveSetupId(
    const MasterSeed256& masterSeed,
    const ByteVector& canonicalSetupDescriptor) {

    if (canonicalSetupDescriptor.empty()) {
        throw std::invalid_argument(
            "canonical setup descriptor must not be empty");
    }

    return SetupId{DeriveSeed128(
        masterSeed,
        "SETUP-ID-V11",
        canonicalSetupDescriptor)};
}

Seed128 DeriveClientGroupIndexKey(
    const MasterSeed256& masterSeed) {

    return DeriveSeed128(masterSeed, "CLIENT-GROUP-IDX-KEY");
}

Seed128 DeriveClientGroupIndexKey(
    const MasterSeed256& masterSeed,
    const SetupId& setupId) {

    return DeriveSeed128(
        masterSeed,
        "CLIENT-GROUP-IDX-KEY-V11",
        ByteVector(setupId.bytes.begin(), setupId.bytes.end()));
}

RoundId DeriveRoundId(
    const MasterSeed256& masterSeed,
    std::uint64_t roundNumber) {

    const ByteVector context = EncodeBigEndian64(roundNumber);
    const Seed128 seed = DeriveSeed128(masterSeed, "ROUND-ID", context);

    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value = (value << 8U) | seed[index];
    }

    return RoundId{value};
}

Seed128 DeriveIndexPermutationSeed(
    const Seed128& clientGroupIndexKey,
    const SetupId& setupId,
    RoundId roundId,
    std::size_t chunkCount) {

    if (chunkCount == 0) {
        throw std::invalid_argument("chunk count must be positive");
    }

    ByteVector key(clientGroupIndexKey.begin(), clientGroupIndexKey.end());
    ByteVector message;
    AppendString(&message, "IDX-PERM");
    AppendArray(&message, setupId.bytes);
    AppendBigEndian64(&message, roundId.value);
    AppendBigEndian64(&message, chunkCount);

    const Digest256 digest = HmacSha256(key, message);
    Seed128 seed{};
    std::copy_n(digest.begin(), seed.size(), seed.begin());
    return seed;
}

std::vector<std::size_t> GenerateIndexPermutation(
    const Seed128& clientGroupIndexKey,
    const SetupId& setupId,
    RoundId roundId,
    std::size_t chunkCount) {

    const Seed128 permutationSeed =
        DeriveIndexPermutationSeed(
            clientGroupIndexKey,
            setupId,
            roundId,
            chunkCount);

    ByteVector input;
    AppendString(&input, "IDX-FY");
    AppendArray(&input, permutationSeed);
    AppendBigEndian64(&input, chunkCount);

    Shake128Xof xof(input);
    std::vector<std::size_t> permutation(chunkCount);

    for (std::size_t index = 0; index < chunkCount; ++index) {
        permutation[index] = index;
    }

    for (std::size_t upper = chunkCount; upper > 1; --upper) {
        const std::uint64_t selected =
            SampleUniformBelow(&xof, static_cast<std::uint64_t>(upper));

        std::swap(
            permutation[upper - 1],
            permutation[static_cast<std::size_t>(selected)]);
    }

    return permutation;
}

std::vector<std::size_t> InvertPermutation(
    const std::vector<std::size_t>& permutation) {

    std::vector<std::size_t> inverse(
        permutation.size(),
        kInvalidPosition);

    for (std::size_t source = 0; source < permutation.size(); ++source) {
        const std::size_t destination = permutation[source];

        if (destination >= permutation.size() ||
            inverse[destination] != kInvalidPosition) {

            throw std::invalid_argument("input is not a permutation");
        }

        inverse[destination] = source;
    }

    return inverse;
}

ByteVector V17LocalCiphertextAContext(
    const SetupId& setupId, RoundId roundId, ClientId clientId,
    std::uint64_t blockIndex) {
    ByteVector context(setupId.bytes.begin(), setupId.bytes.end());
    AppendBigEndian64(&context, roundId.value);
    AppendBigEndian64(&context, clientId.value);
    AppendBigEndian64(&context, blockIndex);
    return context;
}

ByteVector V17LocalCiphertextATag(
    const SetupId& setupId,
    RoundId roundId,
    ClientId clientId,
    std::uint64_t blockIndex,
    const std::string& path) {

    const char* domain = nullptr;
    if (path == "SP-MODEL") {
        domain = "ENC-SP-A";
    }
    else if (path == "FULL-MODEL") {
        domain = "ENC-FULL-A";
    }
    else if (path == "AUX") {
        domain = "ENC-AUX-A";
    }
    else {
        throw std::invalid_argument("invalid v17 local ciphertext path");
    }
    ByteVector tag;
    AppendString(&tag, domain);
    const auto context = V17LocalCiphertextAContext(setupId, roundId, clientId, blockIndex);
    tag.insert(tag.end(), context.begin(), context.end());
    return tag;
}

ByteVector V17DecryptionContext(
    const SetupId& setupId,
    RoundId roundId,
    const std::string& path,
    std::uint64_t instanceIndex) {

    if (path != "SP-MODEL" && path != "FULL-MODEL" && path != "AUX") {
        throw std::invalid_argument("invalid v17 decryption path tag");
    }
    ByteVector context(setupId.bytes.begin(), setupId.bytes.end());
    AppendBigEndian64(&context, roundId.value);
    AppendString(&context, path);
    AppendBigEndian64(&context, instanceIndex);
    return context;
}

ByteVector V17PublicOffsetTag(
    const SetupId& setupId,
    RoundId roundId,
    const std::string& path,
    std::uint64_t instanceIndex) {

    ByteVector tag;
    AppendString(&tag, "C2-OFFSET");
    const ByteVector context = V17DecryptionContext(
        setupId, roundId, path, instanceIndex);
    tag.insert(tag.end(), context.begin(), context.end());
    return tag;
}

ByteVector V17PublicDitherTag(
    const SetupId& setupId,
    RoundId roundId,
    const std::string& path,
    std::uint64_t instanceIndex,
    ClientId clientId) {

    ByteVector tag;
    AppendString(&tag, "C2-DITH");
    const ByteVector context = V17DecryptionContext(
        setupId, roundId, path, instanceIndex);
    tag.insert(tag.end(), context.begin(), context.end());
    AppendBigEndian64(&tag, clientId.value);
    return tag;
}

std::vector<std::uint64_t> GenerateUniformPolynomial(
    const Seed128& seed,
    const ByteVector& tag,
    std::size_t ringDimension,
    std::uint64_t modulus) {

    if (tag.empty()) {
        throw std::invalid_argument("polynomial tag must not be empty");
    }

    if (ringDimension == 0) {
        throw std::invalid_argument("ring dimension must be positive");
    }

    if (modulus < 2) {
        throw std::invalid_argument("polynomial modulus must be at least two");
    }

    ByteVector input;
    AppendString(&input, "UPOLY");
    AppendArray(&input, seed);
    input.insert(input.end(), tag.begin(), tag.end());
    AppendBigEndian64(&input, ringDimension);
    AppendBigEndian64(&input, modulus);

    Shake128Xof xof(input);
    std::vector<std::uint64_t> coefficients(ringDimension);

    for (std::uint64_t& coefficient : coefficients) {
        coefficient = SampleUniformBelow(&xof, modulus);
    }

    return coefficients;
}

std::vector<std::uint64_t> GenerateUniformPolynomial(
    const Seed128& seed,
    const std::string& tag,
    std::size_t ringDimension,
    std::uint64_t modulus) {

    return GenerateUniformPolynomial(
        seed,
        ByteVector(tag.begin(), tag.end()),
        ringDimension,
        modulus);
}

std::vector<double> GenerateClippedGaussian(
    const MasterSeed256& masterSeed,
    const std::string& domain,
    const ByteVector& context,
    std::size_t valueCount,
    long double mean,
    long double standardDeviation,
    long double minimum,
    long double maximum) {

    if (domain.empty()) {
        throw std::invalid_argument("Gaussian domain must not be empty");
    }

    if (!std::isfinite(mean) ||
        !std::isfinite(standardDeviation) ||
        standardDeviation <= 0.0L ||
        !std::isfinite(minimum) ||
        !std::isfinite(maximum) ||
        minimum > mean || mean > maximum ||
        minimum >= maximum) {

        throw std::invalid_argument("invalid clipped-Gaussian parameters");
    }

    const Seed128 seed = DeriveSeed128(masterSeed, domain, context);
    ByteVector input;
    AppendString(&input, "SIM-CLIPPED-GAUSSIAN");
    AppendBigEndian64(&input, domain.size());
    AppendString(&input, domain);
    AppendArray(&input, seed);
    AppendBigEndian64(&input, valueCount);

    Shake128Xof xof(input);
    std::vector<double> values;
    values.reserve(valueCount);

    constexpr long double twoPi =
        6.283185307179586476925286766559005768L;

    while (values.size() < valueCount) {
        const long double uniform1 = SampleOpenUnitInterval(&xof);
        const long double uniform2 = SampleOpenUnitInterval(&xof);
        const long double radius = std::sqrt(-2.0L * std::log(uniform1));
        const long double angle = twoPi * uniform2;

        const std::array<long double, 2> standardSamples{
            radius * std::cos(angle),
            radius * std::sin(angle)};

        for (const long double standardSample : standardSamples) {
            if (values.size() == valueCount) {
                break;
            }

            const long double sample =
                mean + standardDeviation * standardSample;

            const long double clipped =
                std::max(minimum, std::min(sample, maximum));

            values.push_back(static_cast<double>(clipped));
        }
    }

    return values;
}

}  // namespace deterministic_randomness
}  // namespace federated
