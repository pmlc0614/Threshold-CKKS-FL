#include "protocol-contract.h"

#include "deterministic-randomness.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace federated {
namespace {

constexpr std::array<std::uint8_t, 8> kLayoutMagic{
    'F', 'L', 'V', '1', '7', 'L', 'A', 'Y'};
constexpr std::array<std::uint8_t, 8> kBinaryMagic{
    'F', 'L', 'V', '1', '7', 'B', 'I', 'N'};

void AppendUint16(ByteVector* output, std::uint16_t value) {
    output->push_back(static_cast<std::uint8_t>(value & 0xffU));
    output->push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
}

void AppendUint32(ByteVector* output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output->push_back(
            static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void AppendUint64(ByteVector* output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output->push_back(
            static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

std::uint16_t ReadUint16(const ByteVector& input, std::size_t* offset) {
    if (*offset + 2 > input.size()) {
        throw std::invalid_argument("truncated uint16 in contract header");
    }
    const std::uint16_t result =
        static_cast<std::uint16_t>(input[*offset]) |
        static_cast<std::uint16_t>(input[*offset + 1]) << 8U;
    *offset += 2;
    return result;
}

std::uint64_t ReadUint64(const ByteVector& input, std::size_t* offset) {
    if (*offset + 8 > input.size()) {
        throw std::invalid_argument("truncated uint64 in contract header");
    }
    std::uint64_t result = 0;
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        result |= static_cast<std::uint64_t>(input[*offset + shift / 8])
                  << shift;
    }
    *offset += 8;
    return result;
}

template <std::size_t N>
std::array<std::uint8_t, N> ReadArray(
    const ByteVector& input,
    std::size_t* offset) {
    if (*offset + N > input.size()) {
        throw std::invalid_argument("truncated byte array in contract header");
    }
    std::array<std::uint8_t, N> result{};
    std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(*offset), N, result.begin());
    *offset += N;
    return result;
}

bool IsKnownMessageType(ContractMessageType type) {
    switch (type) {
        case ContractMessageType::SETUP_MANIFEST:
        case ContractMessageType::CLIENT_SELECTION:
        case ContractMessageType::ADMISSION_RESULT:
        case ContractMessageType::ACCEPTED_CHUNK_VALUES:
        case ContractMessageType::AUXILIARY_DELTA:
        case ContractMessageType::ROUND_RESULT:
            return true;
    }
    return false;
}

}  // namespace

void ValidateProtocolTensorLayout(
    const ProtocolTensorLayout& layout) {
    if (layout.modelName.empty()) {
        throw std::invalid_argument("layout model name must not be empty");
    }
    if (layout.entries.empty()) {
        throw std::invalid_argument("layout must contain at least one entry");
    }
    if (layout.kind != LayoutKind::PARAMETER &&
        layout.kind != LayoutKind::RUNNING_MEAN &&
        layout.kind != LayoutKind::RUNNING_VARIANCE) {
        throw std::invalid_argument("unsupported layout kind");
    }

    std::uint64_t expectedOffset = 0;
    std::set<std::string> names;
    for (const auto& entry : layout.entries) {
        if (entry.name.empty() || !names.insert(entry.name).second) {
            throw std::invalid_argument("layout entry name is empty or duplicated");
        }
        if (entry.dtype != WireDType::FLOAT32) {
            throw std::invalid_argument("v17 wire layouts require float32");
        }
        if (entry.shape.empty() || entry.numel == 0 || entry.offset != expectedOffset) {
            throw std::invalid_argument("invalid layout shape, numel, or offset");
        }

        std::uint64_t shapeNumel = 1;
        for (const std::uint64_t dimension : entry.shape) {
            if (dimension == 0 || shapeNumel >
                    std::numeric_limits<std::uint64_t>::max() / dimension) {
                throw std::invalid_argument("invalid or overflowing layout shape");
            }
            shapeNumel *= dimension;
        }
        if (shapeNumel != entry.numel ||
            expectedOffset > std::numeric_limits<std::uint64_t>::max() - entry.numel) {
            throw std::invalid_argument("layout shape/numel mismatch or overflow");
        }
        expectedOffset += entry.numel;
    }
}

ByteVector CanonicalProtocolLayoutBytes(
    const ProtocolTensorLayout& layout) {
    ValidateProtocolTensorLayout(layout);
    if (layout.modelName.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("layout model name is too long");
    }

    ByteVector result(kLayoutMagic.begin(), kLayoutMagic.end());
    AppendUint16(&result, kSchemaVersionMajor);
    AppendUint16(&result, kSchemaVersionMinor);
    result.push_back(static_cast<std::uint8_t>(layout.kind));
    AppendUint32(&result, static_cast<std::uint32_t>(layout.modelName.size()));
    result.insert(result.end(), layout.modelName.begin(), layout.modelName.end());
    AppendUint64(&result, static_cast<std::uint64_t>(layout.entries.size()));

    for (const auto& entry : layout.entries) {
        if (entry.name.size() > std::numeric_limits<std::uint32_t>::max() ||
            entry.shape.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("layout entry name or rank is too large");
        }
        AppendUint32(&result, static_cast<std::uint32_t>(entry.name.size()));
        result.insert(result.end(), entry.name.begin(), entry.name.end());
        result.push_back(static_cast<std::uint8_t>(entry.dtype));
        AppendUint32(&result, static_cast<std::uint32_t>(entry.shape.size()));
        for (const std::uint64_t dimension : entry.shape) {
            AppendUint64(&result, dimension);
        }
        AppendUint64(&result, entry.offset);
        AppendUint64(&result, entry.numel);
    }
    return result;
}

ContractHash256 ProtocolLayoutSha256(
    const ProtocolTensorLayout& layout) {
    return deterministic_randomness::Sha256(
        CanonicalProtocolLayoutBytes(layout));
}

std::vector<std::uint32_t> QuantizeRegisteredClientWeights(
    const std::vector<RegisteredClientWeightInput>& clients,
    std::uint32_t denominator) {
    if (denominator == 0 || clients.empty()) {
        throw std::invalid_argument("weight denominator and clients must be nonzero");
    }
    if (clients.size() > denominator) {
        throw std::invalid_argument(
            "positive weights are impossible when clients exceed denominator");
    }

    std::set<std::uint64_t> clientIds;
    std::uint64_t totalExamples = 0;
    for (const auto& client : clients) {
        if (client.clientId.value == kNoClientId || client.exampleCount == 0 ||
            !clientIds.insert(client.clientId.value).second ||
            totalExamples > std::numeric_limits<std::uint64_t>::max() - client.exampleCount) {
            throw std::invalid_argument("invalid client ID, example count, or total");
        }
        totalExamples += client.exampleCount;
    }

    struct RemainderPosition {
        std::uint64_t remainder;
        std::uint64_t clientId;
        std::size_t position;
    };
    std::vector<std::uint32_t> result(clients.size(), 0);
    std::vector<RemainderPosition> order;
    std::uint32_t assigned = 0;
    for (std::size_t position = 0; position < clients.size(); ++position) {
        if (clients[position].exampleCount >
            std::numeric_limits<std::uint64_t>::max() / denominator) {
            throw std::invalid_argument("weight numerator overflows uint64");
        }
        const std::uint64_t numerator =
            clients[position].exampleCount * denominator;
        result[position] = static_cast<std::uint32_t>(numerator / totalExamples);
        assigned += result[position];
        order.push_back(RemainderPosition{
            numerator % totalExamples,
            clients[position].clientId.value,
            position});
    }

    std::sort(order.begin(), order.end(), [](const auto& left, const auto& right) {
        if (left.remainder != right.remainder) {
            return left.remainder > right.remainder;
        }
        return left.clientId < right.clientId;
    });
    const std::uint32_t remaining = denominator - assigned;
    for (std::uint32_t index = 0; index < remaining; ++index) {
        ++result[order[index].position];
    }
    if (std::any_of(result.begin(), result.end(), [](std::uint32_t alpha) {
            return alpha == 0;
        })) {
        throw std::invalid_argument(
            "v17 requires alpha_i >= 1; revise the registered setup");
    }
    return result;
}

void ValidateClientSelectionRequest(
    const ClientSelectionRequest& request) {
    if (request.clientId.value == kNoClientId ||
        request.selectedActualChunkIds.empty()) {
        throw std::invalid_argument("invalid client selection identity or empty support");
    }
    std::set<std::uint64_t> chunkIds;
    for (const ActualChunkId chunkId : request.selectedActualChunkIds) {
        if (!chunkIds.insert(chunkId.value).second) {
            throw std::invalid_argument("client selection contains duplicate chunk IDs");
        }
    }
}

void ValidateAcceptedChunkValues(
    const AcceptedChunkValues& values) {
    if (values.clientId.value == kNoClientId || values.chunkSize == 0) {
        throw std::invalid_argument("invalid accepted-value identity or chunk size");
    }
    if (values.chunkSize > std::numeric_limits<std::size_t>::max() ||
        values.actualChunkIds.size() >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(values.chunkSize) ||
        values.values.size() !=
            values.actualChunkIds.size() * static_cast<std::size_t>(values.chunkSize)) {
        throw std::invalid_argument("accepted chunk value count does not match metadata");
    }
    std::set<std::uint64_t> chunkIds;
    for (const ActualChunkId chunkId : values.actualChunkIds) {
        if (!chunkIds.insert(chunkId.value).second) {
            throw std::invalid_argument("accepted values contain duplicate chunk IDs");
        }
    }
    if (std::any_of(values.values.begin(), values.values.end(), [](float value) {
            return !std::isfinite(value);
        })) {
        throw std::invalid_argument("accepted chunk values must be finite");
    }
}

void ValidateAuxiliaryDeltaPayload(
    const AuxiliaryDeltaPayload& payload) {
    if (payload.clientId.value == kNoClientId ||
        payload.runningMeanDelta.empty() ||
        payload.runningVarianceDelta.empty()) {
        throw std::invalid_argument("invalid or incomplete auxiliary delta payload");
    }
    const auto nonfinite = [](float value) { return !std::isfinite(value); };
    if (std::any_of(payload.runningMeanDelta.begin(), payload.runningMeanDelta.end(), nonfinite) ||
        std::any_of(payload.runningVarianceDelta.begin(), payload.runningVarianceDelta.end(), nonfinite)) {
        throw std::invalid_argument("auxiliary deltas must be finite");
    }
}

bool IsAllowedRoundTransition(
    RoundPhase current,
    RoundPhase following) {
    if (following == RoundPhase::ABORTED) {
        return current != RoundPhase::COMMITTED && current != RoundPhase::ABORTED;
    }
    switch (current) {
        case RoundPhase::CREATED:
            return following == RoundPhase::SELECTION_COLLECTING ||
                following == RoundPhase::VALUES_COLLECTING;
        case RoundPhase::SELECTION_COLLECTING:
            return following == RoundPhase::ADMISSION_FINALIZED;
        case RoundPhase::ADMISSION_FINALIZED:
            return following == RoundPhase::VALUES_COLLECTING;
        case RoundPhase::VALUES_COLLECTING:
            return following == RoundPhase::ENCRYPTED;
        case RoundPhase::ENCRYPTED:
            return following == RoundPhase::DECRYPTING;
        case RoundPhase::DECRYPTING:
            return following == RoundPhase::COMMITTED;
        case RoundPhase::COMMITTED:
        case RoundPhase::ABORTED:
            return false;
    }
    return false;
}

BinaryContractHeader MakeBinaryContractHeader(
    ContractMessageType messageType,
    const SetupId& setupId,
    RoundId roundId,
    ClientId clientId,
    const ContractHash256& parameterLayoutHash,
    const ContractHash256& runningMeanLayoutHash,
    const ContractHash256& runningVarianceLayoutHash,
    const ByteVector& payload,
    std::uint16_t flags) {
    BinaryContractHeader result;
    result.messageType = messageType;
    result.flags = flags;
    result.setupId = setupId;
    result.roundId = roundId;
    result.clientId = clientId;
    result.parameterLayoutHash = parameterLayoutHash;
    result.runningMeanLayoutHash = runningMeanLayoutHash;
    result.runningVarianceLayoutHash = runningVarianceLayoutHash;
    result.payloadLength = static_cast<std::uint64_t>(payload.size());
    result.payloadHash = deterministic_randomness::Sha256(payload);
    return result;
}

ByteVector SerializeBinaryContractHeader(
    const BinaryContractHeader& header) {
    if (!IsKnownMessageType(header.messageType)) {
        throw std::invalid_argument("unknown contract message type");
    }
    if (header.flags != 0) {
        throw std::invalid_argument("schema 1.0 requires flags=0");
    }
    ByteVector result(kBinaryMagic.begin(), kBinaryMagic.end());
    AppendUint16(&result, kSchemaVersionMajor);
    AppendUint16(&result, kSchemaVersionMinor);
    AppendUint16(&result, static_cast<std::uint16_t>(header.messageType));
    AppendUint16(&result, header.flags);
    result.insert(result.end(), header.setupId.bytes.begin(), header.setupId.bytes.end());
    AppendUint64(&result, header.roundId.value);
    AppendUint64(&result, header.clientId.value);
    result.insert(result.end(), header.parameterLayoutHash.begin(), header.parameterLayoutHash.end());
    result.insert(result.end(), header.runningMeanLayoutHash.begin(), header.runningMeanLayoutHash.end());
    result.insert(result.end(), header.runningVarianceLayoutHash.begin(), header.runningVarianceLayoutHash.end());
    AppendUint64(&result, header.payloadLength);
    result.insert(result.end(), header.payloadHash.begin(), header.payloadHash.end());
    if (result.size() != kBinaryHeaderSize) {
        throw std::logic_error("binary contract header size is not 184 bytes");
    }
    return result;
}

BinaryContractHeader ParseBinaryContractHeader(
    const ByteVector& bytes) {
    if (bytes.size() != kBinaryHeaderSize ||
        !std::equal(kBinaryMagic.begin(), kBinaryMagic.end(), bytes.begin())) {
        throw std::invalid_argument("invalid binary contract header size or magic");
    }
    std::size_t offset = kBinaryMagic.size();
    const std::uint16_t major = ReadUint16(bytes, &offset);
    const std::uint16_t minor = ReadUint16(bytes, &offset);
    if (major != kSchemaVersionMajor || minor != kSchemaVersionMinor) {
        throw std::invalid_argument("unsupported binary contract schema");
    }

    BinaryContractHeader result;
    result.messageType = static_cast<ContractMessageType>(ReadUint16(bytes, &offset));
    if (!IsKnownMessageType(result.messageType)) {
        throw std::invalid_argument("unknown contract message type");
    }
    result.flags = ReadUint16(bytes, &offset);
    if (result.flags != 0) {
        throw std::invalid_argument("schema 1.0 requires flags=0");
    }
    result.setupId.bytes = ReadArray<16>(bytes, &offset);
    result.roundId.value = ReadUint64(bytes, &offset);
    result.clientId.value = ReadUint64(bytes, &offset);
    result.parameterLayoutHash = ReadArray<32>(bytes, &offset);
    result.runningMeanLayoutHash = ReadArray<32>(bytes, &offset);
    result.runningVarianceLayoutHash = ReadArray<32>(bytes, &offset);
    result.payloadLength = ReadUint64(bytes, &offset);
    result.payloadHash = ReadArray<32>(bytes, &offset);
    if (offset != bytes.size()) {
        throw std::logic_error("binary contract parser did not consume the header");
    }
    return result;
}

ByteVector SerializePublicSetupBinding(
    const SetupId& setupId, const Seed128& crsSeed,
    const ByteVector& parameterDescriptor, const std::vector<ClientId>& externalIds,
    const ContractHash256& parameterHash, const ContractHash256& meanHash,
    const ContractHash256& varianceHash) {
    if (parameterDescriptor.empty() || externalIds.empty()) {
        throw std::invalid_argument("empty public setup binding");
    }
    ByteVector out{'S','D','V','1','7','P','U','B'};
    auto append = [&](const auto& bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); };
    auto integer = [&](std::uint64_t value) {
        append(deterministic_randomness::EncodeBigEndian64(value));
    };
    integer(1);
    append(setupId.bytes);
    append(crsSeed);
    integer(parameterDescriptor.size());
    append(parameterDescriptor);
    integer(externalIds.size());
    std::set<std::uint64_t> seen;
    for (std::size_t position = 0; position < externalIds.size(); ++position) {
        if (!seen.insert(externalIds[position].value).second) {
            throw std::invalid_argument("duplicate public setup external ID");
        }
        integer(position);
        integer(externalIds[position].value);
    }
    append(parameterHash); append(meanHash); append(varianceHash);
    return out;
}

std::string ContractHex(const ByteVector& bytes) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : bytes) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

std::string ContractHex(const ContractHash256& bytes) {
    return ContractHex(ByteVector(bytes.begin(), bytes.end()));
}

}  // namespace federated
