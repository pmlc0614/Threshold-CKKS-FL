#ifndef FEDERATED1_PROTOCOL_CONTRACT_H
#define FEDERATED1_PROTOCOL_CONTRACT_H

#include "protocol-types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace federated {

inline constexpr const char* kProtocolVersion = "v17";
inline constexpr std::uint16_t kSchemaVersionMajor = 1;
inline constexpr std::uint16_t kSchemaVersionMinor = 0;
inline constexpr std::uint16_t kBindingAbiVersion = 22;
inline constexpr std::uint32_t kWeightDenominator = 128;
inline constexpr std::size_t kBinaryHeaderSize = 184;
inline constexpr std::uint64_t kNoClientId =
    std::numeric_limits<std::uint64_t>::max();

using ContractHash256 = std::array<std::uint8_t, 32>;

enum class LayoutKind : std::uint8_t {
    PARAMETER = 1,
    RUNNING_MEAN = 2,
    RUNNING_VARIANCE = 3
};

enum class WireDType : std::uint8_t {
    FLOAT32 = 1
};

enum class ContractMessageType : std::uint16_t {
    SETUP_MANIFEST = 1,
    CLIENT_SELECTION = 2,
    ADMISSION_RESULT = 3,
    ACCEPTED_CHUNK_VALUES = 4,
    AUXILIARY_DELTA = 5,
    ROUND_RESULT = 6
};

struct ProtocolLayoutEntry {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0;
    std::uint64_t numel = 0;
    WireDType dtype = WireDType::FLOAT32;
};

struct ProtocolTensorLayout {
    LayoutKind kind = LayoutKind::PARAMETER;
    std::string modelName;
    std::vector<ProtocolLayoutEntry> entries;
};

struct RegisteredClientWeightInput {
    ClientId clientId;
    std::uint64_t exampleCount = 0;
};

/* Python -> local C++ client role. Never expose actualChunkIds to the server. */
struct ClientSelectionRequest {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    ContractHash256 parameterLayoutHash{};
    std::vector<ActualChunkId> selectedActualChunkIds;
};

/* Local C++ client role -> Python after server-public admission finalization. */
struct ClientAdmissionResult {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::vector<ActualChunkId> acceptedActualChunkIds;
};

struct AcceptedChunkValues {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::uint64_t chunkSize = 0;
    std::vector<ActualChunkId> actualChunkIds;
    std::vector<float> values;
};

struct AuxiliaryDeltaPayload {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::vector<float> runningMeanDelta;
    std::vector<float> runningVarianceDelta;
};

struct BinaryContractHeader {
    ContractMessageType messageType =
        ContractMessageType::SETUP_MANIFEST;
    std::uint16_t flags = 0;
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    ContractHash256 parameterLayoutHash{};
    ContractHash256 runningMeanLayoutHash{};
    ContractHash256 runningVarianceLayoutHash{};
    std::uint64_t payloadLength = 0;
    ContractHash256 payloadHash{};
};

void ValidateProtocolTensorLayout(
    const ProtocolTensorLayout& layout);

ByteVector CanonicalProtocolLayoutBytes(
    const ProtocolTensorLayout& layout);

ContractHash256 ProtocolLayoutSha256(
    const ProtocolTensorLayout& layout);

std::vector<std::uint32_t> QuantizeRegisteredClientWeights(
    const std::vector<RegisteredClientWeightInput>& clients,
    std::uint32_t denominator = kWeightDenominator);

void ValidateClientSelectionRequest(
    const ClientSelectionRequest& request);

void ValidateAcceptedChunkValues(
    const AcceptedChunkValues& values);

void ValidateAuxiliaryDeltaPayload(
    const AuxiliaryDeltaPayload& payload);

bool IsAllowedRoundTransition(
    RoundPhase current,
    RoundPhase following);

BinaryContractHeader MakeBinaryContractHeader(
    ContractMessageType messageType,
    const SetupId& setupId,
    RoundId roundId,
    ClientId clientId,
    const ContractHash256& parameterLayoutHash,
    const ContractHash256& runningMeanLayoutHash,
    const ContractHash256& runningVarianceLayoutHash,
    const ByteVector& payload,
    std::uint16_t flags = 0);

ByteVector SerializeBinaryContractHeader(
    const BinaryContractHeader& header);

BinaryContractHeader ParseBinaryContractHeader(
    const ByteVector& bytes);

// Public setup envelope v1: BE64 fields; roster order defines crypto IDs 0..n-1.
// Contains no private index key, secret key or server-only authorized weights.
ByteVector SerializePublicSetupBinding(
    const SetupId& setupId, const Seed128& crsSeed,
    const ByteVector& parameterDescriptor, const std::vector<ClientId>& externalIds,
    const ContractHash256& parameterHash, const ContractHash256& meanHash,
    const ContractHash256& varianceHash);

std::string ContractHex(const ByteVector& bytes);
std::string ContractHex(const ContractHash256& bytes);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_CONTRACT_H
