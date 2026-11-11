#ifndef FEDERATED1_PROTOCOL_TYPES_H
#define FEDERATED1_PROTOCOL_TYPES_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace federated {

// SD v17 is the only executable protocol version. ABI versions are independent.
inline bool UsesV17WireProtocol(const std::string& version) {
    return version == "v17";
}

inline bool UsesCertifiedGaussianProfile(const std::string& version) {
    return version == "v17";
}

enum class ProtocolMode {
    SP,
    FULL
};

using ByteVector = std::vector<std::uint8_t>;
using Seed128 = std::array<std::uint8_t, 16>;
using MasterSeed256 = std::array<std::uint8_t, 32>;

struct SetupId {
    Seed128 bytes{};
};

struct RoundId {
    std::uint64_t value = 0;
};

struct ClientId {
    std::uint64_t value = 0;
};

struct ActualChunkId {
    std::uint64_t value = 0;
};

struct ShuffledChunkLabel {
    std::uint64_t value = 0;
};

inline bool operator==(const SetupId& left, const SetupId& right) {
    return left.bytes == right.bytes;
}

inline bool operator==(const RoundId& left, const RoundId& right) {
    return left.value == right.value;
}

inline bool operator==(const ClientId& left, const ClientId& right) {
    return left.value == right.value;
}

inline bool operator<(const ClientId& left, const ClientId& right) {
    return left.value < right.value;
}

inline bool operator==(const ActualChunkId& left, const ActualChunkId& right) {
    return left.value == right.value;
}

inline bool operator<(const ActualChunkId& left, const ActualChunkId& right) {
    return left.value < right.value;
}

inline bool operator==(
    const ShuffledChunkLabel& left,
    const ShuffledChunkLabel& right) {

    return left.value == right.value;
}

inline bool operator<(
    const ShuffledChunkLabel& left,
    const ShuffledChunkLabel& right) {

    return left.value < right.value;
}

constexpr std::size_t kInvalidPosition =
    std::numeric_limits<std::size_t>::max();

/* ActualChunkId is client-private. Server messages contain only ShuffledChunkLabel. */
struct SparseIndexAnnouncement {
    ClientId clientId;
    /* CHUNK-BITMAP-MSB0-V1 over the shuffled-label universe [0,J). */
    ByteVector supportBitmap;
};

struct ShuffledChunkCohort {
    ShuffledChunkLabel label;
    std::vector<ClientId> clients;
    bool accepted = false;
};

struct SurvivorSet {
    std::vector<ClientId> clients;
};

struct RegisteredClientInfo {
    ClientId clientId;
    std::uint32_t integerWeight = 0;
    std::uint64_t dataSize = 0;
};

/* Owned by a client or privileged test oracle, never server state. */
struct ClientPersistentState {
    ClientId clientId;
    std::vector<double> model;
    std::vector<double> residual;
    std::vector<double> runningMean;
    std::vector<double> runningVariance;
};

struct SyntheticClientUpdate {
    ClientId clientId;
    std::vector<double> modelDelta;
    std::vector<double> runningMeanDelta;
    std::vector<double> runningVarianceDelta;
};

struct PackedShuffledLabels {
    std::vector<ShuffledChunkLabel> labels;
};

enum class RoundPhase {
    CREATED,
    SELECTION_COLLECTING,
    ADMISSION_FINALIZED,
    VALUES_COLLECTING,
    ENCRYPTED,
    DECRYPTING,
    COMMITTED,
    ABORTED
};

enum class ValidationSeverity {
    WARNING,
    ERROR
};

struct ProtocolPhaseAudit {
    std::vector<std::string> phaseResidues;
    std::vector<std::string> integerReferences;
};

struct ValidationIssue {
    ValidationSeverity severity = ValidationSeverity::ERROR;
    std::string code;
    std::string message;
};

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_TYPES_H
