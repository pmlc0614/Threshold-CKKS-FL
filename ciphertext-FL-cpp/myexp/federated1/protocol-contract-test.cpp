#include "protocol-contract-test.h"

#include "protocol-contract.h"
#include "support-bitmap.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

bool Check(bool condition, const std::string& description) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << description << '\n';
    return condition;
}

federated::ProtocolTensorLayout MakeLayout(
    federated::LayoutKind kind,
    const std::string& name,
    std::vector<std::uint64_t> shape,
    std::uint64_t numel) {
    federated::ProtocolTensorLayout layout;
    layout.kind = kind;
    layout.modelName = "resnet20-cifar10";
    layout.entries.push_back(
        federated::ProtocolLayoutEntry{
            name, std::move(shape), 0, numel,
            federated::WireDType::FLOAT32});
    return layout;
}

bool TestWeightGoldenVectors() {
    using namespace federated;
    bool passed = true;
    const auto equal = QuantizeRegisteredClientWeights({
        {{30}, 1}, {{10}, 1}, {{20}, 1}});
    passed &= Check(equal == std::vector<std::uint32_t>({42, 43, 43}),
                    "largest-remainder ties use ascending client ID");
    const auto weighted = QuantizeRegisteredClientWeights(
        {{{10}, 1}, {{20}, 3}}, 4);
    passed &= Check(weighted == std::vector<std::uint32_t>({1, 3}),
                    "sample-count weights match the golden vector");
    bool rejectedZero = false;
    try {
        (void)QuantizeRegisteredClientWeights({{{1}, 1}, {{2}, 1000}});
    }
    catch (const std::invalid_argument&) {
        rejectedZero = true;
    }
    passed &= Check(rejectedZero, "a setup producing alpha_i=0 is rejected");
    return passed;
}

bool TestLayoutAndHeaderGoldenVectors() {
    using namespace federated;
    bool passed = true;
    const auto parameter = MakeLayout(
        LayoutKind::PARAMETER, "conv1.weight", {16, 3, 3, 3}, 432);
    const auto mean = MakeLayout(
        LayoutKind::RUNNING_MEAN, "bn1.running_mean", {16}, 16);
    const auto variance = MakeLayout(
        LayoutKind::RUNNING_VARIANCE, "bn1.running_var", {16}, 16);
    const auto parameterHash = ProtocolLayoutSha256(parameter);
    const auto meanHash = ProtocolLayoutSha256(mean);
    const auto varianceHash = ProtocolLayoutSha256(variance);
    passed &= Check(
        ContractHex(parameterHash) ==
            "78500415f35edcad4ab057a2d2ecb5614f74eab963e536845b695110602ecaf6",
        "ParameterLayout SHA-256 matches Python");
    passed &= Check(
        ContractHex(meanHash) ==
            "d8204987221095582c7b13ada6e4b9b5fcad7dc2e47555d6ef81918a76ac86a4",
        "RunningMeanLayout SHA-256 matches Python");
    passed &= Check(
        ContractHex(varianceHash) ==
            "c27aad243b2c8f93c795fb193743167e4ad75e153b7d54d69e8bdb672e51de87",
        "RunningVarianceLayout SHA-256 matches Python");

    SetupId setupId;
    for (std::size_t index = 0; index < setupId.bytes.size(); ++index) {
        setupId.bytes[index] = static_cast<std::uint8_t>(index);
    }
    const ByteVector payload{1, 2, 3, 4, 5};
    const auto header = MakeBinaryContractHeader(
        ContractMessageType::CLIENT_SELECTION,
        setupId,
        RoundId{42},
        ClientId{7},
        parameterHash,
        meanHash,
        varianceHash,
        payload);
    const ByteVector encoded = SerializeBinaryContractHeader(header);
    const auto decoded = ParseBinaryContractHeader(encoded);
    passed &= Check(encoded.size() == kBinaryHeaderSize,
                    "binary replay header is exactly 184 bytes");
    passed &= Check(
        ContractHex(encoded) ==
            "464c56313742494e0100000002000000000102030405060708090a0b0c0d0e0f"
            "2a000000000000000700000000000000"
            "78500415f35edcad4ab057a2d2ecb5614f74eab963e536845b695110602ecaf6"
            "d8204987221095582c7b13ada6e4b9b5fcad7dc2e47555d6ef81918a76ac86a4"
            "c27aad243b2c8f93c795fb193743167e4ad75e153b7d54d69e8bdb672e51de87"
            "0500000000000000"
            "74f81fe167d99b4cb41d6d0ccda82278caee9f3e2f25d5e5a3936ff3dcec60d0",
        "binary replay header matches Python byte-for-byte");
    passed &= Check(
        decoded.messageType == ContractMessageType::CLIENT_SELECTION &&
            decoded.roundId.value == 42 && decoded.clientId.value == 7 &&
            decoded.payloadLength == payload.size(),
        "binary replay header round-trips");
    return passed;
}

bool TestRoundStateMachine() {
    using namespace federated;
    bool passed = true;
    passed &= Check(
        IsAllowedRoundTransition(
            RoundPhase::ADMISSION_FINALIZED,
            RoundPhase::VALUES_COLLECTING),
        "admission finalization enables accepted-value collection");
    passed &= Check(
        IsAllowedRoundTransition(RoundPhase::DECRYPTING, RoundPhase::ABORTED),
        "a pre-commit decryption failure can abort");
    passed &= Check(
        !IsAllowedRoundTransition(RoundPhase::COMMITTED, RoundPhase::ABORTED),
        "commit is terminal");
    return passed;
}

bool TestTwoPhaseBoundaryMessages() {
    using namespace federated;
    bool passed = true;
    ClientSelectionRequest selection;
    selection.clientId = ClientId{2};
    selection.selectedActualChunkIds = {ActualChunkId{4}, ActualChunkId{4}};
    bool rejectedDuplicate = false;
    try {
        ValidateClientSelectionRequest(selection);
    }
    catch (const std::invalid_argument&) {
        rejectedDuplicate = true;
    }
    passed &= Check(rejectedDuplicate,
                    "client selection rejects duplicate actual chunk IDs");

    AcceptedChunkValues values;
    values.clientId = ClientId{2};
    values.chunkSize = 4;
    values.actualChunkIds = {ActualChunkId{4}, ActualChunkId{7}};
    values.values = {1.0F, 2.0F, 3.0F};
    bool rejectedMisaligned = false;
    try {
        ValidateAcceptedChunkValues(values);
    }
    catch (const std::invalid_argument&) {
        rejectedMisaligned = true;
    }
    passed &= Check(rejectedMisaligned,
                    "accepted values reject a mismatched flattened payload");
    return passed;
}

bool TestV17SupportBitmapCodec() {
    using namespace federated;
    bool passed = true;
    const std::vector<ShuffledChunkLabel> labels{
        ShuffledChunkLabel{0},
        ShuffledChunkLabel{3},
        ShuffledChunkLabel{8}};
    const ByteVector encoded = EncodeSupportBitmapMsb0(10, labels);
    passed &= Check(
        encoded == ByteVector({0x90, 0x80}),
        "v17 J=10 MSB0 bitmap matches the normative vector");
    passed &= Check(
        DecodeSupportBitmapMsb0(10, encoded, 3) == labels,
        "v17 support bitmap round-trips in increasing label order");
    passed &= Check(
        EncodeSupportBitmapMsb0(1, {}) == ByteVector({0x00}) &&
            EncodeSupportBitmapMsb0(1, {ShuffledChunkLabel{0}}) ==
                ByteVector({0x80}) &&
            EncodeSupportBitmapMsb0(
                9,
                {ShuffledChunkLabel{0}, ShuffledChunkLabel{8}}) ==
                ByteVector({0x80, 0x80}) &&
            EncodeSupportBitmapMsb0(
                8,
                {ShuffledChunkLabel{0}, ShuffledChunkLabel{1},
                 ShuffledChunkLabel{2}, ShuffledChunkLabel{3},
                 ShuffledChunkLabel{4}, ShuffledChunkLabel{5},
                 ShuffledChunkLabel{6}, ShuffledChunkLabel{7}}) ==
                ByteVector({0xff}) &&
            SupportBitmapByteCount(231) == 29,
        "v17 boundary vectors J=1, J=8, J=9, and J=231 are canonical");
    bool rejectedLength = false;
    try {
        (void)DecodeSupportBitmapMsb0(9, ByteVector({0x80}));
    }
    catch (const std::invalid_argument&) {
        rejectedLength = true;
    }
    passed &= Check(
        rejectedLength,
        "v17 rejects a bitmap with the wrong byte length");
    bool rejectedPadding = false;
    try {
        (void)DecodeSupportBitmapMsb0(10, ByteVector({0x90, 0x81}));
    }
    catch (const std::invalid_argument&) {
        rejectedPadding = true;
    }
    passed &= Check(
        rejectedPadding,
        "v17 rejects nonzero unused low bits");
    const auto intersection = IntersectSupportBitmapsMsb0(
        10,
        encoded,
        EncodeSupportBitmapMsb0(
            10,
            {ShuffledChunkLabel{3}, ShuffledChunkLabel{8},
             ShuffledChunkLabel{9}}));
    passed &= Check(
        DecodeSupportBitmapMsb0(10, intersection) ==
            std::vector<ShuffledChunkLabel>(
                {ShuffledChunkLabel{3}, ShuffledChunkLabel{8}}),
        "v17 upload support is the exact bitmap intersection");
    return passed;
}

bool TestPublicSetupBindingGolden() {
    using namespace federated;
    SetupId setup{};
    Seed128 crs{};
    for (std::size_t i = 0; i < 16; ++i) {
        setup.bytes[i] = static_cast<std::uint8_t>(i);
        crs[i] = static_cast<std::uint8_t>(i + 16);
    }
    ContractHash256 parameter{}, mean{}, variance{};
    parameter.fill(1); mean.fill(2); variance.fill(3);
    const auto actual = SerializePublicSetupBinding(
        setup, crs, ByteVector{1, 2, 3}, {{30}, {10}, {20}, {40}},
        parameter, mean, variance);
    bool passed = Check(ContractHex(actual) ==
        "53445631375055420000000000000001000102030405060708090a0b0c0d0e0f1011121314151617"
        "18191a1b1c1d1e1f0000000000000003010203000000000000000400000000000000000000000000"
        "00001e0000000000000001000000000000000a000000000000000200000000000000140000000000"
        "00000300000000000000280101010101010101010101010101010101010101010101010101010101"
        "01010102020202020202020202020202020202020202020202020202020202020202020303030303"
        "030303030303030303030303030303030303030303030303030303",
        "public setup matches shared SD_V17_BYTE_VECTORS golden");
    bool rejected = false;
    try {
        (void)SerializePublicSetupBinding(setup, crs, ByteVector{1},
            {{30}, {30}}, parameter, mean, variance);
    } catch (const std::invalid_argument&) { rejected = true; }
    passed &= Check(rejected, "public setup rejects duplicate external IDs");
    return passed;
}

}  // namespace

int RunProtocolContractTests() {
    const bool passed = TestWeightGoldenVectors() &&
                        TestLayoutAndHeaderGoldenVectors() &&
                        TestRoundStateMachine() &&
                        TestTwoPhaseBoundaryMessages() &&
                        TestV17SupportBitmapCodec() &&
                        TestPublicSetupBindingGolden();
    std::cout << (passed ? "All protocol contract tests passed.\n"
                         : "One or more protocol contract tests failed.\n");
    return passed ? 0 : 1;
}
