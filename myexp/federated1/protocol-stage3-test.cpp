#include "protocol-stage3-test.h"

#include "client-to-collective-kswitch.h"
#include "deterministic-randomness.h"
#include "protocol-parameters.h"
#include "protocol-setup.h"
#include "rns-polynomial.h"
#include "seeded-rlwe.h"
#include "survivor-keys.h"

#include "openfhe.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

bool Check(bool condition, const std::string& description) {
    std::cout
        << (condition ? "[PASS] " : "[FAIL] ")
        << description
        << '\n';
    return condition;
}

bool CheckThrows(
    const std::function<void()>& operation,
    const std::string& description) {

    try {
        operation();
    }
    catch (const std::exception&) {
        return Check(true, description);
    }

    return Check(false, description);
}

lbcrypto::CryptoContext<DCRTPoly> MakeStage3Context(
    const federated::ProtocolExperimentConfig& config) {

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(
        static_cast<std::uint32_t>(
            config.modulus.qModulusBits.size() - 1));
    parameters.SetFirstModSize(config.modulus.qModulusBits.front());
    /* This backend uses FirstPrime(bits,2N) for the scaling tower: SetScalingModSize(b) produces b+1 actual bits. qModulusBits records actual tower bit lengths. */
    parameters.SetScalingModSize(
        config.modulus.qModulusBits.size() > 1
            ? config.modulus.qModulusBits[1] - 1
            : config.modulus.qModulusBits.front() - 1);
    parameters.SetBatchSize(config.modulus.ringDimension / 2);
    parameters.SetRingDim(config.modulus.ringDimension);
    parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetScalingTechnique(lbcrypto::FIXEDMANUAL);
    parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(lbcrypto::BV);

    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    return cc;
}

DCRTPoly MakeCompositeSparseMessage(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    std::size_t mlweRank,
    std::size_t activeStreamCount) {

    const std::size_t ringDimension = parameters->GetRingDimension();
    const lbcrypto::BigInteger modulus = parameters->GetModulus();
    std::vector<lbcrypto::BigInteger> coefficients(
        ringDimension,
        lbcrypto::BigInteger(0));

    for (std::size_t index = 0;
         index < ringDimension;
         ++index) {

        if (index % mlweRank >= activeStreamCount) {
            continue;
        }

        const std::int64_t signedValue =
            static_cast<std::int64_t>(index % 31) - 15;
        coefficients[index] = signedValue < 0
            ? modulus - lbcrypto::BigInteger(
                  static_cast<std::uint64_t>(-signedValue))
            : lbcrypto::BigInteger(
                  static_cast<std::uint64_t>(signedValue));
    }

    DCRTPoly result =
        federated::rns_polynomial::FromCompositeCoefficients(
            parameters,
            coefficients);
    result.SetFormat(Format::EVALUATION);
    return result;
}

bool EqualDCRT(DCRTPoly left, DCRTPoly right) {
    left.SetFormat(Format::COEFFICIENT);
    right.SetFormat(Format::COEFFICIENT);

    const auto& leftTowers = left.GetAllElements();
    const auto& rightTowers = right.GetAllElements();

    if (leftTowers.size() != rightTowers.size()) {
        return false;
    }

    for (std::size_t tower = 0;
         tower < leftTowers.size();
         ++tower) {

        if (leftTowers[tower].GetModulus() !=
                rightTowers[tower].GetModulus() ||
            leftTowers[tower].GetValues().GetLength() !=
                rightTowers[tower].GetValues().GetLength()) {

            return false;
        }

        for (std::size_t index = 0;
             index < leftTowers[tower].GetValues().GetLength();
             ++index) {

            if (leftTowers[tower].GetValues()[index] !=
                rightTowers[tower].GetValues()[index]) {

                return false;
            }
        }
    }

    return true;
}

std::uint64_t MaxCenteredDifferenceAtIndices(
    DCRTPoly left,
    DCRTPoly right,
    const std::vector<std::size_t>& indices) {

    left.SetFormat(Format::COEFFICIENT);
    right.SetFormat(Format::COEFFICIENT);

    const auto leftComposite = left.CRTInterpolate();
    const auto rightComposite = right.CRTInterpolate();
    const std::uint64_t modulusValue = left.GetParams()->GetModulus().ConvertToInt();
    const auto& leftValues = leftComposite.GetValues();
    const auto& rightValues = rightComposite.GetValues();

    std::uint64_t maximum = 0;

    for (const std::size_t index : indices) {
        const std::uint64_t leftValue =
            leftValues[index].ConvertToInt();
        const std::uint64_t rightValue =
            rightValues[index].ConvertToInt();
        const std::uint64_t forward =
            leftValue >= rightValue
                ? leftValue - rightValue
                : modulusValue - (rightValue - leftValue);
        const std::uint64_t distance =
            std::min(forward, modulusValue - forward);
        maximum = std::max(maximum, distance);
    }

    return maximum;
}

bool OmittedCoefficientsAreZero(
    DCRTPoly polynomial,
    std::size_t mlweRank,
    std::size_t activeStreamCount) {

    polynomial.SetFormat(Format::COEFFICIENT);
    const auto& values =
        polynomial.GetElementAtIndex(0).GetValues();

    for (std::size_t index = 0;
         index < values.GetLength();
         ++index) {

        if (index % mlweRank >= activeStreamCount &&
            values[index] != lbcrypto::NativeInteger(0)) {

            return false;
        }
    }

    return true;
}

DCRTPoly SumSurvivorSecretQ(
    const federated::ProtocolSetupResult& setup,
    const federated::SurvivorSet& survivors) {

    DCRTPoly result(
        setup.common.rns.paramsQ,
        Format::EVALUATION,
        true);

    for (const federated::ClientId clientId : survivors.clients) {
        result += setup.clientSecrets[clientId.value]
                      .secretKeyQ
                      ->GetPrivateElement();
    }

    return result;
}

bool TestTwoTowerSeededPacketAndKeySwitch() {
    using namespace federated;

    ProtocolExperimentConfig config = MakeSmokeConfig();
    config.clientCount = 2;
    config.integerWeights = {64, 64};
    config.sparsification.minimumCohort = 2;
    config.modulus.qModulusBits = {52, 52};
    config.modulus.hybridLargeDigits = 2;

    auto cc = MakeStage3Context(config);
    const ProtocolSetupResult setup = GenerateProtocolSetup(cc, config);

    SurvivorSet survivors;
    survivors.clients = {ClientId{0}, ClientId{1}};

    const ClientId sourceClient{0};
    const RoundId roundId{17};
    constexpr std::size_t activeStreamCount = 2;
    constexpr std::size_t packGroupIndex = 0;
    const Seed128 aSeed{
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};

    const DCRTPoly message = MakeCompositeSparseMessage(
        setup.common.rns.paramsQ,
        config.sparsification.mlweRank,
        activeStreamCount);

    const SeededSparseRLWEPacket packet =
        EncryptSeededSparseRLWEUncompressed(
            message,
            setup.clientSecrets[sourceClient.value],
            setup.common,
            roundId,
            packGroupIndex,
            activeStreamCount,
            aSeed,
            config.setupNoise.encryptionStandardDeviation);

    const RestoredSparseRLWECiphertext restored =
        RestoreSeededSparseRLWE(packet, setup.common);

    ClientToCollectiveKeyMaterial exactKey;
    exactKey.setupId = setup.common.setupId;
    exactKey.sourceClientId = sourceClient;
    exactKey.survivors = survivors;
    exactKey.decompositionDigitCount =
        setup.common.rns.decompositionDigitCount;

    DCRTPoly sourceSecret =
        setup.clientSecrets[sourceClient.value]
            .encryptionKeyQ
            ->GetPrivateElement();

    for (std::size_t digitIndex = 0;
         digitIndex < exactKey.decompositionDigitCount;
         ++digitIndex) {

        exactKey.a.emplace_back(
            setup.common.rns.paramsQP,
            Format::EVALUATION,
            true);
        exactKey.b.push_back(BuildM2RGadgetTerm(
            sourceSecret,
            digitIndex,
            setup.common.rns));
    }

    HybridModDownTrace trace;
    const CollectiveRLWECiphertext switched =
        KeySwitchClientToCollective(
            restored,
            exactKey,
            setup.common,
            &trace);

    DCRTPoly sourcePhase = restored.b + restored.a * sourceSecret;
    DCRTPoly switchedPhase = switched.b;

    bool passed = Check(
        setup.common.rns.qTowerCount == 2 &&
            packet.uncompressedB.size() ==
                restored.transmittedBIndices.size(),
        "two-tower seeded sparse packet stores one composite-Q coefficient per transmitted slot");

    passed &= Check(
        EqualDCRT(sourcePhase, switchedPhase),
        "two-tower digit KSwitch exactly preserves phase with an exact test key");

    passed &= Check(
        EqualDCRT(
            switched.b - restored.b,
            ApproxModDownSinglePToQ(trace.bQP, setup.common.rns)) &&
        EqualDCRT(
            switched.a,
            ApproxModDownSinglePToQ(trace.aQP, setup.common.rns)),
        "client KSwitch applies one ModDown to the QP digit sum");

    DCRTPoly combinedGadgetQP(
        setup.common.rns.paramsQP,
        Format::EVALUATION,
        true);

    for (std::size_t digitIndex = 0;
         digitIndex < setup.common.rns.decompositionDigitCount;
         ++digitIndex) {

        combinedGadgetQP += BuildM2RGadgetTerm(
            message,
            digitIndex,
            setup.common.rns);
    }

    passed &= Check(
        EqualDCRT(
            ApproxModDownSinglePToQ(
                combinedGadgetQP, setup.common.rns),
            message),
        "one ModDown of the two-digit gadget sum reconstructs Q");

    DCRTPoly carryInput(
        setup.common.rns.paramsQP, Format::COEFFICIENT, true);
    const std::size_t pIndex = setup.common.rns.qTowerCount;
    auto& pTower = carryInput.GetAllElements()[pIndex];
    const auto p = pTower.GetModulus();
    lbcrypto::NativeVector pValues(
        setup.common.rns.paramsQP->GetRingDimension(), p);
    for (std::size_t index = 0; index < pValues.GetLength(); ++index) {
        pValues[index] = lbcrypto::NativeInteger(0);
    }
    pValues[0] = p - lbcrypto::NativeInteger(1);
    pTower.SetValues(std::move(pValues), Format::COEFFICIENT);
    carryInput.SetFormat(Format::EVALUATION);
    const DCRTPoly oneModDown = ApproxModDownSinglePToQ(
        carryInput + carryInput, setup.common.rns);
    const DCRTPoly twoModDowns =
        ApproxModDownSinglePToQ(carryInput, setup.common.rns) +
        ApproxModDownSinglePToQ(carryInput, setup.common.rns);
    passed &= Check(
        !EqualDCRT(oneModDown, twoModDowns),
        "P-limb carry distinguishes sum-then-ModDown from per-digit ModDown");

    return passed;
}

federated::ByteVector SeedContext(
    federated::RoundId roundId,
    federated::ClientId clientId,
    std::size_t packGroupIndex) {

    federated::ByteVector result;
    const auto round =
        federated::deterministic_randomness::EncodeBigEndian64(
            roundId.value);
    const auto client =
        federated::deterministic_randomness::EncodeBigEndian64(
            clientId.value);
    const auto group =
        federated::deterministic_randomness::EncodeBigEndian64(
            packGroupIndex);
    result.insert(result.end(), round.begin(), round.end());
    result.insert(result.end(), client.begin(), client.end());
    result.insert(result.end(), group.begin(), group.end());
    return result;
}

}  // namespace

int RunProtocolStage3Tests() {
    using namespace federated;

    bool passed = true;

    try {
        ProtocolExperimentConfig config = MakeSmokeConfig();
        auto cc = MakeStage3Context(config);

        const ProtocolSetupResult setup =
            GenerateProtocolSetup(cc, config);

        // Constant half-residues exercise both signed lifts across every CRT limb.
        const auto& rns = setup.common.rns;
        const auto q0 = rns.paramsQ->GetParams()[0]->GetModulus();
        const auto halfQ = q0 >> 1;
        DCRTPoly input(rns.paramsQ, Format::COEFFICIENT, true);
        for (auto& tower : input.GetAllElements()) {
            auto values = tower.GetValues();
            for (std::size_t i = 0; i < values.GetLength(); ++i)
                values[i] = halfQ.Mod(tower.GetModulus());
            tower.SetValues(values, Format::COEFFICIENT);
        }
        auto lifted = ApproxModUpQDigitToQP(input, 0, rns, true);
        lifted.SetFormat(Format::COEFFICIENT);
        std::size_t negatives = 0;
        const auto n = rns.paramsQ->GetRingDimension();
        const auto pmod = rns.paramsP->GetParams()[0]->GetModulus();
        for (std::size_t i = 0; i < n; ++i) {
            const bool negative = lifted.GetElementAtIndex(rns.qTowerCount).GetValues()[i]
                != halfQ.Mod(pmod);
            negatives += negative;
            for (const auto& tower : lifted.GetAllElements()) {
                const auto modulus = tower.GetModulus();
                const auto expected = negative
                    ? halfQ.Mod(modulus).ModSub(q0.Mod(modulus), modulus)
                    : halfQ.Mod(modulus);
                if (tower.GetValues()[i] != expected) passed = false;
            }
        }
        passed &= Check(negatives > n/4 && negatives < 3*n/4,
            "ZM digit has consistent signed CRT lifts and approximately zero mean");
        DCRTPoly mdInput(rns.paramsQP, Format::COEFFICIENT, true);
        const auto halfP = pmod >> 1;
        for (auto& tower : mdInput.GetAllElements()) {
            auto values = tower.GetValues();
            for (std::size_t i = 0; i < n; ++i) values[i] = halfP.Mod(tower.GetModulus());
            tower.SetValues(values, Format::COEFFICIENT);
        }
        auto md = ApproxModDownSinglePToQ(mdInput, rns, false, true);
        auto freshMd = ApproxModDownSinglePToQ(mdInput, rns, false, true);
        passed &= Check(!EqualDCRT(md, freshMd), "ZM ModDown uses fresh coins per call");
        md.SetFormat(Format::COEFFICIENT);
        std::size_t ones = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto v = md.GetElementAtIndex(0).GetValues()[i];
            if (v == lbcrypto::NativeInteger(1)) ++ones;
            else if (v != lbcrypto::NativeInteger(0)) passed = false;
            for (const auto& tower : md.GetAllElements())
                if (tower.GetValues()[i] != v) passed = false;
        }
        passed &= Check(ones > n/4 && ones < 3*n/4,
            "ZM ModDown is unbiased and consistent across CRT limbs");

        SurvivorSet survivors;
        for (std::size_t client = 0;
             client < config.clientCount;
             ++client) {

            survivors.clients.push_back(ClientId{client});
        }

        const SurvivorKeySet survivorKeys =
            AggregateSurvivorKeys(setup, survivors);

        const ClientId sourceClient{0};
        const RoundId roundId =
            deterministic_randomness::DeriveRoundId(
                config.masterSeed,
                1);
        constexpr std::size_t packGroupIndex = 2;
        constexpr std::size_t activeStreamCount = 2;

        const Seed128 aSeed =
            deterministic_randomness::DeriveSeed128(
                config.masterSeed,
                "ENC-SEED-TEST",
                SeedContext(
                    roundId,
                    sourceClient,
                    packGroupIndex));

        const DCRTPoly message = MakeCompositeSparseMessage(
            setup.common.rns.paramsQ,
            config.sparsification.mlweRank,
            activeStreamCount);

        const SeededSparseRLWEPacket packet =
            EncryptSeededSparseRLWE(
                message,
                setup.clientSecrets[sourceClient.value],
                setup.common,
                roundId,
                packGroupIndex,
                activeStreamCount,
                aSeed,
                config.compression.c1);

        const std::size_t expectedCoefficientCount =
            config.modulus.ringDimension /
            config.sparsification.mlweRank *
            activeStreamCount;

        passed &= Check(
            packet.compressedB.coefficientCount ==
                expectedCoefficientCount,
            "packet contains only the active interleaved b streams");

        passed &= Check(
            packet.compressedB.bytes.size() ==
                (expectedCoefficientCount * config.compression.c1 + 7) / 8,
            "sparse b payload has the canonical packed byte size");

        const RestoredSparseRLWECiphertext restored =
            RestoreSeededSparseRLWE(
                packet,
                setup.common);

        const RestoredSparseRLWECiphertext restoredAgain =
            RestoreSeededSparseRLWE(
                packet,
                setup.common);

        passed &= Check(
            EqualDCRT(restored.a, restoredAgain.a),
            "full a is reproduced deterministically from the packet seed");

        passed &= Check(
            OmittedCoefficientsAreZero(
                restored.b,
                config.sparsification.mlweRank,
                activeStreamCount),
            "untransmitted b coefficients are restored as zero");

        DCRTPoly sourceSecret =
            setup.clientSecrets[sourceClient.value]
                .encryptionKeyQ
                ->GetPrivateElement();

        const DCRTPoly sourcePhase =
            restored.b + restored.a * sourceSecret;

        const std::uint64_t q =
            setup.common.rns.paramsQ
                ->GetModulus()
                .ConvertToInt();
        const std::uint64_t compressionErrorBound =
            q / (std::uint64_t{1} << config.compression.c1) + 64;
        const std::uint64_t sourceError =
            MaxCenteredDifferenceAtIndices(
                sourcePhase,
                message,
                restored.transmittedBIndices);

        std::cout
            << "maximum seeded-encryption error at transmitted coefficients: "
            << sourceError << " (Q=" << q << ", bound=" << compressionErrorBound << ")"
            << '\n';

        passed &= Check(
            sourceError <= compressionErrorBound,
            "seeded encryption is correct at every transmitted b coefficient");

        DCRTPoly gadget(setup.common.rns.paramsQP, Format::EVALUATION, true);
        for (std::size_t digit = 0; digit < setup.common.rns.decompositionDigitCount; ++digit) {
            gadget += BuildM2RGadgetTerm(message, digit, setup.common.rns);
        }
        const DCRTPoly modDownGadget =
            ApproxModDownSinglePToQ(
                gadget,
                setup.common.rns);

        passed &= Check(
            EqualDCRT(modDownGadget, message),
            "single-P mod-down removes P after summing both gadget digits");

        const CollectiveRLWECiphertext switched =
            KeySwitchClientToCollective(
                restored,
                survivorKeys,
                setup.common);

        const DCRTPoly collectiveSecret =
            SumSurvivorSecretQ(setup, survivors);
        const DCRTPoly collectivePhase =
            switched.b + switched.a * collectiveSecret;

        const std::uint64_t switchError =
            MaxCenteredDifferenceAtIndices(
                collectivePhase,
                sourcePhase,
                switched.meaningfulCoefficientIndices);

        std::cout
            << "maximum i-to-S KSwitch error at meaningful coefficients: "
            << switchError
            << '\n';

        passed &= Check(
            switchError <= (std::uint64_t{1} << 30),
            "i-to-S KSwitch preserves the sparse ciphertext phase");

        passed &= Check(
            switched.survivors.clients == survivors.clients &&
                switched.sourceClientId == sourceClient &&
                switched.roundId == roundId,
            "KSwitch output preserves authenticated routing metadata");

        SeededSparseRLWEPacket wrongSetupPacket = packet;
        wrongSetupPacket.setupId.bytes[0] ^= 0x01U;
        passed &= CheckThrows(
            [&]() {
                (void)RestoreSeededSparseRLWE(
                    wrongSetupPacket,
                    setup.common);
            },
            "restoration rejects a packet from another setup");

        passed &= CheckThrows(
            [&]() {
                (void)FindClientToCollectiveKey(
                    survivorKeys,
                    ClientId{config.clientCount});
            },
            "KSwitch lookup rejects an unregistered source client");
    }
    catch (const std::exception& error) {
        std::cout
            << "[FAIL] protocol stage-3 test threw: "
            << error.what()
            << '\n';
        passed = false;
    }

    try {
        passed &= TestTwoTowerSeededPacketAndKeySwitch();
    }
    catch (const std::exception& error) {
        std::cout
            << "[FAIL] two-tower stage-3 test threw: "
            << error.what()
            << '\n';
        passed = false;
    }

    std::cout << '\n';
    if (passed) {
        std::cout << "All protocol stage-3 tests passed.\n";
        return 0;
    }

    std::cout << "One or more protocol stage-3 tests failed.\n";
    return 1;
}
