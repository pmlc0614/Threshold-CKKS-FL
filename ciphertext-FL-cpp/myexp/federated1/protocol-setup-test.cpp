#include "coefficient-codec.h"
#include "client-to-collective-kswitch.h"
#include "rns-polynomial.h"
#include "protocol-rounding.h"
#include "coefficient-compression.h"
#include "protocol-setup-test.h"

#include "deterministic-randomness.h"
#include "protocol-parameters.h"
#include "protocol-sparse-path.h"
#include "protocol-setup.h"
#include "ring-conversion.h"
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

bool EqualDCRT(DCRTPoly left, DCRTPoly right) {
    left.SetFormat(Format::COEFFICIENT);
    right.SetFormat(Format::COEFFICIENT);

    const auto& leftTowers = left.GetAllElements();
    const auto& rightTowers = right.GetAllElements();

    if (leftTowers.size() != rightTowers.size()) {
        return false;
    }

    for (std::size_t towerIndex = 0;
         towerIndex < leftTowers.size();
         ++towerIndex) {

        if (leftTowers[towerIndex].GetModulus() !=
            rightTowers[towerIndex].GetModulus()) {

            return false;
        }

        const auto& leftValues = leftTowers[towerIndex].GetValues();
        const auto& rightValues = rightTowers[towerIndex].GetValues();

        if (leftValues.GetLength() != rightValues.GetLength()) {
            return false;
        }

        for (std::size_t index = 0;
             index < leftValues.GetLength();
             ++index) {

            if (leftValues[index] != rightValues[index]) {
                return false;
            }
        }
    }

    return true;
}

std::uint64_t MaxCenteredMagnitude(DCRTPoly polynomial) {
    polynomial.SetFormat(Format::COEFFICIENT);
    lbcrypto::Poly interpolated = polynomial.CRTInterpolate();
    interpolated.SetFormat(Format::COEFFICIENT);

    const lbcrypto::BigInteger modulus = interpolated.GetModulus();
    const lbcrypto::BigInteger half = modulus >> 1;
    std::uint64_t maximum = 0;

    for (std::size_t index = 0;
         index < interpolated.GetValues().GetLength();
         ++index) {

        const lbcrypto::BigInteger residue = interpolated.GetValues()[index];
        const lbcrypto::BigInteger magnitude =
            residue > half ? modulus - residue : residue;

        if (magnitude.GetMSB() > 64) {
            return std::numeric_limits<std::uint64_t>::max();
        }

        maximum = std::max(maximum, magnitude.ConvertToInt());
    }

    return maximum;
}

lbcrypto::CryptoContext<DCRTPoly> MakeSetupTestContext(
    const federated::ProtocolExperimentConfig& config,
    lbcrypto::SecretKeyDist distribution = lbcrypto::UNIFORM_TERNARY) {

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;

    parameters.SetMultiplicativeDepth(1);
    parameters.SetFirstModSize(config.modulus.qModulusBits.front());
    parameters.SetScalingModSize(config.modulus.qModulusBits[1] - 1);
    parameters.SetBatchSize(config.modulus.ringDimension / 2);
    parameters.SetRingDim(config.modulus.ringDimension);
    parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetScalingTechnique(lbcrypto::FIXEDMANUAL);
    parameters.SetSecretKeyDist(distribution);

    /* The protocol setup constructs its custom 60-bit P and QP basis instead of using the context's automatic HYBRID P. */
    parameters.SetKeySwitchTechnique(lbcrypto::BV);

    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    return cc;
}

DCRTPoly SumSurvivorSecretQP(
    const federated::ProtocolSetupResult& setup,
    const federated::SurvivorSet& survivors) {

    DCRTPoly sum(
        setup.common.rns.paramsQP,
        Format::EVALUATION,
        true);

    for (const federated::ClientId clientId : survivors.clients) {
        sum += setup.clientSecrets[clientId.value].secretKeyQP;
    }

    return sum;
}

DCRTPoly SumSurvivorSecretQ(
    const federated::ProtocolSetupResult& setup,
    const federated::SurvivorSet& survivors) {

    DCRTPoly sum(
        setup.common.rns.paramsQ,
        Format::EVALUATION,
        true);

    for (const federated::ClientId clientId : survivors.clients) {
        sum += setup.clientSecrets[clientId.value]
                   .secretKeyQ
                   ->GetPrivateElement();
    }

    return sum;
}

bool CheckSurvivorEquations(
    const federated::ProtocolSetupResult& setup,
    const federated::SurvivorKeySet& keys) {

    constexpr std::uint64_t generousNoiseBound = 4096;
    const DCRTPoly secretQP =
        SumSurvivorSecretQP(setup, keys.publicKey.survivors);

    const DCRTPoly publicKeyResidual =
        keys.publicKey.p0 + keys.publicKey.p1 * secretQP;

    bool passed = Check(
        MaxCenteredMagnitude(publicKeyResidual) <= generousNoiseBound,
        "survivor public-key equation leaves only small setup error");

    if (!keys.m2rKey) {
        return passed;
    }

    const DCRTPoly secretQ =
        SumSurvivorSecretQ(setup, keys.publicKey.survivors);

    DCRTPoly modelSecretQP(setup.common.rns.paramsQP, Format::EVALUATION, true);
    for (const auto id : keys.publicKey.survivors.clients)
        modelSecretQP += setup.clientSecrets[id.value].modelKeyQP;
    const auto splitSecret = ring_conversion::Split(
        secretQ,
        setup.common.mlweRank);

    for (std::size_t splitIndex = 0;
         splitIndex < setup.common.mlweRank;
         ++splitIndex) {

        const DCRTPoly embedded = ring_conversion::RingEmbedding(
            splitSecret[splitIndex],
            setup.common.mlweRank);

        for (std::size_t lambda = 0;
             lambda <
                 setup.common.rns.decompositionDigitCount;
             ++lambda) {

            const DCRTPoly target = federated::BuildM2RGadgetTerm(
                embedded,
                lambda,
                setup.common.rns);

            const DCRTPoly residual =
                keys.m2rKey->b[splitIndex][lambda] +
                keys.m2rKey->a[splitIndex][lambda] * modelSecretQP -
                target;

            passed &= Check(
                MaxCenteredMagnitude(residual) <= generousNoiseBound,
                "survivor M2R equation is valid for split " +
                    std::to_string(splitIndex) +
                    ", digit " + std::to_string(lambda));
        }
    }

    return passed;
}

bool CheckClientToCollectiveEquations(
    const federated::ProtocolSetupResult& setup,
    const federated::SurvivorKeySet& keys) {

    constexpr std::uint64_t generousNoiseBound = 65536;
    const DCRTPoly targetSecretQP =
        SumSurvivorSecretQP(setup, keys.publicKey.survivors);

    bool passed = Check(
        keys.clientToCollectiveKeys.size() ==
            keys.publicKey.survivors.clients.size(),
        "one client-to-collective KSK exists per survivor");

    for (const auto& key : keys.clientToCollectiveKeys) {
        passed &= Check(
            key.decompositionDigitCount ==
                    setup.common.rns.decompositionDigitCount &&
                key.a.size() == key.decompositionDigitCount &&
                key.b.size() == key.decompositionDigitCount,
            "client-to-collective KSK has one pair per Q digit for client " +
                std::to_string(key.sourceClientId.value));

        const DCRTPoly& sourceSecretQ =
            setup.clientSecrets[key.sourceClientId.value]
                .encryptionKeyQ
                ->GetPrivateElement();

        for (std::size_t lambda = 0;
             lambda < key.decompositionDigitCount;
             ++lambda) {

            const DCRTPoly target = federated::BuildM2RGadgetTerm(
                sourceSecretQ,
                lambda,
                setup.common.rns);

            const DCRTPoly residual =
                key.b[lambda] + key.a[lambda] * targetSecretQP - target;

            passed &= Check(
                MaxCenteredMagnitude(residual) <= generousNoiseBound,
                "client-to-collective KSK equation is valid for client " +
                    std::to_string(key.sourceClientId.value) +
                    ", digit " + std::to_string(lambda));
        }
    }

    return passed;
}

}  // namespace

int RunProtocolSetupTests() {
    using namespace federated;

    bool passed = true;

    try {
        ProtocolExperimentConfig config = MakeSmokeConfig();
        auto cc = MakeSetupTestContext(config);
        auto wrongDistribution = MakeSetupTestContext(config, lbcrypto::SPARSE_TERNARY);
        passed &= CheckThrows([&]() { (void)GenerateProtocolSetup(wrongDistribution, config); },
            "setup rejects a non-uniform-ternary context");

        const auto secretKeys =
            GenerateClientSecretKeys(cc, config.clientCount);

        const ProtocolSetupResult setup =
            GenerateProtocolSetupFromSecretKeys(cc, config, secretKeys);

        const auto& rns = setup.common.rns;
        passed &= Check(protocol_rounding::NearestInteger(-1.5L) == -1 &&
            protocol_rounding::NearestInteger(-0.5L) == 0 &&
            protocol_rounding::NearestInteger(0.5L) == 1 &&
            protocol_rounding::NearestInteger(-1.5001L) == -2,
            "nearest-integer ties go toward positive infinity");
        passed &= Check(coefficient_compression::QuantizeSignedWithScale(
            -1.5L, 8, 1.0L) == -1, "delivery uses normative negative tie");
        const auto ties = coefficient_codec::Decode(
            coefficient_codec::Encode(cc, {-1.5, -0.5, 0.5, 1.5}, 1.0), 1.0);
        passed &= Check(ties[0] == -1 && ties[1] == 0 && ties[2] == 1 && ties[3] == 2,
            "coefficient encoding uses normative ties");
        const auto p = rns.modulusP;
        std::vector<lbcrypto::BigInteger> probes(
            rns.paramsQP->GetRingDimension(), lbcrypto::BigInteger(0));
        probes[0] = p >> 1;
        probes[1] = (p >> 1) + lbcrypto::BigInteger(1);
        probes[2] = p - lbcrypto::BigInteger(1);
        probes[3] = p;
        const auto input = rns_polynomial::FromCompositeCoefficients(rns.paramsQP, probes);
        auto rounded = ApproxModDownSinglePToQ(input, rns, true);
        rounded.SetFormat(Format::COEFFICIENT);
        auto legacy = ApproxModDownSinglePToQ(input, rns, false);
        legacy.SetFormat(Format::COEFFICIENT);
        for (const auto& limb : rounded.GetAllElements()) {
            passed &= Check(limb.GetValues()[0] == lbcrypto::NativeInteger(0) &&
                limb.GetValues()[1] == lbcrypto::NativeInteger(1) &&
                limb.GetValues()[2] == lbcrypto::NativeInteger(1) &&
                limb.GetValues()[3] == lbcrypto::NativeInteger(1),
                "centered ModDown crosses the odd-P midpoint exactly");
        }
        passed &= Check(legacy.GetElementAtIndex(0).GetValues()[1] ==
            lbcrypto::NativeInteger(0), "legacy Sparse ModDown remains unchanged");


        passed &= Check(
            rns.qTowerCount == 2,
            "custom protocol basis has two Q towers");

        passed &= Check(
            rns.pTowerCount == 1,
            "custom protocol basis has one P tower");

        passed &= Check(
            rns.decompositionDigitCount == 2 &&
                rns.qTowersPerDigit == 1,
            "lambda maps one-to-one to the Q tower");

        passed &= Check(
            rns.paramsP->GetModulus().GetMSB() == 60,
            "custom special modulus P is 60 bits");

        passed &= Check(
            rns.paramsQP->GetModulus().GetMSB() <=
                config.modulus.maximumQpBits,
            "custom QP satisfies the N=8192 configured security bound");

        passed &= Check(
            setup.clientSecrets.size() == config.clientCount &&
                setup.cachedClientShares.size() == config.clientCount,
            "setup contains one secret and cached share per client");

        passed &= Check(
            setup.mode == ProtocolMode::SP &&
                setup.common.descriptor.mode == ProtocolMode::SP &&
                setup.sparse.has_value() &&
                setup.sparse->m2rCommonA.size() ==
                config.sparsification.mlweRank,
            "SP setup contains one M2R common-a row per MLWE split");

        const DCRTPoly regeneratedP1 = GenerateSeededUniformDCRT(
            rns.paramsQP,
            setup.common.crsSeed,
            "PK-P1",
            ByteVector(
                setup.common.setupId.bytes.begin(),
                setup.common.setupId.bytes.end()));

        passed &= Check(
            EqualDCRT(regeneratedP1, setup.common.publicKeyP1),
            "public p1 is exactly reproducible from the CRS seed");

        ByteVector m2rContext(
            setup.common.setupId.bytes.begin(),
            setup.common.setupId.bytes.end());

        const ByteVector encodedSplit =
            deterministic_randomness::EncodeBigEndian64(3);
        const ByteVector encodedLambda =
            deterministic_randomness::EncodeBigEndian64(0);

        m2rContext.insert(
            m2rContext.end(),
            encodedSplit.begin(),
            encodedSplit.end());

        m2rContext.insert(
            m2rContext.end(),
            encodedLambda.begin(),
            encodedLambda.end());

        const DCRTPoly regeneratedM2RA = GenerateSeededUniformDCRT(
            rns.paramsQP,
            setup.common.crsSeed,
            "M2R-A",
            m2rContext);

        passed &= Check(
            EqualDCRT(
                regeneratedM2RA,
                setup.sparse->m2rCommonA[3][0]),
            "M2R common-a is exactly reproducible from CRS and (j, lambda)");

        SurvivorSet subset{{ClientId{5}, ClientId{0}, ClientId{2}}};
        const SurvivorKeySet subsetKeys =
            AggregateSurvivorKeys(setup, subset);

        passed &= Check(
            subsetKeys.publicKey.survivors.clients ==
                std::vector<ClientId>{ClientId{0}, ClientId{2}, ClientId{5}},
            "survivor IDs are canonicalized without changing membership");

        DCRTPoly manualP0(
            rns.paramsQP,
            Format::EVALUATION,
            true);

        for (const ClientId clientId :
             subsetKeys.publicKey.survivors.clients) {

            manualP0 +=
                setup.cachedClientShares[clientId.value]
                    .publicKeyP0Share;
        }

        passed &= Check(
            EqualDCRT(manualP0, subsetKeys.publicKey.p0),
            "survivor p0 is the exact sum of cached p0 shares");

        const SurvivorKeySet repeatedSubsetKeys =
            AggregateSurvivorKeys(setup, subset);

        passed &= Check(
            EqualDCRT(
                subsetKeys.publicKey.p0,
                repeatedSubsetKeys.publicKey.p0),
            "repeated survivor assembly reuses cached one-time shares");

        for (std::size_t splitIndex = 0;
             splitIndex < config.sparsification.mlweRank;
             ++splitIndex) {

            passed &= Check(
                EqualDCRT(
                    subsetKeys.m2rKey->a[splitIndex][0],
                    setup.sparse->m2rCommonA[splitIndex][0]),
                "survivor M2R key reuses common-a for split " +
                    std::to_string(splitIndex));

            passed &= Check(
                EqualDCRT(
                    subsetKeys.m2rKey->b[splitIndex][0],
                    repeatedSubsetKeys.m2rKey->b[splitIndex][0]),
                "survivor M2R b is stable for split " +
                    std::to_string(splitIndex));
        }

        passed &= CheckSurvivorEquations(setup, subsetKeys);
        passed &= CheckClientToCollectiveEquations(setup, subsetKeys);

        const bool freshKsk =
            !subsetKeys.clientToCollectiveKeys.empty() &&
            !repeatedSubsetKeys.clientToCollectiveKeys.empty() &&
            !subsetKeys.clientToCollectiveKeys[0].a.empty() &&
            !repeatedSubsetKeys.clientToCollectiveKeys[0].a.empty() &&
            !EqualDCRT(
                subsetKeys.clientToCollectiveKeys[0].a[0],
                repeatedSubsetKeys.clientToCollectiveKeys[0].a[0]);

        passed &= Check(
            freshKsk,
            "client-to-collective KSK uses fresh per-survivor preparation randomness");

        SurvivorSet allClients;
        for (std::size_t clientIndex = 0;
             clientIndex < config.clientCount;
             ++clientIndex) {

            allClients.clients.push_back(ClientId{clientIndex});
        }

        const SurvivorKeySet fullKeys =
            AggregateSurvivorKeys(setup, allClients);

        for (const auto& key : setup.clientSecrets) {
            passed &= Check(key.modelKeyQ && key.encryptionKeyQ != key.secretKeyQ &&
                key.modelKeyQ != key.secretKeyQ && key.modelKeyQ != key.encryptionKeyQ,
                "Sparse x/s/t keys are independently allocated");
        }
        passed &= CheckSurvivorEquations(setup, fullKeys);
        passed &= CheckClientToCollectiveEquations(setup, fullKeys);

        ProtocolExperimentConfig fullConfig = config;
        fullConfig.mode = ProtocolMode::FULL;
        const ProtocolSetupResult fullSetup =
            GenerateProtocolSetupFromSecretKeys(
                cc,
                fullConfig,
                secretKeys);
        const SurvivorKeySet fullModeKeys =
            AggregateSurvivorKeys(fullSetup, allClients);

        const bool fullSharesHaveNoM2R = std::all_of(
            fullSetup.cachedClientShares.begin(),
            fullSetup.cachedClientShares.end(),
            [](const ClientCachedSetupShare& share) {
                return !share.m2rShare.has_value();
            });

        passed &= Check(
            fullSetup.mode == ProtocolMode::FULL &&
                fullSetup.common.descriptor.mode == ProtocolMode::FULL &&
                !fullSetup.sparse.has_value() &&
                !fullModeKeys.m2rKey.has_value() &&
                fullSharesHaveNoM2R,
            "FULL setup and survivor keys contain no sparse M2R material");

        passed &= Check(
            setup.common.descriptor.canonicalEncoding !=
                    fullSetup.common.descriptor.canonicalEncoding &&
                !(setup.common.setupId == fullSetup.common.setupId),
            "mode is bound into the descriptor and separates SP/FULL setup IDs");

        passed &= Check(
            fullModeKeys.clientToCollectiveKeys.size() ==
                allClients.clients.size(),
            "FULL setup still generates one client-to-collective KSK per client");

        passed &= CheckClientToCollectiveEquations(fullSetup, fullModeKeys);
        for (const auto& secret : fullSetup.clientSecrets) {
            passed &= Check(secret.encryptionKeyQ &&
                secret.encryptionKeyQ != secret.secretKeyQ,
                "FULL encryption and collective keys are separate objects");
        }

        passed &= CheckThrows(
            [&]() {
                (void)RunProtocolSparsePath(
                    lbcrypto::CryptoContext<DCRTPoly>{},
                    fullConfig,
                    std::vector<ProtocolClientRoundInput>{},
                    ProtocolSparseRoundArtifacts{},
                    fullSetup,
                    fullModeKeys,
                    MakeUncompressedSparseProfile());
            },
            "FULL setup rejects a sparse-path invocation");

        ProtocolExperimentConfig zeroWeight = config;
        zeroWeight.integerWeights[0] = 0;
        zeroWeight.integerWeights[1] += config.integerWeights[0];
        passed &= Check(
            ValidateProtocolParameters(zeroWeight).HasErrors(),
            "a zero registered-client weight is rejected");

        ProtocolExperimentConfig wrongWeightSum = config;
        ++wrongWeightSum.integerWeights[0];
        passed &= Check(
            ValidateProtocolParameters(wrongWeightSum).HasErrors(),
            "integer weights whose sum is not 128 are rejected");

        passed &= CheckThrows(
            [&setup]() {
                AggregateSurvivorKeys(
                    setup,
                    SurvivorSet{{ClientId{0}, ClientId{0}}});
            },
            "duplicate survivor IDs are rejected");

        passed &= CheckThrows(
            [&setup]() {
                AggregateSurvivorKeys(setup, SurvivorSet{});
            },
            "empty survivor set is rejected");
    }
    catch (const std::exception& error) {
        std::cerr
            << "[FAIL] protocol setup test threw: "
            << error.what()
            << '\n';

        passed = false;
    }

    std::cout
        << '\n'
        << (passed
                ? "All protocol setup and survivor-key tests passed."
                : "One or more protocol setup tests failed.")
        << '\n';

    return passed ? 0 : 1;
}
