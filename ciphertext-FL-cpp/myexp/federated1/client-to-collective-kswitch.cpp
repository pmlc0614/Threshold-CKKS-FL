#include "client-to-collective-kswitch.h"
#include "protocol-randomness.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

void ValidateHybridRns(
    const ProtocolRnsParameters& parameters) {

    if (!parameters.paramsQ ||
        !parameters.paramsP ||
        !parameters.paramsQP ||
        parameters.qTowerCount == 0 ||
        parameters.pTowerCount != 1 ||
        parameters.decompositionDigitCount != parameters.qTowerCount ||
        parameters.qTowersPerDigit != 1 ||
        parameters.paramsQ->GetParams().size() != parameters.qTowerCount ||
        parameters.paramsP->GetParams().size() != 1 ||
        parameters.paramsQP->GetParams().size() !=
            parameters.qTowerCount + 1) {

        throw std::invalid_argument(
            "KSwitch requires one HYBRID digit per Q tower and one P tower");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < parameters.qTowerCount;
         ++towerIndex) {

        if (parameters.paramsQP->GetParams()[towerIndex]->GetModulus() !=
            parameters.paramsQ->GetParams()[towerIndex]->GetModulus()) {

            throw std::invalid_argument(
                "QP basis must begin with the configured Q towers");
        }
    }

    if (parameters.paramsQP->GetParams()[parameters.qTowerCount]
            ->GetModulus() !=
        parameters.paramsP->GetParams()[0]->GetModulus()) {

        throw std::invalid_argument("QP basis ordering must be Q followed by P");
    }
}

void ValidatePolynomialBasis(
    const DCRTPoly& polynomial,
    const std::shared_ptr<DCRTPoly::Params>& expected,
    const char* name) {

    if (!expected) {
        throw std::invalid_argument("expected polynomial basis is null");
    }

    const auto& towers = polynomial.GetAllElements();
    const auto& expectedTowers = expected->GetParams();

    if (towers.size() != expectedTowers.size()) {
        throw std::invalid_argument(
            std::string(name) + " tower count is invalid");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < towers.size();
         ++towerIndex) {

        if (towers[towerIndex].GetModulus() !=
                expectedTowers[towerIndex]->GetModulus() ||
            towers[towerIndex].GetValues().GetLength() !=
                expected->GetRingDimension()) {

            throw std::invalid_argument(
                std::string(name) + " basis does not match protocol setup");
        }
    }
}

void ValidateSwitchingKey(
    const ClientToCollectiveKeyMaterial& key,
    const ProtocolRnsParameters& parameters) {

    if (key.survivors.clients.empty()) {
        throw std::invalid_argument("switching key has no survivor set");
    }

    if (key.decompositionDigitCount !=
            parameters.decompositionDigitCount ||
        key.a.size() != parameters.decompositionDigitCount ||
        key.b.size() != parameters.decompositionDigitCount) {

        throw std::invalid_argument(
            "switching key does not match the HYBRID digit profile");
    }

    for (std::size_t digitIndex = 0;
         digitIndex < parameters.decompositionDigitCount;
         ++digitIndex) {

        ValidatePolynomialBasis(
            key.a[digitIndex],
            parameters.paramsQP,
            "KSwitch a");
        ValidatePolynomialBasis(
            key.b[digitIndex],
            parameters.paramsQP,
            "KSwitch b");
    }
}

}  // namespace

const ClientToCollectiveKeyMaterial& FindClientToCollectiveKey(
    const SurvivorKeySet& survivorKeys,
    ClientId sourceClientId) {

    const auto found = std::find_if(
        survivorKeys.clientToCollectiveKeys.begin(),
        survivorKeys.clientToCollectiveKeys.end(),
        [sourceClientId](const ClientToCollectiveKeyMaterial& key) {
            return key.sourceClientId == sourceClientId;
        });

    if (found == survivorKeys.clientToCollectiveKeys.end()) {
        throw std::out_of_range(
            "no client-to-collective KSwitch key for source client");
    }

    return *found;
}

DCRTPoly ApproxModUpQDigitToQP(
    DCRTPoly valueQ,
    std::size_t digitIndex,
    const ProtocolRnsParameters& parameters,
    bool stochastic) {

    ValidateHybridRns(parameters);
    ValidatePolynomialBasis(valueQ, parameters.paramsQ, "mod-up input");

    if (digitIndex >= parameters.decompositionDigitCount) {
        throw std::out_of_range("mod-up digit index is out of range");
    }

    valueQ.SetFormat(Format::COEFFICIENT);

    const std::size_t ringDimension =
        parameters.paramsQ->GetRingDimension();
    const auto& sourceValues =
        valueQ.GetElementAtIndex(digitIndex).GetValues();

    const auto sourceModulus = parameters.paramsQ->GetParams()[digitIndex]->GetModulus();
    std::vector<bool> negative(ringDimension, false);
    if (stochastic) {
        for (std::size_t i = 0; i < ringDimension; ++i)
            negative[i] = SampleProtocolUniformBelow(lbcrypto::BigInteger(sourceModulus.ToString()))
                < lbcrypto::BigInteger(sourceValues[i].ToString());
    }
    DCRTPoly result(parameters.paramsQP, Format::COEFFICIENT, true);
    for (std::size_t towerIndex = 0;
         towerIndex < parameters.paramsQP->GetParams().size();
         ++towerIndex) {

        const lbcrypto::NativeInteger modulus =
            parameters.paramsQP->GetParams()[towerIndex]->GetModulus();
        lbcrypto::NativeVector values(ringDimension, modulus);

        for (std::size_t index = 0;
             index < ringDimension;
             ++index) {

            values[index] = sourceValues[index].Mod(modulus);
            if (negative[index])
                values[index] = values[index].ModSub(sourceModulus.Mod(modulus), modulus);
        }

        result.GetAllElements()[towerIndex].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }
    result.SetFormat(Format::EVALUATION);
    return result;
}

DCRTPoly ApproxModDownSinglePToQ(
    DCRTPoly valueQP,
    const ProtocolRnsParameters& parameters,
    bool centered,
    bool stochastic) {

    ValidateHybridRns(parameters);
    ValidatePolynomialBasis(valueQP, parameters.paramsQP, "mod-down input");

    valueQP.SetFormat(Format::COEFFICIENT);

    const std::size_t ringDimension =
        parameters.paramsQ->GetRingDimension();
    const lbcrypto::NativeInteger p =
        parameters.paramsP->GetParams()[0]->GetModulus();
    const auto& pValues =
        valueQP.GetElementAtIndex(parameters.qTowerCount).GetValues();

    std::vector<bool> negative(ringDimension, false);
    for (std::size_t i = 0; i < ringDimension; ++i) {
        negative[i] = stochastic
            ? SampleProtocolUniformBelow(lbcrypto::BigInteger(p.ToString()))
                < lbcrypto::BigInteger(pValues[i].ToString())
            : centered && pValues[i] > (p >> 1);
    }
    DCRTPoly result(parameters.paramsQ, Format::COEFFICIENT, true);

    for (std::size_t towerIndex = 0;
         towerIndex < parameters.qTowerCount;
         ++towerIndex) {

        const lbcrypto::NativeInteger q =
            parameters.paramsQ->GetParams()[towerIndex]->GetModulus();
        const lbcrypto::NativeInteger pInverseModQ =
            p.Mod(q).ModInverse(q);
        const auto& qValues =
            valueQP.GetElementAtIndex(towerIndex).GetValues();
        lbcrypto::NativeVector resultValues(ringDimension, q);

        for (std::size_t index = 0;
             index < ringDimension;
             ++index) {

            resultValues[index] =
                qValues[index]
                    .ModSub(pValues[index].Mod(q), q)
                    .ModMul(pInverseModQ, q);
            if (negative[index]) {
                resultValues[index] = resultValues[index].ModAdd(lbcrypto::NativeInteger(1), q);
            }
        }

        result.GetAllElements()[towerIndex].SetValues(
            std::move(resultValues),
            Format::COEFFICIENT);
    }
    result.SetFormat(Format::EVALUATION);
    return result;
}

DCRTPoly LiftQPolynomialTimesPToQP(
    DCRTPoly valueQ,
    const ProtocolRnsParameters& parameters) {

    ValidateHybridRns(parameters);
    ValidatePolynomialBasis(valueQ, parameters.paramsQ, "P-lift input");
    DCRTPoly result(parameters.paramsQP, Format::EVALUATION, true);
    for (std::size_t digitIndex = 0;
         digitIndex < parameters.decompositionDigitCount;
         ++digitIndex) {
        result += BuildM2RGadgetTerm(valueQ, digitIndex, parameters);
    }
    return result;
}

DCRTPoly BuildPreModDownPhaseQP(
    DCRTPoly sourceB,
    const HybridModDownTrace& trace,
    DCRTPoly collectiveSecretQP,
    const ProtocolRnsParameters& parameters) {

    ValidateHybridRns(parameters);
    ValidatePolynomialBasis(sourceB, parameters.paramsQ, "source b");
    ValidatePolynomialBasis(trace.bQP, parameters.paramsQP, "trace b");
    ValidatePolynomialBasis(trace.aQP, parameters.paramsQP, "trace a");
    ValidatePolynomialBasis(
        collectiveSecretQP,
        parameters.paramsQP,
        "collective QP secret");
    sourceB.SetFormat(Format::EVALUATION);
    collectiveSecretQP.SetFormat(Format::EVALUATION);
    DCRTPoly phaseQP = LiftQPolynomialTimesPToQP(sourceB, parameters);
    phaseQP += trace.bQP;
    phaseQP += trace.aQP * collectiveSecretQP;
    return phaseQP;
}

CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredSparseRLWECiphertext& ciphertext,
    const ClientToCollectiveKeyMaterial& switchingKey,
    const CommonSetupMaterial& publicMaterial,
    HybridModDownTrace* modDownTrace) {

    ValidateHybridRns(publicMaterial.rns);

    if (!(ciphertext.setupId == publicMaterial.setupId)) {
        throw std::invalid_argument(
            "ciphertext and switching key setup IDs do not match");
    }

    if (!(switchingKey.setupId == publicMaterial.setupId)) {
        throw std::invalid_argument(
            "switching key belongs to a different protocol setup");
    }

    if (!(ciphertext.clientId == switchingKey.sourceClientId)) {
        throw std::invalid_argument(
            "switching key belongs to a different source client");
    }

    ValidatePolynomialBasis(
        ciphertext.a,
        publicMaterial.rns.paramsQ,
        "ciphertext a");
    ValidatePolynomialBasis(
        ciphertext.b,
        publicMaterial.rns.paramsQ,
        "ciphertext b");
    ValidateSwitchingKey(switchingKey, publicMaterial.rns);

    DCRTPoly preModDownB(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);
    DCRTPoly preModDownA(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);

    for (std::size_t digitIndex = 0;
         digitIndex < publicMaterial.rns.decompositionDigitCount;
         ++digitIndex) {

        const DCRTPoly digitQP = ApproxModUpQDigitToQP(
            ciphertext.a,
            digitIndex,
            publicMaterial.rns, publicMaterial.descriptor.mode == ProtocolMode::SP);

        const DCRTPoly contributionB =
            digitQP * switchingKey.b[digitIndex];
        const DCRTPoly contributionA =
            digitQP * switchingKey.a[digitIndex];
        preModDownB += contributionB;
        preModDownA += contributionA;
    }

    // v17: one QP -> Q ModDown per logical KSwitch source, not per digit.
    DCRTPoly switchedB = ApproxModDownSinglePToQ(
        preModDownB, publicMaterial.rns, publicMaterial.descriptor.mode == ProtocolMode::FULL,
        publicMaterial.descriptor.mode == ProtocolMode::SP);
    DCRTPoly switchedA = ApproxModDownSinglePToQ(
        preModDownA, publicMaterial.rns, publicMaterial.descriptor.mode == ProtocolMode::FULL,
        publicMaterial.descriptor.mode == ProtocolMode::SP);

    if (modDownTrace != nullptr) {
        modDownTrace->bQP = std::move(preModDownB);
        modDownTrace->aQP = std::move(preModDownA);
    }

    DCRTPoly sparseB = ciphertext.b;
    sparseB.SetFormat(Format::EVALUATION);

    CollectiveRLWECiphertext result;
    result.setupId = ciphertext.setupId;
    result.roundId = ciphertext.roundId;
    result.sourceClientId = ciphertext.clientId;
    result.packGroupIndex = ciphertext.packGroupIndex;
    result.activeStreamCount = ciphertext.activeStreamCount;
    result.survivors = switchingKey.survivors;
    result.meaningfulCoefficientIndices =
        ciphertext.transmittedBIndices;
    result.b = sparseB + switchedB;
    result.a = switchedA;
    return result;
}

CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredSparseRLWECiphertext& ciphertext,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial) {

    if (!(survivorKeys.setupId == publicMaterial.setupId)) {
        throw std::invalid_argument(
            "survivor key set belongs to a different protocol setup");
    }

    const ClientToCollectiveKeyMaterial& key =
        FindClientToCollectiveKey(
            survivorKeys,
            ciphertext.clientId);

    if (key.survivors.clients !=
        survivorKeys.publicKey.survivors.clients) {

        throw std::invalid_argument(
            "KSwitch key survivor metadata is inconsistent");
    }

    return KeySwitchClientToCollective(
        ciphertext,
        key,
        publicMaterial);
}

CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredDenseRLWECiphertext& ciphertext,
    const ClientToCollectiveKeyMaterial& switchingKey,
    const CommonSetupMaterial& publicMaterial,
    HybridModDownTrace* modDownTrace) {

    ValidateHybridRns(publicMaterial.rns);
    if (!(ciphertext.setupId == publicMaterial.setupId) ||
        !(switchingKey.setupId == publicMaterial.setupId) ||
        !(ciphertext.clientId == switchingKey.sourceClientId)) {

        throw std::invalid_argument(
            "direct ciphertext and switching-key metadata do not match");
    }

    ValidatePolynomialBasis(
        ciphertext.a,
        publicMaterial.rns.paramsQ,
        "direct ciphertext a");
    ValidatePolynomialBasis(
        ciphertext.b,
        publicMaterial.rns.paramsQ,
        "direct ciphertext b");
    ValidateSwitchingKey(switchingKey, publicMaterial.rns);

    DCRTPoly preModDownB(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);
    DCRTPoly preModDownA(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);

    for (std::size_t digitIndex = 0;
         digitIndex < publicMaterial.rns.decompositionDigitCount;
         ++digitIndex) {

        const DCRTPoly digitQP = ApproxModUpQDigitToQP(
            ciphertext.a,
            digitIndex,
            publicMaterial.rns, publicMaterial.descriptor.mode == ProtocolMode::SP);
        const DCRTPoly contributionB =
            digitQP * switchingKey.b[digitIndex];
        const DCRTPoly contributionA =
            digitQP * switchingKey.a[digitIndex];
        preModDownB += contributionB;
        preModDownA += contributionA;
    }

    DCRTPoly switchedB = ApproxModDownSinglePToQ(
        preModDownB, publicMaterial.rns, publicMaterial.descriptor.mode == ProtocolMode::FULL,
        publicMaterial.descriptor.mode == ProtocolMode::SP);
    DCRTPoly switchedA = ApproxModDownSinglePToQ(
        preModDownA, publicMaterial.rns, publicMaterial.descriptor.mode == ProtocolMode::FULL,
        publicMaterial.descriptor.mode == ProtocolMode::SP);

    if (modDownTrace != nullptr) {
        modDownTrace->bQP = std::move(preModDownB);
        modDownTrace->aQP = std::move(preModDownA);
    }

    DCRTPoly directB = ciphertext.b;
    directB.SetFormat(Format::EVALUATION);

    CollectiveRLWECiphertext result;
    result.setupId = ciphertext.setupId;
    result.roundId = ciphertext.roundId;
    result.sourceClientId = ciphertext.clientId;
    result.packGroupIndex = ciphertext.blockIndex;
    result.activeStreamCount = 1;
    result.survivors = switchingKey.survivors;
    result.b = directB + switchedB;
    result.a = switchedA;
    return result;
}

CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredDenseRLWECiphertext& ciphertext,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial) {

    if (!(survivorKeys.setupId == publicMaterial.setupId)) {
        throw std::invalid_argument(
            "survivor key set belongs to a different protocol setup");
    }

    const ClientToCollectiveKeyMaterial& key =
        FindClientToCollectiveKey(survivorKeys, ciphertext.clientId);
    if (key.survivors.clients !=
        survivorKeys.publicKey.survivors.clients) {

        throw std::invalid_argument(
            "KSwitch key survivor metadata is inconsistent");
    }
    return KeySwitchClientToCollective(
        ciphertext,
        key,
        publicMaterial);
}

}  // namespace federated
