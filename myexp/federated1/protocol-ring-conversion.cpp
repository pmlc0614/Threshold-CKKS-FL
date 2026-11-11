#include "protocol-ring-conversion.h"

#include "ring-conversion.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

void MultiplyPolynomialByInteger(
    DCRTPoly& polynomial,
    std::uint32_t scalar) {

    const Format format = polynomial.GetFormat();
    const lbcrypto::NativeInteger factor(scalar);

    for (auto& tower : polynomial.GetAllElements()) {
        const lbcrypto::NativeInteger modulus = tower.GetModulus();
        lbcrypto::NativeVector values = tower.GetValues();

        for (std::size_t index = 0;
             index < values.GetLength();
             ++index) {

            values[index] = values[index].ModMul(factor, modulus);
        }
        tower.SetValues(std::move(values), format);
    }
}

void ValidateMLWE(const LabeledMLWECiphertext& ciphertext) {
    if (ciphertext.rank == 0 ||
        ciphertext.a.size() != ciphertext.rank ||
        ciphertext.ringDimension == 0) {

        throw std::invalid_argument("invalid labeled MLWE ciphertext");
    }
}

M2RComponentResult SwitchM2RComponent(
    DCRTPoly component,
    std::size_t splitIndex,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    bool collectTrace) {

    const std::size_t digitCount =
        publicMaterial.rns.decompositionDigitCount;

    if (publicMaterial.descriptor.mode != ProtocolMode::SP ||
        !survivorKeys.m2rKey ||
        splitIndex >= survivorKeys.m2rKey->a.size() ||
        splitIndex >= survivorKeys.m2rKey->b.size() ||
        survivorKeys.m2rKey->a[splitIndex].size() != digitCount ||
        survivorKeys.m2rKey->b[splitIndex].size() != digitCount) {

        throw std::invalid_argument("M2R switching key shape is invalid");
    }

    DCRTPoly outputBQP(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);
    DCRTPoly outputAQP(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);

    for (std::size_t digitIndex = 0;
         digitIndex < digitCount;
         ++digitIndex) {

        const DCRTPoly digitQP = ApproxModUpQDigitToQP(
            component,
            digitIndex,
            publicMaterial.rns, true);

        const DCRTPoly contributionB =
            digitQP * survivorKeys.m2rKey->b[splitIndex][digitIndex];
        const DCRTPoly contributionA =
            digitQP * survivorKeys.m2rKey->a[splitIndex][digitIndex];
        outputBQP += contributionB;
        outputAQP += contributionA;
    }
    // v17: sum digit contributions in QP, then ModDown once for this source.
    DCRTPoly outputB = ApproxModDownSinglePToQ(
        outputBQP, publicMaterial.rns, false, true);
    DCRTPoly outputA = ApproxModDownSinglePToQ(
        outputAQP, publicMaterial.rns, false, true);
    return {
        std::move(outputB),
        std::move(outputA),
        collectTrace ? std::move(outputBQP) : DCRTPoly{},
        collectTrace ? std::move(outputAQP) : DCRTPoly{}};
}

}  // namespace

std::vector<LabeledMLWECiphertext> ProtocolRLWEToMLWE(
    const CollectiveRLWECiphertext& ciphertext,
    const PackedShuffledLabels& labels,
    std::size_t mlweRank) {

    if (labels.labels.empty() || labels.labels.size() > mlweRank ||
        ciphertext.activeStreamCount != labels.labels.size()) {

        throw std::invalid_argument("R2M labels do not match active streams");
    }

    const auto bSplit = ring_conversion::Split(ciphertext.b, mlweRank);
    const auto aSplit = ring_conversion::Split(ciphertext.a, mlweRank);
    const std::size_t smallDimension =
        ciphertext.a.GetParams()->GetRingDimension() / mlweRank;

    std::vector<LabeledMLWECiphertext> result;
    result.reserve(labels.labels.size());

    for (std::size_t row = 0; row < labels.labels.size(); ++row) {
        LabeledMLWECiphertext output;
        output.label = labels.labels[row];
        output.rank = mlweRank;
        output.ringDimension = smallDimension;
        output.b = bSplit[row];
        output.a.resize(mlweRank);

        for (std::size_t column = 0; column < mlweRank; ++column) {
            const std::size_t splitIndex =
                (row + mlweRank - column) % mlweRank;
            output.a[column] = aSplit[splitIndex];
            if (row < column) {
                output.a[column] = ring_conversion::MultX(output.a[column]);
            }
        }
        result.push_back(std::move(output));
    }
    return result;
}

LabeledMLWECiphertext MultiplyMLWEByInteger(
    const LabeledMLWECiphertext& ciphertext,
    std::uint32_t scalar) {

    ValidateMLWE(ciphertext);
    LabeledMLWECiphertext result = ciphertext;
    for (auto& component : result.a) {
        MultiplyPolynomialByInteger(component, scalar);
    }
    MultiplyPolynomialByInteger(result.b, scalar);
    return result;
}

void AddMLWEInPlace(
    LabeledMLWECiphertext& target,
    const LabeledMLWECiphertext& source) {

    ValidateMLWE(target);
    ValidateMLWE(source);

    if (!(target.label == source.label) ||
        target.rank != source.rank ||
        target.ringDimension != source.ringDimension) {

        throw std::invalid_argument("MLWE aggregation metadata mismatch");
    }

    for (std::size_t column = 0; column < target.rank; ++column) {
        target.a[column] += source.a[column];
    }
    target.b += source.b;
}

PreparedM2R PrepareM2R(
    std::vector<LabeledMLWECiphertext> ciphertexts,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    bool collectTrace) {

    const std::size_t rank = publicMaterial.mlweRank;
    if (!(survivorKeys.setupId == publicMaterial.setupId) ||
        publicMaterial.descriptor.mode != ProtocolMode::SP ||
        !survivorKeys.m2rKey ||
        ciphertexts.empty() ||
        ciphertexts.size() != labels.labels.size() ||
        ciphertexts.size() > rank ||
        survivorKeys.m2rKey->mlweRank != rank) {

        throw std::invalid_argument("invalid M2R input or survivor key metadata");
    }

    for (std::size_t index = 0; index < ciphertexts.size(); ++index) {
        ValidateMLWE(ciphertexts[index]);
        if (!(ciphertexts[index].label == labels.labels[index]) ||
            ciphertexts[index].rank != rank) {

            throw std::invalid_argument("M2R label ordering mismatch");
        }
    }

    PreparedM2R prepared;
    prepared.rows = std::move(ciphertexts);
    prepared.components.resize(rank);
    prepared.collectTrace = collectTrace;
    std::vector<DCRTPoly> bRows;
    bRows.reserve(rank);
    for (const auto& row : prepared.rows) bRows.push_back(row.b);
    while (bRows.size() < rank)
        bRows.emplace_back(bRows.front().GetParams(), bRows.front().GetFormat(), true);
    prepared.combinedB = ring_conversion::Combine(bRows);
    prepared.combinedB.SetFormat(Format::EVALUATION);
    return prepared;
}

void SwitchPreparedM2RComponent(
    PreparedM2R& prepared, std::size_t column,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial) {
    const std::size_t rank = publicMaterial.mlweRank;
    if (column >= rank || prepared.components.size() != rank || prepared.rows.empty())
        throw std::invalid_argument("invalid prepared M2R component");
    std::vector<DCRTPoly> columnRows;
    columnRows.reserve(rank);
    for (const auto& row : prepared.rows) columnRows.push_back(row.a[column]);
    while (columnRows.size() < rank)
        columnRows.emplace_back(columnRows.front().GetParams(), columnRows.front().GetFormat(), true);
    prepared.components[column] = SwitchM2RComponent(
        ring_conversion::Combine(columnRows), column, survivorKeys,
        publicMaterial, prepared.collectTrace);
}

PackedAggregateRLWECiphertext FinishM2R(
    PreparedM2R& prepared,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId, M2RModDownTrace* modDownTrace) {
    const std::size_t rank = publicMaterial.mlweRank;
    DCRTPoly outputB = std::move(prepared.combinedB);
    if (modDownTrace) {
        if (!prepared.collectTrace)
            throw std::invalid_argument("M2R trace was not requested at preparation");
        modDownTrace->combinedB = outputB;
        modDownTrace->switching.bQP = DCRTPoly(publicMaterial.rns.paramsQP, Format::EVALUATION, true);
        modDownTrace->switching.aQP = DCRTPoly(publicMaterial.rns.paramsQP, Format::EVALUATION, true);
    }
    DCRTPoly outputA(publicMaterial.rns.paramsQ, Format::EVALUATION, true);
    // Preserve per-source ModDown and reduce in column order after the barrier.
    for (auto& component : prepared.components) {
        outputB += component.b;
        outputA += component.a;
        if (modDownTrace) {
            modDownTrace->switching.bQP += component.bQP;
            modDownTrace->switching.aQP += component.aQP;
        }
    }
    PackedAggregateRLWECiphertext result;
    result.setupId = publicMaterial.setupId;
    result.roundId = roundId;
    result.survivors = survivorKeys.publicKey.survivors;
    result.shuffledLabels = labels;
    result.meaningfulCoefficientIndices =
        SparseStreamCoefficientIndices(
            publicMaterial.rns.paramsQ->GetRingDimension(),
            rank,
            labels.labels.size());
    result.b = std::move(outputB);
    result.a = std::move(outputA);
    return result;
}

PackedAggregateRLWECiphertext ProtocolMLWEToRLWE(
    const std::vector<LabeledMLWECiphertext>& ciphertexts,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId, M2RModDownTrace* modDownTrace) {
    auto prepared = PrepareM2R(ciphertexts, labels, survivorKeys, publicMaterial,
                              modDownTrace != nullptr);
    for (std::size_t column = 0; column < publicMaterial.mlweRank; ++column)
        SwitchPreparedM2RComponent(prepared, column, survivorKeys, publicMaterial);
    return FinishM2R(prepared, labels, survivorKeys, publicMaterial, roundId, modDownTrace);
}

}  // namespace federated
