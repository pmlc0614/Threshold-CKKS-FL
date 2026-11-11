#include "ring-conversion.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ring_conversion {
namespace {

size_t RingDimension(const lbcrypto::DCRTPoly& polynomial) {
    const auto& towers = polynomial.GetAllElements();

    if (towers.empty()) {
        OPENFHE_THROW(
            "RingDimension: polynomial has no CRT towers");
    }

    return towers[0].GetValues().GetLength();
}

bool IsPowerOfTwo(size_t value) {
    return value != 0 &&
           (value & (value - 1)) == 0;
}

void ValidateRank(
    size_t ringDimension,
    size_t rank,
    const char* functionName) {

    if (rank == 0) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": rank must be greater than zero");
    }

    if (!IsPowerOfTwo(rank)) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": rank must be a power of two");
    }

    if (ringDimension % rank != 0) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": ring dimension must be divisible by rank");
    }

    if (!IsPowerOfTwo(ringDimension / rank)) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": smaller ring dimension must be a power of two");
    }
}

lbcrypto::DCRTPoly ToCoefficient(
    const lbcrypto::DCRTPoly& polynomial) {

    lbcrypto::DCRTPoly result = polynomial;

    if (result.GetFormat() != Format::COEFFICIENT) {
        result.SetFormat(Format::COEFFICIENT);
    }

    return result;
}

std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>
MakeDCRTParams(
    const lbcrypto::DCRTPoly& reference,
    size_t ringDimension) {

    const auto& referenceTowerParams =
        reference.GetParams()->GetParams();

    if (referenceTowerParams.empty()) {
        OPENFHE_THROW(
            "MakeDCRTParams: reference polynomial has no CRT towers");
    }

    std::vector<
        std::shared_ptr<
            lbcrypto::ILParamsImpl<lbcrypto::NativeInteger>>>
        towerParams;

    towerParams.reserve(referenceTowerParams.size());

    const size_t cyclotomicOrder =
        2 * ringDimension;

    for (const auto& parameter : referenceTowerParams) {
        towerParams.emplace_back(
            std::make_shared<
                lbcrypto::ILParamsImpl<
                    lbcrypto::NativeInteger>>(
                cyclotomicOrder,
                parameter->GetModulus()));
    }

    return std::make_shared<
        lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
        cyclotomicOrder,
        towerParams);
}

void ValidateCompatible(
    const lbcrypto::DCRTPoly& reference,
    const lbcrypto::DCRTPoly& candidate,
    const char* functionName) {

    if (RingDimension(reference) !=
        RingDimension(candidate)) {

        OPENFHE_THROW(
            std::string(functionName) +
            ": ring dimensions do not match");
    }

    const auto& referenceParams =
        reference.GetParams()->GetParams();

    const auto& candidateParams =
        candidate.GetParams()->GetParams();

    if (referenceParams.size() !=
        candidateParams.size()) {

        OPENFHE_THROW(
            std::string(functionName) +
            ": CRT tower counts do not match");
    }

    for (size_t tower = 0;
         tower < referenceParams.size();
         ++tower) {

        if (referenceParams[tower]->GetModulus() !=
            candidateParams[tower]->GetModulus()) {

            OPENFHE_THROW(
                std::string(functionName) +
                ": CRT tower moduli do not match");
        }
    }
}

void ValidateMLWECiphertext(
    const MLWECiphertext& ciphertext,
    const char* functionName) {

    if (ciphertext.rank == 0) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": MLWE rank is zero");
    }

    if (ciphertext.a.size() != ciphertext.rank) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": MLWE a component count must equal rank");
    }

    if (ciphertext.ringDimension == 0) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": MLWE ring dimension is zero");
    }

    if (!std::isfinite(ciphertext.scalingFactor) ||
        ciphertext.scalingFactor <= 0.0) {

        OPENFHE_THROW(
            std::string(functionName) +
            ": invalid MLWE scaling factor");
    }

    const auto& reference = ciphertext.a[0];

    if (RingDimension(reference) !=
        ciphertext.ringDimension) {

        OPENFHE_THROW(
            std::string(functionName) +
            ": stored MLWE ring dimension is inconsistent");
    }

    for (const auto& component : ciphertext.a) {
        ValidateCompatible(
            reference,
            component,
            functionName);
    }

    ValidateCompatible(
        reference,
        ciphertext.b,
        functionName);
}

void ValidateMLWECiphertextBundle(
    const MLWECiphertextBundle& bundle,
    const char* functionName) {

    if (bundle.rank == 0) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": bundle rank is zero");
    }

    if (bundle.ciphertexts.size() != bundle.rank) {
        OPENFHE_THROW(
            std::string(functionName) +
            ": bundle must contain exactly rank MLWE ciphertexts");
    }

    ValidateRank(
        bundle.sourceRingDimension,
        bundle.rank,
        functionName);

    for (const auto& ciphertext : bundle.ciphertexts) {
        ValidateMLWECiphertext(
            ciphertext,
            functionName);

        if (ciphertext.rank != bundle.rank) {
            OPENFHE_THROW(
                std::string(functionName) +
                ": ciphertext rank does not match bundle rank");
        }

        if (ciphertext.ringDimension *
                bundle.rank !=
            bundle.sourceRingDimension) {

            OPENFHE_THROW(
                std::string(functionName) +
                ": small and source ring dimensions are inconsistent");
        }
    }

    const auto& reference =
        bundle.ciphertexts[0];

    for (const auto& ciphertext : bundle.ciphertexts) {
        if (ciphertext.scalingFactor !=
            reference.scalingFactor) {

            OPENFHE_THROW(
                std::string(functionName) +
                ": MLWE scaling factors do not match");
        }

        if (ciphertext.level != reference.level) {
            OPENFHE_THROW(
                std::string(functionName) +
                ": MLWE levels do not match");
        }

        if (ciphertext.keyTag != reference.keyTag) {
            OPENFHE_THROW(
                std::string(functionName) +
                ": MLWE key tags do not match");
        }

        for (const auto& component : ciphertext.a) {
            ValidateCompatible(
                reference.a[0],
                component,
                functionName);
        }

        ValidateCompatible(
            reference.a[0],
            ciphertext.b,
            functionName);
    }
}

}  // namespace

std::vector<lbcrypto::DCRTPoly> Split(
    const lbcrypto::DCRTPoly& polynomial,
    size_t rank) {

    lbcrypto::DCRTPoly coefficientPolynomial =
        ToCoefficient(polynomial);

    const size_t largeRingDimension =
        RingDimension(coefficientPolynomial);

    ValidateRank(
        largeRingDimension,
        rank,
        "Split");

    const size_t smallRingDimension =
        largeRingDimension / rank;

    const auto smallParams =
        MakeDCRTParams(
            coefficientPolynomial,
            smallRingDimension);

    std::vector<lbcrypto::DCRTPoly> result;
    result.reserve(rank);

    for (size_t part = 0;
         part < rank;
         ++part) {

        result.emplace_back(
            smallParams,
            Format::COEFFICIENT,
            true);
    }

    const auto& inputTowers =
        coefficientPolynomial.GetAllElements();

    for (size_t tower = 0;
         tower < inputTowers.size();
         ++tower) {

        const auto modulus =
            inputTowers[tower].GetModulus();

        const auto& inputValues =
            inputTowers[tower].GetValues();

        for (size_t part = 0;
             part < rank;
             ++part) {

            lbcrypto::NativeVector values(
                smallRingDimension,
                modulus);

            for (size_t index = 0;
                 index < smallRingDimension;
                 ++index) {

                values[index] =
                    inputValues[index * rank + part];
            }

            result[part]
                .GetAllElements()[tower]
                .SetValues(
                    std::move(values),
                    Format::COEFFICIENT);
        }
    }

    return result;
}

lbcrypto::DCRTPoly Combine(
    const std::vector<lbcrypto::DCRTPoly>& polynomials) {

    if (polynomials.empty()) {
        OPENFHE_THROW(
            "Combine: input polynomial list is empty");
    }

    const size_t rank = polynomials.size();

    const size_t smallRingDimension =
        RingDimension(polynomials[0]);

    ValidateRank(
        smallRingDimension * rank,
        rank,
        "Combine");

    std::vector<lbcrypto::DCRTPoly>
        coefficientPolynomials;

    coefficientPolynomials.reserve(rank);

    for (const auto& polynomial : polynomials) {
        ValidateCompatible(
            polynomials[0],
            polynomial,
            "Combine");

        coefficientPolynomials.emplace_back(
            ToCoefficient(polynomial));
    }

    const size_t largeRingDimension =
        smallRingDimension * rank;

    const auto largeParams =
        MakeDCRTParams(
            coefficientPolynomials[0],
            largeRingDimension);

    lbcrypto::DCRTPoly result(
        largeParams,
        Format::COEFFICIENT,
        true);

    auto& resultTowers =
        result.GetAllElements();

    const size_t towerCount =
        coefficientPolynomials[0]
            .GetAllElements()
            .size();

    for (size_t tower = 0;
         tower < towerCount;
         ++tower) {

        const auto modulus =
            coefficientPolynomials[0]
                .GetAllElements()[tower]
                .GetModulus();

        lbcrypto::NativeVector values(
            largeRingDimension,
            modulus);

        for (size_t index = 0;
             index < largeRingDimension;
             ++index) {

            const size_t part =
                index % rank;

            const size_t smallIndex =
                index / rank;

            values[index] =
                coefficientPolynomials[part]
                    .GetAllElements()[tower]
                    .GetValues()[smallIndex];
        }

        resultTowers[tower].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }

    return result;
}

lbcrypto::DCRTPoly MultX(
    const lbcrypto::DCRTPoly& polynomial) {

    lbcrypto::DCRTPoly coefficientPolynomial =
        ToCoefficient(polynomial);

    const size_t ringDimension =
        RingDimension(coefficientPolynomial);

    if (ringDimension == 0) {
        OPENFHE_THROW(
            "MultX: ring dimension is zero");
    }

    lbcrypto::DCRTPoly result(
        coefficientPolynomial.GetParams(),
        Format::COEFFICIENT,
        true);

    const auto& inputTowers =
        coefficientPolynomial.GetAllElements();

    auto& resultTowers =
        result.GetAllElements();

    for (size_t tower = 0;
         tower < inputTowers.size();
         ++tower) {

        const auto modulus =
            inputTowers[tower].GetModulus();

        const auto& inputValues =
            inputTowers[tower].GetValues();

        lbcrypto::NativeVector values(
            ringDimension,
            modulus);

        const auto last =
            inputValues[ringDimension - 1];

        if (last == lbcrypto::NativeInteger(0)) {
            values[0] =
                lbcrypto::NativeInteger(0);
        }
        else {
            values[0] =
                modulus - last;
        }

        for (size_t index = 1;
             index < ringDimension;
             ++index) {

            values[index] =
                inputValues[index - 1];
        }

        resultTowers[tower].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }

    return result;
}

lbcrypto::DCRTPoly RingEmbedding(
    const lbcrypto::DCRTPoly& polynomial,
    size_t rank) {

    lbcrypto::DCRTPoly coefficientPolynomial =
        ToCoefficient(polynomial);

    const size_t smallRingDimension =
        RingDimension(coefficientPolynomial);

    const size_t largeRingDimension =
        smallRingDimension * rank;

    ValidateRank(
        largeRingDimension,
        rank,
        "RingEmbedding");

    const auto largeParams =
        MakeDCRTParams(
            coefficientPolynomial,
            largeRingDimension);

    lbcrypto::DCRTPoly result(
        largeParams,
        Format::COEFFICIENT,
        true);

    const auto& inputTowers =
        coefficientPolynomial.GetAllElements();

    auto& resultTowers =
        result.GetAllElements();

    for (size_t tower = 0;
         tower < inputTowers.size();
         ++tower) {

        const auto modulus =
            inputTowers[tower].GetModulus();

        const auto& inputValues =
            inputTowers[tower].GetValues();

        lbcrypto::NativeVector values(
            largeRingDimension,
            modulus);

        for (size_t index = 0;
             index < largeRingDimension;
             ++index) {

            if (index % rank == 0) {
                values[index] =
                    inputValues[index / rank];
            }
            else {
                values[index] =
                    lbcrypto::NativeInteger(0);
            }
        }

        resultTowers[tower].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }

    return result;
}

MLWECiphertextBundle RLWEtoMLWE(
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& ciphertext,
    size_t rank) {

    if (ciphertext == nullptr) {
        OPENFHE_THROW(
            "RLWEtoMLWE: ciphertext is null");
    }

    if (ciphertext->GetElements().size() != 2) {
        OPENFHE_THROW(
            "RLWEtoMLWE: ciphertext must have exactly two components");
    }

    const auto& c0 =
        ciphertext->GetElements()[0];

    const auto& c1 =
        ciphertext->GetElements()[1];

    ValidateCompatible(
        c0,
        c1,
        "RLWEtoMLWE");

    const size_t sourceRingDimension =
        RingDimension(c0);

    ValidateRank(
        sourceRingDimension,
        rank,
        "RLWEtoMLWE");

    const auto c0Split =
        Split(c0, rank);

    const auto c1Split =
        Split(c1, rank);

    const size_t smallRingDimension =
        sourceRingDimension / rank;

    MLWECiphertextBundle bundle;

    bundle.rank = rank;
    bundle.sourceRingDimension =
        sourceRingDimension;

    bundle.ciphertexts.reserve(rank);

    for (size_t row = 0;
         row < rank;
         ++row) {

        MLWECiphertext mlwe;

        mlwe.rank = rank;
        mlwe.ringDimension =
            smallRingDimension;

        mlwe.scalingFactor =
            ciphertext->GetScalingFactor();

        mlwe.level =
            ciphertext->GetLevel();

        mlwe.keyTag =
            ciphertext->GetKeyTag();

        mlwe.a.resize(rank);

        for (size_t column = 0;
             column < rank;
             ++column) {

            const size_t splitIndex =
                (row + rank - column) % rank;

            mlwe.a[column] =
                c1Split[splitIndex];

            // Multiply by X on wrap-around in the smaller negacyclic ring.
            if (row < column) {
                mlwe.a[column] =
                    MultX(mlwe.a[column]);
            }
        }

        mlwe.b = c0Split[row];

        bundle.ciphertexts.emplace_back(
            std::move(mlwe));
    }

    ValidateMLWECiphertextBundle(
        bundle,
        "RLWEtoMLWE");

    return bundle;
}

std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
GenerateMLWEtoRLWEKeys(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PrivateKey<
        lbcrypto::DCRTPoly>& secretKey,
    size_t rank) {

    if (cc == nullptr) {
        OPENFHE_THROW(
            "GenerateMLWEtoRLWEKeys: crypto context is null");
    }

    if (secretKey == nullptr) {
        OPENFHE_THROW(
            "GenerateMLWEtoRLWEKeys: secret key is null");
    }

    lbcrypto::DCRTPoly secret =
        secretKey->GetPrivateElement();

    const size_t ringDimension =
        RingDimension(secret);

    ValidateRank(
        ringDimension,
        rank,
        "GenerateMLWEtoRLWEKeys");

    const auto splitSecret =
        Split(secret, rank);

    std::vector<
        lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
        conversionKeys;

    conversionKeys.reserve(rank);

    const auto scheme =
        cc->GetScheme();

    for (size_t index = 0;
         index < rank;
         ++index) {

        lbcrypto::DCRTPoly embeddedSecret =
            RingEmbedding(
                splitSecret[index],
                rank);

        embeddedSecret.SetFormat(
            Format::EVALUATION);

        auto embeddedSecretKey =
            std::make_shared<
                lbcrypto::PrivateKeyImpl<
                    lbcrypto::DCRTPoly>>(cc);

        embeddedSecretKey->SetPrivateElement(
            embeddedSecret);

        conversionKeys.emplace_back(
            scheme->KeySwitchGen(
                embeddedSecretKey,
                secretKey));
    }

    return conversionKeys;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
MLWEtoRLWE(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const MLWECiphertextBundle& bundle,
    const std::vector<
        lbcrypto::EvalKey<
            lbcrypto::DCRTPoly>>& conversionKeys,
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& referenceCiphertext) {

    if (cc == nullptr) {
        OPENFHE_THROW(
            "MLWEtoRLWE: crypto context is null");
    }

    if (referenceCiphertext == nullptr) {
        OPENFHE_THROW(
            "MLWEtoRLWE: reference ciphertext is null");
    }

    if (referenceCiphertext->GetElements().size() != 2) {
        OPENFHE_THROW(
            "MLWEtoRLWE: reference ciphertext must have two components");
    }

    ValidateMLWECiphertextBundle(
        bundle,
        "MLWEtoRLWE");

    const size_t rank =
        bundle.rank;

    if (conversionKeys.size() != rank) {
        OPENFHE_THROW(
            "MLWEtoRLWE: number of conversion keys must equal rank");
    }

    const auto& metadata =
        bundle.ciphertexts[0];

    if (RingDimension(
            referenceCiphertext->GetElements()[0]) !=
        bundle.sourceRingDimension) {

        OPENFHE_THROW(
            "MLWEtoRLWE: reference ciphertext ring dimension mismatch");
    }

    if (referenceCiphertext->GetScalingFactor() !=
        metadata.scalingFactor) {

        OPENFHE_THROW(
            "MLWEtoRLWE: reference ciphertext scaling factor mismatch");
    }

    if (referenceCiphertext->GetLevel() !=
        metadata.level) {

        OPENFHE_THROW(
            "MLWEtoRLWE: reference ciphertext level mismatch");
    }

    if (referenceCiphertext->GetKeyTag() !=
        metadata.keyTag) {

        OPENFHE_THROW(
            "MLWEtoRLWE: reference ciphertext key tag mismatch");
    }

    // Combine matching a columns of the MLWE ciphertexts.
    std::vector<lbcrypto::DCRTPoly>
        combinedA;

    combinedA.reserve(rank);

    for (size_t column = 0;
         column < rank;
         ++column) {

        std::vector<lbcrypto::DCRTPoly>
            columnComponents;

        columnComponents.reserve(rank);

        for (size_t row = 0;
             row < rank;
             ++row) {

            columnComponents.emplace_back(
                bundle.ciphertexts[row].a[column]);
        }

        combinedA.emplace_back(
            Combine(columnComponents));
    }

    // Combine all b components.
    std::vector<lbcrypto::DCRTPoly>
        bComponents;

    bComponents.reserve(rank);

    for (size_t row = 0;
         row < rank;
         ++row) {

        bComponents.emplace_back(
            bundle.ciphertexts[row].b);
    }

    lbcrypto::DCRTPoly combinedB =
        Combine(bComponents);

    for (auto& component : combinedA) {
        component.SetFormat(
            Format::EVALUATION);
    }

    combinedB.SetFormat(
        Format::EVALUATION);

    const auto scheme =
        cc->GetScheme();

    auto firstSwitched =
        scheme->KeySwitchCore(
            combinedA[0],
            conversionKeys[0]);

    if (firstSwitched == nullptr ||
        firstSwitched->size() != 2) {

        OPENFHE_THROW(
            "MLWEtoRLWE: KeySwitchCore returned an invalid result");
    }

    lbcrypto::DCRTPoly outputC0 =
        (*firstSwitched)[0];

    lbcrypto::DCRTPoly outputC1 =
        (*firstSwitched)[1];

    for (size_t index = 1;
         index < rank;
         ++index) {

        auto switched =
            scheme->KeySwitchCore(
                combinedA[index],
                conversionKeys[index]);

        if (switched == nullptr ||
            switched->size() != 2) {

            OPENFHE_THROW(
                "MLWEtoRLWE: KeySwitchCore returned an invalid result");
        }

        outputC0 +=
            (*switched)[0];

        outputC1 +=
            (*switched)[1];
    }

    outputC0 += combinedB;

    auto result =
        std::make_shared<
            lbcrypto::CiphertextImpl<
                lbcrypto::DCRTPoly>>(
            referenceCiphertext);

    result->SetElements(
        std::vector<lbcrypto::DCRTPoly>{
            std::move(outputC0),
            std::move(outputC1)});

    return result;
}

}  // namespace ring_conversion