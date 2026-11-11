#include "rns-polynomial.h"

#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace federated {
namespace rns_polynomial {
namespace {

std::size_t ValidateParameters(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters) {

    if (!parameters || parameters->GetParams().empty()) {
        throw std::invalid_argument("DCRT parameters are null or empty");
    }

    const std::size_t ringDimension = parameters->GetRingDimension();
    if (ringDimension == 0) {
        throw std::invalid_argument("DCRT ring dimension is zero");
    }
    return ringDimension;
}

void ValidateIndices(
    const std::vector<std::size_t>& indices,
    std::size_t ringDimension) {

    std::unordered_set<std::size_t> seen;
    seen.reserve(indices.size());

    for (const std::size_t index : indices) {
        if (index >= ringDimension) {
            throw std::out_of_range("coefficient index exceeds ring dimension");
        }
        if (!seen.insert(index).second) {
            throw std::invalid_argument("coefficient indices contain duplicates");
        }
    }
}

}  // namespace

void ValidateBasis(
    const lbcrypto::DCRTPoly& polynomial,
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::string& name) {

    const std::size_t ringDimension = ValidateParameters(parameters);
    const auto& towers = polynomial.GetAllElements();
    const auto& expected = parameters->GetParams();

    if (towers.size() != expected.size()) {
        throw std::invalid_argument(name + " has an invalid tower count");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < towers.size();
         ++towerIndex) {

        if (towers[towerIndex].GetModulus() !=
                expected[towerIndex]->GetModulus() ||
            towers[towerIndex].GetValues().GetLength() != ringDimension) {

            throw std::invalid_argument(
                name + " does not use the expected DCRT basis");
        }
    }
}

std::vector<lbcrypto::BigInteger> InterpolateCoefficients(
    lbcrypto::DCRTPoly polynomial) {

    ValidateBasis(polynomial, polynomial.GetParams(), "DCRT polynomial");
    polynomial.SetFormat(Format::COEFFICIENT);
    lbcrypto::Poly interpolated = polynomial.CRTInterpolate();
    interpolated.SetFormat(Format::COEFFICIENT);

    const auto& values = interpolated.GetValues();
    std::vector<lbcrypto::BigInteger> coefficients;
    coefficients.reserve(values.GetLength());

    for (std::size_t index = 0;
         index < values.GetLength();
         ++index) {

        coefficients.push_back(values[index]);
    }

    return coefficients;
}

std::vector<lbcrypto::BigInteger> InterpolateCoefficientsAtIndices(
    lbcrypto::DCRTPoly polynomial,
    const std::vector<std::size_t>& coefficientIndices) {

    const std::size_t ringDimension =
        ValidateParameters(polynomial.GetParams());
    ValidateIndices(coefficientIndices, ringDimension);

    const auto all = InterpolateCoefficients(std::move(polynomial));
    std::vector<lbcrypto::BigInteger> selected;
    selected.reserve(coefficientIndices.size());

    for (const std::size_t index : coefficientIndices) {
        selected.push_back(all[index]);
    }
    return selected;
}

lbcrypto::DCRTPoly FromCompositeCoefficients(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<lbcrypto::BigInteger>& coefficients) {

    const std::size_t ringDimension = ValidateParameters(parameters);
    if (coefficients.size() != ringDimension) {
        throw std::invalid_argument(
            "composite coefficient count does not match ring dimension");
    }

    lbcrypto::DCRTPoly result(parameters, Format::COEFFICIENT, true);
    auto& towers = result.GetAllElements();

    for (std::size_t towerIndex = 0;
         towerIndex < towers.size();
         ++towerIndex) {

        const lbcrypto::NativeInteger modulus =
            parameters->GetParams()[towerIndex]->GetModulus();
        const lbcrypto::BigInteger bigModulus(modulus);
        lbcrypto::NativeVector values(ringDimension, modulus);

        for (std::size_t index = 0; index < ringDimension; ++index) {
            values[index] = lbcrypto::NativeInteger(
                coefficients[index].Mod(bigModulus).ConvertToInt());
        }

        towers[towerIndex].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }
    return result;
}

lbcrypto::DCRTPoly FromCompositeCoefficientsAtIndices(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<lbcrypto::BigInteger>& coefficients,
    const std::vector<std::size_t>& coefficientIndices) {

    const std::size_t ringDimension = ValidateParameters(parameters);
    ValidateIndices(coefficientIndices, ringDimension);

    if (coefficients.size() != coefficientIndices.size()) {
        throw std::invalid_argument(
            "coefficient payload and index counts differ");
    }

    std::vector<lbcrypto::BigInteger> expanded(
        ringDimension,
        lbcrypto::BigInteger(0));

    for (std::size_t offset = 0;
         offset < coefficientIndices.size();
         ++offset) {

        expanded[coefficientIndices[offset]] = coefficients[offset];
    }

    return FromCompositeCoefficients(parameters, expanded);
}

}  // namespace rns_polynomial
}  // namespace federated
