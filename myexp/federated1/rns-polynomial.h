#ifndef FEDERATED1_RNS_POLYNOMIAL_H
#define FEDERATED1_RNS_POLYNOMIAL_H

#include "openfhe.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace federated {
namespace rns_polynomial {

void ValidateBasis(
    const lbcrypto::DCRTPoly& polynomial,
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::string& name);

std::vector<lbcrypto::BigInteger> InterpolateCoefficients(
    lbcrypto::DCRTPoly polynomial);

std::vector<lbcrypto::BigInteger> InterpolateCoefficientsAtIndices(
    lbcrypto::DCRTPoly polynomial,
    const std::vector<std::size_t>& coefficientIndices);

lbcrypto::DCRTPoly FromCompositeCoefficients(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<lbcrypto::BigInteger>& coefficients);

lbcrypto::DCRTPoly FromCompositeCoefficientsAtIndices(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const std::vector<lbcrypto::BigInteger>& coefficients,
    const std::vector<std::size_t>& coefficientIndices);

}  // namespace rns_polynomial
}  // namespace federated

#endif  // FEDERATED1_RNS_POLYNOMIAL_H
