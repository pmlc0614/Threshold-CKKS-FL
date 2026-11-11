#ifndef FEDERATED1_MODULUS_DIAGNOSTICS_H
#define FEDERATED1_MODULUS_DIAGNOSTICS_H

#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-cryptoparameters.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace modulus_diagnostics {

struct ModulusInformation {
    std::vector<uint32_t> qTowerBits;
    std::vector<uint32_t> pTowerBits;

    uint32_t qBits = 0;
    uint32_t pBits = 0;
    uint32_t qpBits = 0;
};

inline ModulusInformation Inspect(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc) {

    if (!cc) {
        throw std::invalid_argument(
            "Inspect: CryptoContext is null");
    }

    const auto cryptoParameters =
        std::dynamic_pointer_cast<
            lbcrypto::CryptoParametersCKKSRNS>(
                cc->GetCryptoParameters());

    if (!cryptoParameters) {
        throw std::runtime_error(
            "Inspect: context is not CKKSRNS");
    }

    const auto paramsQ =
        cryptoParameters->GetElementParams();

    const auto paramsP =
        cryptoParameters->GetParamsP();

    const auto paramsQP =
        cryptoParameters->GetParamsQP();

    if (!paramsQ) {
        throw std::runtime_error(
            "Inspect: Q parameters are null");
    }

    if (!paramsP) {
        throw std::runtime_error(
            "Inspect: P parameters are null; "
            "HYBRID key switching may not be enabled");
    }

    if (!paramsQP) {
        throw std::runtime_error(
            "Inspect: QP parameters are null");
    }

    ModulusInformation information;

    for (const auto& tower :
         paramsQ->GetParams()) {

        information.qTowerBits.push_back(
            tower->GetModulus().GetMSB());
    }

    for (const auto& tower :
         paramsP->GetParams()) {

        information.pTowerBits.push_back(
            tower->GetModulus().GetMSB());
    }

    information.qBits =
        paramsQ->GetModulus().GetMSB();

    information.pBits =
        paramsP->GetModulus().GetMSB();

    information.qpBits =
        paramsQP->GetModulus().GetMSB();

    return information;
}

inline void Print(
    const ModulusInformation& information,
    std::ostream& output = std::cout) {

    output
        << "number of Q towers: "
        << information.qTowerBits.size()
        << '\n';

    for (std::size_t index = 0;
         index < information.qTowerBits.size();
         ++index) {

        output
            << "  q[" << index << "] bits: "
            << information.qTowerBits[index]
            << '\n';
    }

    output
        << "number of P towers: "
        << information.pTowerBits.size()
        << '\n';

    for (std::size_t index = 0;
         index < information.pTowerBits.size();
         ++index) {

        output
            << "  p[" << index << "] bits: "
            << information.pTowerBits[index]
            << '\n';
    }

    output
        << "Q bit length:  "
        << information.qBits
        << '\n'
        << "P bit length:  "
        << information.pBits
        << '\n'
        << "QP bit length: "
        << information.qpBits
        << '\n'
        << "N=8192 ternary 128-bit limit: 218\n";
}

inline void ValidateTwoModulusProfile(
    const ModulusInformation& information) {

    constexpr uint32_t maximumQPBits = 218;

    if (information.qTowerBits.size() != 2) {
        throw std::runtime_error(
            "expected exactly two Q towers");
    }

    if (information.pTowerBits.size() != 1) {
        throw std::runtime_error(
            "expected exactly one special-modulus "
            "P tower");
    }

    if (information.qpBits >
        maximumQPBits) {

        throw std::runtime_error(
            "QP exceeds the N=8192 ternary "
            "128-bit security bound");
    }
}

}  // namespace modulus_diagnostics

#endif  // FEDERATED1_MODULUS_DIAGNOSTICS_H