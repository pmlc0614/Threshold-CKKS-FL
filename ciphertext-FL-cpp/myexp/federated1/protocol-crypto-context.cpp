#include "protocol-crypto-context.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace federated {

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> MakeProtocolCryptoContext(
    const ProtocolExperimentConfig& config,
    lbcrypto::SecurityLevel securityLevel) {

    ValidateProtocolParameters(config).ThrowIfInvalid();
    const auto& qBits = config.modulus.qModulusBits;
    if (qBits.size() - 1U >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {

        throw std::invalid_argument("too many Q towers for OpenFHE CKKS");
    }
    if (qBits.size() > 2U &&
        !std::all_of(
            qBits.begin() + 2,
            qBits.end(),
            [&qBits](std::uint32_t bits) { return bits == qBits[1]; })) {

        throw std::invalid_argument(
            "OpenFHE CCParams requires all Q towers after the first "
            "to use the same bit size");
    }

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(
        static_cast<std::uint32_t>(qBits.size() - 1U));
    parameters.SetFirstModSize(qBits.front());
    parameters.SetScalingModSize(
        qBits.size() > 1U ? qBits[1] - 1U : qBits.front() - 1U);
    parameters.SetBatchSize(config.modulus.ringDimension / 2U);
    parameters.SetRingDim(config.modulus.ringDimension);
    parameters.SetSecurityLevel(securityLevel);
    parameters.SetScalingTechnique(lbcrypto::FIXEDMANUAL);
    parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(lbcrypto::BV);

    auto context = lbcrypto::GenCryptoContext(parameters);
    const auto paramsQ =
        context->GetCryptoParameters()->GetElementParams();
    if (!paramsQ || paramsQ->GetParams().size() != qBits.size()) {
        throw std::runtime_error(
            "OpenFHE generated an unexpected Q tower count");
    }
    for (std::size_t tower = 0; tower < qBits.size(); ++tower) {
        if (paramsQ->GetParams()[tower]->GetModulus().GetMSB() !=
            qBits[tower]) {

            throw std::runtime_error(
                "OpenFHE generated an unexpected Q tower bit size");
        }
    }

    context->Enable(lbcrypto::PKE);
    context->Enable(lbcrypto::KEYSWITCH);
    return context;
}

}  // namespace federated
