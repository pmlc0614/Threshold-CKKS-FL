#ifndef FEDERATED1_PROTOCOL_CRYPTO_CONTEXT_H
#define FEDERATED1_PROTOCOL_CRYPTO_CONTEXT_H

#include "protocol-parameters.h"

#include "openfhe.h"

namespace federated {

/*
 * Build the OpenFHE CKKS context that backs the protocol Q basis.
 *
 * CCParams represents the first Q tower separately and uses one repeated
 * scaling-modulus size for every remaining tower.  The protocol profile is
 * validated against that constraint and the generated tower count/bit sizes
 * are checked before the context is returned.
 */
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> MakeProtocolCryptoContext(
    const ProtocolExperimentConfig& config,
    lbcrypto::SecurityLevel securityLevel = lbcrypto::HEStd_NotSet);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_CRYPTO_CONTEXT_H
