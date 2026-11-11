#ifndef FEDERATED1_SURVIVOR_KEYS_H
#define FEDERATED1_SURVIVOR_KEYS_H

#include "openfhe.h"
#include "protocol-setup.h"
#include "protocol-types.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace federated {

struct SurvivorPublicKeyMaterial {
    SurvivorSet survivors;
    lbcrypto::DCRTPoly p0;
    lbcrypto::DCRTPoly p1;
};

struct SurvivorM2RKeyMaterial {
    SurvivorSet survivors;
    std::size_t mlweRank = 0;
    std::size_t decompositionDigitCount = 0;

    /* [splitIndex][lambda] */
    std::vector<std::vector<lbcrypto::DCRTPoly>> a;
    std::vector<std::vector<lbcrypto::DCRTPoly>> b;
};

struct ClientToCollectiveKeyMaterial {
    SetupId setupId;
    ClientId sourceClientId;
    SurvivorSet survivors;
    std::size_t decompositionDigitCount = 0;

    /* HYBRID switching-key pair (b,a) for each digit lambda. */
    std::vector<lbcrypto::DCRTPoly> b;
    std::vector<lbcrypto::DCRTPoly> a;
};

struct SurvivorKeySet {
    SetupId setupId;
    SurvivorPublicKeyMaterial publicKey;
    std::optional<SurvivorM2RKeyMaterial> m2rKey;
    std::vector<ClientToCollectiveKeyMaterial> clientToCollectiveKeys;
    std::vector<double> clientToCollectiveKeyGenerationSeconds;
    double serverTargetReconstructionSeconds = 0.0;
};

/* Sum cached setup p0_i and build x_i -> s_S KSKs for the target. Only Sparse sums M2R b-shares; FULL has no M2R key. */
SurvivorKeySet AggregateSurvivorKeys(
    const ProtocolSetupResult& setup,
    const SurvivorSet& survivors);

}  // namespace federated

#endif  // FEDERATED1_SURVIVOR_KEYS_H
