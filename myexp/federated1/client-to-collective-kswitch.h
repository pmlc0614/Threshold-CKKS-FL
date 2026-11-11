#ifndef FEDERATED1_CLIENT_TO_COLLECTIVE_KSWITCH_H
#define FEDERATED1_CLIENT_TO_COLLECTIVE_KSWITCH_H

#include "protocol-setup.h"
#include "seeded-dense-rlwe.h"
#include "seeded-rlwe.h"
#include "survivor-keys.h"

#include "openfhe.h"

#include <cstddef>
#include <vector>

namespace federated {

struct CollectiveRLWECiphertext {
    SetupId setupId;
    RoundId roundId;
    ClientId sourceClientId;
    std::size_t packGroupIndex = 0;
    std::size_t activeStreamCount = 0;
    SurvivorSet survivors;
    std::vector<std::size_t> meaningfulCoefficientIndices;
    lbcrypto::DCRTPoly b;
    lbcrypto::DCRTPoly a;
};

/* Calibration-only QP digit sums immediately before one ModDown per logical
 * KSwitch source. They are never serialized or used by the online protocol. */
struct HybridModDownTrace {
    lbcrypto::DCRTPoly bQP;
    lbcrypto::DCRTPoly aQP;
};

const ClientToCollectiveKeyMaterial& FindClientToCollectiveKey(
    const SurvivorKeySet& survivorKeys,
    ClientId sourceClientId);

/* Lift a digit to QP. stochastic selects the exact zero-mean signed lift. */
lbcrypto::DCRTPoly ApproxModUpQDigitToQP(
    lbcrypto::DCRTPoly valueQ,
    std::size_t digitIndex,
    const ProtocolRnsParameters& parameters,
    bool stochastic = false);

/* Sourcewise ModDown; stochastic uses fresh independent remainder coins. */
lbcrypto::DCRTPoly ApproxModDownSinglePToQ(
    lbcrypto::DCRTPoly valueQP,
    const ProtocolRnsParameters& parameters,
    bool centered = false,
    bool stochastic = false);

/* Exact QP representation of P*x for a Q-basis polynomial x. */
lbcrypto::DCRTPoly LiftQPolynomialTimesPToQP(
    lbcrypto::DCRTPoly valueQ,
    const ProtocolRnsParameters& parameters);

/* Calibration-only decrypted QP phase immediately before ModDown. */
lbcrypto::DCRTPoly BuildPreModDownPhaseQP(
    lbcrypto::DCRTPoly sourceB,
    const HybridModDownTrace& trace,
    lbcrypto::DCRTPoly collectiveSecretQP,
    const ProtocolRnsParameters& parameters);

/* Switch (b,a) from x_i to the survivor collective secret s_S. Add the original sparse b to the switched a contribution. */
CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredSparseRLWECiphertext& ciphertext,
    const ClientToCollectiveKeyMaterial& switchingKey,
    const CommonSetupMaterial& publicMaterial,
    HybridModDownTrace* modDownTrace = nullptr);

CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredSparseRLWECiphertext& ciphertext,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial);

/* Fixed-order direct-RLWE blocks have no sparse routing metadata. */
CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredDenseRLWECiphertext& ciphertext,
    const ClientToCollectiveKeyMaterial& switchingKey,
    const CommonSetupMaterial& publicMaterial,
    HybridModDownTrace* modDownTrace = nullptr);

CollectiveRLWECiphertext KeySwitchClientToCollective(
    const RestoredDenseRLWECiphertext& ciphertext,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial);

}  // namespace federated

#endif  // FEDERATED1_CLIENT_TO_COLLECTIVE_KSWITCH_H
