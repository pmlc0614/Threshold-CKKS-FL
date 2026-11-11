#ifndef FEDERATED1_COLLECTIVE_CONVERSION_KEYGEN_H
#define FEDERATED1_COLLECTIVE_CONVERSION_KEYGEN_H

#include "openfhe.h"

#include <cstddef>
#include <vector>

namespace collective_conversion_keygen {

using ConversionEvalKey =
    lbcrypto::EvalKey<lbcrypto::DCRTPoly>;

/* Test-baseline conversion-key shares: keys[j] switches Embed(Split_j(s_i)) to s_i. Each HYBRID key contains A/B vectors per decomposition partition. This is not the v17 independent-t-domain M2R key. */
struct MLWEtoRLWEKeyShare {
    size_t partyIndex = 0;
    size_t rank = 0;

    std::vector<ConversionEvalKey> keys;
};

/* Generate the lead share and its fresh common AVector template for the remaining parties. */
MLWEtoRLWEKeyShare
GenerateLeadMLWEtoRLWEKeyShare(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PrivateKey<
        lbcrypto::DCRTPoly>& secretKeyShare,
    size_t rank);

/* Generate another party's share with MultiKeySwitchGen using the common AVector; partyIndex must be positive. */
MLWEtoRLWEKeyShare
GenerateMainMLWEtoRLWEKeyShare(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PrivateKey<
        lbcrypto::DCRTPoly>& secretKeyShare,
    const MLWEtoRLWEKeyShare&
        commonKeyTemplate,
    size_t partyIndex);

/* Combine BVector shares with MultiAddEvalKeys, retain one common AVector, and use the collective public-key tag. */
std::vector<ConversionEvalKey>
AggregateMLWEtoRLWEKeyShares(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        MLWEtoRLWEKeyShare>& partyKeyShares,
    const lbcrypto::PublicKey<
        lbcrypto::DCRTPoly>& collectivePublicKey);

/* Single-process test helper: generate and combine lead/main shares without reconstructing s = sum_i s_i. */
std::vector<ConversionEvalKey>
GenerateCollectiveMLWEtoRLWEKeys(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>&
        secretKeyShares,
    const lbcrypto::PublicKey<
        lbcrypto::DCRTPoly>& collectivePublicKey,
    size_t rank);

}  // namespace collective_conversion_keygen

#endif  // FEDERATED1_COLLECTIVE_CONVERSION_KEYGEN_H