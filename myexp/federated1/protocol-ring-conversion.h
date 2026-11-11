#ifndef FEDERATED1_PROTOCOL_RING_CONVERSION_H
#define FEDERATED1_PROTOCOL_RING_CONVERSION_H

#include "client-to-collective-kswitch.h"
#include "protocol-sparse-round.h"
#include "survivor-keys.h"

#include "openfhe.h"

#include <cstddef>
#include <vector>

namespace federated {

struct LabeledMLWECiphertext {
    ShuffledChunkLabel label;
    std::size_t rank = 0;
    std::size_t ringDimension = 0;
    std::vector<lbcrypto::DCRTPoly> a;
    lbcrypto::DCRTPoly b;
};

struct PackedAggregateRLWECiphertext {
    SetupId setupId;
    RoundId roundId;
    SurvivorSet survivors;
    PackedShuffledLabels shuffledLabels;
    std::vector<std::size_t> meaningfulCoefficientIndices;
    lbcrypto::DCRTPoly b;
    lbcrypto::DCRTPoly a;
};

struct M2RModDownTrace {
    lbcrypto::DCRTPoly combinedB;
    HybridModDownTrace switching;
};

/* Exact key-free Split/monomial R2M; labels[h] identifies output h. */
std::vector<LabeledMLWECiphertext> ProtocolRLWEToMLWE(
    const CollectiveRLWECiphertext& ciphertext,
    const PackedShuffledLabels& labels,
    std::size_t mlweRank);

LabeledMLWECiphertext MultiplyMLWEByInteger(
    const LabeledMLWECiphertext& ciphertext,
    std::uint32_t scalar);

void AddMLWEInPlace(
    LabeledMLWECiphertext& target,
    const LabeledMLWECiphertext& source);

// One server packing group, with independent component output slots. Prepare,
// switch each column once, then finish after the executor barrier.
struct M2RComponentResult {
    lbcrypto::DCRTPoly b, a, bQP, aQP;
};
struct PreparedM2R {
    std::vector<LabeledMLWECiphertext> rows;
    lbcrypto::DCRTPoly combinedB;
    std::vector<M2RComponentResult> components;
    bool collectTrace = false;
};
PreparedM2R PrepareM2R(
    std::vector<LabeledMLWECiphertext> ciphertexts,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    bool collectTrace);
void SwitchPreparedM2RComponent(
    PreparedM2R& prepared, std::size_t column,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial);
PackedAggregateRLWECiphertext FinishM2R(
    PreparedM2R& prepared,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId, M2RModDownTrace* modDownTrace = nullptr);

/* Combine rows and use cached M2R keys to switch split s_S components into the independent t_S model domain. */
PackedAggregateRLWECiphertext ProtocolMLWEToRLWE(
    const std::vector<LabeledMLWECiphertext>& ciphertexts,
    const PackedShuffledLabels& labels,
    const SurvivorKeySet& survivorKeys,
    const CommonSetupMaterial& publicMaterial,
    RoundId roundId,
    M2RModDownTrace* modDownTrace = nullptr);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_RING_CONVERSION_H
