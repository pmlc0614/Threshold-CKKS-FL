#ifndef FEDERATED1_FEDERATED_ENCRYPTED_H
#define FEDERATED1_FEDERATED_ENCRYPTED_H

#include "federated-types.h"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace federated {

/* Result of the collective-key baseline aggregation used in functional tests. */
struct FederatedEncryptedResult {
    /* Threshold-decrypted values divided by weightDenominator and scattered into model coordinates; rejected coordinates are zero. */
    std::vector<double> aggregate;

    /* The fixed sparse plan used by the ciphertext path. */
    std::uint64_t planHash = 0;

    double scalingFactor = 0.0;

    /* Number of packed RLWE ciphertexts produced by clients. */
    std::size_t clientCiphertextCount = 0;

    /* Number of client/chunk contributions to accepted chunks. */
    std::size_t acceptedClientContributionCount = 0;

    /* Number of aggregated MLWE ciphertexts, normally acceptedChunkIds.size(). */
    std::size_t aggregatedMLWECiphertextCount = 0;

    /* Number of server RLWE ciphertexts converted from MLWE and threshold-decrypted. */
    std::size_t serverCiphertextCount = 0;
};

/* Functional test baseline: pack selected chunks, encrypt under a collective key, R2M, apply integer weights, aggregate admitted chunks, M2R, and threshold-decrypt. Reuses the supplied plan. Requires PKE, KEYSWITCH, LEVELEDSHE, MULTIPARTY and prior ConfigureSmudgingSigma. The production v17 path instead encrypts under x_i and switches to s_S. */
FederatedEncryptedResult
ComputeFederatedEncryptedAggregate(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PublicKey<
        lbcrypto::DCRTPoly>& collectivePublicKey,
    const std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>&
        secretKeyShares,
    const std::vector<
        lbcrypto::EvalKey<lbcrypto::DCRTPoly>>&
        collectiveConversionKeys,
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients,
    const SparseRoundPlan& plan);

}  // namespace federated

#endif  // FEDERATED1_FEDERATED_ENCRYPTED_H