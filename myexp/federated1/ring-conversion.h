#ifndef FEDERATED1_RING_CONVERSION_H
#define FEDERATED1_RING_CONVERSION_H

#include "openfhe.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ring_conversion {

/* MLWE over R_(q,n), with phase b + sum_j a[j]*s[j]; a.size() must equal rank k. */
struct MLWECiphertext {
    size_t rank = 0;
    size_t ringDimension = 0;

    // Metadata for compatibility checks during aggregation.
    double scalingFactor = 1.0;
    size_t level = 0;
    std::string keyTag;

    std::vector<lbcrypto::DCRTPoly> a;
    lbcrypto::DCRTPoly b;
};

/* The k MLWE ciphertexts obtained from one RLWE ciphertext; ciphertexts[i] encrypts Split_i(m). */
struct MLWECiphertextBundle {
    size_t rank = 0;
    size_t sourceRingDimension = 0;

    std::vector<MLWECiphertext> ciphertexts;
};

/* Split R_(q,N) into rank smaller rings: result[j][t] = polynomial[t*rank+j]. */
std::vector<lbcrypto::DCRTPoly> Split(
    const lbcrypto::DCRTPoly& polynomial,
    size_t rank);

/* Combine smaller rings: result[t*rank+j] = polynomials[j][t]. */
lbcrypto::DCRTPoly Combine(
    const std::vector<lbcrypto::DCRTPoly>& polynomials);

/* Multiply by X in the negacyclic small ring: (a0,...,a[n-1]) becomes (-a[n-1],a0,...,a[n-2]). */
lbcrypto::DCRTPoly MultX(
    const lbcrypto::DCRTPoly& polynomial);

/* Embed a small-ring polynomial via a(X) -> a(X^rank). */
lbcrypto::DCRTPoly RingEmbedding(
    const lbcrypto::DCRTPoly& polynomial,
    size_t rank);

/* Convert one RLWE ciphertext into rank MLWE ciphertexts. */
MLWECiphertextBundle RLWEtoMLWE(
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ciphertext,
    size_t rank);

/* Baseline helper: generate keys from Embed(Split_j(s)) to s for each j; not the independent-t-domain v17 key setup. */
std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
GenerateMLWEtoRLWEKeys(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& secretKey,
    size_t rank);

/* Baseline M2R to the original secret; copy OpenFHE metadata from referenceCiphertext. */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> MLWEtoRLWE(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const MLWECiphertextBundle& bundle,
    const std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>&
        conversionKeys,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&
        referenceCiphertext);

}  // namespace ring_conversion

#endif  // FEDERATED1_RING_CONVERSION_H