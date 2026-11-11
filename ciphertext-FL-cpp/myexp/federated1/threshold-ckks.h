#ifndef FEDERATED1_THRESHOLD_H
#define FEDERATED1_THRESHOLD_H

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace threshold_ckks {

/* Default per-party noise for the OpenFHE baseline helper, 2^20. Production v17 uses the selected profile's smudging sigma. */
constexpr double kDefaultSigmaShare =
    1048576.0;

/* Configure the OpenFHE baseline's per-party smudging sigma before key generation/decryption. This does not increase Q; choose sufficient modulus and scale separately. */
void ConfigureSmudgingSigma(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    double sigmaShare =
        kDefaultSigmaShare);

/* Return the configured OpenFHE baseline smudging sigma. */
double GetSmudgingSigma(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc);

/* Single-process test key set: collectivePublicKey corresponds to sum_i s_i. Secret shares are co-located only in this test harness; distributed clients must retain their own shares. */
struct CollectiveKeySet {
    lbcrypto::PublicKey<lbcrypto::DCRTPoly>
        collectivePublicKey;

    std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>
        secretKeyShares;
};

/* Generate partyCount >= 2 shares and their collective public key through sequential OpenFHE multiparty key generation. */
CollectiveKeySet GenerateCollectiveKeys(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    size_t partyCount);

/* Coefficient-encode and encrypt under the collective public key. */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
EncryptCoefficients(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::PublicKey<
        lbcrypto::DCRTPoly>& collectivePublicKey,
    const std::vector<double>& values,
    uint32_t scaleBits);

/* Add ciphertexts sharing a context, level and scale. */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
AggregateCiphertexts(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>&
        ciphertexts);

/* Produce one partial per party; only secretKeyShares[0] produces a lead share, and all others produce main shares. */
std::vector<
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>
CreatePartialDecryptions(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& ciphertext,
    const std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>&
        secretKeyShares);

/* Fuse partials to a Poly, bypassing CKKS slot decoding so the coefficient codec can decode the result. */
lbcrypto::Poly FusePartialDecryptions(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>&
        partialDecryptions);

/* Combine partial shares and coefficient-decode the entire ring. */
std::vector<double> FuseAndDecodeCoefficients(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const std::vector<
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>&
        partialDecryptions,
    double scalingFactor);

/* Convenience wrapper for partial decryption, fusion and coefficient decoding. */
std::vector<double> ThresholdDecryptCoefficients(
    const lbcrypto::CryptoContext<
        lbcrypto::DCRTPoly>& cc,
    const lbcrypto::Ciphertext<
        lbcrypto::DCRTPoly>& ciphertext,
    const std::vector<
        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>&
        secretKeyShares,
    double scalingFactor);

}  // namespace threshold_ckks

#endif  // FEDERATED1_THRESHOLD_H