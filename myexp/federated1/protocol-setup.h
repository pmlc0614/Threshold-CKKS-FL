#ifndef FEDERATED1_PROTOCOL_SETUP_H
#define FEDERATED1_PROTOCOL_SETUP_H

#include "openfhe.h"
#include "protocol-parameters.h"
#include "protocol-types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace federated {

struct ProtocolRnsParameters {
    std::shared_ptr<lbcrypto::DCRTPoly::Params> paramsQ;
    std::shared_ptr<lbcrypto::DCRTPoly::Params> paramsP;
    std::shared_ptr<lbcrypto::DCRTPoly::Params> paramsQP;

    lbcrypto::BigInteger modulusP;
    std::vector<lbcrypto::NativeInteger> pModQ;

    std::size_t qTowerCount = 0;
    std::size_t pTowerCount = 0;
    std::size_t decompositionDigitCount = 0;
    std::size_t qTowersPerDigit = 0;
};

/*
 * HYBRID amplification:
 *   G_hyb = sqrt(N / (12 P^2) * sum_lambda(q_lambda^2)).
 * P is the special-modulus product and q_lambda are individual Q towers.
 */
long double ComputeHybridKeySwitchGain(
    const ProtocolRnsParameters& parameters);

struct ProtocolSetupDescriptor {
    ProtocolMode mode = ProtocolMode::SP;
    std::string protocolVersion;
    std::uint8_t supportCodec = 0;
    ByteVector canonicalEncoding;
};

struct CommonSetupMaterial {
    bool certifiedV17Gaussian = false;
    ProtocolSetupDescriptor descriptor;
    SetupId setupId;
    Seed128 crsSeed{};
    std::size_t mlweRank = 0;
    ProtocolRnsParameters rns;

    /* p1 is a QP polynomial sampled once per setup. */
    lbcrypto::DCRTPoly publicKeyP1;
};

struct SparseSetupMaterial {
    /* [splitIndex][lambda]; empty for rank-one SP direct-RLWE execution. */
    std::vector<std::vector<lbcrypto::DCRTPoly>> m2rCommonA;
};

struct ClientOnlySetupMaterial {
    SetupId setupId;
    Seed128 clientGroupIndexKey{};
};

struct ClientSecretSetupMaterial {
    ClientId clientId;
    // Collective/public-key and threshold-decryption domain s_i.
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> secretKeyQ;
    // Local encryption domain x_i; independently sampled in every mode.
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> encryptionKeyQ;
    lbcrypto::DCRTPoly secretKeyQP;
    // Independent Sparse M2R/threshold domain t_i; absent for FULL and k=1.
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> modelKeyQ;
    lbcrypto::DCRTPoly modelKeyQP;
};

struct CachedM2RKeyShare {
    ClientId clientId;

    /* [splitIndex][lambda] */
    std::vector<std::vector<lbcrypto::DCRTPoly>> b;
};

struct ClientCachedSetupShare {
    ClientId clientId;
    lbcrypto::DCRTPoly publicKeyP0Share;
    std::optional<CachedM2RKeyShare> m2rShare;
};

struct ProtocolSetupTiming {
    std::vector<double> clientSecretKeyGenerationSeconds;
    std::vector<double> clientCachedShareGenerationSeconds;
};

struct ProtocolSetupResult {
    ProtocolMode mode = ProtocolMode::SP;
    CommonSetupMaterial common;
    std::optional<SparseSetupMaterial> sparse;
    SetupNoiseParameters setupNoise;
    std::vector<ClientSecretSetupMaterial> clientSecrets;
    std::vector<ClientCachedSetupShare> cachedClientShares;
    ProtocolSetupTiming timing;
};

/* Derived and retained by clients; it is not part of server setup material. */
ClientOnlySetupMaterial GenerateClientOnlySetupMaterial(
    const ProtocolExperimentConfig& config,
    SetupId setupId);

ByteVector CanonicalSetupDescriptor(
    const ProtocolExperimentConfig& config,
    const ProtocolRnsParameters& parameters);

const SparseSetupMaterial& RequireSparseSetup(
    const ProtocolSetupResult& setup);

ProtocolRnsParameters CreateProtocolRnsParameters(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config);

lbcrypto::DCRTPoly GenerateSeededUniformDCRT(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    const Seed128& seed,
    const std::string& domain,
    const ByteVector& context = {});

std::vector<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>
GenerateClientSecretKeys(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    std::size_t clientCount);

/* Lift a UNIFORM_TERNARY Q secret to QP as the same small signed coefficients; do not copy arbitrary Q residues. */
lbcrypto::DCRTPoly LiftTernarySecretToQP(
    lbcrypto::DCRTPoly secretQ,
    const ProtocolRnsParameters& parameters);

/* Construct P*gamma_lambda*source in QP: P*source in the selected Q tower and zero in all other Q/P towers. */
lbcrypto::DCRTPoly BuildM2RGadgetTerm(
    lbcrypto::DCRTPoly sourceQ,
    std::size_t lambda,
    const ProtocolRnsParameters& parameters);

ProtocolSetupResult GenerateProtocolSetup(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config);

ProtocolSetupResult GenerateProtocolSetupFromSecretKeys(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>& secretKeys);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_SETUP_H
