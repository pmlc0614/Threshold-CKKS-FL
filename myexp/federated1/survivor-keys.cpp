#include "survivor-keys.h"

#include "protocol-randomness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace federated {
namespace {

SurvivorSet CanonicalizeSurvivors(
    const SurvivorSet& survivors,
    std::size_t registeredClientCount) {

    if (survivors.clients.empty()) {
        throw std::invalid_argument("survivor set must not be empty");
    }

    SurvivorSet canonical = survivors;
    std::sort(
        canonical.clients.begin(),
        canonical.clients.end());

    for (std::size_t index = 0;
         index < canonical.clients.size();
         ++index) {

        if (canonical.clients[index].value >= registeredClientCount) {
            throw std::out_of_range("survivor client ID is not registered");
        }

        if (index > 0 &&
            canonical.clients[index] == canonical.clients[index - 1]) {

            throw std::invalid_argument("survivor set contains duplicates");
        }
    }

    return canonical;
}

const ClientCachedSetupShare& FindCachedShare(
    const ProtocolSetupResult& setup,
    ClientId clientId) {

    const auto found = std::find_if(
        setup.cachedClientShares.begin(),
        setup.cachedClientShares.end(),
        [clientId](const ClientCachedSetupShare& share) {
            return share.clientId == clientId;
        });

    if (found == setup.cachedClientShares.end()) {
        throw std::runtime_error("registered client has no cached setup share");
    }

    return *found;
}

const ClientSecretSetupMaterial& FindClientSecret(
    const ProtocolSetupResult& setup,
    ClientId clientId) {

    const auto found = std::find_if(
        setup.clientSecrets.begin(),
        setup.clientSecrets.end(),
        [clientId](const ClientSecretSetupMaterial& secret) {
            return secret.clientId == clientId;
        });

    if (found == setup.clientSecrets.end()) {
        throw std::runtime_error("registered client has no secret setup material");
    }

    return *found;
}

lbcrypto::DCRTPoly SampleSmallError(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    long double standardDeviation, bool certifiedV17) {

    if (!parameters ||
        !std::isfinite(standardDeviation) ||
        standardDeviation <= 0.0L ||
        standardDeviation > std::ldexp(1.0L, 59)) {

        throw std::invalid_argument(
            "invalid client key-switch error standard deviation");
    }

    return SampleProtocolGaussianPolynomial(parameters, standardDeviation, certifiedV17);
}

lbcrypto::DCRTPoly SampleUniformTernary(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters) {

    if (!parameters) {
        throw std::invalid_argument(
            "client key-switch ternary parameters are null");
    }

    lbcrypto::DCRTPoly::TugType generator;
    return lbcrypto::DCRTPoly(
        generator,
        parameters,
        Format::EVALUATION);
}

void ValidateSetupShape(const ProtocolSetupResult& setup) {
    const auto& publicMaterial = setup.common;

    if (!publicMaterial.rns.paramsQP ||
        publicMaterial.descriptor.canonicalEncoding.empty() ||
        publicMaterial.mlweRank == 0 ||
        publicMaterial.rns.decompositionDigitCount == 0) {

        throw std::invalid_argument("protocol setup is incomplete");
    }

    if (setup.clientSecrets.empty() ||
        setup.cachedClientShares.size() != setup.clientSecrets.size()) {

        throw std::invalid_argument("client setup material is incomplete");
    }

    if (setup.mode != publicMaterial.descriptor.mode ||
        (setup.mode == ProtocolMode::SP) != setup.sparse.has_value()) {

        throw std::invalid_argument("protocol setup mode metadata is inconsistent");
    }

    const bool requiresM2r = RequiresSparseRingConversion(
        setup.mode,
        publicMaterial.mlweRank);
    if (setup.sparse && requiresM2r) {
        if (setup.sparse->m2rCommonA.size() != publicMaterial.mlweRank) {
            throw std::invalid_argument(
                "M2R common-a rank does not match setup rank");
        }

        for (const auto& perSplit : setup.sparse->m2rCommonA) {
            if (perSplit.size() !=
                publicMaterial.rns.decompositionDigitCount) {

                throw std::invalid_argument(
                    "M2R common-a decomposition shape is invalid");
            }
        }
    }
    else if (setup.sparse && !setup.sparse->m2rCommonA.empty()) {
        throw std::invalid_argument(
            "rank-one SP setup must not contain M2R common-a material");
    }

    std::vector<bool> seenClientIds(setup.clientSecrets.size(), false);

    for (const auto& share : setup.cachedClientShares) {
        if (share.clientId.value >= seenClientIds.size() ||
            seenClientIds[share.clientId.value]) {

            throw std::invalid_argument(
                "cached setup shares have invalid or duplicate client IDs");
        }

        seenClientIds[share.clientId.value] = true;

        if (share.m2rShare.has_value() != requiresM2r) {
            throw std::invalid_argument(
                "cached M2R share presence does not match conversion mode");
        }

        if (!share.m2rShare) {
            continue;
        }

        if (!(share.m2rShare->clientId == share.clientId)) {
            throw std::invalid_argument(
                "cached public-key and M2R share client IDs differ");
        }

        if (share.m2rShare->b.size() != publicMaterial.mlweRank) {
            throw std::invalid_argument("cached M2R share rank is invalid");
        }

        for (const auto& perSplit : share.m2rShare->b) {
            if (perSplit.size() !=
                publicMaterial.rns.decompositionDigitCount) {

                throw std::invalid_argument(
                    "cached M2R share decomposition shape is invalid");
            }
        }
    }
}

std::vector<ClientToCollectiveKeyMaterial>
GenerateClientToCollectiveKeys(
    const ProtocolSetupResult& setup,
    const SurvivorSet& survivors,
    const SurvivorPublicKeyMaterial& publicKey,
    std::vector<double>* clientSeconds) {

    const auto& publicMaterial = setup.common;
    std::vector<ClientToCollectiveKeyMaterial> result;
    result.reserve(survivors.clients.size());
    if (clientSeconds) {
        clientSeconds->clear();
        clientSeconds->reserve(survivors.clients.size());
    }

    for (const ClientId clientId : survivors.clients) {
        const auto clientStarted = std::chrono::steady_clock::now();
        const ClientSecretSetupMaterial& clientSecret =
            FindClientSecret(setup, clientId);

        if (!clientSecret.encryptionKeyQ) {
            throw std::invalid_argument("local encryption key is missing");
        }

        ClientToCollectiveKeyMaterial key;
        key.setupId = publicMaterial.setupId;
        key.sourceClientId = clientId;
        key.survivors = survivors;
        key.decompositionDigitCount =
            publicMaterial.rns.decompositionDigitCount;

        key.a.reserve(key.decompositionDigitCount);
        key.b.reserve(key.decompositionDigitCount);

        for (std::size_t lambda = 0;
             lambda < key.decompositionDigitCount;
             ++lambda) {

            const lbcrypto::DCRTPoly ephemeral = SampleUniformTernary(
                publicMaterial.rns.paramsQP);

            const lbcrypto::DCRTPoly error0 = SampleSmallError(
                publicMaterial.rns.paramsQP,
                setup.setupNoise.clientKeySwitchStandardDeviation, publicMaterial.certifiedV17Gaussian);

            const lbcrypto::DCRTPoly error1 = SampleSmallError(
                publicMaterial.rns.paramsQP,
                setup.setupNoise.clientKeySwitchStandardDeviation, publicMaterial.certifiedV17Gaussian);

            const lbcrypto::DCRTPoly gadgetTerm = BuildM2RGadgetTerm(
                clientSecret.encryptionKeyQ->GetPrivateElement(),
                lambda,
                publicMaterial.rns);

            key.a.push_back(
                ephemeral * publicKey.p1 + error1);

            key.b.push_back(
                ephemeral * publicKey.p0 + error0 + gadgetTerm);
        }

        result.push_back(std::move(key));
        if (clientSeconds) {
            clientSeconds->push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - clientStarted).count());
        }
    }

    return result;
}

}  // namespace

SurvivorKeySet AggregateSurvivorKeys(
    const ProtocolSetupResult& setup,
    const SurvivorSet& survivors) {

    ValidateSetupShape(setup);

    const SurvivorSet canonical = CanonicalizeSurvivors(
        survivors,
        setup.clientSecrets.size());

    const auto& publicMaterial = setup.common;
    SurvivorKeySet result;
    result.setupId = publicMaterial.setupId;
    const auto reconstructionStarted = std::chrono::steady_clock::now();

    result.publicKey.survivors = canonical;
    result.publicKey.p0 = lbcrypto::DCRTPoly(
        publicMaterial.rns.paramsQP,
        Format::EVALUATION,
        true);

    result.publicKey.p1 = publicMaterial.publicKeyP1;

    if (setup.sparse && RequiresSparseRingConversion(
            setup.mode,
            publicMaterial.mlweRank)) {
        SurvivorM2RKeyMaterial m2rKey;
        m2rKey.survivors = canonical;
        m2rKey.mlweRank = publicMaterial.mlweRank;
        m2rKey.decompositionDigitCount =
            publicMaterial.rns.decompositionDigitCount;
        m2rKey.a = setup.sparse->m2rCommonA;
        m2rKey.b.resize(publicMaterial.mlweRank);

        for (std::size_t splitIndex = 0;
             splitIndex < publicMaterial.mlweRank;
             ++splitIndex) {

            m2rKey.b[splitIndex].reserve(
                publicMaterial.rns.decompositionDigitCount);

            for (std::size_t lambda = 0;
                 lambda < publicMaterial.rns.decompositionDigitCount;
                 ++lambda) {

                m2rKey.b[splitIndex].emplace_back(
                    publicMaterial.rns.paramsQP,
                    Format::EVALUATION,
                    true);
            }
        }
        result.m2rKey = std::move(m2rKey);
    }

    for (const ClientId clientId : canonical.clients) {
        const ClientCachedSetupShare& share =
            FindCachedShare(setup, clientId);

        result.publicKey.p0 += share.publicKeyP0Share;

        if (!result.m2rKey) {
            continue;
        }

        for (std::size_t splitIndex = 0;
             splitIndex < publicMaterial.mlweRank;
             ++splitIndex) {

            for (std::size_t lambda = 0;
                 lambda < publicMaterial.rns.decompositionDigitCount;
                 ++lambda) {

                result.m2rKey->b[splitIndex][lambda] +=
                    share.m2rShare->b[splitIndex][lambda];
            }
        }
    }

    result.serverTargetReconstructionSeconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - reconstructionStarted).count();

    result.clientToCollectiveKeys = GenerateClientToCollectiveKeys(
        setup,
        canonical,
        result.publicKey,
        &result.clientToCollectiveKeyGenerationSeconds);

    return result;
}

}  // namespace federated
