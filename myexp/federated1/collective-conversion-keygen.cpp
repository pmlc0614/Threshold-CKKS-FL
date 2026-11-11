#include "collective-conversion-keygen.h"

#include "ring-conversion.h"

#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace collective_conversion_keygen {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;
using PrivateKey =
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
using PublicKey =
    lbcrypto::PublicKey<lbcrypto::DCRTPoly>;

void ValidateCryptoContext(
    const lbcrypto::CryptoContext<DCRTPoly>& cc) {

    if (cc == nullptr) {
        throw std::invalid_argument(
            "crypto context is null");
    }
}

void ValidatePrivateKey(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const PrivateKey& privateKey,
    const char* keyName) {

    ValidateCryptoContext(cc);

    if (privateKey == nullptr) {
        throw std::invalid_argument(
            std::string(keyName) +
            " is null");
    }

    if (privateKey->GetCryptoContext() != cc) {
        throw std::invalid_argument(
            std::string(keyName) +
            " belongs to a different crypto context");
    }
}

void ValidatePublicKey(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const PublicKey& publicKey,
    const char* keyName) {

    ValidateCryptoContext(cc);

    if (publicKey == nullptr) {
        throw std::invalid_argument(
            std::string(keyName) +
            " is null");
    }

    if (publicKey->GetCryptoContext() != cc) {
        throw std::invalid_argument(
            std::string(keyName) +
            " belongs to a different crypto context");
    }
}

/* Compare all polynomial coefficients to verify that parties reused the same AVector. */
bool EqualDCRTPoly(
    const DCRTPoly& left,
    const DCRTPoly& right) {

    if (left.GetFormat() != right.GetFormat()) {
        return false;
    }

    const auto& leftTowers =
        left.GetAllElements();

    const auto& rightTowers =
        right.GetAllElements();

    if (leftTowers.size() !=
        rightTowers.size()) {

        return false;
    }

    for (size_t towerIndex = 0;
         towerIndex < leftTowers.size();
         ++towerIndex) {

        const auto& leftTower =
            leftTowers[towerIndex];

        const auto& rightTower =
            rightTowers[towerIndex];

        if (leftTower.GetModulus() !=
            rightTower.GetModulus()) {

            return false;
        }

        const auto& leftValues =
            leftTower.GetValues();

        const auto& rightValues =
            rightTower.GetValues();

        if (leftValues.GetLength() !=
            rightValues.GetLength()) {

            return false;
        }

        for (size_t coefficientIndex = 0;
             coefficientIndex <
                 leftValues.GetLength();
             ++coefficientIndex) {

            if (leftValues[coefficientIndex] !=
                rightValues[coefficientIndex]) {

                return false;
            }
        }
    }

    return true;
}

void ValidateEvalKey(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ConversionEvalKey& evalKey,
    const char* functionName) {

    if (evalKey == nullptr) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": evaluation key is null");
    }

    if (evalKey->GetCryptoContext() != cc) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": evaluation key belongs to a "
            "different crypto context");
    }

    const auto& aVector =
        evalKey->GetAVector();

    const auto& bVector =
        evalKey->GetBVector();

    if (aVector.empty()) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": evaluation key AVector is empty");
    }

    if (aVector.size() != bVector.size()) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": evaluation key AVector/BVector "
            "sizes do not match");
    }
}

void ValidateKeyShare(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const MLWEtoRLWEKeyShare& keyShare,
    const char* functionName) {

    if (keyShare.rank == 0) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": key share rank is zero");
    }

    if (keyShare.keys.size() !=
        keyShare.rank) {

        throw std::invalid_argument(
            std::string(functionName) +
            ": number of evaluation keys must "
            "equal rank");
    }

    for (const auto& key : keyShare.keys) {
        ValidateEvalKey(
            cc,
            key,
            functionName);
    }
}

/* Check that candidate and commonTemplate use identical AVectors. */
bool HaveSameAVector(
    const ConversionEvalKey& commonTemplate,
    const ConversionEvalKey& candidate) {

    if (commonTemplate == nullptr ||
        candidate == nullptr) {

        return false;
    }

    const auto& referenceAVector =
        commonTemplate->GetAVector();

    const auto& candidateAVector =
        candidate->GetAVector();

    if (referenceAVector.size() !=
        candidateAVector.size()) {

        return false;
    }

    for (size_t partIndex = 0;
         partIndex < referenceAVector.size();
         ++partIndex) {

        if (!EqualDCRTPoly(
                referenceAVector[partIndex],
                candidateAVector[partIndex])) {

            return false;
        }
    }

    return true;
}

/* Baseline source keys: t_(j,i)=Embed(Split_j(s_i)), supplied as oldPrivateKey to OpenFHE key switching. These are not independent v17 t_i keys. */
std::vector<PrivateKey>
MakeEmbeddedSourceKeys(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const PrivateKey& secretKeyShare,
    size_t rank,
    const char* functionName) {

    ValidatePrivateKey(
        cc,
        secretKeyShare,
        "secret key share");

    if (rank == 0) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": rank must be greater than zero");
    }

    const auto splitSecretShares =
        ring_conversion::Split(
            secretKeyShare
                ->GetPrivateElement(),
            rank);

    if (splitSecretShares.size() != rank) {
        throw std::runtime_error(
            std::string(functionName) +
            ": unexpected number of split "
            "secret-key components");
    }

    std::vector<PrivateKey> sourceKeys;
    sourceKeys.reserve(rank);

    for (size_t splitIndex = 0;
         splitIndex < rank;
         ++splitIndex) {

        auto embeddedShare =
            ring_conversion::RingEmbedding(
                splitSecretShares[splitIndex],
                rank);

        embeddedShare.SetFormat(
            Format::EVALUATION);

        auto sourceKey =
            std::make_shared<
                lbcrypto::PrivateKeyImpl<
                    DCRTPoly>>(cc);

        sourceKey->SetPrivateElement(
            std::move(embeddedShare));

        sourceKeys.emplace_back(
            std::move(sourceKey));
    }

    return sourceKeys;
}

}  // namespace

MLWEtoRLWEKeyShare
GenerateLeadMLWEtoRLWEKeyShare(
    const lbcrypto::CryptoContext<
        DCRTPoly>& cc,
    const PrivateKey& secretKeyShare,
    size_t rank) {

    ValidatePrivateKey(
        cc,
        secretKeyShare,
        "lead secret key share");

    const auto sourceKeys =
        MakeEmbeddedSourceKeys(
            cc,
            secretKeyShare,
            rank,
            "GenerateLeadMLWEtoRLWEKeyShare");

    MLWEtoRLWEKeyShare result;

    result.partyIndex = 0;
    result.rank = rank;
    result.keys.reserve(rank);

    for (size_t splitIndex = 0;
         splitIndex < rank;
         ++splitIndex) {

        /* Generate a fresh common AVector while switching Embed(Split_j(s_0)) to s_0 in this baseline. */
        auto key =
            cc->KeySwitchGen(
                sourceKeys[splitIndex],
                secretKeyShare);

        ValidateEvalKey(
            cc,
            key,
            "GenerateLeadMLWEtoRLWEKeyShare");

        result.keys.emplace_back(
            std::move(key));
    }

    return result;
}

MLWEtoRLWEKeyShare
GenerateMainMLWEtoRLWEKeyShare(
    const lbcrypto::CryptoContext<
        DCRTPoly>& cc,
    const PrivateKey& secretKeyShare,
    const MLWEtoRLWEKeyShare&
        commonKeyTemplate,
    size_t partyIndex) {

    ValidatePrivateKey(
        cc,
        secretKeyShare,
        "main secret key share");

    if (partyIndex == 0) {
        throw std::invalid_argument(
            "GenerateMainMLWEtoRLWEKeyShare: "
            "partyIndex must be greater than zero");
    }

    ValidateKeyShare(
        cc,
        commonKeyTemplate,
        "GenerateMainMLWEtoRLWEKeyShare");

    if (commonKeyTemplate.partyIndex != 0) {
        throw std::invalid_argument(
            "GenerateMainMLWEtoRLWEKeyShare: "
            "common template must be the lead "
            "party share");
    }

    const size_t rank =
        commonKeyTemplate.rank;

    const auto sourceKeys =
        MakeEmbeddedSourceKeys(
            cc,
            secretKeyShare,
            rank,
            "GenerateMainMLWEtoRLWEKeyShare");

    MLWEtoRLWEKeyShare result;

    result.partyIndex = partyIndex;
    result.rank = rank;
    result.keys.reserve(rank);

    for (size_t splitIndex = 0;
         splitIndex < rank;
         ++splitIndex) {

        /* Reuse the template AVector while switching Embed(Split_j(s_i)) to s_i in this baseline. */
        auto key =
            cc->MultiKeySwitchGen(
                sourceKeys[splitIndex],
                secretKeyShare,
                commonKeyTemplate
                    .keys[splitIndex]);

        ValidateEvalKey(
            cc,
            key,
            "GenerateMainMLWEtoRLWEKeyShare");

        if (!HaveSameAVector(
                commonKeyTemplate
                    .keys[splitIndex],
                key)) {

            throw std::runtime_error(
                "GenerateMainMLWEtoRLWEKeyShare: "
                "MultiKeySwitchGen did not reuse "
                "the common AVector");
        }

        result.keys.emplace_back(
            std::move(key));
    }

    return result;
}

std::vector<ConversionEvalKey>
AggregateMLWEtoRLWEKeyShares(
    const lbcrypto::CryptoContext<
        DCRTPoly>& cc,
    const std::vector<
        MLWEtoRLWEKeyShare>& partyKeyShares,
    const PublicKey& collectivePublicKey) {

    ValidateCryptoContext(cc);

    ValidatePublicKey(
        cc,
        collectivePublicKey,
        "collective public key");

    if (partyKeyShares.size() < 2) {
        throw std::invalid_argument(
            "AggregateMLWEtoRLWEKeyShares: "
            "at least two party shares are required");
    }

    if (partyKeyShares[0].partyIndex != 0) {
        throw std::invalid_argument(
            "AggregateMLWEtoRLWEKeyShares: "
            "the first share must be the lead "
            "party share");
    }

    const size_t rank =
        partyKeyShares[0].rank;

    ValidateKeyShare(
        cc,
        partyKeyShares[0],
        "AggregateMLWEtoRLWEKeyShares");

    std::set<size_t> partyIndices;

    for (size_t shareIndex = 0;
         shareIndex < partyKeyShares.size();
         ++shareIndex) {

        const auto& keyShare =
            partyKeyShares[shareIndex];

        ValidateKeyShare(
            cc,
            keyShare,
            "AggregateMLWEtoRLWEKeyShares");

        if (keyShare.rank != rank) {
            throw std::invalid_argument(
                "AggregateMLWEtoRLWEKeyShares: "
                "party key-share ranks do not match");
        }

        if (!partyIndices
                 .insert(keyShare.partyIndex)
                 .second) {

            throw std::invalid_argument(
                "AggregateMLWEtoRLWEKeyShares: "
                "duplicate party index");
        }

        for (size_t splitIndex = 0;
             splitIndex < rank;
             ++splitIndex) {

            if (!HaveSameAVector(
                    partyKeyShares[0]
                        .keys[splitIndex],
                    keyShare.keys[splitIndex])) {

                throw std::invalid_argument(
                    "AggregateMLWEtoRLWEKeyShares: "
                    "party shares do not use the "
                    "same AVector");
            }
        }
    }

    const std::string collectiveKeyTag =
        collectivePublicKey->GetKeyTag();

    std::vector<ConversionEvalKey>
        collectiveKeys;

    collectiveKeys.reserve(rank);

    for (size_t splitIndex = 0;
         splitIndex < rank;
         ++splitIndex) {

        auto aggregate =
            partyKeyShares[0]
                .keys[splitIndex];

        /* MultiAddEvalKeys retains the first AVector and adds only the BVectors. */
        for (size_t shareIndex = 1;
             shareIndex < partyKeyShares.size();
             ++shareIndex) {

            aggregate =
                cc->MultiAddEvalKeys(
                    aggregate,
                    partyKeyShares[shareIndex]
                        .keys[splitIndex],
                    collectiveKeyTag);

            ValidateEvalKey(
                cc,
                aggregate,
                "AggregateMLWEtoRLWEKeyShares");
        }

        if (aggregate->GetKeyTag() !=
            collectiveKeyTag) {

            throw std::runtime_error(
                "AggregateMLWEtoRLWEKeyShares: "
                "final evaluation-key tag does not "
                "match collective public-key tag");
        }

        collectiveKeys.emplace_back(
            std::move(aggregate));
    }

    return collectiveKeys;
}

std::vector<ConversionEvalKey>
GenerateCollectiveMLWEtoRLWEKeys(
    const lbcrypto::CryptoContext<
        DCRTPoly>& cc,
    const std::vector<PrivateKey>&
        secretKeyShares,
    const PublicKey& collectivePublicKey,
    size_t rank) {

    ValidateCryptoContext(cc);

    ValidatePublicKey(
        cc,
        collectivePublicKey,
        "collective public key");

    if (secretKeyShares.size() < 2) {
        throw std::invalid_argument(
            "GenerateCollectiveMLWEtoRLWEKeys: "
            "at least two secret-key shares are "
            "required");
    }

    for (size_t partyIndex = 0;
         partyIndex < secretKeyShares.size();
         ++partyIndex) {

        ValidatePrivateKey(
            cc,
            secretKeyShares[partyIndex],
            "secret key share");
    }

    const auto leadShare =
        GenerateLeadMLWEtoRLWEKeyShare(
            cc,
            secretKeyShares[0],
            rank);

    std::vector<MLWEtoRLWEKeyShare>
        partyKeyShares;

    partyKeyShares.reserve(
        secretKeyShares.size());

    partyKeyShares.push_back(
        leadShare);

    for (size_t partyIndex = 1;
         partyIndex < secretKeyShares.size();
         ++partyIndex) {

        partyKeyShares.emplace_back(
            GenerateMainMLWEtoRLWEKeyShare(
                cc,
                secretKeyShares[partyIndex],
                leadShare,
                partyIndex));
    }

    return AggregateMLWEtoRLWEKeyShares(
        cc,
        partyKeyShares,
        collectivePublicKey);
}

}  // namespace collective_conversion_keygen