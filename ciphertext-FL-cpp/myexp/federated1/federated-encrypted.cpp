#include "federated-encrypted.h"

#include "chunk-packing.h"
#include "coefficient-codec.h"
#include "ring-conversion.h"
#include "sparsification.h"
#include "threshold-ckks.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace federated {
namespace {

const ClientRoundInput& FindClientInput(
    const std::vector<ClientRoundInput>& clients,
    std::size_t clientId) {

    for (const auto& client : clients) {
        if (client.clientId == clientId) {
            return client;
        }
    }

    throw std::invalid_argument(
        "FindClientInput: clientId was not found");
}

void ValidateCryptographicInputs(
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
    const FederatedConfig& config) {

    if (!cc) {
        throw std::invalid_argument(
            "ComputeFederatedEncryptedAggregate: "
            "crypto context is null");
    }

    if (!collectivePublicKey) {
        throw std::invalid_argument(
            "ComputeFederatedEncryptedAggregate: "
            "collective public key is null");
    }

    if (cc->GetRingDimension() !=
        config.ringDimension) {

        throw std::invalid_argument(
            "ComputeFederatedEncryptedAggregate: "
            "crypto-context ring dimension mismatch");
    }

    if (secretKeyShares.size() !=
        config.clientCount) {

        throw std::invalid_argument(
            "ComputeFederatedEncryptedAggregate: "
            "the number of secret shares must equal "
            "clientCount");
    }

    for (const auto& secretKeyShare :
         secretKeyShares) {

        if (!secretKeyShare) {
            throw std::invalid_argument(
                "ComputeFederatedEncryptedAggregate: "
                "a secret-key share is null");
        }
    }

    if (collectiveConversionKeys.size() !=
        config.mlweRank) {

        throw std::invalid_argument(
            "ComputeFederatedEncryptedAggregate: "
            "the number of conversion keys must equal "
            "the MLWE rank");
    }

    for (const auto& conversionKey :
         collectiveConversionKeys) {

        if (!conversionKey) {
            throw std::invalid_argument(
                "ComputeFederatedEncryptedAggregate: "
                "a collective conversion key is null");
        }
    }

    const double configuredSigma =
        threshold_ckks::GetSmudgingSigma(cc);

    const double sigmaTolerance =
        std::max(
            1.0,
            std::abs(config.sigmaShare)) *
        16.0 *
        std::numeric_limits<double>::epsilon();

    if (std::abs(
            configuredSigma -
            config.sigmaShare) >
        sigmaTolerance) {

        throw std::invalid_argument(
            "ComputeFederatedEncryptedAggregate: "
            "configured smudging sigma does not match "
            "config.sigmaShare");
    }
}

/* Multiply every CRT tower by the same integer alpha without changing the scale: the message becomes alpha*round(Delta*x). */
void MultiplyDCRTPolyByInteger(
    lbcrypto::DCRTPoly& polynomial,
    std::uint32_t scalar) {

    auto& towers =
        polynomial.GetAllElements();

    if (towers.empty()) {
        throw std::invalid_argument(
            "MultiplyDCRTPolyByInteger: "
            "polynomial has no CRT towers");
    }

    const Format format =
        polynomial.GetFormat();

    const lbcrypto::NativeInteger factor(
        static_cast<std::uint64_t>(scalar));

    for (auto& tower : towers) {
        const lbcrypto::NativeInteger modulus =
            tower.GetModulus();

        lbcrypto::NativeVector values =
            tower.GetValues();

        for (std::size_t index = 0;
             index < values.GetLength();
             ++index) {

            values[index] =
                values[index].ModMul(
                    factor,
                    modulus);
        }

        tower.SetValues(
            std::move(values),
            format);
    }
}

void ValidateMLWEMetadataMatch(
    const ring_conversion::MLWECiphertext& left,
    const ring_conversion::MLWECiphertext& right) {

    if (left.rank != right.rank ||
        left.ringDimension !=
            right.ringDimension ||
        left.a.size() != right.a.size() ||
        left.scalingFactor !=
            right.scalingFactor ||
        left.level != right.level ||
        left.keyTag != right.keyTag) {

        throw std::invalid_argument(
            "MLWE ciphertext metadata mismatch "
            "during aggregation");
    }

    if (left.a.size() != left.rank ||
        right.a.size() != right.rank) {

        throw std::invalid_argument(
            "invalid MLWE ciphertext component count");
    }

    for (std::size_t column = 0;
         column < left.rank;
         ++column) {

        if (left.a[column].GetFormat() !=
            right.a[column].GetFormat()) {

            throw std::invalid_argument(
                "MLWE ciphertext polynomial "
                "format mismatch");
        }
    }

    if (left.b.GetFormat() !=
        right.b.GetFormat()) {

        throw std::invalid_argument(
            "MLWE ciphertext b-component "
            "format mismatch");
    }
}

ring_conversion::MLWECiphertext
MultiplyMLWECiphertextByInteger(
    const ring_conversion::MLWECiphertext& ciphertext,
    std::uint32_t scalar) {

    if (ciphertext.rank == 0 ||
        ciphertext.a.size() !=
            ciphertext.rank) {

        throw std::invalid_argument(
            "MultiplyMLWECiphertextByInteger: "
            "invalid MLWE ciphertext");
    }

    ring_conversion::MLWECiphertext result =
        ciphertext;

    for (auto& component : result.a) {
        MultiplyDCRTPolyByInteger(
            component,
            scalar);
    }

    MultiplyDCRTPolyByInteger(
        result.b,
        scalar);

    return result;
}

void AddMLWECiphertextInPlace(
    ring_conversion::MLWECiphertext& target,
    const ring_conversion::MLWECiphertext& source) {

    ValidateMLWEMetadataMatch(
        target,
        source);

    for (std::size_t column = 0;
         column < target.rank;
         ++column) {

        target.a[column] +=
            source.a[column];
    }

    target.b +=
        source.b;
}

/* Create a zero MLWE padding row with the prototype's metadata and polynomial parameters. */
ring_conversion::MLWECiphertext
MakeZeroMLWECiphertext(
    const ring_conversion::MLWECiphertext& prototype) {

    return MultiplyMLWECiphertextByInteger(
        prototype,
        0);
}

const ring_conversion::MLWECiphertext&
FindAggregatedPrototype(
    const std::vector<
        std::unique_ptr<
            ring_conversion::MLWECiphertext>>&
        aggregatedByChunk) {

    for (const auto& ciphertext :
         aggregatedByChunk) {

        if (ciphertext) {
            return *ciphertext;
        }
    }

    throw std::runtime_error(
        "no aggregated MLWE ciphertext is available");
}

}  // namespace

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
    const SparseRoundPlan& plan) {

    ValidateFederatedRoundInputs(
        config,
        clients);

    ValidateSparseRoundPlan(
        config,
        clients,
        plan);

    ValidateCryptographicInputs(
        cc,
        collectivePublicKey,
        secretKeyShares,
        collectiveConversionKeys,
        config);

    FederatedEncryptedResult result;

    result.aggregate.assign(
        config.modelDimension,
        0.0);

    result.planHash =
        plan.planHash;

    result.scalingFactor =
        coefficient_codec::ScalingFactorFromBits(
            config.scaleBits);

    /* Store sum_i alpha_i*MLWECiphertext(i,l) for each admitted chunk; rejected chunks remain null. */
    std::vector<
        std::unique_ptr<
            ring_conversion::MLWECiphertext>>
        aggregatedByChunk(
            plan.chunkCount);

    /* Retain a client RLWE ciphertext as the metadata template for baseline M2R. */
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
        referenceCiphertext;

    /* Client stage of the functional baseline. */
    for (const auto& clientPlan :
         plan.clientPlans) {

        const ClientRoundInput& client =
            FindClientInput(
                clients,
                clientPlan.clientId);

        for (const auto& packingGroup :
             clientPlan.packingGroups) {

            const std::vector<double> packed =
                PackInterleavedChunks(
                    client.update,
                    packingGroup,
                    config);

            /* Encrypt the packed update under the baseline collective public key. */
            const auto packedCiphertext =
                threshold_ckks::EncryptCoefficients(
                    cc,
                    collectivePublicKey,
                    packed,
                    config.scaleBits);

            if (!packedCiphertext) {
                throw std::runtime_error(
                    "client encryption returned "
                    "a null ciphertext");
            }

            ++result.clientCiphertextCount;

            if (!referenceCiphertext) {
                referenceCiphertext =
                    packedCiphertext;
            }

            /* Split packed RLWE into rank MLWE ciphertexts; each slot matches packingGroup.chunkIds[slot]. */
            const auto mlweBundle =
                ring_conversion::RLWEtoMLWE(
                    packedCiphertext,
                    config.mlweRank);

            if (mlweBundle.ciphertexts.size() !=
                config.mlweRank) {

                throw std::runtime_error(
                    "RLWEtoMLWE returned an unexpected "
                    "number of ciphertexts");
            }

            if (packingGroup.chunkIds.size() !=
                config.mlweRank) {

                throw std::invalid_argument(
                    "client packing-group size does "
                    "not equal the MLWE rank");
            }

            for (std::size_t slot = 0;
                 slot < config.mlweRank;
                 ++slot) {

                const std::size_t chunkId =
                    packingGroup.chunkIds[slot];

                if (chunkId ==
                    kInvalidChunkId) {

                    continue;
                }

                const ChunkCohort& cohort =
                    FindChunkCohort(
                        plan,
                        chunkId);

                /* Exclude chunks that fail the minimum-cohort condition. */
                if (!cohort.accepted) {
                    continue;
                }

                /* Apply alpha_i after encoding, matching alpha_i*round(Delta*update_i) in the plaintext oracle. */
                auto weighted =
                    MultiplyMLWECiphertextByInteger(
                        mlweBundle.ciphertexts[slot],
                        client.alpha);

                auto& aggregate =
                    aggregatedByChunk[chunkId];

                if (!aggregate) {
                    aggregate =
                        std::make_unique<
                            ring_conversion::
                                MLWECiphertext>(
                            std::move(weighted));

                    ++result
                          .aggregatedMLWECiphertextCount;
                }
                else {
                    AddMLWECiphertextInPlace(
                        *aggregate,
                        weighted);
                }

                ++result
                      .acceptedClientContributionCount;
            }
        }
    }

    if (result.aggregatedMLWECiphertextCount !=
        plan.acceptedChunkIds.size()) {

        throw std::runtime_error(
            "not every accepted chunk produced an "
            "aggregated MLWE ciphertext");
    }

    /* Return an all-zero update when no chunks are admitted. */
    if (plan.acceptedChunkIds.empty()) {
        return result;
    }

    if (!referenceCiphertext) {
        throw std::runtime_error(
            "no reference RLWE ciphertext was generated");
    }

    const auto& zeroPrototype =
        FindAggregatedPrototype(
            aggregatedByChunk);

    /* Server baseline: group admitted MLWE ciphertexts in batches of rank and perform collective M2R. */
    for (const auto& serverGroup :
         plan.serverPackingGroups) {

        if (serverGroup.chunkIds.size() !=
            config.mlweRank) {

            throw std::invalid_argument(
                "server packing-group size does not "
                "equal the MLWE rank");
        }

        ring_conversion::MLWECiphertextBundle
            serverBundle;

        serverBundle.rank =
            config.mlweRank;

        serverBundle.sourceRingDimension =
            config.ringDimension;

        serverBundle.ciphertexts.reserve(
            config.mlweRank);

        for (std::size_t slot = 0;
             slot < config.mlweRank;
             ++slot) {

            const std::size_t chunkId =
                serverGroup.chunkIds[slot];

            if (chunkId ==
                kInvalidChunkId) {

                serverBundle.ciphertexts.emplace_back(
                    MakeZeroMLWECiphertext(
                        zeroPrototype));

                continue;
            }

            if (chunkId >=
                    aggregatedByChunk.size() ||
                !aggregatedByChunk[chunkId]) {

                throw std::runtime_error(
                    "server packing group references "
                    "a missing aggregated chunk");
            }

            serverBundle.ciphertexts.emplace_back(
                *aggregatedByChunk[chunkId]);
        }

        /* Convert to RLWE under the baseline collective secret. */
        const auto restoredCiphertext =
            ring_conversion::MLWEtoRLWE(
                cc,
                serverBundle,
                collectiveConversionKeys,
                referenceCiphertext);

        if (!restoredCiphertext) {
            throw std::runtime_error(
                "MLWEtoRLWE returned a null ciphertext");
        }

        /* Threshold-decrypt using each share; divide the decoded weighted sum by weightDenominator. */
        std::vector<double> decodedPacked =
            threshold_ckks::
                ThresholdDecryptCoefficients(
                    cc,
                    restoredCiphertext,
                    secretKeyShares,
                    result.scalingFactor);

        if (decodedPacked.size() !=
            config.ringDimension) {

            throw std::runtime_error(
                "threshold decryption returned an "
                "unexpected number of coefficients");
        }

        const double weightDenominator =
            static_cast<double>(
                config.weightDenominator);

        for (double& value : decodedPacked) {
            value /=
                weightDenominator;
        }

        const auto unpackedChunks =
            UnpackInterleavedChunks(
                decodedPacked,
                config);

        PlaceUnpackedChunks(
            unpackedChunks,
            serverGroup,
            config,
            result.aggregate);

        ++result.serverCiphertextCount;
    }

    return result;
}

}  // namespace federated