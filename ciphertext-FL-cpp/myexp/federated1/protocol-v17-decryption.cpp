#include "codec-timing.h"
#include "protocol-v17-decryption.h"

#include "deterministic-randomness.h"
#include "protocol-randomness.h"
#include "protocol-setup.h"

#include <stdexcept>
#include <set>
#include <utility>

namespace federated {
namespace {

lbcrypto::DCRTPoly PublicPolynomial(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& paramsQ,
    const Seed128& seedD,
    SetupId setupId,
    RoundId roundId,
    const std::string& path,
    std::size_t instanceIndex,
    const char* domain,
    const ClientId* clientId) {

    codec_timing::Scope timer(codec_timing::PublicPolynomial);
    ByteVector context = deterministic_randomness::V17DecryptionContext(
        setupId, roundId, path, instanceIndex);
    if (clientId != nullptr) {
        const ByteVector id = deterministic_randomness::EncodeBigEndian64(
            clientId->value);
        context.insert(context.end(), id.begin(), id.end());
    }
    return GenerateSeededUniformDCRT(paramsQ, seedD, domain, context);
}

}  // namespace

Seed128 FreshV17DecryptionSeed128() {
    static std::mutex mutex;
    static std::set<Seed128> used;
    std::lock_guard<std::mutex> lock(mutex);
    for (;;) {
        const Seed128 candidate = SampleFreshProtocolSeed128();
        if (used.insert(candidate).second) {
            return candidate;
        }
    }
}

V17DecryptionSeedGate::V17DecryptionSeedGate(
    std::size_t pathCount,
    bool abortAfterRequest)
    : pathCount_(pathCount), abortAfterRequest_(abortAfterRequest) {
    if (pathCount == 0) {
        throw std::invalid_argument("v17 decryption gate needs a path");
    }
}

Seed128 V17DecryptionSeedGate::ArriveAfterAggregateA() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (error_) {
        std::rethrow_exception(error_);
    }
    if (arrived_ >= pathCount_) {
        throw std::logic_error("v17 decryption gate reused");
    }
    if (++arrived_ == pathCount_) {
        seed_ = FreshV17DecryptionSeed128();
        released_ = true;
        if (abortAfterRequest_) {
            error_ = std::make_exception_ptr(V17PostRequestDropout());
        }
        ready_.notify_all();
    }
    else {
        ready_.wait(lock, [this]() { return released_ || error_; });
    }
    if (error_) {
        std::rethrow_exception(error_);
    }
    return seed_;
}

void V17DecryptionSeedGate::Abort(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) {
        error_ = error != nullptr
            ? error
            : std::make_exception_ptr(std::runtime_error(
                  "v17 finalized decryption aborted"));
    }
    ready_.notify_all();
}

coefficient_compression::CompressedCoefficientPayload
MakeV17PartialResponse(
    lbcrypto::DCRTPoly partial,
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& paramsQ,
    const Seed128& seedD,
    SetupId setupId,
    RoundId roundId,
    const std::string& path,
    std::size_t instanceIndex,
    ClientId clientId,
    ClientId offsetHolder,
    std::uint32_t bits,
    const std::vector<std::size_t>& selectedIndices) {

    partial += PublicPolynomial(paramsQ, seedD, setupId, roundId, path,
        instanceIndex, "C2-DITH", &clientId);
    if (clientId == offsetHolder) {
        partial += PublicPolynomial(paramsQ, seedD, setupId, roundId, path,
            instanceIndex, "C2-OFFSET", nullptr);
    }
    return selectedIndices.empty()
        ? coefficient_compression::CompressAllCoefficientsStochastic(
              std::move(partial), bits)
        : coefficient_compression::CompressSelectedCoefficientsStochastic(
              std::move(partial), bits, selectedIndices);
}

lbcrypto::DCRTPoly RecoverV17Phase(
    lbcrypto::DCRTPoly aggregateB,
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& paramsQ,
    const std::vector<coefficient_compression::CompressedCoefficientPayload>&
        responseCodes,
    const std::vector<ClientId>& survivorIds,
    const Seed128& seedD,
    SetupId setupId,
    RoundId roundId,
    const std::string& path,
    std::size_t instanceIndex,
    const std::vector<std::size_t>& selectedIndices) {

    if (responseCodes.size() != survivorIds.size() || survivorIds.empty()) {
        throw std::invalid_argument("v17 response/survivor count mismatch");
    }
    lbcrypto::DCRTPoly summed = selectedIndices.empty()
        ? coefficient_compression::DecompressSummedStochasticCodes(
              paramsQ, responseCodes)
        : coefficient_compression::DecompressSelectedSummedStochasticCodes(
              paramsQ, responseCodes, selectedIndices);
    summed.SetFormat(Format::EVALUATION);
    aggregateB.SetFormat(Format::EVALUATION);
    aggregateB += summed;
    aggregateB -= PublicPolynomial(paramsQ, seedD, setupId, roundId, path,
        instanceIndex, "C2-OFFSET", nullptr);
    for (const ClientId clientId : survivorIds) {
        aggregateB -= PublicPolynomial(paramsQ, seedD, setupId, roundId,
            path, instanceIndex, "C2-DITH", &clientId);
    }
    return aggregateB;
}

}  // namespace federated
