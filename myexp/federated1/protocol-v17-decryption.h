#ifndef FEDERATED1_PROTOCOL_V17_DECRYPTION_H
#define FEDERATED1_PROTOCOL_V17_DECRYPTION_H

#include "coefficient-compression.h"
#include "protocol-types.h"

#include "openfhe.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>

namespace federated {

class V17PostRequestDropout : public std::runtime_error {
public:
    V17PostRequestDropout()
        : std::runtime_error(
              "v17 final survivor dropped after decryption request; round aborted") {}
};

/* The final decryption seed is released only after every participating path
 * has frozen all of its aggregate ciphertexts and deterministic ca bytes. */
class V17DecryptionSeedGate {
public:
    explicit V17DecryptionSeedGate(
        std::size_t pathCount,
        bool abortAfterRequest = false);
    Seed128 ArriveAfterAggregateA();
    void Abort(std::exception_ptr error);

private:
    const std::size_t pathCount_;
    const bool abortAfterRequest_;
    std::size_t arrived_ = 0;
    Seed128 seed_{};
    bool released_ = false;
    std::exception_ptr error_;
    std::mutex mutex_;
    std::condition_variable ready_;
};

/* Process-local non-reuse guard; a retry/new finalized collection gets a
 * fresh seed even if the underlying 128-bit sampler ever collides. */
Seed128 FreshV17DecryptionSeed128();

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
    const std::vector<std::size_t>& selectedIndices = {});

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
    const std::vector<std::size_t>& selectedIndices = {});

}  // namespace federated

#endif
