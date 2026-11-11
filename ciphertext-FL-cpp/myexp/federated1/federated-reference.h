#ifndef FEDERATED1_FEDERATED_REFERENCE_H
#define FEDERATED1_FEDERATED_REFERENCE_H

#include "federated-types.h"

#include <cstdint>
#include <vector>

namespace federated {

/* Plaintext test oracle: idealAggregate = sum_i alpha_i*update_i/weightDenominator. quantizedAggregate first applies the codec rounding to Delta*update_i, then weights and sums integers and divides by weightDenominator*Delta. Use the latter to compare an encode-then-weight ciphertext path. */
struct FederatedReferenceResult {
    std::vector<double> idealAggregate;
    std::vector<double> quantizedAggregate;

    /* One for coordinates in admitted cohorts; zero otherwise. */
    std::vector<std::uint8_t> acceptedCoordinateMask;

    /* Identifies the fixed sparse plan. */
    std::uint64_t planHash = 0;

    double scalingFactor = 0.0;
};

/* Evaluate the plaintext oracle using an existing plan without recomputing chunk scores or selections. */
FederatedReferenceResult ComputeFederatedReference(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients,
    const SparseRoundPlan& plan);

}  // namespace federated

#endif  // FEDERATED1_FEDERATED_REFERENCE_H