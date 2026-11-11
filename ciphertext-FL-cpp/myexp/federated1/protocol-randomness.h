#ifndef FEDERATED1_PROTOCOL_RANDOMNESS_H
#define FEDERATED1_PROTOCOL_RANDOMNESS_H

#include "openfhe.h"
#include "exact-integer.h"
#include "protocol-types.h"

#include <memory>

namespace federated {

/*
 * OpenFHE gives each worker thread an independent PRNG in normal builds.
 * The protocol sampler therefore keeps its Gaussian generator in thread-local
 * storage and permits independent ciphertext tasks to sample concurrently.
 * OpenMP+FIXED_SEED builds are intentionally serialized because OpenFHE uses
 * a shared deterministic PRNG in that configuration.
 */
lbcrypto::DCRTPoly SampleProtocolGaussianPolynomial(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    long double standardDeviation,
    bool certifiedV17 = false);

/* Exact rational rejection, conditioned on |X| <= ceil(24 sigma).
 * Supported sigma: 319/100 (3.19), 16/5 (3.2), or integers in [1,2^128]. */
exact_integer::Integer SampleCertifiedV17GaussianInteger(long double standardDeviation);
// Legacy scalar API rejects sigmas whose support exceeds int64 before drawing.
std::int64_t SampleCertifiedV17Gaussian(long double standardDeviation);

/* Fresh private OpenFHE PRNG output, independent of public SHAKE domains.
 * FIXED_SEED builds are diagnostic only and must not be used for v17 runs. */
Seed128 SampleFreshProtocolSeed128();

/* Exact rejection sampling in Z_modulus for stochastic c1/c2 coins. */
lbcrypto::BigInteger SampleProtocolUniformBelow(
    const lbcrypto::BigInteger& modulus);

bool ProtocolGaussianSamplingSupportsParallelism() noexcept;

}  // namespace federated

#endif
