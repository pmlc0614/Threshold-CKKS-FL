#include "protocol-randomness.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    // Optional deterministic test-only PRNG, never used by production targets.
    if (argc == 2) lbcrypto::PseudoRandomNumberGenerator::InitPRNGEngine(argv[1]);
    auto params = std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
        16384, static_cast<std::uint32_t>(2), static_cast<std::uint32_t>(31));
    const std::uint64_t reference[] = {2735394715718372971ULL,
        11764748295979271162ULL, 17965530706811686575ULL};
    std::size_t caseIndex = 0;
    for (long double sigma : {3.19L, 700000000000.L, 40000000000000.L}) {
        double total = 0;
        std::uint64_t fingerprint = 1469598103934665603ULL;
        for (int trial = 0; trial < 12; ++trial) {
            const auto begin = std::chrono::steady_clock::now();
            auto poly = federated::SampleProtocolGaussianPolynomial(params, sigma, true);
            const auto seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            if (trial >= 2) total += seconds;
            if (poly.GetNumOfElements() != 2) return 1;
            for (const auto& tower : poly.GetAllElements())
                for (std::size_t i = 0; i < 8192; ++i)
                    fingerprint = (fingerprint ^ tower[i].ConvertToInt()) * 1099511628211ULL;
        }
        std::cout << "sigma=" << static_cast<double>(sigma)
                  << " N=8192 measured_polynomials=10 ms/poly=" << total * 100
                  << " fingerprint=" << fingerprint << '\n';
        if (argc == 2 && fingerprint != reference[caseIndex]) {
            std::cerr << "Gaussian replay differs from pre-optimization output\n";
            return 1;
        }
        ++caseIndex;
    }
}
