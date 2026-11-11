#include "codec-timing.h"
#include "protocol-randomness.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace federated {
namespace {

#if defined(WITH_OPENMP) && defined(FIXED_SEED)
std::mutex& ProtocolPrngMutex() {
    static std::mutex mutex;
    return mutex;
}
#endif

lbcrypto::DCRTPoly::DggType& ThreadGaussianGenerator(double sigma) {
    struct CachedGenerator {
        double sigma = std::numeric_limits<double>::quiet_NaN();
        std::optional<lbcrypto::DCRTPoly::DggType> generator;
    };
    thread_local CachedGenerator cached;
    if (!cached.generator || cached.sigma != sigma) {
        cached.generator.emplace(sigma);
        cached.sigma = sigma;
    }
    return *cached.generator;
}

lbcrypto::DCRTPoly SampleWithThreadGenerator(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    double sigma) {

    return lbcrypto::DCRTPoly(
        ThreadGaussianGenerator(sigma),
        parameters,
        Format::EVALUATION);
}

Seed128 SampleFreshSeedUnlocked() {
    Seed128 seed{};
    auto& prng = lbcrypto::PseudoRandomNumberGenerator::GetPRNG();
    for (std::size_t wordIndex = 0; wordIndex < 4; ++wordIndex) {
        const std::uint32_t word = prng();
        for (std::size_t byteIndex = 0; byteIndex < 4; ++byteIndex) {
            seed[4 * wordIndex + byteIndex] = static_cast<std::uint8_t>(
                word >> (24U - 8U * byteIndex));
        }
    }
    return seed;
}

lbcrypto::BigInteger SampleUniformBelowUnlocked(
    const lbcrypto::BigInteger& modulus) {

    auto& prng = lbcrypto::PseudoRandomNumberGenerator::GetPRNG();
    const std::size_t bitCount = modulus.GetMSB();
    const std::size_t wordCount = (bitCount + 31U) / 32U;
    const std::size_t firstWordBits = bitCount - 32U * (wordCount - 1U);

    while (true) {
        std::uint32_t firstWord = prng();
        if (firstWordBits < 32U) {
            firstWord >>= 32U - firstWordBits;
        }
        lbcrypto::BigInteger candidate(
            static_cast<std::uint64_t>(firstWord));
        for (std::size_t wordIndex = 1; wordIndex < wordCount; ++wordIndex) {
            candidate <<= 32U;
            candidate += lbcrypto::BigInteger(
                static_cast<std::uint64_t>(prng()));
        }
        if (candidate < modulus) {
            return candidate;
        }
    }
}

}  // namespace

namespace {
using Integer = lbcrypto::BigInteger;

bool RationalCoin(const Integer& numerator, const Integer& denominator) {
    if (numerator == Integer(0)) return false;
    if (numerator == denominator) return true;
    if (numerator > denominator) throw std::invalid_argument("invalid rational coin");
    return SampleUniformBelowUnlocked(denominator) < numerator;
}

// P(at least k successes) = gamma^k/k!, for 0 <= gamma <= 1.
// Odd stopping index has probability sum (-gamma)^k/k! = exp(-gamma).
bool ExpUnitCoin(const Integer& numerator, const Integer& denominator) {
    Integer k(1);
    bool odd = true;
    while (RationalCoin(numerator, denominator*k)) {
        k += Integer(1);
        odd = !odd;
    }
    return odd;
}

bool ExpCoin(Integer numerator, const Integer& denominator) {
    while (numerator >= denominator) {
        if (!ExpUnitCoin(denominator, denominator)) return false;
        numerator -= denominator;
    }
    return ExpUnitCoin(numerator, denominator);
}

struct ExactGaussianParameters {
    Integer p, q, denominator;
    Integer bigBound;
    std::uint64_t bound=0, pNative=0, qNative=0, smallDenominator=0;
    bool small;
    explicit ExactGaussianParameters(long double sigma) : p(0), q(1), bound(0) {
        if (sigma == 3.19L || sigma == static_cast<long double>(3.19)) {
            p = Integer(319); q = Integer(100);
        } else if (sigma == 3.2L || sigma == static_cast<long double>(3.2)) {
            p = Integer(16); q = Integer(5);
        } else if (std::isfinite(sigma) && sigma >= 1 && sigma <= std::ldexp(1.0L,128)
                   && std::floor(sigma) == sigma) {
            p = Integer(exact_integer::Integer(sigma).str());
        } else {
            throw std::invalid_argument("v17 certified sampler requires sigma=3.19, 3.2 or integer 1..2^128");
        }
        bigBound = (Integer(24)*p+q-Integer(1))/q;
        if (p <= Integer(std::uint64_t{1}<<55)) {
            bound = bigBound.ConvertToInt();
            pNative = p.ConvertToInt();
        }
        qNative = q.ConvertToInt();
        small = pNative != 0 && pNative <= 1000000;
        smallDenominator = small ? 2*pNative*pNative : 0;
        denominator = Integer(2)*p*p;
    }
};

// Exact integer fast path for the common small error sigma. Unsigned rejection
// removes modulo bias; no floating probability or finite iteration cap is used.
std::uint64_t UniformBelow64(std::uint64_t modulus) {
    auto& prng = lbcrypto::PseudoRandomNumberGenerator::GetPRNG();
    const std::uint64_t threshold = (std::uint64_t{0}-modulus)%modulus;
    while (true) {
        const std::uint64_t value = (static_cast<std::uint64_t>(prng())<<32U) | prng();
        if (value>=threshold) return value%modulus;
    }
}

bool ExpUnitCoin64(std::uint64_t numerator, std::uint64_t denominator) {
    std::uint64_t k=1;
    bool odd=true;
    while (k<=std::numeric_limits<std::uint64_t>::max()/denominator) {
        const auto d=denominator*k;
        if (numerator==0 || (numerator!=d && UniformBelow64(d)>=numerator)) return odd;
        odd=!odd;
        if (k==std::numeric_limits<std::uint64_t>::max()) break;
        ++k;
    }
    // Continue the same infinite series if a machine denominator would overflow.
    Integer bigK(k);
    if (k==std::numeric_limits<std::uint64_t>::max()) bigK+=Integer(1);
    while (RationalCoin(Integer(numerator),Integer(denominator)*bigK)) {
        odd=!odd;bigK+=Integer(1);
    }
    return odd;
}

std::int64_t SmallGaussian(const ExactGaussianParameters& pars) {
    const auto q=pars.qNative;
    const std::uint64_t denominator=pars.smallDenominator;
    while (true) {
        const auto value=static_cast<std::int64_t>(UniformBelow64(2*pars.bound+1))-
                         static_cast<std::int64_t>(pars.bound);
        const auto magnitude=static_cast<std::uint64_t>(value<0 ? -value : value);
        std::uint64_t numerator=magnitude*magnitude*q*q;
        bool accepted=true;
        while (numerator>=denominator) {
            if (!ExpUnitCoin64(denominator,denominator)) { accepted=false;break; }
            numerator-=denominator;
        }
        if (accepted && ExpUnitCoin64(numerator,denominator)) return value;
    }
}

exact_integer::Integer ExactGaussianUnlocked(const ExactGaussianParameters& pars) {
    // Supported q is 1, 5 or 100; these limits keep every proposal product in u64.
    if (pars.small) return SmallGaussian(pars);
    // All remaining supported sigmas are integers. An exact discrete-Laplace
    // proposal avoids ~19 uniform proposals per large-sigma Gaussian sample.
    if (pars.pNative == 0) {
        const Integer& sigma=pars.p;
        while (true) {
            Integer magnitude;
            do { magnitude=SampleUniformBelowUnlocked(sigma); }
            while (!ExpUnitCoin(magnitude,sigma));
            while (ExpUnitCoin64(1,1)) {
                magnitude+=sigma;
                if (magnitude>pars.bigBound) break;
            }
            if (magnitude>pars.bigBound) continue;
            const bool negative=UniformBelow64(2)!=0;
            if (magnitude==Integer(0) && negative) continue;
            const Integer distance=magnitude>=sigma ? magnitude-sigma : sigma-magnitude;
            if (ExpCoin(distance*distance,pars.denominator)) {
                const exact_integer::Integer value(magnitude.ToString());
                return negative ? -value : value;
            }
        }
    }
    const auto sigma=pars.pNative;
    const Integer& denominator=pars.denominator;
    while (true) {
        std::uint64_t magnitude;
        do { magnitude=UniformBelow64(sigma); }
        while (!ExpUnitCoin64(magnitude,sigma));
        while (ExpUnitCoin64(1,1)) {
            magnitude+=sigma;
            if (magnitude>pars.bound) break;
        }
        if (magnitude>pars.bound) continue;
        const bool negative=UniformBelow64(2)!=0;
        if (magnitude==0 && negative) continue;
        const Integer distance(magnitude>=sigma ? magnitude-sigma : sigma-magnitude);
        // exp(-|x|/sigma) * exp(-(|x|-sigma)^2/(2*sigma^2))
        // is a constant times exp(-x^2/(2*sigma^2)).
        if (ExpCoin(distance*distance,denominator))
            return negative ? -static_cast<std::int64_t>(magnitude) : static_cast<std::int64_t>(magnitude);
    }
}
} // namespace

exact_integer::Integer SampleCertifiedV17GaussianInteger(long double standardDeviation) {
    const ExactGaussianParameters pars(standardDeviation);
#if defined(WITH_OPENMP) && defined(FIXED_SEED)
    std::lock_guard<std::mutex> lock(ProtocolPrngMutex());
#endif
    return ExactGaussianUnlocked(pars);
}

std::int64_t SampleCertifiedV17Gaussian(long double standardDeviation) {
    // Compatibility API: reject before sampling rather than truncate a big sample.
    const ExactGaussianParameters pars(standardDeviation);
    if (pars.bigBound > Integer(static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        throw std::invalid_argument("use SampleCertifiedV17GaussianInteger for large sigma");
    return SampleCertifiedV17GaussianInteger(standardDeviation).convert_to<std::int64_t>();
}

lbcrypto::DCRTPoly SampleProtocolGaussianPolynomial(
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    long double standardDeviation,
    bool certifiedV17) {

    codec_timing::Scope timer(codec_timing::Gaussian);
    if (!parameters || !std::isfinite(standardDeviation) ||
        standardDeviation <= 0.0L ||
        standardDeviation > std::ldexp(1.0L, certifiedV17 ? 128 : 59)) {
        throw std::invalid_argument(
            "invalid protocol Gaussian standard deviation");
    }

    const double sigma = static_cast<double>(standardDeviation);
#if defined(WITH_OPENMP) && defined(FIXED_SEED)
    // OpenFHE deliberately shares its deterministic PRNG in FIXED_SEED builds.
    // Keep that diagnostic configuration race-free, even though it cannot
    // benefit from concurrent Gaussian sampling.
    std::lock_guard<std::mutex> lock(ProtocolPrngMutex());
#endif
    if (certifiedV17) {
        const ExactGaussianParameters pars(standardDeviation);
        std::vector<exact_integer::Integer> coefficients(parameters->GetRingDimension());
        {
            codec_timing::Scope sampling(pars.small ? codec_timing::GaussianSmall : codec_timing::GaussianLarge);
            for (auto& value : coefficients) value = ExactGaussianUnlocked(pars);
        }
        lbcrypto::DCRTPoly result(parameters, Format::COEFFICIENT, true);
        // The generic vector assignment assumes |x| < each tower modulus.
        // Smudging exceeds individual RNS primes: reduce before signed lifting.
        {
        codec_timing::Scope lifting(codec_timing::GaussianRns);
        for (auto& tower : result.GetAllElements()) {
            const auto modulus = tower.GetModulus();
            for (std::size_t i=0;i<coefficients.size();++i) {
                const auto residue=exact_integer::Residue(coefficients[i], Integer(modulus.ConvertToInt()));
                tower[i]=lbcrypto::NativeInteger(residue.ConvertToInt());
            }
        }
        }
        {
            codec_timing::Scope transform(codec_timing::GaussianNtt);
            result.SetFormat(Format::EVALUATION);
        }
        return result;
    }
    return SampleWithThreadGenerator(parameters, sigma);
}

bool ProtocolGaussianSamplingSupportsParallelism() noexcept {
#if defined(WITH_OPENMP) && defined(FIXED_SEED)
    return false;
#else
    return true;
#endif
}

Seed128 SampleFreshProtocolSeed128() {
#if defined(WITH_OPENMP) && defined(FIXED_SEED)
    std::lock_guard<std::mutex> lock(ProtocolPrngMutex());
#endif
    return SampleFreshSeedUnlocked();
}

lbcrypto::BigInteger SampleProtocolUniformBelow(
    const lbcrypto::BigInteger& modulus) {

    if (modulus < lbcrypto::BigInteger(2)) {
        throw std::invalid_argument(
            "private uniform modulus must be at least two");
    }
#if defined(WITH_OPENMP) && defined(FIXED_SEED)
    std::lock_guard<std::mutex> lock(ProtocolPrngMutex());
#endif
    return SampleUniformBelowUnlocked(modulus);
}

}  // namespace federated
