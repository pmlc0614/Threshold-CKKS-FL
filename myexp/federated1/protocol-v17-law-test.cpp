#include "protocol-v17-law-test.h"
#include "coefficient-compression.h"
#include "protocol-randomness.h"
#include "protocol-stage1-test.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using lbcrypto::BigInteger;
using namespace federated;
using namespace federated::coefficient_compression;

long double Centered(const BigInteger& x, const BigInteger& q) {
    const BigInteger reduced = x.Mod(q);
    return reduced > q / BigInteger(2)
        ? -static_cast<long double>((q - reduced).ConvertToDouble())
        : static_cast<long double>(reduced.ConvertToDouble());
}

struct Statistics {
    std::vector<long double> values;
    std::size_t algebraFailures = 0;
    std::size_t supportViolations = 0;

    void Write(std::ostream& out, std::size_t trials) const {
        long double sum = 0, squares = 0;
        std::vector<long double> absolute;
        absolute.reserve(values.size());
        for (const auto x : values) {
            sum += x; squares += x*x; absolute.push_back(std::abs(x));
        }
        std::sort(absolute.begin(), absolute.end());
        const long double mean = sum / values.size();
        const auto quantile = [&](long double p) {
            return absolute[static_cast<std::size_t>(std::ceil(p*absolute.size()))-1];
        };
        out << "{\"sample_count\":" << values.size()
            << ",\"independent_trials\":" << trials
            << ",\"mean\":" << mean
            << ",\"stddev\":" << std::sqrt(std::max(0.0L, squares/values.size()-mean*mean))
            << ",\"second_moment\":" << squares/values.size()
            << ",\"q99\":" << quantile(0.99L)
            << ",\"q999\":" << quantile(0.999L)
            << ",\"max_abs\":" << absolute.back()
            << ",\"algebra_failures\":" << algebraFailures
            << ",\"support_violations\":" << supportViolations << "}";
    }
};

std::uint64_t OracleCode(const BigInteger& x, const BigInteger& coin,
    const BigInteger& q, const BigInteger& k) {
    const BigInteger product = x*k;
    BigInteger code = product/q;
    if (coin < product.Mod(q)) code += BigInteger(1);
    return code.Mod(k).ConvertToInt();
}

BigInteger OracleReconstruction(const BigInteger& code,
    const BigInteger& q, const BigInteger& k) {
    return ((code.Mod(k)*q + k/BigInteger(2))/k).Mod(q);
}

Statistics Measure(const BigInteger& q, std::uint32_t bits,
    std::size_t cohort, std::size_t count, bool codeSum) {
    const BigInteger k = BigInteger(1).LShift(bits);
    const long double step = q.ConvertToDouble()/std::ldexp(1.0L, bits);
    Statistics result;
    result.values.reserve(count);
    for (std::size_t sample = 0; sample < count; ++sample) {
        BigInteger sumX(0), sumCodes(0);
        std::vector<std::uint64_t> codes;
        for (std::size_t client = 0; client < cohort; ++client) {
            // In E4 x is the actual DD-quantizer input: uniform residue after dither.
            const auto x = SampleProtocolUniformBelow(q);
            const auto coin = SampleProtocolUniformBelow(q);
            const auto code = CompressStochasticCoefficientWithCoin(x,q,bits,coin);
            result.algebraFailures += code != OracleCode(x,coin,q,k);
            codes.push_back(code); sumCodes += BigInteger(code);
            sumX = sumX.ModAdd(x,q);
        }
        const auto actual = codeSum ? ReconstructStochasticCodeSum(codes,q,bits)
                                   : DecompressCoefficient(codes[0],q,bits);
        result.algebraFailures += actual != OracleReconstruction(sumCodes,q,k);
        const auto error = Centered(actual.ModSub(sumX,q),q);
        result.values.push_back(error);
        // Only this floating diagnostic gets a representational tolerance;
        // exact code/reconstruction equality above has no tolerance.
        const auto bound = std::min(q.ConvertToDouble()/2.0L, cohort*step+0.5L);
        const auto tolerance = 8*std::numeric_limits<double>::epsilon()*q.ConvertToDouble();
        result.supportViolations += std::abs(error)>bound+tolerance;
    }
    return result;
}
} // namespace

bool MeasureV17CodecLaws(const federated::ProtocolExperimentConfig& config,
    const federated::ProtocolSetupResult& setup, std::size_t trials,
    const std::string& outputPath) {
    if (!federated::UsesV17WireProtocol(config.protocolVersion) || trials == 0 ||
        trials > std::numeric_limits<std::size_t>::max()/config.modulus.ringDimension)
        throw std::invalid_argument("invalid v17 codec measurement plan");
    std::ofstream out(outputPath);
    if (!out) throw std::runtime_error("cannot write codec law diagnostics");
    bool passed = RunProtocolStage1Tests() == 0;
    out << std::scientific << std::setprecision(17)
        << "{\"schema\":\"v17-native-codec-diagnostics-v1\","
        << "\"e1_regressions_passed\":" << (passed ? "true" : "false")
        << ",\"seed_rule\":\"fresh private OpenFHE PRNG per input and coin\","
        << "\"scope\":\"honest-coin samples; all-coin support checked; no rare-tail proof\","
        << "\"measurements\":[";
    const auto q=setup.common.rns.paramsQ->GetModulus();
    bool first=true;
    for (const bool aux : {false,true}) {
        for (const bool c2 : {false,true}) {
            for (const auto cohort : {config.sparsification.minimumCohort, config.clientCount}) {
                if (!c2 && cohort != config.sparsification.minimumCohort) continue;
                const auto bits=c2 ? (aux ? config.compression.auxC2 : config.compression.c2)
                                   : (aux ? config.compression.auxC1 : config.compression.c1);
                const auto statistics=Measure(q,bits,c2 ? cohort : 1,
                    trials*config.modulus.ringDimension,c2);
                passed &= statistics.algebraFailures==0 && statistics.supportViolations==0;
                out << (first ? "" : ",") << "{\"path\":\"" << (aux ? "aux" : "model")
                    << "\",\"stage\":\"" << (c2 ? "E4" : "E3")
                    << "\",\"cohort\":" << (c2 ? cohort : 1) << ",\"bits\":" << bits
                    << ",\"statistics\":";
                statistics.Write(out,trials); out << "}"; first=false;
            }
        }
    }
    out << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
    return passed;
}
