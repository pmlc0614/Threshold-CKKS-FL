#ifndef FEDERATED_EXACT_INTEGER_H
#define FEDERATED_EXACT_INTEGER_H
#include "openfhe.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace exact_integer {
using Integer = boost::multiprecision::cpp_int;
// Bounded public scale format; enough for the supported RNS parameter space.
inline bool ValidScale(const Integer& scale) { return scale > 0 && scale < (Integer(1) << 256); }
inline Integer ParseScale(const std::string& text) {
    if (text.empty() || text.size()>78 || text.front()=='0') throw std::invalid_argument("noncanonical integer scale");
    Integer value=0;
    for(char c:text) {
        if(c<'0'||c>'9') throw std::invalid_argument("scale must be a positive decimal integer");
        value=value*10+(c-'0');
    }
    if(!ValidScale(value)) throw std::invalid_argument("scale must be in [1, 2^256)");
    return value;
}
// Exact arithmetic on the supplied binary64 input, with floor(x + 1/2) ties.
inline Integer RoundProduct(double value, const Integer& scale) {
    if(!std::isfinite(value) || !ValidScale(scale)) throw std::invalid_argument("invalid coefficient or integer scale");
    if(value==0) return 0;
    int exponent=0;
    double mantissa=std::frexp(std::abs(value), &exponent);
    Integer product=Integer(static_cast<std::uint64_t>(std::ldexp(mantissa,53)))*scale;
    exponent-=53;
    if(exponent>=0) product <<= exponent;
    else {
        const Integer divisor=Integer(1)<<(-exponent);
        const Integer remainder=product%divisor;
        product/=divisor;
        if(value<0 ? 2*remainder>divisor : 2*remainder>=divisor) ++product;
    }
    return value<0 ? -product : product;
}
inline lbcrypto::BigInteger Residue(const Integer& value, const lbcrypto::BigInteger& modulus) {
    const Integer magnitude=value<0 ? -value:value;
    auto result=lbcrypto::BigInteger(magnitude.str()).Mod(modulus);
    return value<0 && result!=lbcrypto::BigInteger(0) ? modulus-result:result;
}
}
#endif
