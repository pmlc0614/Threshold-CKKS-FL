#ifndef FEDERATED1_PROTOCOL_ROUNDING_H
#define FEDERATED1_PROTOCOL_ROUNDING_H
#include <cmath>
// nint(x) = floor(x + 1/2), including negative ties. Avoid adding 1/2
// directly to large integral inputs, where that addition can lose precision.
namespace protocol_rounding {
inline long double NearestInteger(long double value) {
    const long double lower = std::floor(value);
    return value - lower >= 0.5L ? lower + 1.0L : lower;
}
}
#endif
