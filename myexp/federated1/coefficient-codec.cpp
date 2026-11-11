#include "protocol-rounding.h"
#include "coefficient-codec.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace coefficient_codec {
namespace {

void ValidateScalingFactor(double scalingFactor) {
    if (!std::isfinite(scalingFactor) ||
        scalingFactor <= 0.0) {

        throw std::invalid_argument(
            "scalingFactor must be finite and positive");
    }
}

exact_integer::Integer IntegerScale(double scale) {
    ValidateScalingFactor(scale);
    if (std::floor(scale)!=scale) throw std::invalid_argument("coefficient scale must be integral");
    return exact_integer::RoundProduct(scale, exact_integer::Integer(1));
}

double RestoreCoefficient(
    const lbcrypto::BigInteger& residue,
    const lbcrypto::BigInteger& modulus,
    double scalingFactor) {

    const lbcrypto::BigInteger half =
        modulus >> 1;

    const bool negative =
        residue > half;

    const lbcrypto::BigInteger magnitude =
        negative
            ? modulus - residue
            : residue;

    long double value = std::stold(magnitude.ToString());

    if (negative) {
        value = -value;
    }

    value /=
        static_cast<long double>(
            scalingFactor);

    return static_cast<double>(value);
}

}  // namespace

double ScalingFactorFromBits(uint32_t scaleBits) {
    if (scaleBits >=
        static_cast<uint32_t>(
            std::numeric_limits<double>::max_exponent)) {

        throw std::invalid_argument(
            "scaleBits is too large");
    }

    return std::ldexp(
        1.0,
        static_cast<int>(scaleBits));
}

void ValidateEncodableExact(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values,
    const exact_integer::Integer& scalingFactor) {

    if (!cc) {
        throw std::invalid_argument(
            "crypto context is null");
    }
    if (!exact_integer::ValidScale(scalingFactor)) throw std::invalid_argument("invalid scale");
    if (values.size() > cc->GetRingDimension()) {
        throw std::invalid_argument(
            "the number of input values exceeds the ring dimension");
    }
    const auto paramsQ = cc->GetCryptoParameters()->GetElementParams();
    if (paramsQ->GetParams().empty()) {
        throw std::runtime_error(
            "the crypto context has no CRT moduli");
    }
    const lbcrypto::BigInteger halfModulusQ = paramsQ->GetModulus() >> 1;
    for (const double value : values) {
        const auto rounded = exact_integer::RoundProduct(value, scalingFactor);
        if (lbcrypto::BigInteger(exact_integer::Integer(rounded<0 ? -rounded:rounded).str()) > halfModulusQ) {
            throw std::overflow_error(
                "scaled coefficient exceeds the centered plaintext range");
        }
    }
}

lbcrypto::DCRTPoly EncodeExact(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values,
    const exact_integer::Integer& scalingFactor) {

    if (!cc) {
        throw std::invalid_argument(
            "crypto context is null");
    }

    ValidateEncodableExact(cc, values, scalingFactor);

    const size_t ringDimension =
        cc->GetRingDimension();

    if (values.size() > ringDimension) {
        throw std::invalid_argument(
            "the number of input values exceeds "
            "the ring dimension");
    }

    const auto paramsQ =
        cc->GetCryptoParameters()
            ->GetElementParams();

    const auto& towerParams =
        paramsQ->GetParams();

    if (towerParams.empty()) {
        throw std::runtime_error(
            "the crypto context has no CRT moduli");
    }

    std::vector<lbcrypto::NativeInteger> moduli;
    moduli.reserve(
        towerParams.size());

    for (const auto& towerParam : towerParams) {
        moduli.push_back(
            towerParam->GetModulus());
    }

    const lbcrypto::BigInteger halfModulusQ =
        paramsQ->GetModulus() >> 1;

    lbcrypto::DCRTPoly encoded(
        paramsQ,
        Format::COEFFICIENT,
        true);

    auto& towers =
        encoded.GetAllElements();

    std::vector<lbcrypto::NativeVector>
        valuesPerTower;

    valuesPerTower.reserve(
        moduli.size());

    for (const auto& modulus : moduli) {
        valuesPerTower.emplace_back(
            ringDimension,
            modulus);
    }

    for (size_t coefficientIndex = 0;
         coefficientIndex < ringDimension;
         ++coefficientIndex) {

        // Coefficients beyond the input vector
        // are zero-padded.
        const double value =
            coefficientIndex < values.size()
                ? values[coefficientIndex]
                : 0.0;

        const auto rounded = exact_integer::RoundProduct(value, scalingFactor);
        const exact_integer::Integer magnitudeValue = rounded<0 ? -rounded:rounded;
        const lbcrypto::BigInteger magnitude(magnitudeValue.str());
        if (magnitude > halfModulusQ) throw std::overflow_error("scaled coefficient exceeds centered plaintext range");

        for (size_t towerIndex = 0;
             towerIndex < moduli.size();
             ++towerIndex) {

            const auto& modulus =
                moduli[towerIndex];

            lbcrypto::NativeInteger residue =
                lbcrypto::NativeInteger(magnitude.Mod(
                    lbcrypto::BigInteger(modulus.ConvertToInt())).ConvertToInt());

            if (rounded < 0 &&
                residue !=
                    lbcrypto::NativeInteger(0)) {

                // Represent a negative coefficient -r modulo q
                // by q-r.
                residue =
                    modulus - residue;
            }

            valuesPerTower[towerIndex]
                          [coefficientIndex] =
                residue;
        }
    }

    for (size_t towerIndex = 0;
         towerIndex < towers.size();
         ++towerIndex) {

        towers[towerIndex].SetValues(
            std::move(
                valuesPerTower[towerIndex]),
            Format::COEFFICIENT);
    }

    return encoded;
}

void ValidateEncodable(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values, double scale) {
    ValidateEncodableExact(cc, values, IntegerScale(scale));
}
lbcrypto::DCRTPoly Encode(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values, double scale) {
    return EncodeExact(cc, values, IntegerScale(scale));
}

lbcrypto::Plaintext MakePlaintext(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<double>& values,
    uint32_t scaleBits) {

    if (!cc) {
        throw std::invalid_argument(
            "crypto context is null");
    }

    if (scaleBits >=
        std::numeric_limits<int64_t>::digits) {

        throw std::invalid_argument(
            "scaleBits is too large");
    }

    const double scalingFactor =
        ScalingFactorFromBits(
            scaleBits);

    const lbcrypto::NativeInteger
        scalingFactorInt =
            lbcrypto::NativeInteger(1)
            << scaleBits;

    auto plaintext =
        std::make_shared<
            lbcrypto::CKKSPackedEncoding>(
                cc->GetElementParams(),
                cc->GetEncodingParams(),
                lbcrypto::REAL);

    plaintext->SetLevel(0);
    plaintext->SetNoiseScaleDeg(1);

    plaintext->SetScalingFactor(
        scalingFactor);

    plaintext->SetScalingFactorInt(
        scalingFactorInt);

    auto encoded =
        Encode(
            cc,
            values,
            scalingFactor);

    /* OpenFHE encryption-of-zero is in EVALUATION form; convert the plaintext polynomial before adding it to the ciphertext's first component. */
    encoded.SetFormat(
        Format::EVALUATION);

    plaintext
        ->GetElement<lbcrypto::DCRTPoly>() =
        std::move(encoded);

    return plaintext;
}

std::vector<double> Decode(
    lbcrypto::Poly polynomial,
    double scalingFactor) {

    ValidateScalingFactor(
        scalingFactor);

    polynomial.SetFormat(
        Format::COEFFICIENT);

    const auto& residues =
        polynomial.GetValues();

    const auto& modulus =
        polynomial.GetModulus();

    const size_t ringDimension =
        residues.GetLength();

    std::vector<double> decoded(
        ringDimension);

    for (size_t index = 0;
         index < ringDimension;
         ++index) {

        decoded[index] =
            RestoreCoefficient(
                residues[index],
                modulus,
                scalingFactor);
    }

    return decoded;
}

std::vector<double> Decode(
    lbcrypto::DCRTPoly polynomial,
    double scalingFactor) {

    ValidateScalingFactor(
        scalingFactor);

    polynomial.SetFormat(
        Format::COEFFICIENT);

    /* CRT-interpolate all towers into a polynomial modulo Q. */
    lbcrypto::Poly interpolated =
        polynomial.CRTInterpolate();

    return Decode(
        std::move(interpolated),
        scalingFactor);
}

std::vector<double> Decrypt(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ciphertext,
    const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
    double scalingFactor) {

    if (!cc) {
        throw std::invalid_argument(
            "crypto context is null");
    }

    if (!ciphertext) {
        throw std::invalid_argument(
            "ciphertext is null");
    }

    if (!privateKey) {
        throw std::invalid_argument(
            "private key is null");
    }

    ValidateScalingFactor(
        scalingFactor);

    /* The Poly output overload CRT-interpolates all remaining towers into modulus Q. */
    lbcrypto::Poly decryptedPolynomial;

    const lbcrypto::DecryptResult result =
        cc->GetScheme()->Decrypt(
            ciphertext,
            privateKey,
            &decryptedPolynomial);

    if (!result.isValid) {
        throw std::runtime_error(
            "OpenFHE decryption failed");
    }

    return Decode(
        std::move(decryptedPolynomial),
        scalingFactor);
}

}  // namespace coefficient_codec
