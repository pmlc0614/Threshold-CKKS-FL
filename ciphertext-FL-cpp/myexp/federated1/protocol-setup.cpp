#include "protocol-setup.h"

#include "protocol-randomness.h"
#include "support-bitmap.h"

#include "deterministic-randomness.h"
#include "ring-conversion.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace federated {
namespace {

using DCRTPoly = lbcrypto::DCRTPoly;
using PrivateKey = lbcrypto::PrivateKey<DCRTPoly>;

SetupId FreshUniqueV17SetupId() {
    static std::mutex mutex;
    static std::set<Seed128> used;
    std::lock_guard<std::mutex> lock(mutex);
    for (;;) {
        const Seed128 candidate = SampleFreshProtocolSeed128();
        if (used.insert(candidate).second) {
            return SetupId{candidate};
        }
    }
}

void ValidateCryptoContext(
    const lbcrypto::CryptoContext<DCRTPoly>& cc);

std::vector<PrivateKey> GenerateClientSecretKeysWithTiming(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    std::size_t clientCount,
    std::vector<double>* clientSeconds) {

    ValidateCryptoContext(cc);
    if (clientCount == 0) {
        throw std::invalid_argument("client count must be positive");
    }
    if (clientSeconds) {
        clientSeconds->clear();
        clientSeconds->reserve(clientCount);
    }

    std::vector<PrivateKey> secretKeys;
    secretKeys.reserve(clientCount);
    for (std::size_t clientIndex = 0;
         clientIndex < clientCount;
         ++clientIndex) {

        const auto started = std::chrono::steady_clock::now();
        const auto keyPair = cc->KeyGen();
        if (!keyPair.good()) {
            throw std::runtime_error(
                "client key generation failed at index " +
                std::to_string(clientIndex));
        }
        secretKeys.push_back(keyPair.secretKey);
        if (clientSeconds) {
            clientSeconds->push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count());
        }
    }
    return secretKeys;
}

void ValidateCryptoContext(
    const lbcrypto::CryptoContext<DCRTPoly>& cc) {

    if (!cc) {
        throw std::invalid_argument("crypto context is null");
    }
    const auto parameters = std::dynamic_pointer_cast<
        lbcrypto::CryptoParametersRLWE<DCRTPoly>>(cc->GetCryptoParameters());
    if (!parameters || parameters->GetSecretKeyDist() != lbcrypto::UNIFORM_TERNARY) {
        throw std::invalid_argument("SD setup requires UNIFORM_TERNARY key generation");
    }
#if defined(WITH_OPENMP) && defined(FIXED_SEED)
    throw std::invalid_argument("SD setup forbids deterministic FIXED_SEED PRNG builds");
#endif
}

void ValidatePrivateKey(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const PrivateKey& privateKey,
    std::size_t clientIndex) {

    if (!privateKey) {
        throw std::invalid_argument(
            "secret key is null for client " +
            std::to_string(clientIndex));
    }

    if (privateKey->GetCryptoContext() != cc) {
        throw std::invalid_argument(
            "secret key belongs to a different context for client " +
            std::to_string(clientIndex));
    }
}

void ValidateRnsParameters(
    const ProtocolRnsParameters& parameters) {

    if (!parameters.paramsQ ||
        !parameters.paramsP ||
        !parameters.paramsQP) {

        throw std::invalid_argument("protocol RNS parameters are incomplete");
    }

    if (parameters.qTowerCount == 0 ||
        parameters.pTowerCount != 1 ||
        parameters.decompositionDigitCount !=
            parameters.qTowerCount ||
        parameters.qTowersPerDigit != 1) {

        throw std::invalid_argument(
            "protocol RNS profile requires one Q tower per digit and one P tower");
    }

    if (parameters.pModQ.size() != parameters.qTowerCount) {
        throw std::invalid_argument("P mod Q table has an invalid size");
    }

    if (parameters.paramsQ->GetParams().size() !=
            parameters.qTowerCount ||
        parameters.paramsP->GetParams().size() !=
            parameters.pTowerCount ||
        parameters.paramsQP->GetParams().size() !=
            parameters.qTowerCount + parameters.pTowerCount) {

        throw std::invalid_argument(
            "protocol RNS parameter tower counts are inconsistent");
    }

    const std::uint32_t cyclotomicOrder =
        parameters.paramsQ->GetCyclotomicOrder();

    if (parameters.paramsP->GetCyclotomicOrder() != cyclotomicOrder ||
        parameters.paramsQP->GetCyclotomicOrder() != cyclotomicOrder) {

        throw std::invalid_argument(
            "Q, P, and QP must use the same cyclotomic order");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < parameters.qTowerCount;
         ++towerIndex) {

        if (parameters.paramsQ->GetParams()[towerIndex]->GetModulus() !=
            parameters.paramsQP->GetParams()[towerIndex]->GetModulus()) {

            throw std::invalid_argument(
                "QP does not begin with the configured Q basis");
        }
    }

    if (parameters.paramsP->GetParams()[0]->GetModulus() !=
        parameters.paramsQP->GetParams()[parameters.qTowerCount]
            ->GetModulus()) {

        throw std::invalid_argument(
            "QP does not end with the configured P basis");
    }
}

ByteVector SetupTagContext(
    const SetupId& setupId,
    const std::vector<std::uint64_t>& indices = {}) {

    ByteVector context(setupId.bytes.begin(), setupId.bytes.end());

    for (const std::uint64_t index : indices) {
        const ByteVector encoded =
            deterministic_randomness::EncodeBigEndian64(index);

        context.insert(context.end(), encoded.begin(), encoded.end());
    }

    return context;
}

DCRTPoly GenerateUniformDCRT(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    const Seed128& seed,
    const std::string& domain,
    const ByteVector& context = {}) {

    if (!parameters) {
        throw std::invalid_argument("uniform DCRT parameters are null");
    }

    const std::size_t ringDimension = parameters->GetRingDimension();
    DCRTPoly result(parameters, Format::COEFFICIENT, true);
    auto& towers = result.GetAllElements();

    if (towers.size() != parameters->GetParams().size()) {
        throw std::runtime_error("uniform DCRT tower count mismatch");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < towers.size();
         ++towerIndex) {

        const lbcrypto::NativeInteger modulus =
            parameters->GetParams()[towerIndex]->GetModulus();

        ByteVector tag(domain.begin(), domain.end());
        tag.insert(tag.end(), context.begin(), context.end());

        const auto coefficients =
            deterministic_randomness::GenerateUniformPolynomial(
                seed,
                tag,
                ringDimension,
                modulus.ConvertToInt());

        lbcrypto::NativeVector values(ringDimension, modulus);

        for (std::size_t coefficientIndex = 0;
             coefficientIndex < ringDimension;
             ++coefficientIndex) {

            values[coefficientIndex] =
                lbcrypto::NativeInteger(coefficients[coefficientIndex]);
        }

        towers[towerIndex].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }

    result.SetFormat(Format::EVALUATION);
    return result;
}

DCRTPoly SampleError(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    long double standardDeviation, bool certifiedV17) {

    if (!parameters ||
        !std::isfinite(standardDeviation) ||
        standardDeviation <= 0.0L ||
        standardDeviation > std::ldexp(1.0L, 59)) {

        throw std::invalid_argument("invalid setup error standard deviation");
    }

    return SampleProtocolGaussianPolynomial(parameters, standardDeviation, certifiedV17);
}

void AppendDescriptorInteger(
    ByteVector* descriptor,
    std::uint64_t value) {

    const ByteVector encoded =
        deterministic_randomness::EncodeBigEndian64(value);
    descriptor->insert(descriptor->end(), encoded.begin(), encoded.end());
}

void AppendDescriptorString(
    ByteVector* descriptor,
    const std::string& value) {

    AppendDescriptorInteger(descriptor, value.size());
    descriptor->insert(descriptor->end(), value.begin(), value.end());
}

void AppendDescriptorIntegers(
    ByteVector* descriptor,
    const std::vector<std::uint32_t>& values) {

    AppendDescriptorInteger(descriptor, values.size());
    for (const std::uint32_t value : values) {
        AppendDescriptorInteger(descriptor, value);
    }
}

void AppendDescriptorReal(
    ByteVector* descriptor,
    long double value) {

    static_assert(sizeof(double) == sizeof(std::uint64_t));
    const double narrowed = static_cast<double>(value);
    std::uint64_t bits = 0;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    AppendDescriptorInteger(descriptor, bits);
}

std::vector<std::vector<DCRTPoly>> GenerateCommonM2RA(
    const CommonSetupMaterial& publicMaterial) {

    std::vector<std::vector<DCRTPoly>> result(
        publicMaterial.mlweRank);

    for (std::size_t splitIndex = 0;
         splitIndex < publicMaterial.mlweRank;
         ++splitIndex) {

        result[splitIndex].reserve(
            publicMaterial.rns.decompositionDigitCount);

        for (std::size_t lambda = 0;
             lambda < publicMaterial.rns.decompositionDigitCount;
             ++lambda) {

            const ByteVector context = SetupTagContext(
                publicMaterial.setupId,
                {splitIndex, lambda});

            result[splitIndex].push_back(
                GenerateUniformDCRT(
                    publicMaterial.rns.paramsQP,
                    publicMaterial.crsSeed,
                    "M2R-A",
                    context));
        }
    }

    return result;
}

ClientCachedSetupShare GenerateCachedClientShare(
    const CommonSetupMaterial& publicMaterial,
    const SparseSetupMaterial* sparseMaterial,
    const ClientSecretSetupMaterial& clientSecret,
    const SetupNoiseParameters& noiseParameters) {

    ClientCachedSetupShare share;
    share.clientId = clientSecret.clientId;

    const DCRTPoly publicKeyError = SampleError(
        publicMaterial.rns.paramsQP,
        noiseParameters.publicKeyShareStandardDeviation, publicMaterial.certifiedV17Gaussian);

    share.publicKeyP0Share =
        -(publicMaterial.publicKeyP1 * clientSecret.secretKeyQP) +
        publicKeyError;

    if (!sparseMaterial || sparseMaterial->m2rCommonA.empty()) {
        return share;
    }

    CachedM2RKeyShare m2rShare;
    m2rShare.clientId = clientSecret.clientId;
    m2rShare.b.resize(publicMaterial.mlweRank);

    const auto splitSecrets = ring_conversion::Split(
        clientSecret.secretKeyQ->GetPrivateElement(),
        publicMaterial.mlweRank);

    if (splitSecrets.size() != publicMaterial.mlweRank) {
        throw std::runtime_error("unexpected number of split secret components");
    }

    for (std::size_t splitIndex = 0;
         splitIndex < publicMaterial.mlweRank;
         ++splitIndex) {

        DCRTPoly embedded = ring_conversion::RingEmbedding(
            splitSecrets[splitIndex],
            publicMaterial.mlweRank);

        m2rShare.b[splitIndex].reserve(
            publicMaterial.rns.decompositionDigitCount);

        for (std::size_t lambda = 0;
             lambda < publicMaterial.rns.decompositionDigitCount;
             ++lambda) {

            const DCRTPoly gadgetTerm = BuildM2RGadgetTerm(
                embedded,
                lambda,
                publicMaterial.rns);

            const DCRTPoly error = SampleError(
                publicMaterial.rns.paramsQP,
                noiseParameters.m2rKeyShareStandardDeviation, publicMaterial.certifiedV17Gaussian);

            const DCRTPoly& commonA =
                sparseMaterial->m2rCommonA[splitIndex][lambda];

            m2rShare.b[splitIndex].push_back(
                -(commonA * clientSecret.modelKeyQP) +
                gadgetTerm + error);
        }
    }

    share.m2rShare = std::move(m2rShare);
    return share;
}

}  // namespace

long double ComputeHybridKeySwitchGain(
    const ProtocolRnsParameters& parameters) {

    if (!parameters.paramsQ || !parameters.paramsP ||
        parameters.paramsQ->GetParams().empty() ||
        parameters.paramsP->GetParams().empty() ||
        parameters.paramsQ->GetRingDimension() == 0) {
        throw std::invalid_argument(
            "hybrid gain requires non-empty Q and P towers");
    }

    long double squaredQTowerNorm = 0.0L;
    for (const auto& tower : parameters.paramsQ->GetParams()) {
        const long double modulus = static_cast<long double>(
            tower->GetModulus().ConvertToInt());
        squaredQTowerNorm += modulus * modulus;
    }
    const long double specialModulus = std::stold(
        parameters.paramsP->GetModulus().ToString());
    const long double ringDimension = static_cast<long double>(
        parameters.paramsQ->GetRingDimension());
    if (!std::isfinite(squaredQTowerNorm) ||
        squaredQTowerNorm <= 0.0L ||
        !std::isfinite(specialModulus) || specialModulus <= 0.0L) {
        throw std::invalid_argument("invalid modulus in hybrid gain");
    }
    return std::sqrt(ringDimension / 12.0L) *
        std::sqrt(squaredQTowerNorm) / specialModulus;
}

ByteVector CanonicalSetupDescriptor(
    const ProtocolExperimentConfig& config,
    const ProtocolRnsParameters& parameters) {

    const ValidationReport validation = ValidateProtocolParameters(config);
    validation.ThrowIfInvalid();
    ValidateRnsParameters(parameters);

    ByteVector descriptor;
    AppendDescriptorString(&descriptor, "FEDERATED-SETUP-DESCRIPTOR-2");
    AppendDescriptorString(&descriptor, config.protocolVersion);
    AppendDescriptorInteger(
        &descriptor,
        config.mode == ProtocolMode::SP ? 0U : 1U);
    if (config.protocolVersion == "v17") {
        descriptor.push_back(
            config.mode == ProtocolMode::SP
                ? kChunkBitmapMsb0V1
                : kFullSupportCodec);
    }
    AppendDescriptorInteger(&descriptor, config.clientCount);
    AppendDescriptorInteger(&descriptor, config.modelDimension);
    AppendDescriptorInteger(&descriptor, config.modulus.ringDimension);
    if (UsesCertifiedGaussianProfile(config.protocolVersion)) {
        const auto& qParameters = parameters.paramsQ->GetParams();
        const auto& pParameters = parameters.paramsP->GetParams();
        AppendDescriptorInteger(&descriptor, qParameters.size());
        for (const auto& tower : qParameters) {
            AppendDescriptorInteger(
                &descriptor,
                tower->GetModulus().ConvertToInt());
        }
        AppendDescriptorInteger(&descriptor, pParameters.size());
        for (const auto& tower : pParameters) {
            AppendDescriptorInteger(
                &descriptor,
                tower->GetModulus().ConvertToInt());
        }
    }
    else {
        AppendDescriptorIntegers(&descriptor, config.modulus.qModulusBits);
        AppendDescriptorIntegers(&descriptor, config.modulus.pModulusBits);
    }
    AppendDescriptorInteger(
        &descriptor,
        config.modulus.hybridLargeDigits);
    AppendDescriptorInteger(
        &descriptor,
        config.modulus.maximumQpBits);
    AppendDescriptorInteger(
        &descriptor,
        config.sparsification.mlweRank);
    AppendDescriptorInteger(&descriptor, config.weightDenominator);
    if (!UsesCertifiedGaussianProfile(config.protocolVersion)) {
        AppendDescriptorIntegers(&descriptor, config.integerWeights);
    }
    AppendDescriptorReal(
        &descriptor,
        config.setupNoise.publicKeyShareStandardDeviation);
    AppendDescriptorReal(
        &descriptor,
        config.setupNoise.m2rKeyShareStandardDeviation);
    AppendDescriptorReal(
        &descriptor,
        config.setupNoise.clientKeySwitchStandardDeviation);
    AppendDescriptorReal(
        &descriptor,
        config.setupNoise.encryptionStandardDeviation);
    if (UsesCertifiedGaussianProfile(config.protocolVersion)) {
        AppendDescriptorString(&descriptor, "exact-rational-truncated-dg-v3");
        AppendDescriptorInteger(&descriptor, config.syntheticUpdates.runningMeanDimension + config.syntheticUpdates.runningVarianceDimension);
        AppendDescriptorString(&descriptor, config.scaling.factor.str());
        AppendDescriptorString(&descriptor, config.auxScaling.factor.str());
        for (const std::uint64_t value : {
                static_cast<std::uint64_t>(config.sparsification.minimumCohort),
                static_cast<std::uint64_t>(config.sparsification.collusionBound),
                static_cast<std::uint64_t>(config.sparsification.errorFeedbackEnabled),
                static_cast<std::uint64_t>(config.precisionBits),
                static_cast<std::uint64_t>(config.auxPrecisionBits)})
            AppendDescriptorInteger(&descriptor, value);
        for (const std::uint32_t value : {config.compression.c1, config.compression.ca,
                config.compression.c2, config.compression.cd, config.compression.auxC1,
                config.compression.auxCa, config.compression.auxC2, config.compression.auxCd})
            AppendDescriptorInteger(&descriptor, value);
        for (const long double value : {config.sparsification.selectionRatio,
                config.updateBound, config.auxUpdateBound, config.smudging.sigmaSmg,
                config.smudging.auxSigmaSmg, config.modelDeliveryScale, config.auxDeliveryScale})
            AppendDescriptorReal(&descriptor, value);
    }
    if (config.experimentalDynamicCohort) {
        AppendDescriptorString(&descriptor, "v17-post-upload-pre-request-dropout");
        AppendDescriptorInteger(&descriptor, 1); // attempt cap
        AppendDescriptorInteger(&descriptor, config.dropoutTargetCap);
    }
    return descriptor;
}

const SparseSetupMaterial& RequireSparseSetup(
    const ProtocolSetupResult& setup) {

    if (setup.mode != ProtocolMode::SP ||
        setup.common.descriptor.mode != ProtocolMode::SP ||
        !setup.sparse) {

        throw std::invalid_argument(
            "sparse protocol operation requires an SP setup");
    }
    return *setup.sparse;
}

DCRTPoly GenerateSeededUniformDCRT(
    const std::shared_ptr<DCRTPoly::Params>& parameters,
    const Seed128& seed,
    const std::string& domain,
    const ByteVector& context) {

    return GenerateUniformDCRT(parameters, seed, domain, context);
}

ProtocolRnsParameters CreateProtocolRnsParameters(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config) {

    ValidateCryptoContext(cc);

    const ValidationReport report = ValidateProtocolParameters(config);
    report.ThrowIfInvalid();

    if (config.modulus.pModulusBits.size() != 1) {
        throw std::invalid_argument("exactly one P tower is required");
    }

    ProtocolRnsParameters result;
    result.paramsQ = cc->GetCryptoParameters()->GetElementParams();

    if (!result.paramsQ) {
        throw std::runtime_error("crypto context Q parameters are null");
    }

    const auto& qParameters = result.paramsQ->GetParams();
    result.qTowerCount = qParameters.size();
    result.pTowerCount = 1;
    result.decompositionDigitCount =
        config.modulus.hybridLargeDigits;

    if (result.qTowerCount != config.modulus.qModulusBits.size() ||
        result.decompositionDigitCount != result.qTowerCount) {

        throw std::invalid_argument(
            "actual Q tower count does not match the one-tower-per-digit profile");
    }

    if (result.paramsQ->GetRingDimension() !=
        config.modulus.ringDimension) {

        throw std::invalid_argument(
            "actual crypto-context ring dimension does not match the protocol profile");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < qParameters.size();
         ++towerIndex) {

        if (qParameters[towerIndex]->GetModulus().GetMSB() !=
            config.modulus.qModulusBits[towerIndex]) {

            throw std::invalid_argument(
                "actual Q tower bit size does not match the protocol profile");
        }
    }

    result.qTowersPerDigit = 1;

    const std::uint32_t cyclotomicOrder =
        result.paramsQ->GetCyclotomicOrder();

    lbcrypto::NativeInteger specialPrime =
        lbcrypto::LastPrime<lbcrypto::NativeInteger>(
            config.modulus.pModulusBits[0],
            cyclotomicOrder);

    bool collidesWithQ = true;
    while (collidesWithQ) {
        collidesWithQ = false;

        for (const auto& qParameter : qParameters) {
            if (qParameter->GetModulus() == specialPrime) {
                collidesWithQ = true;
                specialPrime =
                    lbcrypto::PreviousPrime(specialPrime, cyclotomicOrder);
                break;
            }
        }
    }

    const lbcrypto::NativeInteger specialRoot =
        lbcrypto::RootOfUnity<lbcrypto::NativeInteger>(
            cyclotomicOrder,
            specialPrime);

    if (specialPrime.GetMSB() != config.modulus.pModulusBits[0]) {
        throw std::runtime_error(
            "generated P tower bit size does not match the protocol profile");
    }

    result.paramsP =
        std::make_shared<DCRTPoly::Params>(
            cyclotomicOrder,
            std::vector<lbcrypto::NativeInteger>{specialPrime},
            std::vector<lbcrypto::NativeInteger>{specialRoot});

    std::vector<lbcrypto::NativeInteger> qpModuli;
    std::vector<lbcrypto::NativeInteger> qpRoots;
    qpModuli.reserve(qParameters.size() + 1);
    qpRoots.reserve(qParameters.size() + 1);

    for (const auto& qParameter : qParameters) {
        qpModuli.push_back(qParameter->GetModulus());
        qpRoots.push_back(qParameter->GetRootOfUnity());
    }

    qpModuli.push_back(specialPrime);
    qpRoots.push_back(specialRoot);

    result.paramsQP =
        std::make_shared<DCRTPoly::Params>(
            cyclotomicOrder,
            qpModuli,
            qpRoots);

    result.modulusP = lbcrypto::BigInteger(specialPrime);
    result.pModQ.reserve(qParameters.size());

    for (const auto& qParameter : qParameters) {
        result.pModQ.push_back(
            lbcrypto::NativeInteger(
                result.modulusP
                    .Mod(qParameter->GetModulus())
                    .ConvertToInt()));
    }

    if (result.paramsQP->GetModulus().GetMSB() >
        config.modulus.maximumQpBits) {

        throw std::invalid_argument(
            "actual custom QP basis exceeds the configured security limit");
    }

    ValidateRnsParameters(result);
    return result;
}

std::vector<PrivateKey> GenerateClientSecretKeys(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    std::size_t clientCount) {
    return GenerateClientSecretKeysWithTiming(cc, clientCount, nullptr);
}

DCRTPoly LiftTernarySecretToQP(
    DCRTPoly secretQ,
    const ProtocolRnsParameters& parameters) {

    ValidateRnsParameters(parameters);
    secretQ.SetFormat(Format::COEFFICIENT);

    const auto& qTowers = secretQ.GetAllElements();
    if (qTowers.size() != parameters.qTowerCount) {
        throw std::invalid_argument("secret Q tower count mismatch");
    }

    const std::size_t ringDimension = parameters.paramsQ->GetRingDimension();
    if (qTowers[0].GetValues().GetLength() != ringDimension) {
        throw std::invalid_argument("secret Q ring dimension mismatch");
    }

    const lbcrypto::NativeInteger referenceModulus =
        qTowers[0].GetModulus();
    const std::uint64_t referenceModulusValue =
        referenceModulus.ConvertToInt();

    std::vector<std::int8_t> signedCoefficients(ringDimension, 0);

    for (std::size_t index = 0; index < ringDimension; ++index) {
        const std::uint64_t residue =
            qTowers[0].GetValues()[index].ConvertToInt();

        std::int64_t centered = 0;
        if (residue <= referenceModulusValue / 2U) {
            centered = static_cast<std::int64_t>(residue);
        }
        else {
            centered = -static_cast<std::int64_t>(
                referenceModulusValue - residue);
        }

        if (centered < -1 || centered > 1) {
            throw std::invalid_argument(
                "secret polynomial is not uniform ternary");
        }

        signedCoefficients[index] = static_cast<std::int8_t>(centered);
    }

    for (std::size_t towerIndex = 0;
         towerIndex < qTowers.size();
         ++towerIndex) {

        const auto& tower = qTowers[towerIndex];
        const lbcrypto::NativeInteger modulus = tower.GetModulus();

        if (modulus !=
                parameters.paramsQ->GetParams()[towerIndex]->GetModulus() ||
            tower.GetValues().GetLength() != ringDimension) {

            throw std::invalid_argument("secret Q parameters mismatch");
        }

        for (std::size_t index = 0; index < ringDimension; ++index) {
            lbcrypto::NativeInteger expected(0);

            if (signedCoefficients[index] < 0) {
                expected = modulus - lbcrypto::NativeInteger(1);
            }
            else {
                expected = lbcrypto::NativeInteger(
                    static_cast<std::uint64_t>(signedCoefficients[index]));
            }

            if (tower.GetValues()[index] != expected) {
                throw std::invalid_argument(
                    "secret Q towers do not encode one ternary polynomial");
            }
        }
    }

    DCRTPoly lifted(parameters.paramsQP, Format::COEFFICIENT, true);
    auto& liftedTowers = lifted.GetAllElements();

    for (std::size_t towerIndex = 0;
         towerIndex < liftedTowers.size();
         ++towerIndex) {

        const lbcrypto::NativeInteger modulus =
            parameters.paramsQP->GetParams()[towerIndex]->GetModulus();

        lbcrypto::NativeVector values(ringDimension, modulus);

        for (std::size_t index = 0; index < ringDimension; ++index) {
            if (signedCoefficients[index] < 0) {
                values[index] = modulus - lbcrypto::NativeInteger(1);
            }
            else {
                values[index] = lbcrypto::NativeInteger(
                    static_cast<std::uint64_t>(signedCoefficients[index]));
            }
        }

        liftedTowers[towerIndex].SetValues(
            std::move(values),
            Format::COEFFICIENT);
    }

    lifted.SetFormat(Format::EVALUATION);
    return lifted;
}

DCRTPoly BuildM2RGadgetTerm(
    DCRTPoly sourceQ,
    std::size_t lambda,
    const ProtocolRnsParameters& parameters) {

    ValidateRnsParameters(parameters);

    if (lambda >= parameters.decompositionDigitCount) {
        throw std::out_of_range("M2R decomposition digit is out of range");
    }

    sourceQ.SetFormat(Format::EVALUATION);
    if (sourceQ.GetAllElements().size() != parameters.qTowerCount) {
        throw std::invalid_argument("M2R source Q tower count mismatch");
    }

    for (std::size_t towerIndex = 0;
         towerIndex < parameters.qTowerCount;
         ++towerIndex) {

        if (sourceQ.GetElementAtIndex(towerIndex).GetModulus() !=
            parameters.paramsQ->GetParams()[towerIndex]->GetModulus()) {

            throw std::invalid_argument("M2R source Q modulus mismatch");
        }
    }

    DCRTPoly result(parameters.paramsQP, Format::EVALUATION, true);
    const std::size_t qTowerIndex = lambda;

    result.SetElementAtIndex(
        qTowerIndex,
        sourceQ.GetElementAtIndex(qTowerIndex) *
            parameters.pModQ[qTowerIndex]);

    return result;
}

ProtocolSetupResult GenerateProtocolSetup(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config) {
    std::vector<double> keyGenerationSeconds;
    const auto secretKeys = GenerateClientSecretKeysWithTiming(
        cc,
        config.clientCount,
        &keyGenerationSeconds);
    ProtocolSetupResult result = GenerateProtocolSetupFromSecretKeys(
        cc,
        config,
        secretKeys);
    result.timing.clientSecretKeyGenerationSeconds =
        std::move(keyGenerationSeconds);
    return result;
}

ClientOnlySetupMaterial GenerateClientOnlySetupMaterial(
    const ProtocolExperimentConfig& config,
    SetupId setupId) {

    ValidateProtocolParameters(config).ThrowIfInvalid();
    if (config.mode != ProtocolMode::SP) {
        throw std::invalid_argument(
            "Kidx material is available only for the SP profile");
    }
    ClientOnlySetupMaterial result;
    result.setupId = setupId;
    result.clientGroupIndexKey = federated::UsesV17WireProtocol(config.protocolVersion)
        ? SampleFreshProtocolSeed128()
        : deterministic_randomness::DeriveClientGroupIndexKey(
              config.masterSeed, setupId);
    return result;
}

ProtocolSetupResult GenerateProtocolSetupFromSecretKeys(
    const lbcrypto::CryptoContext<DCRTPoly>& cc,
    const ProtocolExperimentConfig& config,
    const std::vector<PrivateKey>& secretKeys) {

    ValidateCryptoContext(cc);
    const ValidationReport validation = ValidateProtocolParameters(config);
    validation.ThrowIfInvalid();

    if (secretKeys.size() != config.clientCount) {
        throw std::invalid_argument(
            "there must be one secret key for each registered client");
    }

    for (std::size_t clientIndex = 0;
         clientIndex < secretKeys.size();
         ++clientIndex) {

        ValidatePrivateKey(cc, secretKeys[clientIndex], clientIndex);
    }

    ProtocolSetupResult result;
    result.mode = config.mode;
    result.setupNoise = config.setupNoise;
    result.common.certifiedV17Gaussian =
        UsesCertifiedGaussianProfile(config.protocolVersion);
    result.common.rns = CreateProtocolRnsParameters(cc, config);
    result.common.descriptor.mode = config.mode;
    result.common.descriptor.protocolVersion = config.protocolVersion;
    result.common.descriptor.supportCodec =
        config.protocolVersion == "v17" && config.mode == ProtocolMode::SP
            ? kChunkBitmapMsb0V1
            : kFullSupportCodec;
    result.common.descriptor.canonicalEncoding =
        CanonicalSetupDescriptor(config, result.common.rns);
    result.common.setupId = federated::UsesV17WireProtocol(config.protocolVersion)
        ? FreshUniqueV17SetupId()
        : deterministic_randomness::DeriveSetupId(
              config.masterSeed,
              result.common.descriptor.canonicalEncoding);

    result.common.crsSeed = federated::UsesV17WireProtocol(config.protocolVersion)
        ? SampleFreshProtocolSeed128()
        : deterministic_randomness::DeriveSeed128(
              config.masterSeed,
              "CRS-SEED-V11",
              SetupTagContext(result.common.setupId));

    result.common.mlweRank = config.sparsification.mlweRank;

    result.common.publicKeyP1 = GenerateUniformDCRT(
        result.common.rns.paramsQP,
        result.common.crsSeed,
        "PK-P1",
        SetupTagContext(result.common.setupId));

    if (config.mode == ProtocolMode::SP) {
        SparseSetupMaterial sparse;
        if (RequiresSparseRingConversion(config)) {
            sparse.m2rCommonA = GenerateCommonM2RA(result.common);
        }
        result.sparse = std::move(sparse);
    }

    result.clientSecrets.reserve(config.clientCount);
    result.cachedClientShares.reserve(config.clientCount);
    result.timing.clientCachedShareGenerationSeconds.reserve(
        config.clientCount);

    for (std::size_t clientIndex = 0;
         clientIndex < config.clientCount;
         ++clientIndex) {

        const auto clientStarted = std::chrono::steady_clock::now();

        ClientSecretSetupMaterial clientSecret;
        clientSecret.clientId = ClientId{clientIndex};
        clientSecret.secretKeyQ = secretKeys[clientIndex];
        {
            const auto encryptionKeys = cc->KeyGen();
            if (!encryptionKeys.good()) {
                throw std::runtime_error("encryption-domain key generation failed");
            }
            clientSecret.encryptionKeyQ = encryptionKeys.secretKey;
        }
        if (config.mode == ProtocolMode::SP && result.common.mlweRank > 1) {
            const auto modelKeys = cc->KeyGen();
            if (!modelKeys.good()) throw std::runtime_error("model-domain key generation failed");
            clientSecret.modelKeyQ = modelKeys.secretKey;
            clientSecret.modelKeyQP = LiftTernarySecretToQP(
                modelKeys.secretKey->GetPrivateElement(), result.common.rns);
        }
        clientSecret.secretKeyQP = LiftTernarySecretToQP(
            secretKeys[clientIndex]->GetPrivateElement(),
            result.common.rns);

        result.cachedClientShares.push_back(
            GenerateCachedClientShare(
                result.common,
                result.sparse ? &*result.sparse : nullptr,
                clientSecret,
                config.setupNoise));

        result.clientSecrets.push_back(std::move(clientSecret));
        result.timing.clientCachedShareGenerationSeconds.push_back(
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - clientStarted).count());
    }

    return result;
}

}  // namespace federated
