#include "protocol-parameters.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace federated {
namespace {

bool IsPowerOfTwo(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

void AddIssue(
    ValidationReport* report,
    ValidationSeverity severity,
    const std::string& code,
    const std::string& message) {

    report->issues.push_back(
        ValidationIssue{severity, code, message});
}

void AddError(
    ValidationReport* report,
    const std::string& code,
    const std::string& message) {

    AddIssue(
        report,
        ValidationSeverity::ERROR,
        code,
        message);
}

void AddWarning(
    ValidationReport* report,
    const std::string& code,
    const std::string& message) {

    AddIssue(
        report,
        ValidationSeverity::WARNING,
        code,
        message);
}

bool IsFinitePositive(long double value) {
    return std::isfinite(value) && value > 0.0L;
}

bool IsValidGaussian(
    const ClippedGaussianParameters& parameters) {

    return std::isfinite(parameters.mean) &&
           IsFinitePositive(parameters.standardDeviation) &&
           std::isfinite(parameters.minimum) &&
           std::isfinite(parameters.maximum) &&
           parameters.minimum <= parameters.mean &&
           parameters.mean <= parameters.maximum &&
           parameters.minimum < parameters.maximum;
}

void ValidateCompressionBits(
    ValidationReport* report,
    std::uint32_t bits,
    const std::string& name) {

    if (bits == 0 || bits > 63) {
        AddError(
            report,
            "compression." + name,
            name + " must be in [1, 63]");
    }
}

ValidationReport MergeReports(
    ValidationReport first,
    const ValidationReport& second) {

    first.issues.insert(
        first.issues.end(),
        second.issues.begin(),
        second.issues.end());

    return first;
}

}  // namespace

bool ValidationReport::HasErrors() const {
    return std::any_of(
        issues.begin(),
        issues.end(),
        [](const ValidationIssue& issue) {
            return issue.severity == ValidationSeverity::ERROR;
        });
}

bool ValidationReport::HasWarnings() const {
    return std::any_of(
        issues.begin(),
        issues.end(),
        [](const ValidationIssue& issue) {
            return issue.severity == ValidationSeverity::WARNING;
        });
}

void ValidationReport::ThrowIfInvalid() const {
    if (!HasErrors()) {
        return;
    }

    std::ostringstream message;
    message << "invalid protocol parameters";

    for (const auto& issue : issues) {
        if (issue.severity == ValidationSeverity::ERROR) {
            message << "; " << issue.code << ": " << issue.message;
        }
    }

    throw std::invalid_argument(message.str());
}

ProtocolExperimentConfig MakeSmokeConfig() {
    return ProtocolExperimentConfig{};
}

ProtocolExperimentConfig MakeResNet20Cifar10FixtureConfig() {
    ProtocolExperimentConfig result = MakeSmokeConfig();
    result.modelDimension = 272474;
    result.syntheticUpdates.runningMeanDimension = 784;
    result.syntheticUpdates.runningVarianceDimension = 784;
    result.sparsification.selectionRatio = 0.30L;
    result.sparsification.selectedChunksPerClient = 0;
    return result;
}

bool RequiresSparseRingConversion(
    ProtocolMode mode,
    std::size_t mlweRank) {

    return mode == ProtocolMode::SP && mlweRank > 1;
}

bool RequiresSparseRingConversion(
    const ProtocolExperimentConfig& config) {

    return RequiresSparseRingConversion(
        config.mode,
        config.sparsification.mlweRank);
}

DerivedDimensions DeriveDimensions(
    const ProtocolExperimentConfig& config) {

    if (config.sparsification.mlweRank == 0) {
        throw std::invalid_argument("mlweRank must be positive");
    }

    if (config.modulus.ringDimension == 0 ||
        config.modulus.ringDimension %
                config.sparsification.mlweRank !=
            0) {

        throw std::invalid_argument(
            "ringDimension must be divisible by mlweRank");
    }

    DerivedDimensions dimensions;
    dimensions.smallRingDimension =
        config.modulus.ringDimension /
        config.sparsification.mlweRank;

    dimensions.chunkCount =
        (config.modelDimension +
         dimensions.smallRingDimension - 1) /
        dimensions.smallRingDimension;

    if (config.sparsification.selectionRatio > 0.0L) {
        const long double selectedCapacity =
            config.sparsification.selectionRatio *
            static_cast<long double>(config.modelDimension) /
            static_cast<long double>(dimensions.smallRingDimension);
        dimensions.selectedChunksPerClient = std::min(
            dimensions.chunkCount,
            std::max<std::size_t>(
                1,
                static_cast<std::size_t>(std::ceil(selectedCapacity))));
    }
    else {
        dimensions.selectedChunksPerClient =
            config.sparsification.selectedChunksPerClient;
    }

    dimensions.clientCiphertextsPerRound =
        (dimensions.selectedChunksPerClient +
         config.sparsification.mlweRank - 1) /
        config.sparsification.mlweRank;

    return dimensions;
}

long double ScalingFactor(
    const ScalingParameters& parameters) {

    if (!exact_integer::ValidScale(parameters.factor)) {
        throw std::invalid_argument("scaling factor must be in [1, 2^256)");
    }
    return static_cast<long double>(parameters.factor);
}

long double SigmaSmgFromModulus(
    std::uint64_t modulus,
    std::uint32_t divisorBits) {

    if (modulus < 2 || divisorBits > 63) {
        throw std::invalid_argument("invalid sigma_smg modulus profile");
    }
    return static_cast<long double>(modulus) /
        std::ldexp(1.0L, static_cast<int>(divisorBits));
}

NumericDiagnostics ComputeNumericDiagnostics(
    const ProtocolExperimentConfig& config,
    long double actualQModulus) {

    if (config.clientCount == 0 ||
        config.weightDenominator == 0 ||
        !IsFinitePositive(actualQModulus)) {

        throw std::invalid_argument(
            "client count, weight denominator, and actual Q must be positive");
    }

    NumericDiagnostics diagnostics;
    diagnostics.modulus = actualQModulus;

    diagnostics.scalingFactor =
        ScalingFactor(config.scaling);

    diagnostics.auxScalingFactor =
        ScalingFactor(config.auxScaling);

    diagnostics.targetPrecision =
        config.updateBound * std::ldexp(
            1.0L,
            -static_cast<int>(config.precisionBits));

    diagnostics.auxTargetPrecision =
        config.auxUpdateBound * std::ldexp(
            1.0L,
            -static_cast<int>(config.auxPrecisionBits));

    diagnostics.plaintextMagnitude =
        static_cast<long double>(config.weightDenominator) *
        diagnostics.scalingFactor *
        config.updateBound;

    diagnostics.centeredCorrectnessMargin =
        diagnostics.modulus / 2.0L -
        diagnostics.plaintextMagnitude;

    diagnostics.auxPlaintextMagnitude =
        static_cast<long double>(config.weightDenominator) *
        diagnostics.auxScalingFactor *
        config.auxUpdateBound;

    diagnostics.auxCenteredCorrectnessMargin =
        diagnostics.modulus / 2.0L -
        diagnostics.auxPlaintextMagnitude;

    const long double decodedDenominator =
        static_cast<long double>(config.weightDenominator) *
        diagnostics.scalingFactor;

    const long double auxDecodedDenominator =
        static_cast<long double>(config.weightDenominator) *
        diagnostics.auxScalingFactor;

    const long double squareRootClientCount =
        std::sqrt(static_cast<long double>(config.clientCount));

    diagnostics.decodedSigmaSmgStandardDeviation =
        squareRootClientCount *
        config.smudging.sigmaSmg /
        decodedDenominator;

    diagnostics.auxDecodedSigmaSmgStandardDeviation =
        squareRootClientCount *
        config.smudging.auxSigmaSmg /
        auxDecodedDenominator;

    const long double weightSum =
        static_cast<long double>(std::accumulate(
            config.integerWeights.begin(),
            config.integerWeights.end(),
            std::uint64_t{0}));

    diagnostics.c1WorstCaseDecodedError =
        weightSum * diagnostics.modulus /
        std::ldexp(
            decodedDenominator,
            static_cast<int>(config.compression.c1 + 1));

    diagnostics.c2WorstCaseDecodedError =
        static_cast<long double>(config.clientCount) *
        diagnostics.modulus /
        std::ldexp(
            decodedDenominator,
            static_cast<int>(config.compression.c2 + 1));

    return diagnostics;
}

ValidationReport ValidateProtocolParameters(
    const ProtocolExperimentConfig& config) {

    ValidationReport report;

    if (config.protocolVersion != "v17") {
        AddError(
            &report,
            "protocol.version",
            "protocol version must be v17");
    }

    if (config.mode != ProtocolMode::SP &&
        config.mode != ProtocolMode::FULL) {

        AddError(
            &report,
            "protocol.mode",
            "protocol mode must be SP or FULL");
    }

    if (config.clientCount == 0) {
        AddError(&report, "clients.count", "client count must be positive");
    }

    if (config.modelDimension == 0) {
        AddError(&report, "model.dimension", "model dimension must be positive");
    }

    if (!IsPowerOfTwo(config.modulus.ringDimension)) {
        AddError(
            &report,
            "modulus.ring_dimension",
            "ring dimension must be a power of two");
    }

    if (federated::UsesV17WireProtocol(config.protocolVersion) &&
        (config.modulus.qModulusBits.size() != 2 ||
         config.modulus.hybridLargeDigits != 2)) {
        AddError(&report, "v17.hybrid",
            "SD v17 requires exactly two Q primes and m=2");
    }
    if (config.modulus.qModulusBits.empty()) {
        AddError(&report, "modulus.q", "at least one Q tower is required");
    }

    if (std::any_of(
            config.modulus.qModulusBits.begin(),
            config.modulus.qModulusBits.end(),
            [](std::uint32_t bits) { return bits < 2 || bits > 60; })) {

        AddError(
            &report,
            "modulus.q_bits",
            "every Q tower bit size must be in [2, 60]");
    }

    if (config.modulus.pModulusBits.size() != 1) {
        AddError(
            &report,
            "modulus.p",
            "the smoke profile requires one special-modulus P tower");
    }

    if (std::any_of(
            config.modulus.pModulusBits.begin(),
            config.modulus.pModulusBits.end(),
            [](std::uint32_t bits) { return bits < 2 || bits > 60; })) {

        AddError(
            &report,
            "modulus.p_bits",
            "every P tower bit size must be in [2, 60]");
    }

    if (config.modulus.hybridLargeDigits !=
        config.modulus.qModulusBits.size()) {

        AddError(
            &report,
            "modulus.hybrid_digits",
            "one HYBRID decomposition digit per Q tower is required");
    }

    const std::uint64_t estimatedQpBits =
        std::accumulate(
            config.modulus.qModulusBits.begin(),
            config.modulus.qModulusBits.end(),
            std::uint64_t{0}) +
        std::accumulate(
            config.modulus.pModulusBits.begin(),
            config.modulus.pModulusBits.end(),
            std::uint64_t{0});

    if (estimatedQpBits > config.modulus.maximumQpBits) {
        AddError(
            &report,
            "security.qp_bits",
            "the estimated QP bit length exceeds the configured security limit");
    }

    if (config.sparsification.mlweRank == 0 ||
        config.modulus.ringDimension %
                config.sparsification.mlweRank !=
            0) {

        AddError(
            &report,
            "sparsification.rank",
            "MLWE rank must be positive and divide the ring dimension");
    }

    if (config.sparsification.minimumCohort > config.clientCount) {
        AddError(
            &report,
            "sparsification.cohort",
            "minimum cohort cannot exceed the client count");
    }

    if (config.sparsification.minimumCohort <
        config.sparsification.collusionBound + 2) {

        AddError(
            &report,
            "sparsification.collusion",
            "minimum cohort must satisfy tau >= collusionBound + 2");
    }

    if (config.weightDenominator != 128) {
        AddError(
            &report,
            "weights.denominator",
            "the v17 integer-weight denominator must equal 128");
    }

    if (config.integerWeights.size() != config.clientCount) {
        AddError(
            &report,
            "weights.count",
            "there must be one integer weight per client");
    }
    else {
        if (std::any_of(
                config.integerWeights.begin(),
                config.integerWeights.end(),
                [](std::uint32_t weight) { return weight == 0; })) {

            AddError(
                &report,
                "weights.positive",
                "every registered client weight must be positive");
        }

        const std::uint64_t weightSum =
            std::accumulate(
                config.integerWeights.begin(),
                config.integerWeights.end(),
                std::uint64_t{0});

        if (weightSum != config.weightDenominator) {
            AddError(
                &report,
                "weights.sum",
                "integer weights must sum to the weight denominator");
        }
    }

    if (!exact_integer::ValidScale(config.scaling.factor)) {
        AddError(
            &report,
            "scaling.factor",
            "model scaling factor must be in [1, 2^256)");
    }

    if (!exact_integer::ValidScale(config.auxScaling.factor)) {
        AddError(
            &report,
            "aux.scaling.factor",
            "auxiliary scaling factor must be in [1, 2^256)");
    }

    if (!IsFinitePositive(config.updateBound)) {
        AddError(
            &report,
            "precision.update_bound",
            "update bound must be finite and positive");
    }

    if (!IsFinitePositive(config.auxUpdateBound)) {
        AddError(
            &report,
            "aux.precision.update_bound",
            "auxiliary update bound must be finite and positive");
    }

    if (!std::isfinite(config.smudging.sigmaSmg) ||
        config.smudging.sigmaSmg < 0.0L ||
        !std::isfinite(config.smudging.auxSigmaSmg) ||
        config.smudging.auxSigmaSmg < 0.0L) {

        AddError(
            &report,
            "smudging.sigma_smg",
            "sigma_smg values must be finite and nonnegative");
    }

    if (!IsFinitePositive(
            config.setupNoise.publicKeyShareStandardDeviation) ||
        !IsFinitePositive(
            config.setupNoise.m2rKeyShareStandardDeviation) ||
        !IsFinitePositive(
            config.setupNoise.clientKeySwitchStandardDeviation) ||
        !IsFinitePositive(
            config.setupNoise.encryptionStandardDeviation)) {

        AddError(
            &report,
            "setup.noise",
            "all setup and key-switch noise standard deviations must be finite and positive");
    }

    ValidateCompressionBits(&report, config.compression.c1, "c1");
    ValidateCompressionBits(&report, config.compression.c2, "c2");
    ValidateCompressionBits(&report, config.compression.ca, "ca");
    ValidateCompressionBits(&report, config.compression.cd, "cd");
    ValidateCompressionBits(&report, config.compression.auxC1, "auxC1");
    ValidateCompressionBits(&report, config.compression.auxC2, "auxC2");
    ValidateCompressionBits(&report, config.compression.auxCa, "auxCa");
    ValidateCompressionBits(&report, config.compression.auxCd, "auxCd");

    if (!IsFinitePositive(config.modelDeliveryScale)) {
        AddError(
            &report,
            "delivery.model_scale",
            "modelDeliveryScale must be finite and positive");
    }
    else if (config.compression.cd >= 2 &&
             config.compression.cd <= 63) {
        const long double maximum = std::ldexp(
            1.0L,
            static_cast<int>(config.compression.cd - 1U)) - 1.0L;
        if (config.modelDeliveryScale * config.updateBound > maximum) {
            AddError(
                &report,
                "delivery.model_range",
                "modelDeliveryScale*U_max exceeds the signed cd range; clipping is forbidden");
        }
    }

    if (!IsFinitePositive(config.auxDeliveryScale)) {
        AddError(
            &report,
            "delivery.aux_scale",
            "auxDeliveryScale must be finite and positive");
    }
    else if (config.compression.auxCd >= 2 &&
             config.compression.auxCd <= 63) {
        const long double maximum = std::ldexp(
            1.0L,
            static_cast<int>(config.compression.auxCd - 1U)) - 1.0L;
        if (config.auxDeliveryScale * config.auxUpdateBound > maximum) {
            AddError(
                &report,
                "delivery.aux_range",
                "auxDeliveryScale*U_max_aux exceeds the signed cd_aux range; clipping is forbidden");
        }
    }

    if (!IsValidGaussian(config.syntheticUpdates.modelDelta) ||
        !IsValidGaussian(config.syntheticUpdates.runningMeanDelta) ||
        !IsValidGaussian(config.syntheticUpdates.runningVarianceDelta)) {

        AddError(
            &report,
            "synthetic.gaussian",
            "all clipped-Gaussian parameters must be finite and ordered");
    }

    if (config.syntheticUpdates.runningMeanDelta.minimum <
            -config.auxUpdateBound ||
        config.syntheticUpdates.runningMeanDelta.maximum >
            config.auxUpdateBound ||
        config.syntheticUpdates.runningVarianceDelta.minimum <
            -config.auxUpdateBound ||
        config.syntheticUpdates.runningVarianceDelta.maximum >
            config.auxUpdateBound) {

        AddError(
            &report,
            "aux.synthetic.update_bound",
            "auxiliary clipped-Gaussian intervals exceed auxUpdateBound");
    }

    if (config.syntheticUpdates.modelDelta.minimum <
            -config.updateBound ||
        config.syntheticUpdates.modelDelta.maximum >
            config.updateBound) {

        AddError(
            &report,
            "synthetic.update_bound",
            "model-delta generation interval exceeds U_max");
    }

    if (config.syntheticUpdates.runningMeanDimension == 0 ||
        config.syntheticUpdates.runningVarianceDimension == 0) {

        AddError(
            &report,
            "synthetic.aux_dimension",
            "auxiliary running-statistic dimensions must be positive");
    }

    if (config.syntheticUpdates.initialRunningVariance +
            config.syntheticUpdates.runningVarianceDelta.minimum <=
        0.0L) {

        AddWarning(
            &report,
            "synthetic.running_variance",
            "one auxiliary update can make running variance nonpositive");
    }

    if (config.dropoutEnabled) {
        AddWarning(
            &report,
            "round.dropout",
            "dropout state transitions are intentionally deferred in stage 1");
    }

    if (!std::isfinite(config.sparsification.selectionRatio) ||
        config.sparsification.selectionRatio < 0.0L ||
        config.sparsification.selectionRatio > 1.0L) {

        AddError(
            &report,
            "sparsification.selection_ratio",
            "selection ratio must be zero or in (0, 1]");
    }

    if (!report.HasErrors()) {
        const DerivedDimensions dimensions = DeriveDimensions(config);

        if (dimensions.selectedChunksPerClient == 0 ||
            dimensions.selectedChunksPerClient > dimensions.chunkCount) {

            AddError(
                &report,
                "sparsification.selected_chunks",
                "selected chunk count must be in [1, chunkCount]");
        }
    }

    return report;
}

ValidationReport ValidateProtocolParameters(
    const ProtocolExperimentConfig& config,
    long double actualQModulus) {

    ValidationReport report = ValidateProtocolParameters(config);

    if (report.HasErrors()) {
        return report;
    }

    ValidationReport numericReport;
    const NumericDiagnostics diagnostics =
        ComputeNumericDiagnostics(config, actualQModulus);

    if (diagnostics.centeredCorrectnessMargin <= 0.0L) {
        AddError(
            &numericReport,
            "correctness.plaintext",
            "weighted plaintext reaches or exceeds Q/2");
    }

    if (diagnostics.auxCenteredCorrectnessMargin <= 0.0L) {
        AddError(
            &numericReport,
            "correctness.aux_plaintext",
            "weighted auxiliary plaintext reaches or exceeds Q/2");
    }

    if (diagnostics.decodedSigmaSmgStandardDeviation >
        diagnostics.targetPrecision) {

        AddWarning(
            &numericReport,
            "precision.smudging",
            "predicted model smudging standard deviation exceeds U_max*2^-p");
    }

    if (diagnostics.auxDecodedSigmaSmgStandardDeviation >
        diagnostics.auxTargetPrecision) {

        AddWarning(
            &numericReport,
            "precision.aux_smudging",
            "predicted auxiliary smudging standard deviation exceeds U_max_aux*2^-p_aux");
    }

    if (diagnostics.c1WorstCaseDecodedError >
        diagnostics.targetPrecision) {

        AddWarning(
            &numericReport,
            "precision.c1",
            "the standalone deterministic c1 error bound exceeds U_max*2^-p");
    }

    if (diagnostics.c2WorstCaseDecodedError >
        diagnostics.targetPrecision) {

        AddWarning(
            &numericReport,
            "precision.c2",
            "the standalone deterministic c2 error bound exceeds U_max*2^-p");
    }

    return MergeReports(std::move(report), numericReport);
}

}  // namespace federated
