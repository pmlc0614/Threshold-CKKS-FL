#ifndef FEDERATED1_PROTOCOL_PARAMETERS_H
#define FEDERATED1_PROTOCOL_PARAMETERS_H

#include "protocol-types.h"
#include "exact-integer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace federated {

struct ModulusProfile {
    std::size_t ringDimension = 8192;
    std::vector<std::uint32_t> qModulusBits{26, 26};
    std::vector<std::uint32_t> pModulusBits{60};
    std::size_t hybridLargeDigits = 2;
    std::uint32_t maximumQpBits = 218;
};

struct ScalingParameters {
    /* Exact protocol coefficient-encoding scale; never derived from Q. */
    exact_integer::Integer factor = std::uint64_t{1} << 43U;
};

struct CompressionParameters {
    std::uint32_t c1 = 19;
    std::uint32_t c2 = 15;
    std::uint32_t ca = 48;
    std::uint32_t cd = 13;

    std::uint32_t auxC1 = 19;
    std::uint32_t auxC2 = 15;
    std::uint32_t auxCa = 48;
    std::uint32_t auxCd = 13;
};

struct SmudgingParameters {
    /* Each client adds an independent sample to its partial decryption. */
    long double sigmaSmg = 68719476736.0L;  // nominal Q/2^16
    long double auxSigmaSmg = 68719476736.0L;  // nominal Q/2^16
};

struct SetupNoiseParameters {
    long double publicKeyShareStandardDeviation = 3.2L;
    long double m2rKeyShareStandardDeviation = 3.2L;
    long double clientKeySwitchStandardDeviation = 3.2L;
    long double encryptionStandardDeviation = 3.2L;
};

struct ClippedGaussianParameters {
    long double mean = 0.0L;
    long double standardDeviation = 0.01L;
    long double minimum = -0.06L;
    long double maximum = 0.06L;
};

struct SyntheticUpdateParameters {
    ClippedGaussianParameters modelDelta;

    ClippedGaussianParameters runningMeanDelta{
        0.0L, 0.005L, -0.03L, 0.03L};

    ClippedGaussianParameters runningVarianceDelta{
        0.0L, 0.002L, -0.012L, 0.012L};

    std::size_t runningMeanDimension = 688;
    std::size_t runningVarianceDimension = 688;
    long double initialRunningMean = 0.0L;
    long double initialRunningVariance = 1.0L;
};

struct SparsificationParameters {
    std::size_t mlweRank = 4;
    /* Zero selects the fixed selectedChunksPerClient count; otherwise use ceil(selectionRatio * ceil(d/chunkSize)). */
    long double selectionRatio = 0.0L;
    std::size_t selectedChunksPerClient = 4;
    std::size_t minimumCohort = 3;
    std::size_t collusionBound = 0;
    bool errorFeedbackEnabled = true;
};

struct ProtocolExperimentConfig {
    std::string protocolVersion = "v17";
    // Explicit legacy research opt-in; outside the fixed-cohort SD contract.
    bool experimentalDynamicCohort = false;
    std::size_t dropoutTargetCap = 128;
    ProtocolMode mode = ProtocolMode::SP;
    std::size_t clientCount = 8;
    std::size_t modelDimension = 24576;

    ModulusProfile modulus;
    ScalingParameters scaling;
    /* Configure auxiliary scaling independently of model-update scaling. */
    ScalingParameters auxScaling;
    CompressionParameters compression;
    SmudgingParameters smudging;
    SetupNoiseParameters setupNoise;
    SyntheticUpdateParameters syntheticUpdates;
    SparsificationParameters sparsification;

    /* Aggregate U_max: sigma_final <= U_max * 2^-precisionBits. */
    std::uint32_t precisionBits = 12;
    long double updateBound = 1.0L;
    /* Delta_d: requires Delta_d*aggregate U_max <= 2^(c_d-1)-1. */
    long double modelDeliveryScale = 4095.0L;
    /* Aggregate U'_max: sigma_final_aux <= U'_max * 2^-auxPrecisionBits. */
    std::uint32_t auxPrecisionBits = 12;
    long double auxUpdateBound = 1.0L;
    /* Delta_d_aux: independent no-clipping scale for auxiliary deltas. */
    long double auxDeliveryScale = 4095.0L;

    std::uint32_t weightDenominator = 128;
    std::vector<std::uint32_t> integerWeights{
        16, 16, 16, 16, 16, 16, 16, 16};

    bool dropoutEnabled = false;

    MasterSeed256 masterSeed{
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};
};

struct DerivedDimensions {
    std::size_t smallRingDimension = 0;
    std::size_t chunkCount = 0;
    std::size_t selectedChunksPerClient = 0;
    std::size_t clientCiphertextsPerRound = 0;
};

struct NumericDiagnostics {
    long double modulus = 0.0L;
    long double scalingFactor = 0.0L;
    long double auxScalingFactor = 0.0L;
    long double targetPrecision = 0.0L;
    long double auxTargetPrecision = 0.0L;
    long double plaintextMagnitude = 0.0L;
    long double centeredCorrectnessMargin = 0.0L;
    long double auxPlaintextMagnitude = 0.0L;
    long double auxCenteredCorrectnessMargin = 0.0L;
    long double decodedSigmaSmgStandardDeviation = 0.0L;
    long double auxDecodedSigmaSmgStandardDeviation = 0.0L;
    long double c1WorstCaseDecodedError = 0.0L;
    long double c2WorstCaseDecodedError = 0.0L;
};

struct ValidationReport {
    std::vector<ValidationIssue> issues;

    bool HasErrors() const;
    bool HasWarnings() const;
    void ThrowIfInvalid() const;
};

ProtocolExperimentConfig MakeSmokeConfig();

ProtocolExperimentConfig MakeResNet20Cifar10FixtureConfig();

/* Rank-one SP already lives in the large RLWE ring, so R2M/M2R is needed
 * only for SP profiles whose MLWE rank is greater than one. */
bool RequiresSparseRingConversion(
    ProtocolMode mode,
    std::size_t mlweRank);

bool RequiresSparseRingConversion(
    const ProtocolExperimentConfig& config);

DerivedDimensions DeriveDimensions(
    const ProtocolExperimentConfig& config);

long double ScalingFactor(
    const ScalingParameters& parameters);

long double SigmaSmgFromModulus(
    std::uint64_t modulus,
    std::uint32_t divisorBits);

NumericDiagnostics ComputeNumericDiagnostics(
    const ProtocolExperimentConfig& config,
    long double actualQModulus);

ValidationReport ValidateProtocolParameters(
    const ProtocolExperimentConfig& config);

ValidationReport ValidateProtocolParameters(
    const ProtocolExperimentConfig& config,
    long double actualQModulus);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_PARAMETERS_H
