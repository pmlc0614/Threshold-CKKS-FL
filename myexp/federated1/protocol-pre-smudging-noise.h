#ifndef FEDERATED1_PROTOCOL_PRE_SMUDGING_NOISE_H
#define FEDERATED1_PROTOCOL_PRE_SMUDGING_NOISE_H

#include <cstddef>
#include <vector>

namespace federated {

struct ProtocolNoiseGroupStatistics {
    std::size_t sampleCount = 0;
    long double sum = 0.0L;
    long double squaredSum = 0.0L;
    long double mean = 0.0L;
    long double standardDeviation = 0.0L;
    long double maximumAbsoluteValue = 0.0L;
    long double absoluteQ99 = 0.0L;
    long double absoluteQ999 = 0.0L;
};

/* Test-only phase residuals before smudging, grouped by sparse label or dense block. maximumGroupStandardDeviation is the maximum measured group deviation, not a static certificate. */
struct ProtocolPreSmudgingNoiseMeasurement {
    std::vector<ProtocolNoiseGroupStatistics> groups;
    std::size_t totalSampleCount = 0;
    long double pooledMean = 0.0L;
    long double pooledStandardDeviation = 0.0L;
    long double maximumGroupStandardDeviation = 0.0L;
    long double maximumAbsoluteValue = 0.0L;
};

/*
 * Privileged calibration-only decomposition of the pre-smudging residual.
 * KS_CORE and M2R_CORE are measured before their respective ModDown steps;
 * MD_KS and MD_M2R are the paired pre/post-ModDown differences.  TOTAL is
 * retained independently so that the component sum can be audited.
 */
struct ProtocolNoiseComponentResiduals {
    std::vector<long double> encryption;
    std::vector<long double> keySwitchCore;
    std::vector<long double> keySwitchModDown;
    std::vector<long double> m2rCore;
    std::vector<long double> m2rModDown;
    std::vector<long double> aggregateACompression;
    std::vector<long double> total;
};

struct ProtocolNoiseBreakdownMeasurement {
    ProtocolPreSmudgingNoiseMeasurement encryption;
    ProtocolPreSmudgingNoiseMeasurement keySwitchCore;
    ProtocolPreSmudgingNoiseMeasurement keySwitchModDown;
    ProtocolPreSmudgingNoiseMeasurement keySwitch;
    ProtocolPreSmudgingNoiseMeasurement m2rCore;
    ProtocolPreSmudgingNoiseMeasurement m2rModDown;
    ProtocolPreSmudgingNoiseMeasurement m2r;
    ProtocolPreSmudgingNoiseMeasurement aggregateACompression;
    ProtocolPreSmudgingNoiseMeasurement total;
    std::size_t reconstructionSampleCount = 0;
    long double maximumReconstructionAbsoluteError = 0.0L;
};

ProtocolNoiseGroupStatistics MeasureProtocolNoiseGroup(
    const std::vector<long double>& centeredResiduals);

void AppendProtocolNoiseGroup(
    ProtocolPreSmudgingNoiseMeasurement* measurement,
    const ProtocolNoiseGroupStatistics& group);

void MergeProtocolNoiseMeasurement(
    ProtocolPreSmudgingNoiseMeasurement* target,
    const ProtocolPreSmudgingNoiseMeasurement& source);

void AppendProtocolNoiseBreakdownGroup(
    ProtocolNoiseBreakdownMeasurement* measurement,
    const ProtocolNoiseComponentResiduals& residuals);

void MergeProtocolNoiseBreakdownMeasurement(
    ProtocolNoiseBreakdownMeasurement* target,
    const ProtocolNoiseBreakdownMeasurement& source);

}  // namespace federated

#endif  // FEDERATED1_PROTOCOL_PRE_SMUDGING_NOISE_H
