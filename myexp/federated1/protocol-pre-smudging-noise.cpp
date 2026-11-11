#include "protocol-pre-smudging-noise.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace federated {
namespace {

void RefreshPooledStatistics(
    ProtocolPreSmudgingNoiseMeasurement* measurement) {

    if (measurement->totalSampleCount == 0) {
        measurement->pooledMean = 0.0L;
        measurement->pooledStandardDeviation = 0.0L;
        return;
    }

    long double sum = 0.0L;
    long double squaredSum = 0.0L;
    for (const auto& group : measurement->groups) {
        sum += group.sum;
        squaredSum += group.squaredSum;
    }
    const long double count = measurement->totalSampleCount;
    measurement->pooledMean = sum / count;
    const long double variance = std::max(
        0.0L,
        squaredSum / count -
            measurement->pooledMean * measurement->pooledMean);
    measurement->pooledStandardDeviation = std::sqrt(variance);
}

}  // namespace

ProtocolNoiseGroupStatistics MeasureProtocolNoiseGroup(
    const std::vector<long double>& centeredResiduals) {

    if (centeredResiduals.empty()) {
        throw std::invalid_argument("pre-smudging noise group is empty");
    }

    ProtocolNoiseGroupStatistics result;
    result.sampleCount = centeredResiduals.size();
    for (const long double value : centeredResiduals) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("pre-smudging residual is non-finite");
        }
        result.sum += value;
        result.squaredSum += value * value;
        result.maximumAbsoluteValue = std::max(
            result.maximumAbsoluteValue,
            std::abs(value));
    }
    std::vector<long double> absolute;
    absolute.reserve(centeredResiduals.size());
    for (const auto value : centeredResiduals) absolute.push_back(std::abs(value));
    std::sort(absolute.begin(), absolute.end());
    result.absoluteQ99 = absolute[static_cast<std::size_t>(std::ceil(0.99L*absolute.size()))-1];
    result.absoluteQ999 = absolute[static_cast<std::size_t>(std::ceil(0.999L*absolute.size()))-1];
    const long double count = result.sampleCount;
    result.mean = result.sum / count;
    result.standardDeviation = std::sqrt(std::max(
        0.0L,
        result.squaredSum / count - result.mean * result.mean));
    return result;
}

void AppendProtocolNoiseGroup(
    ProtocolPreSmudgingNoiseMeasurement* measurement,
    const ProtocolNoiseGroupStatistics& group) {

    if (measurement == nullptr || group.sampleCount == 0 ||
        !std::isfinite(group.standardDeviation)) {

        throw std::invalid_argument("invalid pre-smudging noise group");
    }
    measurement->groups.push_back(group);
    measurement->totalSampleCount += group.sampleCount;
    measurement->maximumGroupStandardDeviation = std::max(
        measurement->maximumGroupStandardDeviation,
        group.standardDeviation);
    measurement->maximumAbsoluteValue = std::max(
        measurement->maximumAbsoluteValue,
        group.maximumAbsoluteValue);
    RefreshPooledStatistics(measurement);
}

void MergeProtocolNoiseMeasurement(
    ProtocolPreSmudgingNoiseMeasurement* target,
    const ProtocolPreSmudgingNoiseMeasurement& source) {

    if (target == nullptr) {
        throw std::invalid_argument("pre-smudging target is null");
    }
    for (const auto& group : source.groups) {
        AppendProtocolNoiseGroup(target, group);
    }
}

void AppendProtocolNoiseBreakdownGroup(
    ProtocolNoiseBreakdownMeasurement* measurement,
    const ProtocolNoiseComponentResiduals& residuals) {

    if (measurement == nullptr) {
        throw std::invalid_argument("noise-breakdown target is null");
    }
    const std::size_t count = residuals.total.size();
    if (count == 0 || residuals.encryption.size() != count ||
        residuals.keySwitchCore.size() != count ||
        residuals.keySwitchModDown.size() != count ||
        residuals.m2rCore.size() != count ||
        residuals.m2rModDown.size() != count ||
        residuals.aggregateACompression.size() != count) {
        throw std::invalid_argument(
            "noise-breakdown component sizes are inconsistent");
    }

    std::vector<long double> keySwitch(count);
    std::vector<long double> m2r(count);
    for (std::size_t index = 0; index < count; ++index) {
        keySwitch[index] = residuals.keySwitchCore[index] +
            residuals.keySwitchModDown[index];
        m2r[index] = residuals.m2rCore[index] +
            residuals.m2rModDown[index];
    }

    AppendProtocolNoiseGroup(
        &measurement->encryption,
        MeasureProtocolNoiseGroup(residuals.encryption));
    AppendProtocolNoiseGroup(
        &measurement->keySwitchCore,
        MeasureProtocolNoiseGroup(residuals.keySwitchCore));
    AppendProtocolNoiseGroup(
        &measurement->keySwitchModDown,
        MeasureProtocolNoiseGroup(residuals.keySwitchModDown));
    AppendProtocolNoiseGroup(
        &measurement->keySwitch,
        MeasureProtocolNoiseGroup(keySwitch));
    AppendProtocolNoiseGroup(
        &measurement->m2rCore,
        MeasureProtocolNoiseGroup(residuals.m2rCore));
    AppendProtocolNoiseGroup(
        &measurement->m2rModDown,
        MeasureProtocolNoiseGroup(residuals.m2rModDown));
    AppendProtocolNoiseGroup(
        &measurement->m2r,
        MeasureProtocolNoiseGroup(m2r));
    AppendProtocolNoiseGroup(
        &measurement->aggregateACompression,
        MeasureProtocolNoiseGroup(residuals.aggregateACompression));
    AppendProtocolNoiseGroup(
        &measurement->total,
        MeasureProtocolNoiseGroup(residuals.total));

    for (std::size_t index = 0; index < count; ++index) {
        const long double reconstructed =
            residuals.encryption[index] +
            residuals.keySwitchCore[index] +
            residuals.keySwitchModDown[index] +
            residuals.m2rCore[index] +
            residuals.m2rModDown[index] +
            residuals.aggregateACompression[index];
        measurement->maximumReconstructionAbsoluteError = std::max(
            measurement->maximumReconstructionAbsoluteError,
            std::abs(residuals.total[index] - reconstructed));
    }
    measurement->reconstructionSampleCount += count;
}

void MergeProtocolNoiseBreakdownMeasurement(
    ProtocolNoiseBreakdownMeasurement* target,
    const ProtocolNoiseBreakdownMeasurement& source) {

    if (target == nullptr) {
        throw std::invalid_argument("noise-breakdown merge target is null");
    }
    MergeProtocolNoiseMeasurement(&target->encryption, source.encryption);
    MergeProtocolNoiseMeasurement(
        &target->keySwitchCore,
        source.keySwitchCore);
    MergeProtocolNoiseMeasurement(
        &target->keySwitchModDown,
        source.keySwitchModDown);
    MergeProtocolNoiseMeasurement(&target->keySwitch, source.keySwitch);
    MergeProtocolNoiseMeasurement(&target->m2rCore, source.m2rCore);
    MergeProtocolNoiseMeasurement(&target->m2rModDown, source.m2rModDown);
    MergeProtocolNoiseMeasurement(&target->m2r, source.m2r);
    MergeProtocolNoiseMeasurement(
        &target->aggregateACompression,
        source.aggregateACompression);
    MergeProtocolNoiseMeasurement(&target->total, source.total);
    target->reconstructionSampleCount += source.reconstructionSampleCount;
    target->maximumReconstructionAbsoluteError = std::max(
        target->maximumReconstructionAbsoluteError,
        source.maximumReconstructionAbsoluteError);
}

}  // namespace federated
