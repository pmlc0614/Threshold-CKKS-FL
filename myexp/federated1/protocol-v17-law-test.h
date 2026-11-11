#pragma once

#include "protocol-parameters.h"
#include "protocol-setup.h"
#include <cstddef>
#include <string>

// Native codec diagnostics. These do not certify rare-tail security.
bool MeasureV17CodecLaws(const federated::ProtocolExperimentConfig& config,
    const federated::ProtocolSetupResult& setup, std::size_t trials,
    const std::string& outputPath);
