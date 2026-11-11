#include "codec-timing.h"
#include "external-input-crypto-engine.h"
#include "protocol-contract.h"
#include "deterministic-randomness.h"
#include "coefficient-compression.h"
#include "coefficient-codec.h"
#include "protocol-randomness.h"
#include "support-bitmap.h"
#include "seeded-rlwe.h"
#include "protocol-crypto-context.h"
#include "protocol-parameters.h"

#include "openfhe.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace federated {
namespace {

template <typename T>
std::optional<T> ReadOption(const py::dict& options, const char* name) {
    const py::str key(name);
    if (!options.contains(key) || options[key].is_none()) {
        return std::nullopt;
    }
    return py::cast<T>(options[key]);
}

bool HasOption(const py::dict& options, const char* name) {
    const py::str key(name);
    return options.contains(key) && !options[key].is_none();
}

void ValidateOptionNames(const py::dict& options) {
    static const std::set<std::string> allowed{
        "profile",
        "protocol_mode",
        "protocol_version",
        "experimental_dynamic_cohort",
        "dropout_target_cap",
        "collusion_bound",
        "ring_dimension",
        "mlwe_rank",
        "q_modulus_bits",
        "p_modulus_bits",
        "hybrid_large_digits",
        "maximum_qp_bits",
        "model_scaling_factor",
        "aux_scaling_factor",
        "precision_bits",
        "aux_precision_bits",
        "selection_ratio",
        "selected_chunks_per_client",
        "update_bound",
        "aux_update_bound",
        "compression_c1",
        "compression_ca",
        "compression_c2",
        "compression_cd",
        "aux_compression_c1",
        "aux_compression_ca",
        "aux_compression_c2",
        "aux_compression_cd",
        "smudging_sigma",
        "aux_smudging_sigma",
        "model_delivery_scale",
        "aux_delivery_scale",
        "use_sparse_compression",
        "use_auxiliary_compression",
        "measure_pre_smudging_noise",
        "crypto_threads",
        "master_seed"};

    for (const auto item : options) {
        const std::string key = py::cast<std::string>(item.first);
        if (allowed.count(key) == 0) {
            throw std::invalid_argument(
                "unknown native engine option: " + key);
        }
    }
}

ProtocolExperimentConfig MakeConfiguredProtocol(const py::dict& options) {
    ValidateOptionNames(options);
    const std::string profile =
        ReadOption<std::string>(options, "profile")
            .value_or("resnet20-cifar10");

    ProtocolExperimentConfig config;
    if (profile == "resnet20-cifar10") {
        config = MakeResNet20Cifar10FixtureConfig();
    }
    else if (profile == "smoke") {
        config = MakeSmokeConfig();
    }
    else {
        throw std::invalid_argument(
            "profile must be 'resnet20-cifar10' or 'smoke'");
    }

    if (const auto value = ReadOption<std::string>(
            options, "protocol_version")) {
        if (*value != "v17") {
            throw std::invalid_argument(
                "protocol_version must be 'v17'");
        }
        config.protocolVersion = *value;
    }

    config.experimentalDynamicCohort = ReadOption<bool>(
        options, "experimental_dynamic_cohort").value_or(false);
    if (const auto cap = ReadOption<int>(options, "dropout_target_cap")) {
        if (*cap < 0) throw std::invalid_argument("dropout_target_cap must be nonnegative");
        config.dropoutTargetCap = static_cast<std::size_t>(*cap);
    }
    if (const auto c = ReadOption<int>(options, "collusion_bound")) {
        if (*c < 0) throw std::invalid_argument("collusion_bound must be nonnegative");
        config.sparsification.collusionBound = static_cast<std::size_t>(*c);
    }

    if (const auto value = ReadOption<std::string>(
            options, "protocol_mode")) {
        if (*value == "SP" || *value == "SPARSE") {
            config.mode = ProtocolMode::SP;
        }
        else if (*value == "FULL") {
            config.mode = ProtocolMode::FULL;
        }
        else {
            throw std::invalid_argument(
                "protocol_mode must be 'SPARSE' or 'FULL'");
        }
    }

    if (const auto value = ReadOption<std::size_t>(
            options, "ring_dimension")) {
        config.modulus.ringDimension = *value;
    }
    if (const auto value = ReadOption<std::size_t>(options, "mlwe_rank")) {
        config.sparsification.mlweRank = *value;
    }
    if (const auto value = ReadOption<std::vector<std::uint32_t>>(
            options, "q_modulus_bits")) {
        config.modulus.qModulusBits = *value;
    }
    if (const auto value = ReadOption<std::vector<std::uint32_t>>(
            options, "p_modulus_bits")) {
        config.modulus.pModulusBits = *value;
    }
    if (const auto value = ReadOption<std::size_t>(
            options, "hybrid_large_digits")) {
        config.modulus.hybridLargeDigits = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "maximum_qp_bits")) {
        config.modulus.maximumQpBits = *value;
    }
    for (const char* name : {"model_scaling_factor", "aux_scaling_factor"}) {
        if (options.contains(name) && !options[name].is_none()) {
            const auto value=options[name];
            if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
                throw std::invalid_argument("scale must be an exact Python integer");
            const auto parsed=exact_integer::ParseScale(py::str(value).cast<std::string>());
            (std::string(name)=="model_scaling_factor" ? config.scaling.factor : config.auxScaling.factor)=parsed;
        }
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "precision_bits")) {
        config.precisionBits = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "aux_precision_bits")) {
        config.auxPrecisionBits = *value;
    }

    if (HasOption(options, "selection_ratio") &&
        HasOption(options, "selected_chunks_per_client")) {
        throw std::invalid_argument(
            "set only selection_ratio or selected_chunks_per_client");
    }
    if (const auto value = ReadOption<double>(options, "selection_ratio")) {
        config.sparsification.selectionRatio =
            static_cast<long double>(*value);
        config.sparsification.selectedChunksPerClient = 0;
    }
    if (const auto value = ReadOption<std::size_t>(
            options, "selected_chunks_per_client")) {
        config.sparsification.selectionRatio = 0.0L;
        config.sparsification.selectedChunksPerClient = *value;
    }
    if (const auto value = ReadOption<double>(options, "update_bound")) {
        config.updateBound = static_cast<long double>(*value);
    }
    if (const auto value = ReadOption<double>(options, "aux_update_bound")) {
        config.auxUpdateBound = static_cast<long double>(*value);
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "compression_c1")) {
        config.compression.c1 = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "compression_ca")) {
        config.compression.ca = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "compression_c2")) {
        config.compression.c2 = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "compression_cd")) {
        config.compression.cd = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "aux_compression_c1")) {
        config.compression.auxC1 = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "aux_compression_ca")) {
        config.compression.auxCa = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "aux_compression_c2")) {
        config.compression.auxC2 = *value;
    }
    if (const auto value = ReadOption<std::uint32_t>(
            options, "aux_compression_cd")) {
        config.compression.auxCd = *value;
    }
    if (const auto value = ReadOption<double>(options, "smudging_sigma")) {
        config.smudging.sigmaSmg = static_cast<long double>(*value);
    }
    if (const auto value = ReadOption<double>(
            options, "aux_smudging_sigma")) {
        config.smudging.auxSigmaSmg = static_cast<long double>(*value);
    }
    if (const auto value = ReadOption<double>(
            options, "model_delivery_scale")) {
        config.modelDeliveryScale = static_cast<long double>(*value);
    }
    if (const auto value = ReadOption<double>(
            options, "aux_delivery_scale")) {
        config.auxDeliveryScale = static_cast<long double>(*value);
    }
    if (const auto value = ReadOption<py::bytes>(options, "master_seed")) {
        const std::string bytes = *value;
        if (bytes.size() != config.masterSeed.size()) {
            throw std::invalid_argument("master_seed must contain 32 bytes");
        }
        std::copy(bytes.begin(), bytes.end(), config.masterSeed.begin());
    }
    return config;
}

py::dict EffectiveProtocolPayload(const ProtocolExperimentConfig& config) {
    const DerivedDimensions dimensions = DeriveDimensions(config);
    py::dict result;
    result["protocol_version"] = config.protocolVersion;
    result["client_count"] = config.clientCount;
    result["model_dimension"] = config.modelDimension;
    result["auxiliary_dimension"] = config.syntheticUpdates.runningMeanDimension + config.syntheticUpdates.runningVarianceDimension;
    result["weight_denominator"] = config.weightDenominator;
    result["minimum_cohort_size"] = config.sparsification.minimumCohort;
    result["collusion_bound"] = config.sparsification.collusionBound;
    result["error_feedback_enabled"] = config.sparsification.errorFeedbackEnabled;
    result["setup_noise_sigmas"] = std::vector<double>{
        static_cast<double>(config.setupNoise.publicKeyShareStandardDeviation),
        static_cast<double>(config.setupNoise.m2rKeyShareStandardDeviation),
        static_cast<double>(config.setupNoise.clientKeySwitchStandardDeviation),
        static_cast<double>(config.setupNoise.encryptionStandardDeviation)};
    result["sampler_id"] = "exact-rational-truncated-dg-v3";
    result["experimental_dynamic_cohort"] = config.experimentalDynamicCohort;
    result["dropout_attempt_cap"] = 1;
    result["dropout_target_cap"] = config.dropoutTargetCap;
    result["protocol_mode"] =
        config.mode == ProtocolMode::SP ? "SPARSE" : "FULL";
    result["ring_dimension"] = config.modulus.ringDimension;
    result["mlwe_rank"] = config.sparsification.mlweRank;
    result["q_modulus_bits"] = config.modulus.qModulusBits;
    result["p_modulus_bits"] = config.modulus.pModulusBits;
    result["hybrid_large_digits"] = config.modulus.hybridLargeDigits;
    result["maximum_qp_bits"] = config.modulus.maximumQpBits;
    result["model_scaling_factor"] = py::module_::import("builtins").attr("int")(config.scaling.factor.str());
    result["aux_scaling_factor"] = py::module_::import("builtins").attr("int")(config.auxScaling.factor.str());
    result["precision_bits"] = config.precisionBits;
    result["aux_precision_bits"] = config.auxPrecisionBits;
    result["selection_ratio"] = static_cast<double>(
        config.sparsification.selectionRatio);
    result["selected_chunks_per_client"] =
        dimensions.selectedChunksPerClient;
    if (config.protocolVersion == "v17") {
        result["support_codec"] = config.mode == ProtocolMode::SP
            ? "CHUNK-BITMAP-MSB0-V1"
            : "NONE";
        result["support_codec_code"] =
            config.mode == ProtocolMode::SP ? 1 : 0;
        result["support_bitmap_bytes"] = config.mode == ProtocolMode::SP
            ? (dimensions.chunkCount + 7U) / 8U
            : 0U;
    }
    result["update_bound"] = static_cast<double>(config.updateBound);
    result["aux_update_bound"] = static_cast<double>(config.auxUpdateBound);
    result["compression_c1"] = config.compression.c1;
    result["compression_ca"] = config.compression.ca;
    result["compression_c2"] = config.compression.c2;
    result["compression_cd"] = config.compression.cd;
    result["aux_compression_c1"] = config.compression.auxC1;
    result["aux_compression_ca"] = config.compression.auxCa;
    result["aux_compression_c2"] = config.compression.auxC2;
    result["aux_compression_cd"] = config.compression.auxCd;
    result["smudging_sigma"] = static_cast<double>(
        config.smudging.sigmaSmg);
    result["aux_smudging_sigma"] = static_cast<double>(
        config.smudging.auxSigmaSmg);
    result["model_delivery_scale"] = static_cast<double>(
        config.modelDeliveryScale);
    result["aux_delivery_scale"] = static_cast<double>(
        config.auxDeliveryScale);
    return result;
}

ExternalEngineExecutionOptions MakeExecutionOptions(const py::dict& options) {
    ExternalEngineExecutionOptions result;
    if (const auto value = ReadOption<bool>(
            options, "use_sparse_compression")) {
        result.useSparseCompression = *value;
    }
    if (const auto value = ReadOption<bool>(
            options, "use_auxiliary_compression")) {
        result.useAuxiliaryCompression = *value;
    }
    if (const auto value = ReadOption<bool>(
            options, "measure_pre_smudging_noise")) {
        result.measurePreSmudgingNoise = *value;
    }
    if (const auto value = ReadOption<int>(options, "crypto_threads")) {
        if (*value <= 0 ||
            static_cast<std::size_t>(*value) > kMaximumCryptoThreads) {
            throw std::invalid_argument(
                "crypto_threads must be between 1 and 256");
        }
        result.cryptoThreads = static_cast<std::size_t>(*value);
    }
    return result;
}

LayoutKind ParseLayoutKind(const std::string& value) {
    if (value == "parameter") {
        return LayoutKind::PARAMETER;
    }
    if (value == "running_mean") {
        return LayoutKind::RUNNING_MEAN;
    }
    if (value == "running_variance") {
        return LayoutKind::RUNNING_VARIANCE;
    }
    throw std::invalid_argument("unknown tensor layout kind: " + value);
}

ProtocolTensorLayout ParseLayout(const py::dict& source) {
    ProtocolTensorLayout result;
    result.kind = ParseLayoutKind(py::cast<std::string>(source["kind"]));
    result.modelName = py::cast<std::string>(source["model_name"]);
    const py::sequence entries = py::cast<py::sequence>(source["entries"]);
    result.entries.reserve(entries.size());
    for (const py::handle item : entries) {
        const py::dict entry = py::cast<py::dict>(item);
        const std::string dtype = py::cast<std::string>(entry["dtype"]);
        if (dtype != "float32") {
            throw std::invalid_argument("v17 layouts require float32");
        }
        result.entries.push_back(ProtocolLayoutEntry{
            py::cast<std::string>(entry["name"]),
            py::cast<std::vector<std::uint64_t>>(entry["shape"]),
            py::cast<std::uint64_t>(entry["offset"]),
            py::cast<std::uint64_t>(entry["numel"]),
            WireDType::FLOAT32});
    }
    ValidateProtocolTensorLayout(result);
    return result;
}

std::vector<RegisteredClientWeightInput> ParseClients(
    const py::sequence& source) {

    std::vector<RegisteredClientWeightInput> result;
    result.reserve(source.size());
    for (const py::handle item : source) {
        const py::sequence pair = py::cast<py::sequence>(item);
        if (pair.size() != 2) {
            throw std::invalid_argument(
                "each client must be a (client_id, example_count) pair");
        }
        result.push_back(RegisteredClientWeightInput{
            ClientId{py::cast<std::uint64_t>(pair[0])},
            py::cast<std::uint64_t>(pair[1])});
    }
    return result;
}

std::vector<ActualChunkId> ParseChunkIds(const py::sequence& source) {
    std::vector<ActualChunkId> result;
    result.reserve(source.size());
    for (const py::handle item : source) {
        result.push_back(ActualChunkId{py::cast<std::uint64_t>(item)});
    }
    return result;
}

py::tuple ChunkIdsToTuple(const std::vector<ActualChunkId>& source) {
    py::tuple result(source.size());
    for (std::size_t position = 0; position < source.size(); ++position) {
        result[position] = py::int_(source[position].value);
    }
    return result;
}

template <std::size_t N>
py::bytes ToPythonBytes(const std::array<std::uint8_t, N>& source) {
    return py::bytes(
        reinterpret_cast<const char*>(source.data()),
        static_cast<py::ssize_t>(source.size()));
}

std::string PhaseName(RoundPhase phase) {
    switch (phase) {
        case RoundPhase::CREATED:
            return "created";
        case RoundPhase::SELECTION_COLLECTING:
            return "selection_collecting";
        case RoundPhase::ADMISSION_FINALIZED:
            return "admission_finalized";
        case RoundPhase::VALUES_COLLECTING:
            return "values_collecting";
        case RoundPhase::ENCRYPTED:
            return "encrypted";
        case RoundPhase::DECRYPTING:
            return "decrypting";
        case RoundPhase::COMMITTED:
            return "committed";
        case RoundPhase::ABORTED:
            return "aborted";
    }
    throw std::logic_error("unknown native round phase");
}

std::vector<float> CopyFloat32Array(
    const py::array& values,
    const std::vector<py::ssize_t>& expectedShape,
    const char* name) {

    if (!values.dtype().is(py::dtype::of<float>())) {
        throw py::type_error(std::string(name) + " must use float32");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument(
            std::string(name) + " must be C-contiguous");
    }
    const py::buffer_info info = values.request();
    if (info.ndim != static_cast<py::ssize_t>(expectedShape.size()) ||
        !std::equal(
            expectedShape.begin(), expectedShape.end(), info.shape.begin())) {
        throw std::invalid_argument(std::string(name) + " has an invalid shape");
    }
    const auto* begin = static_cast<const float*>(info.ptr);
    return std::vector<float>(begin, begin + info.size);
}

py::array_t<float> MakeFloat32Array(const std::vector<float>& source) {
    py::array_t<float> result(source.size());
    if (!source.empty()) {
        std::memcpy(
            result.mutable_data(),
            source.data(),
            source.size() * sizeof(float));
    }
    return result;
}

py::dict StageTimingsToDict(const CryptoStageTimings& timings) {
    py::dict result;
    for (std::size_t index = 0; index < kCryptoStageCount; ++index) {
        const auto stage = static_cast<CryptoStage>(index);
        const auto& timing = timings[index];
        py::dict item;
        item["wall_seconds"] = timing.wallSeconds;
        item["executor_wait_seconds"] = timing.executorWaitSeconds;
        item["elapsed_including_executor_wait_seconds"] =
            timing.wallSeconds + timing.executorWaitSeconds;
        item["work_seconds"] = timing.workSeconds;
        item["task_count"] = timing.taskCount;
        result[py::str(CryptoStageName(stage))] = std::move(item);
    }
    return result;
}

py::list ClientStageTimingsToList(
    const std::vector<CryptoStageTimings>& timings) {

    py::list result;
    for (const auto& clientTimings : timings) {
        result.append(StageTimingsToDict(clientTimings));
    }
    return result;
}

py::bytes ToPythonBytes(const ByteVector& value) {
    return py::bytes(reinterpret_cast<const char*>(value.data()), value.size());
}

// Stateless coefficient conformance surface; no keys or live engine state.
py::dict CoefficientScaleVectors(const py::dict& options, const std::vector<double>& values) {
    const auto config=MakeConfiguredProtocol(options);
    const auto cc=MakeProtocolCryptoContext(config, lbcrypto::HEStd_NotSet);
    auto encoded=coefficient_codec::EncodeExact(cc, values, config.scaling.factor);
    auto poly=encoded.CRTInterpolate();
    std::vector<std::string> rounded, residues;
    for(std::size_t i=0;i<values.size();++i) {
        rounded.push_back(exact_integer::RoundProduct(values[i],config.scaling.factor).str());
        residues.push_back(poly.GetValues()[i].ToString());
    }
    auto decoded=coefficient_codec::Decode(encoded, static_cast<double>(ScalingFactor(config.scaling)));
    decoded.resize(values.size());
    py::dict result;
    result["rounded"]=rounded;result["residues"]=residues;result["decoded"]=decoded;
    result["modulus"]=poly.GetModulus().ToString();
    return result;
}
// Pure conformance surface: accepts explicit test bytes, never reads live keys.
py::dict WireVectors(const py::dict& input) {
    using namespace deterministic_randomness;
    auto seed = [&](const char* name) {
        const std::string bytes = py::cast<py::bytes>(input[name]);
        if (bytes.size() != 16) throw std::invalid_argument("wire vector seed size");
        Seed128 result{};
        std::copy(bytes.begin(), bytes.end(), result.begin());
        return result;
    };
    const auto root = seed("seed");
    const SetupId setupId{seed("setup_id")};
    const RoundId round{py::cast<std::uint64_t>(input["round_id"])};
    const ClientId client{py::cast<std::uint64_t>(input["client_id"])};
    const auto block = py::cast<std::uint64_t>(input["block"]);
    const auto n = py::cast<std::size_t>(input["dimension"]);
    const auto j = py::cast<std::size_t>(input["chunks"]);
    if (n == 0 || n > 4096 || j == 0 || j > 4096)
        throw std::invalid_argument("wire vector dimensions");
    const auto prime = py::cast<std::uint64_t>(input["prime"]);
    py::dict tags, masks;
    auto record = [&](const std::string& name, const ByteVector& tag) {
        tags[py::str(name)] = ContractHex(tag);
        masks[py::str(name)] = GenerateUniformPolynomial(root, tag, n, prime);
    };
    for (const std::string path : {"FULL-MODEL", "SP-MODEL", "AUX"}) {
        record("local/" + path, V17LocalCiphertextATag(setupId, round, client, block, path));
        record("offset/" + path, V17PublicOffsetTag(setupId, round, path, block));
        record("dither/" + path, V17PublicDitherTag(setupId, round, path, block, client));
    }
    ByteVector pk{'P','K','-','P','1'};
    pk.insert(pk.end(), setupId.bytes.begin(), setupId.bytes.end());
    record("PK-P1", pk);
    ByteVector m2r{'M','2','R','-','A'};
    m2r.insert(m2r.end(), setupId.bytes.begin(), setupId.bytes.end());
    for (auto value : {block, client.value}) {
        const auto bytes = EncodeBigEndian64(value);
        m2r.insert(m2r.end(), bytes.begin(), bytes.end());
    }
    record("M2R-A", m2r);
    const auto key = seed("index_key");
    const auto codes = py::cast<std::vector<std::uint64_t>>(input["codes"]);
    const auto bits = py::cast<std::uint32_t>(input["bits"]);
    auto packed = coefficient_compression::PackUnsignedValues(codes, bits);
    if (input.contains("packed_override")) {
        const std::string raw = py::cast<py::bytes>(input["packed_override"]);
        packed.assign(raw.begin(), raw.end());
    }
    std::vector<ShuffledChunkLabel> labels;
    for (auto value : py::cast<std::vector<std::uint64_t>>(input["labels"])) labels.push_back({value});
    auto bitmap = EncodeSupportBitmapMsb0(j, labels);
    if (input.contains("bitmap_override")) {
        const std::string raw = py::cast<py::bytes>(input["bitmap_override"]);
        bitmap.assign(raw.begin(), raw.end());
    }
    std::vector<std::uint64_t> decodedLabels;
    for (const auto label : DecodeSupportBitmapMsb0(j, bitmap)) decodedLabels.push_back(label.value);
    py::dict result;
    result["tags"] = tags; result["masks"] = masks;
    const auto permutationSeed = DeriveIndexPermutationSeed(key, setupId, round, j);
    result["index_seed"] = ContractHex(ByteVector(permutationSeed.begin(), permutationSeed.end()));
    result["permutation"] = GenerateIndexPermutation(key, setupId, round, j);
    result["packed"] = ContractHex(packed);
    result["unpacked"] = coefficient_compression::UnpackUnsignedValues(packed, codes.size(), bits);
    result["bitmap"] = ContractHex(bitmap); result["labels"] = decodedLabels;
    result["slots"] = SparseStreamCoefficientIndices(n,
        py::cast<std::size_t>(input["rank"]), py::cast<std::size_t>(input["active_slots"]));
    return result;
}

class NativeEngineBinding {
public:
    explicit NativeEngineBinding(py::dict options)
        : options_(std::move(options)) {
        ValidateOptionNames(options_);
    }

    py::dict Setup(
        const py::dict& parameterLayout,
        const py::dict& runningMeanLayout,
        const py::dict& runningVarianceLayout,
        const py::sequence& clients,
        std::uint64_t chunkSize,
        std::size_t minimumCohortSize) {

        ExternalEngineSetupRequest request;
        request.protocol = MakeConfiguredProtocol(options_);
        request.parameterLayout = ParseLayout(parameterLayout);
        request.runningMeanLayout = ParseLayout(runningMeanLayout);
        request.runningVarianceLayout = ParseLayout(runningVarianceLayout);
        request.clients = ParseClients(clients);
        request.protocol.sparsification.minimumCohort = minimumCohortSize;
        request.execution = MakeExecutionOptions(options_);

        request.protocol.modelDimension = static_cast<std::size_t>(
            request.parameterLayout.entries.back().offset +
            request.parameterLayout.entries.back().numel);
        const DerivedDimensions dimensions = DeriveDimensions(request.protocol);
        const std::size_t expectedBlockSize =
            request.protocol.mode == ProtocolMode::SP
                ? dimensions.smallRingDimension
                : request.protocol.modulus.ringDimension;
        if (chunkSize != expectedBlockSize) {
            throw std::invalid_argument(
                "Python block size does not match the protocol mode");
        }

        ExternalEngineSetupResult setup;
        const auto setupStarted = std::chrono::steady_clock::now();
        {
            py::gil_scoped_release release;
            auto context = MakeProtocolCryptoContext(request.protocol);
            auto engine = std::make_unique<ExternalInputCryptoEngine>(context);
            setup = engine->Setup(request);
            engine_ = std::move(engine);
        }
        const double setupSequentialSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setupStarted).count();
        setup_ = setup;

        py::dict result;
        result["setup_id"] = ToPythonBytes(setup.setupId.bytes);
        result["crs_seed"] = ToPythonBytes(setup.crsSeed);
        result["setup_descriptor"] = ToPythonBytes(setup.setupDescriptor);
        result["public_setup_binding"] = ToPythonBytes(setup.publicSetupBinding);
        result["setup_descriptor_bytes"] = setup.setupDescriptorBytes;
        result["q_primes"] = setup.qPrimes;
        result["p_primes"] = setup.pPrimes;
        result["protocol_mode"] =
            engine_->EffectiveProtocolConfig().mode == ProtocolMode::SP
                ? "SPARSE"
                : "FULL";
        result["parameter_layout_hash"] =
            ToPythonBytes(setup.parameterLayoutHash);
        result["running_mean_layout_hash"] =
            ToPythonBytes(setup.runningMeanLayoutHash);
        result["running_variance_layout_hash"] =
            ToPythonBytes(setup.runningVarianceLayoutHash);
        result["parameter_numel"] = setup.parameterNumel;
        result["running_mean_numel"] = setup.runningMeanNumel;
        result["running_variance_numel"] = setup.runningVarianceNumel;
        result["chunk_size"] = setup.chunkSize;
        result["chunk_count"] = setup.chunkCount;
        py::list clientIds;
        for (const auto clientId : setup.clientIds) {
            clientIds.append(clientId.value);
        }
        result["client_ids"] = std::move(clientIds);
        result["alphas"] = setup.alphas;
        const double clientInitialTotal = std::accumulate(
            setup.clientInitialSetupSeconds.begin(),
            setup.clientInitialSetupSeconds.end(),
            0.0);
        const double clientCollectiveTotal = std::accumulate(
            setup.clientCollectiveKeyGenerationSeconds.begin(),
            setup.clientCollectiveKeyGenerationSeconds.end(),
            0.0);
        const double setupServerSeconds = std::max(
            0.0,
            setupSequentialSeconds - clientInitialTotal -
                clientCollectiveTotal);
        const double maximumInitialClientSeconds = *std::max_element(
            setup.clientInitialSetupSeconds.begin(),
            setup.clientInitialSetupSeconds.end());
        const double maximumCollectiveClientSeconds = *std::max_element(
            setup.clientCollectiveKeyGenerationSeconds.begin(),
            setup.clientCollectiveKeyGenerationSeconds.end());
        py::dict setupTiming;
        setupTiming["model"] =
            "barrier-critical-path-independent-clients-v1";
        setupTiming["network_latency_included"] = false;
        setupTiming["sequential_seconds"] = setupSequentialSeconds;
        setupTiming["server_seconds"] = setupServerSeconds;
        setupTiming["client_initial_setup_seconds"] =
            setup.clientInitialSetupSeconds;
        setupTiming["client_collective_key_generation_seconds"] =
            setup.clientCollectiveKeyGenerationSeconds;
        setupTiming["maximum_initial_client_seconds"] =
            maximumInitialClientSeconds;
        setupTiming["maximum_collective_key_generation_seconds"] =
            maximumCollectiveClientSeconds;
        setupTiming["estimated_parallel_seconds"] =
            setupServerSeconds + maximumInitialClientSeconds +
            maximumCollectiveClientSeconds;
        result["setup_timing"] = std::move(setupTiming);
        py::dict effectiveProtocol = EffectiveProtocolPayload(
            engine_->EffectiveProtocolConfig());
        effectiveProtocol["setup_descriptor_bytes"] =
            setup.setupDescriptorBytes;
        effectiveProtocol["q_primes"] = setup.qPrimes;
        effectiveProtocol["p_primes"] = setup.pPrimes;
        result["effective_protocol"] = std::move(effectiveProtocol);
        return result;
    }

    py::object Phase() const {
        RequireEngine();
        const auto phase = engine_->Phase();
        if (!phase.has_value()) {
            return py::none();
        }
        return py::str(PhaseName(*phase));
    }

    py::object LastAbortedRoundTraffic() const {
        RequireEngine();
        const auto traffic = engine_->LastAbortedRoundTraffic();
        if (!traffic.has_value()) {
            return py::none();
        }
        py::dict result;
        result["round_id"] = traffic->roundId.value;
        result["sparse_client_ciphertext_count"] =
            traffic->sparseClientCiphertextCount;
        result["auxiliary_client_ciphertext_count"] =
            traffic->auxiliaryClientCiphertextCount;
        result["auxiliary_aggregate_ciphertext_count"] =
            traffic->auxiliaryAggregateCiphertextCount;
        result["sparse_server_ciphertext_count"] =
            traffic->sparseServerCiphertextCount;
        result["upload_accepted_chunk_count"] =
            traffic->uploadAcceptedChunkCount;
        result["accepted_chunk_count"] = traffic->acceptedChunkCount;
        result["uploaded_sparse_contribution_count"] =
            traffic->uploadedSparseContributionCount;
        result["accepted_sparse_contribution_count"] =
            traffic->acceptedSparseContributionCount;
        result["selection_bitmap_count"] =
            traffic->selectionBitmapCount;
        result["admission_bitmap_recipient_count"] =
            traffic->admissionBitmapRecipientCount;
        result["upload_support_bitmap_count"] =
            traffic->uploadSupportBitmapCount;
        py::list survivors;
        for (const auto clientId : traffic->survivorClientIds) {
            survivors.append(clientId.value);
        }
        result["survivor_client_ids"] = std::move(survivors);
        py::list dropped;
        for (const auto clientId : traffic->droppedClientIds) {
            dropped.append(clientId.value);
        }
        result["dropped_client_ids"] = std::move(dropped);
        result["survivor_new_target_sizes"] =
            traffic->survivorNewTargetSizes;
        return result;
    }

    void BeginRound(
        std::uint64_t roundId,
        std::optional<std::uint64_t> logicalRoundIndex) {
        RequireEngine();
        engine_->BeginRound(RoundId{roundId}, logicalRoundIndex);
    }

    void SubmitSelection(
        std::uint64_t clientId,
        const py::sequence& selectedChunkIds) {

        RequireEngine();
        ClientSelectionRequest request;
        request.setupId = setup_->setupId;
        request.roundId = CurrentRoundId();
        request.clientId = ClientId{clientId};
        request.parameterLayoutHash = setup_->parameterLayoutHash;
        request.selectedActualChunkIds = ParseChunkIds(selectedChunkIds);
        engine_->SubmitSelection(request);
    }

    py::dict FinalizeAdmission() {
        RequireEngine();
        ExternalEngineAdmissionPlan admission;
        {
            py::gil_scoped_release release;
            admission = engine_->FinalizeAdmission();
        }

        py::dict result;
        result["setup_id"] = ToPythonBytes(admission.setupId.bytes);
        result["round_id"] = admission.roundId.value;
        result["accepted_actual_chunk_ids"] =
            ChunkIdsToTuple(admission.acceptedActualChunkIds);
        result["cohort_sizes"] = admission.cohortSizes;
        py::list clients;
        for (const auto& client : admission.clientResults) {
            py::dict item;
            item["setup_id"] = ToPythonBytes(client.setupId.bytes);
            item["round_id"] = client.roundId.value;
            item["client_id"] = client.clientId.value;
            item["accepted_actual_chunk_ids"] =
                ChunkIdsToTuple(client.acceptedActualChunkIds);
            clients.append(std::move(item));
        }
        result["client_results"] = std::move(clients);
        return result;
    }

    void SubmitAcceptedValues(
        std::uint64_t clientId,
        const py::sequence& acceptedChunkIds,
        const py::array& values) {

        RequireEngine();
        const auto chunkIds = ParseChunkIds(acceptedChunkIds);
        AcceptedChunkValues payload;
        payload.setupId = setup_->setupId;
        payload.roundId = CurrentRoundId();
        payload.clientId = ClientId{clientId};
        payload.chunkSize = setup_->chunkSize;
        payload.actualChunkIds = chunkIds;
        payload.values = CopyFloat32Array(
            values,
            {static_cast<py::ssize_t>(chunkIds.size()),
             static_cast<py::ssize_t>(setup_->chunkSize)},
            "accepted values");
        engine_->SubmitAcceptedValues(payload);
    }

    void SubmitFullParameterDelta(
        std::uint64_t clientId,
        const py::array& values) {

        RequireEngine();
        ExternalFullParameterDeltaPayload payload;
        payload.setupId = setup_->setupId;
        payload.roundId = CurrentRoundId();
        payload.clientId = ClientId{clientId};
        payload.parameterDelta = CopyFloat32Array(
            values,
            {static_cast<py::ssize_t>(setup_->parameterNumel)},
            "FULL parameter delta");
        engine_->SubmitFullParameterDelta(payload);
    }

    void SubmitAuxiliaryDeltas(
        std::uint64_t clientId,
        const py::array& runningMeanDelta,
        const py::array& runningVarianceDelta) {

        RequireEngine();
        AuxiliaryDeltaPayload payload;
        payload.setupId = setup_->setupId;
        payload.roundId = CurrentRoundId();
        payload.clientId = ClientId{clientId};
        payload.runningMeanDelta = CopyFloat32Array(
            runningMeanDelta,
            {static_cast<py::ssize_t>(setup_->runningMeanNumel)},
            "running mean delta");
        payload.runningVarianceDelta = CopyFloat32Array(
            runningVarianceDelta,
            {static_cast<py::ssize_t>(setup_->runningVarianceNumel)},
            "running variance delta");
        engine_->SubmitAuxiliaryDeltas(payload);
    }

    py::dict SetRoundSurvivors(const py::sequence& survivorClientIds) {
        RequireEngine();
        std::vector<ClientId> survivors;
        survivors.reserve(static_cast<std::size_t>(
            py::len(survivorClientIds)));
        for (const auto item : survivorClientIds) {
            survivors.push_back(ClientId{py::cast<std::uint64_t>(item)});
        }
        ExternalEngineSurvivorTransition transition;
        {
            py::gil_scoped_release release;
            transition = engine_->SetRoundSurvivors(survivors);
        }
        py::dict result;
        result["setup_id"] = ToPythonBytes(transition.setupId.bytes);
        result["round_id"] = transition.roundId.value;
        py::list survivorIds;
        for (const auto clientId : transition.survivorClientIds) {
            survivorIds.append(clientId.value);
        }
        py::list droppedIds;
        for (const auto clientId : transition.droppedClientIds) {
            droppedIds.append(clientId.value);
        }
        result["survivor_client_ids"] = std::move(survivorIds);
        result["dropped_client_ids"] = std::move(droppedIds);
        result["accepted_actual_chunk_ids"] = ChunkIdsToTuple(
            transition.admission.acceptedActualChunkIds);
        result["cohort_sizes"] = transition.admission.cohortSizes;
        py::list clients;
        for (const auto& client : transition.admission.clientResults) {
            py::dict item;
            item["setup_id"] = ToPythonBytes(client.setupId.bytes);
            item["round_id"] = client.roundId.value;
            item["client_id"] = client.clientId.value;
            item["accepted_actual_chunk_ids"] = ChunkIdsToTuple(
                client.acceptedActualChunkIds);
            clients.append(std::move(item));
        }
        result["client_results"] = std::move(clients);
        result["target_key_cache_hit"] = transition.targetKeyCacheHit;
        result["server_target_reconstruction_seconds"] =
            transition.serverTargetReconstructionSeconds;
        result["client_collective_key_generation_seconds"] =
            transition.clientCollectiveKeyGenerationSeconds;
        result["maximum_client_collective_key_generation_seconds"] =
            transition.maximumClientCollectiveKeyGenerationSeconds;
        return result;
    }

    std::vector<std::uint64_t> PrepareFinalization() {
        RequireEngine();
        return engine_->PrepareFinalization();
    }
    void AcknowledgeFinalization(std::uint64_t clientId,
        const std::vector<std::uint64_t>& context) {
        RequireEngine();
        engine_->AcknowledgeFinalization(ClientId{clientId}, context);
    }

    py::dict FinalizeRound(
        const py::sequence& postRequestDropoutIds) {
        RequireEngine();
        std::vector<ClientId> dropouts;
        dropouts.reserve(static_cast<std::size_t>(
            py::len(postRequestDropoutIds)));
        for (const auto item : postRequestDropoutIds) {
            dropouts.push_back(ClientId{py::cast<std::uint64_t>(item)});
        }
        const auto codecBefore = codec_timing::Read();
        ExternalEngineRoundResult round;
        {
            py::gil_scoped_release release;
            round = engine_->FinalizeRound(dropouts);
        }

        py::dict result;
        result["setup_id"] = ToPythonBytes(round.setupId.bytes);
        result["round_id"] = round.roundId.value;
        result["accepted_actual_chunk_ids"] =
            ChunkIdsToTuple(round.acceptedActualChunkIds);
        result["cohort_sizes"] = round.cohortSizes;
        result["parameter_delta"] = MakeFloat32Array(round.parameterDelta);
        py::dict phaseAudit;
        for (const auto& item : {std::make_pair("model", &round.modelPhaseAudit),
                                 std::make_pair("aux", &round.auxPhaseAudit)}) {
            py::dict path;
            path["phase_residues"]=item.second->phaseResidues;
            path["integer_references"]=item.second->integerReferences;
            phaseAudit[item.first]=path;
        }
        result["phase_audit"]=phaseAudit;
        if (codec_timing::Enabled()) {
            const auto after = codec_timing::Read();
            py::dict timings;
            for (int i = 0; i < codec_timing::GaussianSmall; ++i) {
                py::dict row;
                row["work_seconds"] = (after[i].ns - codecBefore[i].ns) / 1e9;
                row["calls"] = after[i].calls - codecBefore[i].calls;
                if (i == codec_timing::Gaussian) {
                    py::dict components;
                    for (int j = codec_timing::GaussianSmall; j < codec_timing::Count; ++j) {
                        py::dict component;
                        component["work_seconds"] = (after[j].ns - codecBefore[j].ns) / 1e9;
                        component["calls"] = after[j].calls - codecBefore[j].calls;
                        components[codec_timing::names[j]] = component;
                    }
                    row["components"] = components;
                }
                timings[codec_timing::names[i]] = row;
            }
            result["codec_work_timings"] = timings;
        }

        result["running_mean_delta"] =
            MakeFloat32Array(round.runningMeanDelta);
        result["running_variance_delta"] =
            MakeFloat32Array(round.runningVarianceDelta);
        result["sparse_plan_hash"] = round.sparsePlanHash;
        result["sparse_client_ciphertext_count"] =
            round.sparseClientCiphertextCount;
        result["sparse_survivor_client_ciphertext_count"] =
            round.sparseSurvivorClientCiphertextCount;
        result["sparse_discarded_client_ciphertext_count"] =
            round.sparseDiscardedClientCiphertextCount;
        result["accepted_sparse_contribution_count"] =
            round.acceptedSparseContributionCount;
        result["aggregated_mlwe_ciphertext_count"] =
            round.aggregatedMLWECiphertextCount;
        result["sparse_server_ciphertext_count"] =
            round.sparseServerCiphertextCount;
        result["selection_bitmap_count"] = round.selectionBitmapCount;
        result["admission_bitmap_recipient_count"] =
            round.admissionBitmapRecipientCount;
        result["upload_support_bitmap_count"] =
            round.uploadSupportBitmapCount;
        result["result_recipient_count"] = round.resultRecipientCount;
        result["full_model_client_ciphertext_count"] =
            round.fullModelClientCiphertextCount;
        result["full_model_aggregate_ciphertext_count"] =
            round.fullModelAggregateCiphertextCount;
        result["r2m_ciphertext_count"] = round.r2mCiphertextCount;
        result["m2r_ciphertext_count"] = round.m2rCiphertextCount;
        result["auxiliary_client_ciphertext_count"] =
            round.auxiliaryClientCiphertextCount;
        result["auxiliary_survivor_client_ciphertext_count"] =
            round.auxiliarySurvivorClientCiphertextCount;
        result["auxiliary_discarded_client_ciphertext_count"] =
            round.auxiliaryDiscardedClientCiphertextCount;
        result["auxiliary_aggregate_ciphertext_count"] =
            round.auxiliaryAggregateCiphertextCount;
        result["crypto_sequential_seconds"] = round.cryptoSequentialSeconds;
        result["crypto_timing_model"] = "client-batch-service-v3";
        result["crypto_server_seconds"] = round.cryptoServerSeconds;
        result["estimated_parallel_crypto_seconds"] =
            round.estimatedParallelCryptoSeconds;
        result["client_encryption_seconds"] =
            round.clientEncryptionSeconds;
        result["client_key_switch_seconds"] =
            round.clientKeySwitchSeconds;
        result["client_partial_decryption_seconds"] =
            round.clientPartialDecryptionSeconds;
        result["client_total_crypto_seconds"] =
            round.clientTotalCryptoSeconds;
        result["maximum_client_pre_aggregation_seconds"] =
            round.maximumClientPreAggregationSeconds;
        result["maximum_client_partial_decryption_seconds"] =
            round.maximumClientPartialDecryptionSeconds;
        result["maximum_client_total_crypto_seconds"] =
            round.maximumClientTotalCryptoSeconds;
        result["crypto_thread_count"] = round.cryptoThreadCount;
        result["crypto_stage_timings"] =
            StageTimingsToDict(round.cryptoStageTimings);
        result["full_model_client_stage_timings"] =
            ClientStageTimingsToList(round.fullModelClientStageTimings);
        result["sparse_model_client_stage_timings"] =
            ClientStageTimingsToList(round.sparseModelClientStageTimings);
        result["auxiliary_client_stage_timings"] =
            ClientStageTimingsToList(round.auxiliaryClientStageTimings);
        result["full_model_server_stage_timings"] =
            StageTimingsToDict(round.fullModelServerStageTimings);
        result["sparse_model_server_stage_timings"] =
            StageTimingsToDict(round.sparseModelServerStageTimings);
        result["auxiliary_server_stage_timings"] =
            StageTimingsToDict(round.auxiliaryServerStageTimings);
        result["survivor_reconfiguration_applied"] =
            round.survivorReconfigurationApplied;
        result["survivor_target_key_cache_hit"] =
            round.survivorTargetKeyCacheHit;
        result["survivor_new_target_sizes"] =
            round.survivorNewTargetSizes;
        py::list survivorIds;
        for (const auto clientId : round.survivorClientIds) {
            survivorIds.append(clientId.value);
        }
        py::list droppedIds;
        for (const auto clientId : round.droppedClientIds) {
            droppedIds.append(clientId.value);
        }
        result["survivor_client_ids"] = std::move(survivorIds);
        result["dropped_client_ids"] = std::move(droppedIds);
        result["survivor_target_reconstruction_seconds"] =
            round.survivorTargetReconstructionSeconds;
        result["survivor_collective_key_generation_seconds"] =
            round.survivorCollectiveKeyGenerationSeconds;
        result["maximum_survivor_collective_key_generation_seconds"] =
            round.maximumSurvivorCollectiveKeyGenerationSeconds;
        result["survivor_reconfiguration_sequential_seconds"] =
            round.survivorReconfigurationSequentialSeconds;
        return result;
    }

    void CommitRound() {
        RequireEngine();
        engine_->CommitRound();
    }

    void AbortRound() {
        RequireEngine();
        engine_->AbortRound();
    }

private:
    void RequireEngine() const {
        if (!engine_ || !setup_.has_value()) {
            throw std::logic_error("native engine setup has not been installed");
        }
    }

    RoundId CurrentRoundId() const {
        RequireEngine();
        const auto phase = engine_->Phase();
        if (!phase.has_value()) {
            throw std::logic_error("native engine round has not begun");
        }
        // The binding owns message construction; the public API supplies the
        // same round ID to BeginRound and no other active round can coexist.
        return currentRoundId_;
    }

public:
    void BeginRoundAndRemember(
        std::uint64_t roundId,
        std::optional<std::uint64_t> logicalRoundIndex) {
        BeginRound(roundId, logicalRoundIndex);
        currentRoundId_ = RoundId{roundId};
    }

private:
    py::dict options_;
    std::unique_ptr<ExternalInputCryptoEngine> engine_;
    std::optional<ExternalEngineSetupResult> setup_;
    RoundId currentRoundId_{};
};

}  // namespace
}  // namespace federated

PYBIND11_MODULE(ciphertext_fl_cpp, module) {
    using federated::NativeEngineBinding;

    module.doc() =
        "OpenFHE external-input federated crypto engine (protocol v17)";
    module.attr("BINDING_ABI_VERSION") =
        py::int_(federated::kBindingAbiVersion);
    module.attr("PROTOCOL_VERSION") = py::str(federated::kProtocolVersion);

    module.def("_gaussian_samples", [](double sigma, std::size_t count) {
        if (count>65536) throw std::invalid_argument("sample count too large");
        std::vector<std::string> result;
        result.reserve(count);
        for (std::size_t i=0;i<count;++i)
            result.push_back(federated::SampleCertifiedV17GaussianInteger(sigma).str());
        return result;
    });
    module.def("_gaussian_polynomial", [](const py::dict& options, double sigma) {
        const auto config=federated::MakeConfiguredProtocol(options);
        const auto cc=federated::MakeProtocolCryptoContext(config, lbcrypto::HEStd_NotSet);
        auto poly=federated::SampleProtocolGaussianPolynomial(cc->GetElementParams(),sigma,true);
        poly.SetFormat(Format::COEFFICIENT);
        const auto crt=poly.CRTInterpolate();
        std::vector<std::string> residues;
        for (std::size_t i=0;i<crt.GetLength();++i) residues.push_back(crt[i].ToString());
        py::dict result;
        result["modulus"]=crt.GetModulus().ToString();result["residues"]=residues;
        return result;
    });
    module.def("_coefficient_scale_vectors", &federated::CoefficientScaleVectors);

    module.def("_wire_vectors", &federated::WireVectors, py::arg("input"));

    py::class_<NativeEngineBinding>(module, "NativeCryptoEngine")
        .def(py::init<py::dict>(), py::arg("options") = py::dict())
        .def("setup", &NativeEngineBinding::Setup,
             py::arg("parameter_layout"),
             py::arg("running_mean_layout"),
             py::arg("running_variance_layout"),
             py::arg("clients"),
             py::arg("chunk_size"),
             py::arg("minimum_cohort_size"))
        .def_property_readonly("phase", &NativeEngineBinding::Phase)
        .def_property_readonly(
            "last_aborted_round_traffic",
            &NativeEngineBinding::LastAbortedRoundTraffic)
        .def("begin_round", &NativeEngineBinding::BeginRoundAndRemember,
             py::arg("round_id"),
             py::arg("logical_round_index") = py::none())
        .def("submit_selection", &NativeEngineBinding::SubmitSelection,
             py::arg("client_id"),
             py::arg("selected_actual_chunk_ids"))
        .def("finalize_admission", &NativeEngineBinding::FinalizeAdmission)
        .def("submit_accepted_values",
             &NativeEngineBinding::SubmitAcceptedValues,
             py::arg("client_id"),
             py::arg("accepted_actual_chunk_ids"),
             py::arg("values"))
        .def("submit_full_parameter_delta",
             &NativeEngineBinding::SubmitFullParameterDelta,
             py::arg("client_id"),
             py::arg("values"))
        .def("submit_auxiliary_deltas",
             &NativeEngineBinding::SubmitAuxiliaryDeltas,
             py::arg("client_id"),
             py::arg("running_mean_delta"),
             py::arg("running_variance_delta"))
        .def("set_round_survivors",
             &NativeEngineBinding::SetRoundSurvivors,
             py::arg("survivor_client_ids"))
        .def("prepare_finalization", &NativeEngineBinding::PrepareFinalization)
        .def("acknowledge_finalization", &NativeEngineBinding::AcknowledgeFinalization)
        .def("finalize_round", &NativeEngineBinding::FinalizeRound,
             py::arg("post_request_dropouts") = py::tuple())
        .def("commit_round", &NativeEngineBinding::CommitRound)
        .def("abort_round", &NativeEngineBinding::AbortRound);
}
