#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

namespace federated::codec_timing {
// Diagnostic process-wide work counters. Parallel durations are summed, not
// wall time. Enable before process startup; do not use overlapping engines to
// attribute a snapshot difference to an individual round.
enum Stage { PublicPolynomial, Gaussian, FormatAndCRT, Quantize, Pack, Unpack,
             CodeSum, Reconstruct, GaussianSmall, GaussianLarge, GaussianRns, GaussianNtt, Count };
inline constexpr const char* names[] = {"public_polynomial", "gaussian",
    "format_and_crt", "quantize", "pack", "unpack", "code_sum", "reconstruct",
    "sampling_small", "sampling_large", "rns_lift", "ntt"};
inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("FL_CODEC_TIMING");
        return value && std::string(value) == "1";
    }();
    return enabled;
}
struct Counter { std::atomic<unsigned long long> ns{0}, calls{0}; };
inline std::array<Counter, Count> counters{};
struct Sample { unsigned long long ns = 0, calls = 0; };
using Snapshot = std::array<Sample, Count>;
inline Snapshot Read() {
    Snapshot result{};
    for (int i = 0; i < Count; ++i)
        result[i] = {counters[i].ns.load(std::memory_order_relaxed),
                     counters[i].calls.load(std::memory_order_relaxed)};
    return result;
}
class Scope {
 public:
    explicit Scope(Stage stage) : stage_(stage), enabled_(Enabled()) {
        if (enabled_) start_ = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (!enabled_) return;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
        counters[stage_].ns.fetch_add(elapsed, std::memory_order_relaxed);
        counters[stage_].calls.fetch_add(1, std::memory_order_relaxed);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
 private:
    Stage stage_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};
} // namespace federated::codec_timing
