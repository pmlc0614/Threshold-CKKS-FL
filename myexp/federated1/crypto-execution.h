#ifndef FEDERATED1_CRYPTO_EXECUTION_H
#define FEDERATED1_CRYPTO_EXECUTION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

namespace federated {

constexpr std::size_t kMaximumCryptoThreads = 256;

enum class CryptoStage : std::size_t {
    ENCRYPTION = 0,
    CLIENT_TO_COLLECTIVE_KEY_SWITCH = 1,
    RLWE_AGGREGATION = 2,
    R2M = 3,
    MLWE_AGGREGATION = 4,
    M2R = 5,
    PARTIAL_DECRYPTION = 6,
    DECODING = 7,
    COUNT = 8,
};

constexpr std::size_t kCryptoStageCount =
    static_cast<std::size_t>(CryptoStage::COUNT);

const char* CryptoStageName(CryptoStage stage);

struct CryptoStageTiming {
    // Batch service elapsed time, excluding contention for the shared executor.
    double wallSeconds = 0.0;
    double executorWaitSeconds = 0.0;
    double workSeconds = 0.0;
    std::uint64_t taskCount = 0;
};

using CryptoStageTimings =
    std::array<CryptoStageTiming, kCryptoStageCount>;

class CryptoStageMetrics {
public:
    void Add(
        CryptoStage stage,
        double wallSeconds,
        double workSeconds,
        std::uint64_t taskCount,
        double executorWaitSeconds = 0.0);
    CryptoStageTimings Snapshot() const;
    void Reset();

private:
    mutable std::mutex mutex_;
    CryptoStageTimings timings_{};
};

/*
 * Reusable bounded executor for independent ciphertext work.  A batch queues
 * at most one runner per configured worker, keeps output ownership with the
 * caller, and propagates the first worker exception after the batch barrier.
 * This count bounds outer ciphertext workers.  OpenFHE operations with an
 * explicit tower-level OpenMP limit may still create nested workers, so this
 * is intentionally not described as a process-wide CPU-thread cap.
 */
class CryptoTaskExecutor {
public:
    explicit CryptoTaskExecutor(std::size_t threadCount);
    ~CryptoTaskExecutor();

    CryptoTaskExecutor(const CryptoTaskExecutor&) = delete;
    CryptoTaskExecutor& operator=(const CryptoTaskExecutor&) = delete;
    CryptoTaskExecutor(CryptoTaskExecutor&&) = delete;
    CryptoTaskExecutor& operator=(CryptoTaskExecutor&&) = delete;

    std::size_t ThreadCount() const noexcept;

    void ParallelFor(
        std::size_t itemCount,
        CryptoStage stage,
        CryptoStageMetrics* metrics,
        const std::function<void(std::size_t)>& task);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace federated

#endif
