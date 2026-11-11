#include "crypto-execution.h"

#include "utils/parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace federated {
namespace {

std::size_t StageIndex(CryptoStage stage) {
    const std::size_t index = static_cast<std::size_t>(stage);
    if (index >= kCryptoStageCount) {
        throw std::invalid_argument("invalid crypto stage");
    }
    return index;
}

thread_local const void* gActiveCryptoExecutor = nullptr;

}  // namespace

const char* CryptoStageName(CryptoStage stage) {
    switch (stage) {
        case CryptoStage::ENCRYPTION:
            return "encryption";
        case CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH:
            return "client_to_collective_key_switch";
        case CryptoStage::RLWE_AGGREGATION:
            return "rlwe_aggregation";
        case CryptoStage::R2M:
            return "r2m";
        case CryptoStage::MLWE_AGGREGATION:
            return "mlwe_aggregation";
        case CryptoStage::M2R:
            return "m2r";
        case CryptoStage::PARTIAL_DECRYPTION:
            return "partial_decryption";
        case CryptoStage::DECODING:
            return "decoding";
        case CryptoStage::COUNT:
            break;
    }
    throw std::invalid_argument("invalid crypto stage");
}

void CryptoStageMetrics::Add(
    CryptoStage stage,
    double wallSeconds,
    double workSeconds,
    std::uint64_t taskCount,
    double executorWaitSeconds) {

    if (!std::isfinite(wallSeconds) || wallSeconds < 0.0 ||
        !std::isfinite(workSeconds) || workSeconds < 0.0 ||
        !std::isfinite(executorWaitSeconds) || executorWaitSeconds < 0.0) {
        throw std::invalid_argument("crypto stage timings must be finite and non-negative");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& timing = timings_.at(StageIndex(stage));
    timing.wallSeconds += wallSeconds;
    timing.executorWaitSeconds += executorWaitSeconds;
    timing.workSeconds += workSeconds;
    timing.taskCount += taskCount;
}

CryptoStageTimings CryptoStageMetrics::Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timings_;
}

void CryptoStageMetrics::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_ = {};
}

struct CryptoTaskExecutor::Impl {
    struct BatchState {
        explicit BatchState(std::size_t runnerCount)
            : remaining(runnerCount), workSeconds(runnerCount, 0.0) {}

        std::mutex mutex;
        std::condition_variable completed;
        std::size_t remaining;
        std::vector<double> workSeconds;
        std::exception_ptr firstException;
    };

    explicit Impl(std::size_t configuredThreadCount)
        : threadCount(configuredThreadCount) {
        if (threadCount == 0 || threadCount > kMaximumCryptoThreads) {
            throw std::invalid_argument(
                "crypto thread count must be between 1 and 256");
        }
        if (threadCount == 1) {
            return;
        }
        workers.reserve(threadCount);
        try {
            for (std::size_t index = 0; index < threadCount; ++index) {
                workers.emplace_back([this]() { WorkerLoop(); });
            }
        }
        catch (...) {
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                stopping = true;
            }
            queueReady.notify_all();
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            stopping = true;
        }
        queueReady.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void Enqueue(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (stopping) {
                throw std::logic_error("crypto executor is stopping");
            }
            queue.push_back(std::move(job));
        }
        queueReady.notify_one();
    }

    void WorkerLoop() {
        gActiveCryptoExecutor = this;
        lbcrypto::OpenFHEParallelControls.SetNumThreads(1);
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueReady.wait(lock, [this]() {
                    return stopping || !queue.empty();
                });
                if (stopping && queue.empty()) {
                    break;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            job();
        }
        gActiveCryptoExecutor = nullptr;
    }

    const std::size_t threadCount;
    std::mutex parallelForMutex;
    std::mutex queueMutex;
    std::condition_variable queueReady;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    bool stopping = false;
};

CryptoTaskExecutor::CryptoTaskExecutor(std::size_t threadCount)
    : impl_(std::make_unique<Impl>(threadCount)) {}

CryptoTaskExecutor::~CryptoTaskExecutor() = default;

std::size_t CryptoTaskExecutor::ThreadCount() const noexcept {
    return impl_->threadCount;
}

void CryptoTaskExecutor::ParallelFor(
    std::size_t itemCount,
    CryptoStage stage,
    CryptoStageMetrics* metrics,
    const std::function<void(std::size_t)>& task) {

    (void)StageIndex(stage);
    if (itemCount == 0) {
        return;
    }
    if (!task) {
        throw std::invalid_argument("crypto parallel task is empty");
    }

    const auto started = std::chrono::steady_clock::now();
    // Serialize independent batches even for a one-worker executor. Never mix
    // different clients' work in one batch. Recursive calls already own it.
    std::unique_lock<std::mutex> batchLock(impl_->parallelForMutex, std::defer_lock);
    if (gActiveCryptoExecutor != impl_.get()) batchLock.lock();
    const auto serviceStarted = std::chrono::steady_clock::now();
    const double executorWait = std::chrono::duration<double>(
        serviceStarted - started).count();
    if (impl_->threadCount == 1 || gActiveCryptoExecutor == impl_.get()) {
        struct ActiveScope {
            const void* previous;
            ~ActiveScope() { gActiveCryptoExecutor = previous; }
        } active{gActiveCryptoExecutor};
        gActiveCryptoExecutor = impl_.get();
        for (std::size_t index = 0; index < itemCount; ++index) {
            task(index);
        }
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - serviceStarted).count();
        if (metrics != nullptr) {
            metrics->Add(stage, elapsed, elapsed, itemCount, executorWait);
        }
        return;
    }

    const std::size_t runnerCount = std::min(itemCount, impl_->threadCount);
    auto batch = std::make_shared<Impl::BatchState>(runnerCount);
    for (std::size_t runner = 0; runner < runnerCount; ++runner) {
        const std::size_t begin = itemCount * runner / runnerCount;
        const std::size_t end = itemCount * (runner + 1) / runnerCount;
        impl_->Enqueue([batch, begin, end, runner, &task]() {
            const auto runnerStarted = std::chrono::steady_clock::now();
            try {
                for (std::size_t index = begin; index < end; ++index) {
                    task(index);
                }
            }
            catch (...) {
                std::lock_guard<std::mutex> lock(batch->mutex);
                if (!batch->firstException) {
                    batch->firstException = std::current_exception();
                }
            }
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - runnerStarted).count();
            {
                std::lock_guard<std::mutex> lock(batch->mutex);
                batch->workSeconds.at(runner) = elapsed;
                --batch->remaining;
            }
            batch->completed.notify_one();
        });
    }

    std::exception_ptr firstException;
    double workSeconds = 0.0;
    {
        std::unique_lock<std::mutex> lock(batch->mutex);
        batch->completed.wait(lock, [&batch]() {
            return batch->remaining == 0;
        });
        firstException = batch->firstException;
        workSeconds = std::accumulate(
            batch->workSeconds.begin(),
            batch->workSeconds.end(),
            0.0);
    }
    const double wallSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - serviceStarted).count();
    if (metrics != nullptr) {
        metrics->Add(stage, wallSeconds, workSeconds, itemCount, executorWait);
    }
    if (firstException) {
        std::rethrow_exception(firstException);
    }
}

}  // namespace federated
