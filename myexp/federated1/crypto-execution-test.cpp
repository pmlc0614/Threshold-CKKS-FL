#include "crypto-execution-test.h"

#include "crypto-execution.h"
#include "protocol-randomness.h"

#include "math/distributiongenerator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void TestSingleThreadPreservesOrder() {
    federated::CryptoTaskExecutor executor(1);
    federated::CryptoStageMetrics metrics;
    std::vector<std::size_t> order;
    executor.ParallelFor(
        32,
        federated::CryptoStage::ENCRYPTION,
        &metrics,
        [&order](std::size_t index) { order.push_back(index); });

    Require(executor.ThreadCount() == 1, "single-thread executor count mismatch");
    Require(order.size() == 32, "single-thread executor omitted work");
    for (std::size_t index = 0; index < order.size(); ++index) {
        Require(order[index] == index, "single-thread executor changed work order");
    }
    const auto timing = metrics.Snapshot().at(
        static_cast<std::size_t>(federated::CryptoStage::ENCRYPTION));
    Require(timing.taskCount == 32, "single-thread task count mismatch");
    Require(timing.wallSeconds >= 0.0, "single-thread wall timing is negative");
    Require(timing.workSeconds >= 0.0, "single-thread work timing is negative");
}

void TestExecutorWaitIsNotClientService() {
    for (std::size_t workers : {1U, 4U}) {
        federated::CryptoTaskExecutor executor(workers);
        federated::CryptoStageMetrics metrics;
        std::promise<void> entered, release, waiterStarted;
        auto gate = release.get_future().share();
        std::atomic<bool> firstActive{false}, overlapped{false};
        auto first = std::async(std::launch::async, [&] {
            executor.ParallelFor(1, federated::CryptoStage::ENCRYPTION, nullptr,
                [&](std::size_t) {
                    firstActive = true;
                    entered.set_value();
                    gate.wait();
                    firstActive = false;
                });
        });
        entered.get_future().wait();
        auto second = std::async(std::launch::async, [&] {
            waiterStarted.set_value();
            executor.ParallelFor(1, federated::CryptoStage::PARTIAL_DECRYPTION,
                &metrics, [&](std::size_t) { overlapped = firstActive.load(); });
        });
        waiterStarted.get_future().wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        release.set_value();
        first.get(); second.get();
        const auto t = metrics.Snapshot().at(static_cast<std::size_t>(
            federated::CryptoStage::PARTIAL_DECRYPTION));
        Require(!overlapped, "independent client batches overlapped");
        Require(t.executorWaitSeconds > 0.1, "executor contention was not recorded");
        Require(t.wallSeconds < t.executorWaitSeconds,
            "blocked time was charged to client service");
        Require(t.taskCount == 1, "wait accounting changed task count");
        metrics.Reset();
        Require(metrics.Snapshot().at(static_cast<std::size_t>(
            federated::CryptoStage::PARTIAL_DECRYPTION)).executorWaitSeconds == 0,
            "reset retained executor wait");
        executor.ParallelFor(1, federated::CryptoStage::ENCRYPTION, nullptr,
            [&](std::size_t) {
                executor.ParallelFor(1, federated::CryptoStage::ENCRYPTION,
                    nullptr, [](std::size_t) {});
            });
    }
}

void TestBoundedParallelExecutionAndMetrics() {
    federated::CryptoTaskExecutor executor(4);
    federated::CryptoStageMetrics metrics;
    std::vector<std::size_t> output(101, 0);
    executor.ParallelFor(
        output.size(),
        federated::CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH,
        &metrics,
        [&output](std::size_t index) { output[index] = index * index; });

    Require(executor.ThreadCount() == 4, "parallel executor count mismatch");
    for (std::size_t index = 0; index < output.size(); ++index) {
        Require(output[index] == index * index, "parallel executor corrupted output");
    }
    const auto timing = metrics.Snapshot().at(static_cast<std::size_t>(
        federated::CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH));
    Require(timing.taskCount == output.size(), "parallel task count mismatch");
    Require(timing.wallSeconds >= 0.0, "parallel wall timing is negative");
    Require(timing.workSeconds >= 0.0, "parallel work timing is negative");

    metrics.Reset();
    for (const auto& reset : metrics.Snapshot()) {
        Require(reset.taskCount == 0, "crypto metrics reset retained tasks");
        Require(reset.wallSeconds == 0.0, "crypto metrics reset retained wall time");
        Require(reset.workSeconds == 0.0, "crypto metrics reset retained work time");
    }
}

void TestWorkerExceptionPropagationAndReuse() {
    federated::CryptoTaskExecutor executor(3);
    bool threw = false;
    try {
        executor.ParallelFor(
            12,
            federated::CryptoStage::R2M,
            nullptr,
            [](std::size_t index) {
                if (index == 5) {
                    throw std::runtime_error("expected worker failure");
                }
            });
    }
    catch (const std::runtime_error&) {
        threw = true;
    }
    Require(threw, "crypto worker exception was not propagated");

    std::vector<std::size_t> output(9, 0);
    executor.ParallelFor(
        output.size(),
        federated::CryptoStage::DECODING,
        nullptr,
        [&output](std::size_t index) { output[index] = index + 1; });
    Require(output.back() == output.size(), "executor was not reusable after failure");
}

void TestThreadCountValidation() {
    bool rejectedZero = false;
    try {
        federated::CryptoTaskExecutor executor(0);
    }
    catch (const std::invalid_argument&) {
        rejectedZero = true;
    }
    Require(rejectedZero, "zero crypto threads were accepted");

    bool rejectedExcessive = false;
    try {
        federated::CryptoTaskExecutor executor(
            federated::kMaximumCryptoThreads + 1);
    }
    catch (const std::invalid_argument&) {
        rejectedExcessive = true;
    }
    Require(rejectedExcessive, "excessive crypto threads were accepted");
}

std::shared_ptr<lbcrypto::DCRTPoly::Params> MakeGaussianTestParameters(
    std::uint32_t ringDimension = 2048) {

    return std::make_shared<lbcrypto::DCRTPoly::Params>(
        2U * ringDimension,
        2U,
        32U);
}

void TestWorkerPrngIndependence() {
    // Match the 64-worker FULL experiment configuration: every persistent
    // executor worker must own a distinct OpenFHE PRNG instance.
    constexpr std::size_t kThreads = 64;
    federated::CryptoTaskExecutor executor(kThreads);
    std::vector<const void*> prngAddresses(kThreads, nullptr);
    std::vector<std::thread::id> workerIds(kThreads);
    std::mutex barrierMutex;
    std::condition_variable barrierReady;
    std::size_t arrived = 0;

    executor.ParallelFor(
        kThreads,
        federated::CryptoStage::ENCRYPTION,
        nullptr,
        [&](std::size_t index) {
            prngAddresses[index] = static_cast<const void*>(
                std::addressof(lbcrypto::PseudoRandomNumberGenerator::GetPRNG()));
            workerIds[index] = std::this_thread::get_id();
            std::unique_lock<std::mutex> lock(barrierMutex);
            ++arrived;
            barrierReady.notify_all();
            barrierReady.wait(lock, [&]() { return arrived == kThreads; });
        });

    const std::unordered_set<std::thread::id> uniqueWorkers(
        workerIds.begin(), workerIds.end());
    const std::unordered_set<const void*> uniquePrngs(
        prngAddresses.begin(), prngAddresses.end());
    Require(uniqueWorkers.size() == kThreads,
            "PRNG test did not occupy every crypto worker");
    if (federated::ProtocolGaussianSamplingSupportsParallelism()) {
        Require(uniquePrngs.size() == kThreads,
                "OpenFHE PRNG is shared by independent crypto workers");
    }
    else {
        Require(uniquePrngs.size() == 1,
                "FIXED_SEED build unexpectedly created independent PRNGs");
    }
    std::cout << "Gaussian sampler PRNGs: workers=" << uniqueWorkers.size()
              << " unique_prngs=" << uniquePrngs.size()
              << " parallel="
              << (federated::ProtocolGaussianSamplingSupportsParallelism()
                      ? "yes" : "no (FIXED_SEED)")
              << '\n';
}

std::string GaussianFingerprint(lbcrypto::DCRTPoly polynomial) {
    polynomial.SetFormat(Format::COEFFICIENT);
    const auto& values = polynomial.GetElementAtIndex(0).GetValues();
    std::ostringstream stream;
    const std::size_t count = std::min<std::size_t>(values.GetLength(), 64U);
    for (std::size_t index = 0; index < count; ++index) {
        stream << values[index].ConvertToInt() << ',';
    }
    return stream.str();
}

void TestParallelGaussianStatisticsAndDuplicates() {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kSamples = 32;
    constexpr long double kSigma = 3.2L;
    const auto parameters = MakeGaussianTestParameters();
    // Initialize OpenFHE's transform tables before independent workers use
    // the same DCRT parameters. Protocol setup performs the same warm-up.
    (void)federated::SampleProtocolGaussianPolynomial(parameters, kSigma);
    federated::CryptoTaskExecutor executor(kThreads);
    std::vector<lbcrypto::DCRTPoly> samples(kSamples);
    executor.ParallelFor(
        samples.size(),
        federated::CryptoStage::ENCRYPTION,
        nullptr,
        [&](std::size_t index) {
            samples[index] = federated::SampleProtocolGaussianPolynomial(
                parameters, kSigma);
        });

    std::set<std::string> fingerprints;
    long double sum = 0.0L;
    long double squaredSum = 0.0L;
    std::size_t coefficientCount = 0;
    for (auto& sample : samples) {
        fingerprints.insert(GaussianFingerprint(sample));
        sample.SetFormat(Format::COEFFICIENT);
        const auto& tower = sample.GetElementAtIndex(0);
        const std::uint64_t modulus = tower.GetModulus().ConvertToInt();
        const std::uint64_t halfModulus = modulus / 2U;
        const auto& values = tower.GetValues();
        for (std::size_t index = 0; index < values.GetLength(); ++index) {
            const std::uint64_t residue = values[index].ConvertToInt();
            const long double centered = residue > halfModulus
                ? static_cast<long double>(residue) -
                      static_cast<long double>(modulus)
                : static_cast<long double>(residue);
            sum += centered;
            squaredSum += centered * centered;
            ++coefficientCount;
        }
    }
    Require(fingerprints.size() == kSamples,
            "parallel Gaussian sampling produced a duplicate polynomial");
    const long double mean = sum / coefficientCount;
    const long double variance =
        squaredSum / coefficientCount - mean * mean;
    const long double standardDeviation = std::sqrt(variance);
    Require(std::abs(mean) < 0.15L,
            "parallel Gaussian sample mean is unexpectedly biased");
    Require(standardDeviation > 2.9L && standardDeviation < 3.5L,
            "parallel Gaussian sample standard deviation is unexpected");
    std::cout << std::fixed << std::setprecision(4)
              << "Gaussian sampler statistics: n=" << coefficientCount
              << " mean=" << static_cast<double>(mean)
              << " std=" << static_cast<double>(standardDeviation)
              << " unique_polynomials=" << fingerprints.size() << '\n';
}

double MeasureGaussianBatch(
    federated::CryptoTaskExecutor& executor,
    const std::shared_ptr<lbcrypto::DCRTPoly::Params>& parameters,
    std::vector<lbcrypto::DCRTPoly>& output,
    std::mutex* compatibilityMutex) {

    const auto started = std::chrono::steady_clock::now();
    executor.ParallelFor(
        output.size(),
        federated::CryptoStage::ENCRYPTION,
        nullptr,
        [&](std::size_t index) {
            if (compatibilityMutex != nullptr) {
                std::lock_guard<std::mutex> lock(*compatibilityMutex);
                output[index] =
                    federated::SampleProtocolGaussianPolynomial(
                        parameters, 3.2L);
            }
            else {
                output[index] =
                    federated::SampleProtocolGaussianPolynomial(
                        parameters, 3.2L);
            }
        });
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

void TestGaussianSamplerConcurrencyBenchmark() {
    constexpr std::size_t kThreads = 64;
    constexpr std::size_t kSamples = 58;
    const auto parameters = MakeGaussianTestParameters(8192U);
    (void)federated::SampleProtocolGaussianPolynomial(parameters, 3.2L);
    federated::CryptoTaskExecutor executor(kThreads);
    std::vector<lbcrypto::DCRTPoly> warmup(kThreads);
    std::vector<lbcrypto::DCRTPoly> locked(kSamples);
    std::vector<lbcrypto::DCRTPoly> parallel(kSamples);
    std::mutex compatibilityMutex;

    (void)MeasureGaussianBatch(executor, parameters, warmup, nullptr);
    const double lockedSeconds = MeasureGaussianBatch(
        executor, parameters, locked, &compatibilityMutex);
    const double parallelSeconds = MeasureGaussianBatch(
        executor, parameters, parallel, nullptr);
    Require(lockedSeconds > 0.0 && parallelSeconds > 0.0,
            "Gaussian sampler benchmark returned invalid timing");
    const double speedup = lockedSeconds / parallelSeconds;
    std::cout << std::fixed << std::setprecision(4)
              << "Gaussian sampler benchmark: tasks=" << kSamples
              << " ring_dimension=8192 towers=2 threads=" << kThreads
              << " serialized=" << lockedSeconds << "s"
              << " parallel=" << parallelSeconds << "s"
              << " speedup=" << speedup << "x\n";
}

}  // namespace

int RunCryptoExecutionTests() {
    try {
        for (long double sigma : {1.0L, 3.19L, 3.2L, 1000000.0L, 1000001.0L, std::ldexp(1.0L,55), std::ldexp(1.0L,71), std::ldexp(1.0L,79), std::ldexp(1.0L,128)}) {
            long double sum=0, squares=0;
            for (int i=0;i<8192;++i) {
                const long double x=federated::SampleCertifiedV17GaussianInteger(sigma).convert_to<long double>();
                Require(std::abs(x)<=std::ceil(24*sigma), "certified sampler support");
                sum+=x/sigma; squares+=(x/sigma)*(x/sigma);
            }
            // Wide statistical smoke checks, not distribution certificates.
            Require(std::abs(sum/8192)<0.12L, "certified sampler centered smoke test");
            Require(squares/8192>0.8L && squares/8192<1.2L, "certified sampler moment smoke test");
        }
        bool invalidSigmaRejected=false;
        try { (void)federated::SampleCertifiedV17GaussianInteger(3.21L); }
        catch (const std::invalid_argument&) { invalidSigmaRejected=true; }
        Require(invalidSigmaRejected, "uncertified sigma rejected");
        TestSingleThreadPreservesOrder();
        TestExecutorWaitIsNotClientService();
        TestBoundedParallelExecutionAndMetrics();
        TestWorkerExceptionPropagationAndReuse();
        TestThreadCountValidation();
        TestWorkerPrngIndependence();
        TestParallelGaussianStatisticsAndDuplicates();
        TestGaussianSamplerConcurrencyBenchmark();
        std::cout << "Crypto execution foundation tests passed.\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Crypto execution foundation tests failed: "
                  << error.what() << '\n';
        return 1;
    }
}
