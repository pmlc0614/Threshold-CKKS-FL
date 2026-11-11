#ifndef FEDERATED1_EXTERNAL_INPUT_CRYPTO_ENGINE_H
#define FEDERATED1_EXTERNAL_INPUT_CRYPTO_ENGINE_H

#include "crypto-execution.h"
#include "protocol-contract.h"
#include "protocol-parameters.h"
#include "protocol-types.h"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace federated {

struct ExternalEngineExecutionOptions {
    bool useSparseCompression = true;
    bool useAuxiliaryCompression = true;
    bool measurePreSmudgingNoise = false;
    std::size_t cryptoThreads = 1;
};

/* Setup input supplied by the future pybind11 adapter. */
struct ExternalEngineSetupRequest {
    ProtocolExperimentConfig protocol;
    ProtocolTensorLayout parameterLayout;
    ProtocolTensorLayout runningMeanLayout;
    ProtocolTensorLayout runningVarianceLayout;
    std::vector<RegisteredClientWeightInput> clients;
    ExternalEngineExecutionOptions execution;
};

/* C++ creates cryptographic setup material once and returns its identity. */
struct ExternalEngineSetupResult {
    SetupId setupId;
    std::uint64_t setupDescriptorBytes = 0;
    Seed128 crsSeed{};
    ByteVector setupDescriptor;
    ByteVector publicSetupBinding;
    std::vector<std::uint64_t> qPrimes;
    std::vector<std::uint64_t> pPrimes;
    ContractHash256 parameterLayoutHash{};
    ContractHash256 runningMeanLayoutHash{};
    ContractHash256 runningVarianceLayoutHash{};
    std::uint64_t parameterNumel = 0;
    std::uint64_t runningMeanNumel = 0;
    std::uint64_t runningVarianceNumel = 0;
    std::uint64_t chunkSize = 0;
    std::uint64_t chunkCount = 0;
    std::vector<ClientId> clientIds;
    std::vector<std::uint32_t> alphas;
    std::vector<double> clientInitialSetupSeconds;
    std::vector<double> clientCollectiveKeyGenerationSeconds;
};

struct ExternalEngineAdmissionPlan {
    SetupId setupId;
    RoundId roundId;
    std::vector<ClientAdmissionResult> clientResults;
    std::vector<ActualChunkId> acceptedActualChunkIds;
    std::vector<std::uint64_t> cohortSizes;

    const ClientAdmissionResult& ForClient(ClientId clientId) const;
};

struct ExternalFullParameterDeltaPayload {
    SetupId setupId;
    RoundId roundId;
    ClientId clientId;
    std::vector<float> parameterDelta;
};

struct ExternalEngineSurvivorTransition {
    SetupId setupId;
    RoundId roundId;
    std::vector<ClientId> survivorClientIds;
    std::vector<ClientId> droppedClientIds;
    ExternalEngineAdmissionPlan admission;
    bool targetKeyCacheHit = false;
    double serverTargetReconstructionSeconds = 0.0;
    std::vector<double> clientCollectiveKeyGenerationSeconds;
    double maximumClientCollectiveKeyGenerationSeconds = 0.0;
};

struct ExternalEngineRoundResult {
    ProtocolPhaseAudit modelPhaseAudit;
    ProtocolPhaseAudit auxPhaseAudit;
    SetupId setupId;
    RoundId roundId;
    std::vector<ActualChunkId> acceptedActualChunkIds;
    std::vector<std::uint64_t> cohortSizes;
    std::vector<float> parameterDelta;
    std::vector<float> runningMeanDelta;
    std::vector<float> runningVarianceDelta;

    std::uint64_t sparsePlanHash = 0;
    std::size_t sparseClientCiphertextCount = 0;
    std::size_t sparseSurvivorClientCiphertextCount = 0;
    std::size_t sparseDiscardedClientCiphertextCount = 0;
    std::size_t acceptedSparseContributionCount = 0;
    std::size_t aggregatedMLWECiphertextCount = 0;
    std::size_t sparseServerCiphertextCount = 0;
    std::size_t selectionBitmapCount = 0;
    std::size_t admissionBitmapRecipientCount = 0;
    std::size_t uploadSupportBitmapCount = 0;
    std::size_t resultRecipientCount = 0;
    std::size_t fullModelClientCiphertextCount = 0;
    std::size_t fullModelAggregateCiphertextCount = 0;
    std::size_t r2mCiphertextCount = 0;
    std::size_t m2rCiphertextCount = 0;
    std::size_t auxiliaryClientCiphertextCount = 0;
    std::size_t auxiliarySurvivorClientCiphertextCount = 0;
    std::size_t auxiliaryDiscardedClientCiphertextCount = 0;
    std::size_t auxiliaryAggregateCiphertextCount = 0;

    /* Preserve single-host wall-clock while exposing isolated client timings
       from client-sequential, ciphertext-parallel FULL execution. */
    double cryptoSequentialSeconds = 0.0;
    double cryptoServerSeconds = 0.0;
    double estimatedParallelCryptoSeconds = 0.0;
    std::vector<double> clientEncryptionSeconds;
    std::vector<double> clientKeySwitchSeconds;
    std::vector<double> clientPartialDecryptionSeconds;
    std::vector<double> clientTotalCryptoSeconds;
    double maximumClientPreAggregationSeconds = 0.0;
    double maximumClientPartialDecryptionSeconds = 0.0;
    double maximumClientTotalCryptoSeconds = 0.0;
    std::size_t cryptoThreadCount = 1;
    std::array<CryptoStageTiming, kCryptoStageCount> cryptoStageTimings{};
    std::vector<CryptoStageTimings> fullModelClientStageTimings;
    std::vector<CryptoStageTimings> sparseModelClientStageTimings;
    std::vector<CryptoStageTimings> auxiliaryClientStageTimings;
    CryptoStageTimings fullModelServerStageTimings{};
    CryptoStageTimings sparseModelServerStageTimings{};
    CryptoStageTimings auxiliaryServerStageTimings{};

    bool survivorReconfigurationApplied = false;
    bool survivorTargetKeyCacheHit = false;
    std::vector<std::size_t> survivorNewTargetSizes;
    std::vector<ClientId> survivorClientIds;
    std::vector<ClientId> droppedClientIds;
    double survivorTargetReconstructionSeconds = 0.0;
    std::vector<double> survivorCollectiveKeyGenerationSeconds;
    double maximumSurvivorCollectiveKeyGenerationSeconds = 0.0;
    double survivorReconfigurationSequentialSeconds = 0.0;
};

struct ExternalEngineAbortedRoundTraffic {
    RoundId roundId;
    std::size_t sparseClientCiphertextCount = 0;
    std::size_t auxiliaryClientCiphertextCount = 0;
    std::size_t auxiliaryAggregateCiphertextCount = 0;
    std::size_t sparseServerCiphertextCount = 0;
    std::size_t uploadAcceptedChunkCount = 0;
    std::size_t acceptedChunkCount = 0;
    std::size_t uploadedSparseContributionCount = 0;
    std::size_t acceptedSparseContributionCount = 0;
    std::size_t selectionBitmapCount = 0;
    std::size_t admissionBitmapRecipientCount = 0;
    std::size_t uploadSupportBitmapCount = 0;
    std::vector<ClientId> survivorClientIds;
    std::vector<ClientId> droppedClientIds;
    std::vector<std::size_t> survivorNewTargetSizes;
};

/*
 * Stateful façade matching the Python FederatedCryptoEngine call order.
 * It never generates synthetic updates and never performs top-k selection.
 */
class ExternalInputCryptoEngine {
public:
    explicit ExternalInputCryptoEngine(
        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext);
    ~ExternalInputCryptoEngine();

    ExternalInputCryptoEngine(ExternalInputCryptoEngine&&) noexcept;
    ExternalInputCryptoEngine& operator=(
        ExternalInputCryptoEngine&&) noexcept;

    ExternalInputCryptoEngine(const ExternalInputCryptoEngine&) = delete;
    ExternalInputCryptoEngine& operator=(
        const ExternalInputCryptoEngine&) = delete;

    ExternalEngineSetupResult Setup(
        const ExternalEngineSetupRequest& request);

    bool IsConfigured() const;
    const ExternalEngineSetupResult& SetupResult() const;
    const ProtocolExperimentConfig& EffectiveProtocolConfig() const;
    std::optional<RoundPhase> Phase() const;
    std::optional<ExternalEngineAbortedRoundTraffic>
    LastAbortedRoundTraffic() const;

    void BeginRound(
        RoundId roundId,
        std::optional<std::uint64_t> logicalRoundIndex = std::nullopt);
    void SubmitSelection(const ClientSelectionRequest& request);
    ExternalEngineAdmissionPlan FinalizeAdmission();
    void SubmitAcceptedValues(const AcceptedChunkValues& values);
    void SubmitFullParameterDelta(
        const ExternalFullParameterDeltaPayload& payload);
    void SubmitAuxiliaryDeltas(const AuxiliaryDeltaPayload& payload);
    ExternalEngineSurvivorTransition SetRoundSurvivors(
        const std::vector<ClientId>& survivorClientIds);
    // Exact serialized control context; acknowledgements are simulated in-process.
    std::vector<std::uint64_t> PrepareFinalization();
    void AcknowledgeFinalization(ClientId clientId,
        const std::vector<std::uint64_t>& context);
    ExternalEngineRoundResult FinalizeRound(
        const std::vector<ClientId>& postRequestDropouts = {});
    void CommitRound();
    void AbortRound();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace federated

#endif  // FEDERATED1_EXTERNAL_INPUT_CRYPTO_ENGINE_H
