#include "external-input-crypto-engine-test.h"

#include "external-input-crypto-engine.h"
#include "protocol-crypto-context.h"
#include "protocol-v17-decryption.h"

#include "openfhe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using DCRTPoly = lbcrypto::DCRTPoly;

federated::ExternalEngineRoundResult FinalizeAcknowledged(
    federated::ExternalInputCryptoEngine& engine) {
    if (engine.EffectiveProtocolConfig().experimentalDynamicCohort) {
        const auto context = engine.PrepareFinalization();
        const auto countPos = 2 + engine.SetupResult().setupId.bytes.size() + 3;
        for (std::size_t i = 0; i < context.at(countPos); ++i)
            engine.AcknowledgeFinalization(
                federated::ClientId{context.at(countPos + 1 + i)}, context);
    }
    return engine.FinalizeRound();
}


bool Check(bool condition, const std::string& description) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ")
        << description << '\n';
    return condition;
}

federated::ProtocolTensorLayout MakeLayout(
    federated::LayoutKind kind,
    const std::string& name,
    std::uint64_t numel) {

    federated::ProtocolTensorLayout layout;
    layout.kind = kind;
    layout.modelName = "external-tiny-model";
    layout.entries.push_back(federated::ProtocolLayoutEntry{
        name,
        {numel},
        0,
        numel,
        federated::WireDType::FLOAT32});
    return layout;
}

federated::ExternalEngineSetupRequest MakeSetupRequest() {
    using namespace federated;
    ExternalEngineSetupRequest request;
    request.protocol = MakeSmokeConfig();
    request.protocol.experimentalDynamicCohort = true; // legacy coverage
    request.protocol.modulus.ringDimension = 1024;
    request.protocol.modulus.qModulusBits = {28, 28};
    request.protocol.modulus.pModulusBits = {55};
    request.protocol.modulus.hybridLargeDigits = 2;
    request.protocol.modulus.maximumQpBits = 160;
    request.protocol.sparsification.mlweRank = 4;
    request.protocol.sparsification.selectionRatio = 0.0L;
    request.protocol.sparsification.selectedChunksPerClient = 2;
    request.protocol.sparsification.minimumCohort = 3;
    request.protocol.sparsification.collusionBound = 0;
    request.protocol.updateBound = 1.0L;
    request.protocol.auxUpdateBound = 1.0L;
    request.protocol.scaling.factor = std::uint64_t{1} << 35U;
    request.protocol.auxScaling.factor = std::uint64_t{1} << 35U;
    request.protocol.smudging.sigmaSmg = 0.0L;
    request.protocol.smudging.auxSigmaSmg = 0.0L;
    request.parameterLayout = MakeLayout(
        LayoutKind::PARAMETER,
        "weight",
        600);
    request.runningMeanLayout = MakeLayout(
        LayoutKind::RUNNING_MEAN,
        "bn.running_mean",
        2);
    request.runningVarianceLayout = MakeLayout(
        LayoutKind::RUNNING_VARIANCE,
        "bn.running_var",
        2);
    request.clients = {
        {ClientId{30}, 1},
        {ClientId{10}, 1},
        {ClientId{20}, 1},
        {ClientId{40}, 1}};
    request.execution.useSparseCompression = false;
    request.execution.useAuxiliaryCompression = false;
    request.execution.cryptoThreads = 3;
    return request;
}

federated::ExternalEngineSetupRequest MakeFullSetupRequest() {
    auto request = MakeSetupRequest();
    request.protocol.mode = federated::ProtocolMode::FULL;
    request.protocol.sparsification.mlweRank = 1;
    request.protocol.sparsification.selectionRatio = 1.0L;
    request.protocol.sparsification.selectedChunksPerClient = 0;
    request.protocol.sparsification.errorFeedbackEnabled = false;
    request.execution.cryptoThreads = 3;
    return request;
}

federated::ExternalEngineSetupRequest MakeRankOneSparseSetupRequest() {
    auto request = MakeSetupRequest();
    request.protocol.sparsification.mlweRank = 1;
    request.protocol.sparsification.selectedChunksPerClient = 1;
    request.parameterLayout = MakeLayout(
        federated::LayoutKind::PARAMETER,
        "weight",
        1400);
    request.execution.measurePreSmudgingNoise = true;
    return request;
}

bool Near(float actual, float expected, float tolerance = 2.0e-4F) {
    return std::abs(actual - expected) <= tolerance;
}

bool TestV17CompressedRound(bool sparse, bool fixed = false, std::size_t rank = 4, std::size_t threads = 3) {
    using namespace federated;
    auto request = sparse ? MakeSetupRequest() : MakeFullSetupRequest();
    request.protocol.experimentalDynamicCohort = !fixed;
    if (sparse) request.protocol.sparsification.mlweRank = rank;
    request.execution.cryptoThreads = threads;
    request.clients.push_back({ClientId{50}, 1});
    request.protocol.protocolVersion = "v17";
    request.protocol.compression.c1 = 25;
    request.protocol.compression.c2 = 24;
    request.protocol.compression.ca = 52;
    request.protocol.compression.auxC1 = 25;
    request.protocol.compression.auxC2 = 24;
    request.protocol.compression.auxCa = 52;
    request.execution.useSparseCompression = true;
    request.execution.useAuxiliaryCompression = true;

    auto cc = MakeProtocolCryptoContext(request.protocol);
    ExternalInputCryptoEngine engine(cc);
    const auto setup = engine.Setup(request);
    const auto finalize = [&](const std::vector<ClientId>& dropped = std::vector<ClientId>{}) {
        const auto ctx = engine.PrepareFinalization();
        // Context prefix: version, phase, setup ID bytes, t, rid, revision, count.
        const std::size_t countPos = 2 + setup.setupId.bytes.size() + 3;
        for (std::size_t i = 0; i < ctx.at(countPos); ++i)
            engine.AcknowledgeFinalization(ClientId{ctx.at(countPos + 1 + i)}, ctx);
        return engine.FinalizeRound(dropped);
    };
    const auto submitRound = [&](RoundId roundId,
                                 std::uint64_t logicalRoundIndex) {
        engine.BeginRound(roundId, logicalRoundIndex);
        std::vector<ActualChunkId> accepted;
        if (sparse) {
        for (const ClientId clientId : setup.clientIds) {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = roundId;
            selection.clientId = clientId;
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = {
                ActualChunkId{0}, ActualChunkId{1}};
            engine.SubmitSelection(selection);
        }
        accepted = engine.FinalizeAdmission().acceptedActualChunkIds;
        }
        for (const ClientId clientId : setup.clientIds) {
        if (sparse) {
            AcceptedChunkValues model;
            model.setupId = setup.setupId;
            model.roundId = roundId;
            model.clientId = clientId;
            model.chunkSize = setup.chunkSize;
            model.actualChunkIds = accepted;
            model.values.assign(accepted.size() * setup.chunkSize, 0.1F);
            engine.SubmitAcceptedValues(model);
        }
        else {
            ExternalFullParameterDeltaPayload model;
            model.setupId = setup.setupId;
            model.roundId = roundId;
            model.clientId = clientId;
            model.parameterDelta.assign(setup.parameterNumel, 0.1F);
            engine.SubmitFullParameterDelta(model);
        }
        AuxiliaryDeltaPayload auxiliary;
        auxiliary.setupId = setup.setupId;
        auxiliary.roundId = roundId;
        auxiliary.clientId = clientId;
        auxiliary.runningMeanDelta = {0.1F, 0.1F};
        auxiliary.runningVarianceDelta = {0.1F, 0.1F};
        engine.SubmitAuxiliaryDeltas(auxiliary);
        }
    };
    submitRound(RoundId{7}, 7);
    const auto result = finalize();
    if (sparse && result.sparseModelServerStageTimings[
            static_cast<std::size_t>(CryptoStage::M2R)].taskCount !=
            result.sparseServerCiphertextCount * (rank + 2))
        throw std::runtime_error("M2R must schedule preparation, each component, and reduction");
    bool modelCorrect = result.parameterDelta.size() == setup.parameterNumel;
    for (std::size_t index = 0; index < result.parameterDelta.size(); ++index) {
        const float expected = !sparse || index < 2 * setup.chunkSize
            ? 0.1F : 0.0F;
        modelCorrect &= Near(result.parameterDelta[index], expected, 0.03F);
    }
    bool auxCorrect = result.runningMeanDelta.size() == 2 &&
        result.runningVarianceDelta.size() == 2;
    for (const float value : result.runningMeanDelta) {
        auxCorrect &= Near(value, 0.1F, 0.03F);
    }
    for (const float value : result.runningVarianceDelta) {
        auxCorrect &= Near(value, 0.1F, 0.03F);
    }
    engine.CommitRound();
    bool passed = Check(modelCorrect && auxCorrect,
        sparse
            ? "v17 sparse/aux stochastic c1 and shared late-seed c2 round"
            : "v17 FULL/aux stochastic c1 and shared late-seed c2 round");

    if (fixed) {
        submitRound(RoundId{8}, 8);
        bool aborted = false;
        try { (void)finalize({setup.clientIds.front()}); }
        catch (const V17PostRequestDropout&) { aborted = engine.Phase() == RoundPhase::ABORTED; }
        bool blocked = false;
        try { engine.BeginRound(RoundId{9}, 8); }
        catch (const std::logic_error&) { blocked = true; }
        return passed && Check(aborted && blocked,
            "fixed cohort aborts late dropout and rejects retry without result");
    }
    const std::vector<ClientId> firstSurvivors(
        setup.clientIds.begin(), setup.clientIds.begin() + 4);
    const std::vector<ClientId> survivors(
        setup.clientIds.begin(), setup.clientIds.begin() + 3);
    submitRound(RoundId{8}, 8);
    const auto firstTransition = engine.SetRoundSurvivors(firstSurvivors);
    const auto secondTransition = engine.SetRoundSurvivors(survivors);
    passed &= Check(
        firstTransition.droppedClientIds.size() == 1 &&
            firstTransition.survivorClientIds.size() == 4 &&
            secondTransition.droppedClientIds.size() == 2 &&
            secondTransition.survivorClientIds.size() == 3,
        sparse ? "v17 SP rebuilds a pre-request survivor target"
               : "v17 FULL rebuilds a pre-request survivor target");
    bool survivorExpansionRejected = false;
    try {
        (void)engine.SetRoundSurvivors(setup.clientIds);
    }
    catch (const std::invalid_argument&) {
        survivorExpansionRejected = true;
    }
    passed &= Check(survivorExpansionRejected,
        "v17 pre-request survivor set cannot expand");
    bool lateDropoutAborted = false;
    try {
        (void)finalize({survivors.front()});
    }
    catch (const V17PostRequestDropout&) {
        lateDropoutAborted = engine.Phase() == RoundPhase::ABORTED;
    }
    passed &= Check(lateDropoutAborted,
        sparse ? "v17 SP late dropout aborts without plaintext"
               : "v17 FULL late dropout aborts without plaintext");
    const auto abortedTraffic = engine.LastAbortedRoundTraffic();
    passed &= Check(
        abortedTraffic.has_value() &&
            abortedTraffic->roundId == RoundId{8} &&
            abortedTraffic->auxiliaryClientCiphertextCount > 0 &&
            abortedTraffic->survivorNewTargetSizes ==
                std::vector<std::size_t>{4, 3} &&
            (!sparse ||
             (abortedTraffic->sparseClientCiphertextCount > 0 &&
              abortedTraffic->uploadAcceptedChunkCount >=
                  abortedTraffic->acceptedChunkCount)),
        "v17 late abort retains pre-response traffic counters");
    bool wrongLogicalRoundRejected = false;
    try {
        engine.BeginRound(RoundId{9}, 8);
    }
    catch (const std::invalid_argument&) {
        wrongLogicalRoundRejected = true;
    }
    passed &= Check(wrongLogicalRoundRejected,
        "v17 abort closes the logical round and forbids retry");
    submitRound(RoundId{9}, 9);
    const auto cachedTransition = engine.SetRoundSurvivors(survivors);
    passed &= Check(
        cachedTransition.targetKeyCacheHit &&
        cachedTransition.serverTargetReconstructionSeconds == 0.0 &&
            cachedTransition.maximumClientCollectiveKeyGenerationSeconds == 0.0,
        "v17 next round reuses setup-scoped survivor target and KSK cache");
    const auto retried = finalize();
    const float survivorWeightedValue = 0.1F * static_cast<float>(
        setup.alphas[0] + setup.alphas[1] + setup.alphas[2]) / 128.0F;
    const bool retryModelCorrect = !retried.parameterDelta.empty() &&
        Near(retried.parameterDelta.front(), survivorWeightedValue, 0.03F);
    const bool retryAuxCorrect = !retried.runningMeanDelta.empty() &&
        Near(retried.runningMeanDelta.front(), survivorWeightedValue, 0.03F);
    passed &= Check(retryModelCorrect && retryAuxCorrect &&
                        retried.survivorTargetKeyCacheHit &&
                        retried.survivorNewTargetSizes.empty(),
        "v17 next round aggregates only final survivors without renormalizing weights");
    engine.CommitRound();
    return passed;
}

}  // namespace

int RunExternalInputCryptoEngineTests() {
    using namespace federated;
    bool passed = true;

    try {
        for (bool full : {false, true}) {
            auto request = full ? MakeFullSetupRequest() : MakeSetupRequest();
            request.protocol.experimentalDynamicCohort = false;
            auto cc = MakeProtocolCryptoContext(request.protocol);
            ExternalInputCryptoEngine engine(cc);
            engine.Setup(request);
            engine.BeginRound(RoundId{0});
            bool missingAborted = false;
            try {
                if (full) (void)FinalizeAcknowledged(engine);
                else (void)engine.FinalizeAdmission();
            } catch (const std::logic_error&) {
                missingAborted = engine.Phase() == RoundPhase::ABORTED;
            }
            bool retryRejected = false;
            try { engine.BeginRound(RoundId{1}); }
            catch (const std::logic_error&) { retryRejected = true; }
            passed &= Check(missingAborted && retryRejected,
                "missing mandatory messages abort fixed-cohort execution");
            engine.Setup(request);
            engine.BeginRound(RoundId{0});
            bool changed = false;
            try { (void)engine.SetRoundSurvivors({ClientId{10}, ClientId{20}, ClientId{40}}); }
            catch (const std::logic_error&) { changed = engine.Phase() == RoundPhase::ABORTED; }
            passed &= Check(changed, "fixed cohort refuses target-key reconfiguration");
        }
        passed &= TestV17CompressedRound(false, true);
        passed &= TestV17CompressedRound(true, true);
        passed &= TestV17CompressedRound(false);
        passed &= TestV17CompressedRound(true);
        // A single incomplete rank-64 group must exercise component parallelism
        // and zero padding with both one worker and multiple workers.
        passed &= TestV17CompressedRound(true, true, 64, 1);
        passed &= TestV17CompressedRound(true, true, 64, 3);
    }
    catch (const std::exception& error) {
        std::cout << "[FAIL] v17 compressed execution test: "
            << error.what() << '\n';
        passed = false;
    }

    try {
        const ExternalEngineSetupRequest request = MakeSetupRequest();
        auto cc = MakeProtocolCryptoContext(request.protocol);
        ExternalInputCryptoEngine engine(cc);
        const ExternalEngineSetupResult setup = engine.Setup(request);

        passed &= Check(
            setup.parameterNumel == 600 &&
                setup.runningMeanNumel == 2 &&
                setup.runningVarianceNumel == 2 &&
                setup.chunkSize == 256 &&
                setup.chunkCount == 3 &&
                setup.clientIds == std::vector<ClientId>{
                    ClientId{30}, ClientId{10}, ClientId{20}, ClientId{40}} &&
                setup.alphas == std::vector<std::uint32_t>{32, 32, 32, 32},
            "setup accepts external layouts and non-contiguous client IDs");

        const RoundId roundId{0};
        engine.BeginRound(roundId);
        const std::vector<std::vector<ActualChunkId>> selections{
            {ActualChunkId{0}, ActualChunkId{1}},
            {ActualChunkId{0}, ActualChunkId{1}},
            {ActualChunkId{1}, ActualChunkId{2}},
            {ActualChunkId{1}, ActualChunkId{2}}};
        for (std::size_t position = 0; position < setup.clientIds.size(); ++position) {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = roundId;
            selection.clientId = setup.clientIds[position];
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = selections[position];
            engine.SubmitSelection(selection);
        }

        const ExternalEngineAdmissionPlan admission =
            engine.FinalizeAdmission();
        bool clientAdmissionsAreExact = true;
        for (const auto clientId : setup.clientIds) {
            clientAdmissionsAreExact &=
                admission.ForClient(clientId).acceptedActualChunkIds ==
                    std::vector<ActualChunkId>{ActualChunkId{1}};
        }
        passed &= Check(
            admission.cohortSizes == std::vector<std::uint64_t>{2, 4, 2} &&
                admission.acceptedActualChunkIds ==
                    std::vector<ActualChunkId>{ActualChunkId{1}} &&
                clientAdmissionsAreExact,
            "minimum cohort is finalized before values are accepted");

        AcceptedChunkValues rejectedAttempt;
        rejectedAttempt.setupId = setup.setupId;
        rejectedAttempt.roundId = roundId;
        rejectedAttempt.clientId = setup.clientIds.front();
        rejectedAttempt.chunkSize = setup.chunkSize;
        rejectedAttempt.actualChunkIds = {ActualChunkId{0}};
        rejectedAttempt.values.assign(setup.chunkSize, 0.1F);
        bool rejectedUnapprovedValues = false;
        try {
            engine.SubmitAcceptedValues(rejectedAttempt);
        }
        catch (const std::invalid_argument&) {
            rejectedUnapprovedValues = true;
        }
        passed &= Check(
            rejectedUnapprovedValues,
            "the engine rejects values for a non-admitted chunk");

        const std::vector<float> localValues{2.0F, 2.0F, -1.5F, -1.5F};
        for (std::size_t position = 0; position < setup.clientIds.size(); ++position) {
            const float value = localValues[position];
            AcceptedChunkValues accepted;
            accepted.setupId = setup.setupId;
            accepted.roundId = roundId;
            accepted.clientId = setup.clientIds[position];
            accepted.chunkSize = setup.chunkSize;
            accepted.actualChunkIds = {ActualChunkId{1}};
            accepted.values.assign(setup.chunkSize, value);
            engine.SubmitAcceptedValues(accepted);

            AuxiliaryDeltaPayload auxiliary;
            auxiliary.setupId = setup.setupId;
            auxiliary.roundId = roundId;
            auxiliary.clientId = setup.clientIds[position];
            auxiliary.runningMeanDelta = {value, 2.0F * value};
            auxiliary.runningVarianceDelta = {value, 2.0F * value};
            engine.SubmitAuxiliaryDeltas(auxiliary);
        }

        const ExternalEngineRoundResult result = FinalizeAcknowledged(engine);
        bool parameterResultIsCorrect = result.parameterDelta.size() == 600;
        for (std::size_t index = 0; index < result.parameterDelta.size(); ++index) {
            const float expected = index >= 256 && index < 512 ? 0.25F : 0.0F;
            parameterResultIsCorrect &= Near(result.parameterDelta[index], expected);
        }
        passed &= Check(
            parameterResultIsCorrect &&
                result.runningMeanDelta.size() == 2 &&
                result.runningVarianceDelta.size() == 2 &&
                Near(result.runningMeanDelta[0], 0.25F) &&
                Near(result.runningMeanDelta[1], 0.50F) &&
                Near(result.runningVarianceDelta[0], 0.25F) &&
                Near(result.runningVarianceDelta[1], 0.50F),
            "local values above the aggregate bound complete encryption when their aggregate is in range");
        passed &= Check(
            result.sparseClientCiphertextCount == setup.clientIds.size() &&
                result.acceptedSparseContributionCount == setup.clientIds.size() &&
                result.aggregatedMLWECiphertextCount == 1 &&
                result.sparseServerCiphertextCount == 1 &&
                result.auxiliaryClientCiphertextCount > 0 &&
                result.auxiliaryAggregateCiphertextCount > 0,
            "external engine reports the real encrypted-path ciphertext counts");
        passed &= Check(
            result.cryptoSequentialSeconds > 0.0 &&
                result.cryptoServerSeconds >= 0.0 &&
                result.estimatedParallelCryptoSeconds > 0.0 &&
                result.estimatedParallelCryptoSeconds <=
                    result.cryptoSequentialSeconds &&
                result.clientEncryptionSeconds.size() ==
                    setup.clientIds.size() &&
                result.clientPartialDecryptionSeconds.size() ==
                    setup.clientIds.size(),
            "external engine reports sequential and parallel crypto timings");
        const auto sparseServerHasTasks = [&](CryptoStage stage) {
            return result.sparseModelServerStageTimings[
                static_cast<std::size_t>(stage)].taskCount > 0;
        };
        const bool sparseClientStagesAreIsolated = std::all_of(
            result.sparseModelClientStageTimings.begin(),
            result.sparseModelClientStageTimings.end(),
            [](const CryptoStageTimings& timings) {
                return timings[static_cast<std::size_t>(
                           CryptoStage::ENCRYPTION)].taskCount > 0 &&
                    timings[static_cast<std::size_t>(
                           CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH)]
                           .taskCount > 0 &&
                    timings[static_cast<std::size_t>(
                           CryptoStage::PARTIAL_DECRYPTION)].taskCount > 0 &&
                    timings[static_cast<std::size_t>(CryptoStage::R2M)]
                           .taskCount == 0 &&
                    timings[static_cast<std::size_t>(CryptoStage::M2R)]
                           .taskCount == 0;
            });
        passed &= Check(
            result.cryptoThreadCount == 3 &&
                result.sparseModelClientStageTimings.size() ==
                    setup.clientIds.size() &&
                sparseClientStagesAreIsolated &&
                sparseServerHasTasks(CryptoStage::R2M) &&
                sparseServerHasTasks(CryptoStage::MLWE_AGGREGATION) &&
                sparseServerHasTasks(CryptoStage::M2R) &&
                sparseServerHasTasks(CryptoStage::PARTIAL_DECRYPTION) &&
                sparseServerHasTasks(CryptoStage::DECODING),
            "SPARSE reports isolated client timings and parallel R2M/MLWE/M2R server stages");
        passed &= Check(
            engine.Phase() == RoundPhase::DECRYPTING,
            "finalize leaves the round uncommitted for Python model apply");

        engine.CommitRound();
        passed &= Check(
            engine.Phase() == RoundPhase::COMMITTED,
            "Python can commit only after applying the decrypted result");

        bool rejectedRoundReplay = false;
        try {
            engine.BeginRound(roundId);
        }
        catch (const std::invalid_argument&) {
            rejectedRoundReplay = true;
        }
        passed &= Check(
            rejectedRoundReplay,
            "committed round IDs cannot be replayed");

        const RoundId secondRoundId{1};
        engine.BeginRound(secondRoundId);
        for (std::size_t position = 0; position < setup.clientIds.size(); ++position) {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = secondRoundId;
            selection.clientId = setup.clientIds[position];
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = selections[position];
            engine.SubmitSelection(selection);
        }
        const ExternalEngineAdmissionPlan secondAdmission =
            engine.FinalizeAdmission();
        for (std::size_t position = 0; position < setup.clientIds.size(); ++position) {
            const float value = localValues[position];
            AcceptedChunkValues accepted;
            accepted.setupId = setup.setupId;
            accepted.roundId = secondRoundId;
            accepted.clientId = setup.clientIds[position];
            accepted.chunkSize = setup.chunkSize;
            accepted.actualChunkIds =
                secondAdmission.ForClient(setup.clientIds[position])
                    .acceptedActualChunkIds;
            accepted.values.assign(
                accepted.actualChunkIds.size() * setup.chunkSize,
                value);
            engine.SubmitAcceptedValues(accepted);

            AuxiliaryDeltaPayload auxiliary;
            auxiliary.setupId = setup.setupId;
            auxiliary.roundId = secondRoundId;
            auxiliary.clientId = setup.clientIds[position];
            auxiliary.runningMeanDelta = {value, 2.0F * value};
            auxiliary.runningVarianceDelta = {value, 2.0F * value};
            engine.SubmitAuxiliaryDeltas(auxiliary);
        }
        const ExternalEngineRoundResult secondResult = FinalizeAcknowledged(engine);
        passed &= Check(
            secondResult.roundId == secondRoundId &&
                secondResult.setupId == setup.setupId &&
                secondResult.acceptedActualChunkIds ==
                    result.acceptedActualChunkIds &&
                secondResult.sparsePlanHash != result.sparsePlanHash &&
                Near(secondResult.parameterDelta[256], 0.25F),
            "a second encrypted round reuses setup and separates round randomness");
        engine.CommitRound();
        passed &= Check(
            engine.Phase() == RoundPhase::COMMITTED,
            "the reused setup commits the second encrypted round");
    }
    catch (const std::exception& error) {
        std::cout << "[FAIL] external-input crypto engine exception: "
            << error.what() << '\n';
        passed = false;
    }

    try {
        const ExternalEngineSetupRequest request = MakeSetupRequest();
        auto cc = MakeProtocolCryptoContext(request.protocol);
        ExternalInputCryptoEngine engine(cc);
        const ExternalEngineSetupResult setup = engine.Setup(request);
        const RoundId roundId{0};
        engine.BeginRound(roundId);
        const std::vector<std::vector<ActualChunkId>> selections{
            {ActualChunkId{0}, ActualChunkId{1}},
            {ActualChunkId{0}, ActualChunkId{1}},
            {ActualChunkId{0}, ActualChunkId{1}},
            {ActualChunkId{1}, ActualChunkId{2}}};
        for (std::size_t position = 0;
             position < setup.clientIds.size();
             ++position) {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = roundId;
            selection.clientId = setup.clientIds[position];
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = selections[position];
            engine.SubmitSelection(selection);
        }
        const ExternalEngineAdmissionPlan uploadAdmission =
            engine.FinalizeAdmission();
        passed &= Check(
            uploadAdmission.cohortSizes ==
                    std::vector<std::uint64_t>{3, 4, 1} &&
                uploadAdmission.acceptedActualChunkIds ==
                    std::vector<ActualChunkId>{
                        ActualChunkId{0}, ActualChunkId{1}},
            "dropout test first admits ciphertext uploads using all clients");

        bool rejectedEarlyTransition = false;
        try {
            (void)engine.SetRoundSurvivors(
                {ClientId{10}, ClientId{20}, ClientId{40}});
        }
        catch (const std::logic_error&) {
            rejectedEarlyTransition = true;
        }
        passed &= Check(
            rejectedEarlyTransition,
            "survivor target cannot be fixed before all uploads arrive");

        const std::vector<float> localValues{2.0F, 2.0F, -1.5F, -1.5F};
        for (std::size_t position = 0;
             position < setup.clientIds.size();
             ++position) {
            AcceptedChunkValues accepted;
            accepted.setupId = setup.setupId;
            accepted.roundId = roundId;
            accepted.clientId = setup.clientIds[position];
            accepted.chunkSize = setup.chunkSize;
            accepted.actualChunkIds = uploadAdmission.ForClient(
                setup.clientIds[position]).acceptedActualChunkIds;
            accepted.values.assign(
                accepted.actualChunkIds.size() * setup.chunkSize,
                localValues[position]);
            engine.SubmitAcceptedValues(accepted);

            AuxiliaryDeltaPayload auxiliary;
            auxiliary.setupId = setup.setupId;
            auxiliary.roundId = roundId;
            auxiliary.clientId = setup.clientIds[position];
            auxiliary.runningMeanDelta = {localValues[position], 0.0F};
            auxiliary.runningVarianceDelta = {localValues[position], 0.0F};
            engine.SubmitAuxiliaryDeltas(auxiliary);
        }

        const ExternalEngineSurvivorTransition transition =
            engine.SetRoundSurvivors(
                {ClientId{10}, ClientId{20}, ClientId{40}});
        passed &= Check(
            transition.survivorClientIds == std::vector<ClientId>{
                    ClientId{10}, ClientId{20}, ClientId{40}} &&
                transition.droppedClientIds ==
                    std::vector<ClientId>{ClientId{30}} &&
                transition.admission.cohortSizes ==
                    std::vector<std::uint64_t>{2, 3, 1} &&
                transition.admission.acceptedActualChunkIds ==
                    std::vector<ActualChunkId>{ActualChunkId{1}} &&
                transition.admission.ForClient(ClientId{30})
                    .acceptedActualChunkIds.empty(),
            "post-upload survivor transition recomputes minimum cohort");
        passed &= Check(
            transition.serverTargetReconstructionSeconds >= 0.0 &&
                transition.clientCollectiveKeyGenerationSeconds.size() == 4 &&
                transition.clientCollectiveKeyGenerationSeconds[0] == 0.0 &&
                transition.clientCollectiveKeyGenerationSeconds[1] > 0.0 &&
                transition.clientCollectiveKeyGenerationSeconds[2] > 0.0 &&
                transition.clientCollectiveKeyGenerationSeconds[3] > 0.0 &&
                transition.maximumClientCollectiveKeyGenerationSeconds > 0.0,
            "survivor target and fresh rekey timings exclude the dropout client");

        const ExternalEngineRoundResult result = FinalizeAcknowledged(engine);
        bool parameterResultIsCorrect = result.parameterDelta.size() == 600;
        for (std::size_t index = 0; index < result.parameterDelta.size(); ++index) {
            const float expected = index >= 256 && index < 512
                ? -0.25F
                : 0.0F;
            parameterResultIsCorrect &= Near(result.parameterDelta[index], expected);
        }
        const auto encryptionStage = static_cast<std::size_t>(
            CryptoStage::ENCRYPTION);
        const auto keySwitchStage = static_cast<std::size_t>(
            CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH);
        const auto partialStage = static_cast<std::size_t>(
            CryptoStage::PARTIAL_DECRYPTION);
        passed &= Check(
            parameterResultIsCorrect &&
                Near(result.runningMeanDelta[0], -0.25F) &&
                Near(result.runningVarianceDelta[0], -0.25F),
            "dropout aggregate keeps fixed denominator-128 client weights");
        passed &= Check(
            result.survivorReconfigurationApplied &&
                result.survivorClientIds == transition.survivorClientIds &&
                result.droppedClientIds == transition.droppedClientIds &&
                result.sparseClientCiphertextCount == 4 &&
                result.sparseSurvivorClientCiphertextCount == 3 &&
                result.sparseDiscardedClientCiphertextCount == 1 &&
                result.auxiliaryClientCiphertextCount == 4 &&
                result.auxiliarySurvivorClientCiphertextCount == 3 &&
                result.auxiliaryDiscardedClientCiphertextCount == 1,
            "all uploads are counted while one dropped model/aux ciphertext is discarded");
        passed &= Check(
            result.sparseModelClientStageTimings[0][encryptionStage]
                    .taskCount > 0 &&
                result.sparseModelClientStageTimings[0][keySwitchStage]
                    .taskCount == 0 &&
                result.sparseModelClientStageTimings[0][partialStage]
                    .taskCount == 0 &&
                result.auxiliaryClientStageTimings[0][encryptionStage]
                    .taskCount > 0 &&
                result.auxiliaryClientStageTimings[0][keySwitchStage]
                    .taskCount == 0 &&
                result.auxiliaryClientStageTimings[0][partialStage]
                    .taskCount == 0,
            "dropout client encrypts/uploads but never rekeys or partially decrypts");
        passed &= Check(
            result.survivorReconfigurationSequentialSeconds > 0.0 &&
                result.maximumSurvivorCollectiveKeyGenerationSeconds > 0.0 &&
                result.cryptoSequentialSeconds >=
                    result.survivorReconfigurationSequentialSeconds,
            "dropout reconfiguration is included in native crypto timing");
        engine.CommitRound();
    }
    catch (const std::exception& error) {
        std::cout << "[FAIL] post-upload dropout test exception: "
            << error.what() << '\n';
        passed = false;
    }

    try {
        auto request = MakeSetupRequest();
        request.parameterLayout = MakeLayout(
            LayoutKind::PARAMETER,
            "weight",
            1400);
        request.protocol.sparsification.selectedChunksPerClient = 5;
        auto cc = MakeProtocolCryptoContext(request.protocol);
        ExternalInputCryptoEngine engine(cc);
        const ExternalEngineSetupResult setup = engine.Setup(request);
        engine.BeginRound(RoundId{0});
        const std::vector<ActualChunkId> selected{
            ActualChunkId{0},
            ActualChunkId{1},
            ActualChunkId{2},
            ActualChunkId{3},
            ActualChunkId{4}};
        for (const auto clientId : setup.clientIds) {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = RoundId{0};
            selection.clientId = clientId;
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = selected;
            engine.SubmitSelection(selection);
        }
        const auto admission = engine.FinalizeAdmission();
        const std::vector<float> localValues{1.25F, -0.75F, 0.5F, 0.0F};
        for (std::size_t client = 0; client < setup.clientIds.size(); ++client) {
            AcceptedChunkValues accepted;
            accepted.setupId = setup.setupId;
            accepted.roundId = RoundId{0};
            accepted.clientId = setup.clientIds[client];
            accepted.chunkSize = setup.chunkSize;
            accepted.actualChunkIds = admission.ForClient(
                setup.clientIds[client]).acceptedActualChunkIds;
            accepted.values.assign(
                accepted.actualChunkIds.size() * setup.chunkSize,
                localValues[client]);
            engine.SubmitAcceptedValues(accepted);

            AuxiliaryDeltaPayload auxiliary;
            auxiliary.setupId = setup.setupId;
            auxiliary.roundId = RoundId{0};
            auxiliary.clientId = setup.clientIds[client];
            auxiliary.runningMeanDelta = {localValues[client], 0.0F};
            auxiliary.runningVarianceDelta = {localValues[client], 0.0F};
            engine.SubmitAuxiliaryDeltas(auxiliary);
        }
        const auto result = FinalizeAcknowledged(engine);
        const auto& server = result.sparseModelServerStageTimings;
        bool valuesAreCorrect = result.parameterDelta.size() == 1400;
        for (std::size_t index = 0; index < result.parameterDelta.size(); ++index) {
            valuesAreCorrect &= Near(
                result.parameterDelta[index],
                index < 5 * setup.chunkSize ? 0.25F : 0.0F);
        }
        passed &= Check(
            valuesAreCorrect &&
                result.sparseClientCiphertextCount == 8 &&
                result.sparseServerCiphertextCount == 2 &&
                result.sparseModelClientStageTimings.front()[
                    static_cast<std::size_t>(CryptoStage::ENCRYPTION)]
                    .taskCount == 2 &&
                server[static_cast<std::size_t>(CryptoStage::R2M)]
                    .taskCount == 8 &&
                server[static_cast<std::size_t>(CryptoStage::M2R)]
                    .taskCount == 2 * (request.protocol.sparsification.mlweRank + 2),
            "SPARSE parallel batches preserve multi-packet and multi-group results");
    }
    catch (const std::exception& error) {
        std::cout << "[FAIL] sparse multi-packet external test exception: "
            << error.what() << '\n';
        passed = false;
    }

    try {
        const ExternalEngineSetupRequest request =
            MakeRankOneSparseSetupRequest();
        auto cc = MakeProtocolCryptoContext(request.protocol);
        ExternalInputCryptoEngine engine(cc);
        const ExternalEngineSetupResult setup = engine.Setup(request);
        engine.BeginRound(RoundId{0});
        for (const auto clientId : setup.clientIds) {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = RoundId{0};
            selection.clientId = clientId;
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = {ActualChunkId{0}};
            engine.SubmitSelection(selection);
        }
        const auto admission = engine.FinalizeAdmission();
        const std::vector<float> localValues{1.25F, -0.75F, 0.5F, 0.0F};
        for (std::size_t client = 0; client < setup.clientIds.size(); ++client) {
            AcceptedChunkValues accepted;
            accepted.setupId = setup.setupId;
            accepted.roundId = RoundId{0};
            accepted.clientId = setup.clientIds[client];
            accepted.chunkSize = setup.chunkSize;
            accepted.actualChunkIds = admission.ForClient(
                setup.clientIds[client]).acceptedActualChunkIds;
            accepted.values.assign(setup.chunkSize, localValues[client]);
            engine.SubmitAcceptedValues(accepted);

            AuxiliaryDeltaPayload auxiliary;
            auxiliary.setupId = setup.setupId;
            auxiliary.roundId = RoundId{0};
            auxiliary.clientId = setup.clientIds[client];
            auxiliary.runningMeanDelta = {localValues[client], 0.0F};
            auxiliary.runningVarianceDelta = {localValues[client], 0.0F};
            engine.SubmitAuxiliaryDeltas(auxiliary);
        }
        const auto result = FinalizeAcknowledged(engine);
        bool valuesAreCorrect = result.parameterDelta.size() == 1400;
        for (std::size_t index = 0; index < result.parameterDelta.size(); ++index) {
            valuesAreCorrect &= Near(
                result.parameterDelta[index],
                index < setup.chunkSize ? 0.25F : 0.0F);
        }
        const auto& server = result.sparseModelServerStageTimings;
        passed &= Check(
            valuesAreCorrect && setup.chunkSize == 1024 &&
                result.sparseClientCiphertextCount == setup.clientIds.size() &&
                result.acceptedSparseContributionCount == setup.clientIds.size() &&
                result.aggregatedMLWECiphertextCount == 0 &&
                result.sparseServerCiphertextCount == 1 &&
                result.r2mCiphertextCount == 0 &&
                result.m2rCiphertextCount == 0 &&
                server[static_cast<std::size_t>(CryptoStage::R2M)]
                    .taskCount == 0 &&
                server[static_cast<std::size_t>(CryptoStage::MLWE_AGGREGATION)]
                    .taskCount == 0 &&
                server[static_cast<std::size_t>(CryptoStage::M2R)]
                    .taskCount == 0 &&
                server[static_cast<std::size_t>(CryptoStage::RLWE_AGGREGATION)]
                    .taskCount > 0,
            "rank-one SP uses label-aware direct RLWE aggregation without R2M/M2R");
    }
    catch (const std::exception& error) {
        std::cout << "[FAIL] rank-one sparse external test exception: "
            << error.what() << '\n';
        passed = false;
    }

    try {
        const ExternalEngineSetupRequest request = MakeFullSetupRequest();
        auto cc = MakeProtocolCryptoContext(request.protocol);
        ExternalInputCryptoEngine engine(cc);
        const ExternalEngineSetupResult setup = engine.Setup(request);
        passed &= Check(
            setup.clientInitialSetupSeconds.size() == request.clients.size() &&
                setup.clientCollectiveKeyGenerationSeconds.size() ==
                    request.clients.size() &&
                std::all_of(
                    setup.clientInitialSetupSeconds.begin(),
                    setup.clientInitialSetupSeconds.end(),
                    [](double value) { return value >= 0.0; }) &&
                std::all_of(
                    setup.clientCollectiveKeyGenerationSeconds.begin(),
                    setup.clientCollectiveKeyGenerationSeconds.end(),
                    [](double value) { return value >= 0.0; }),
            "external setup reports per-client parallel timing phases");
        passed &= Check(
            engine.EffectiveProtocolConfig().mode == ProtocolMode::FULL &&
                setup.chunkSize == request.protocol.modulus.ringDimension &&
                setup.chunkCount == 1,
            "FULL external setup uses direct RLWE blocks");

        const RoundId roundId{0};
        engine.BeginRound(roundId);
        passed &= Check(
            engine.Phase() == RoundPhase::VALUES_COLLECTING,
            "FULL round skips selection and admission phases");
        bool rejectedSelection = false;
        try {
            ClientSelectionRequest selection;
            selection.setupId = setup.setupId;
            selection.roundId = roundId;
            selection.clientId = setup.clientIds.front();
            selection.parameterLayoutHash = setup.parameterLayoutHash;
            selection.selectedActualChunkIds = {ActualChunkId{0}};
            engine.SubmitSelection(selection);
        }
        catch (const std::logic_error&) {
            rejectedSelection = true;
        }
        passed &= Check(
            rejectedSelection,
            "FULL external engine rejects sparse selection messages");

        const std::vector<float> localValues{2.0F, 2.0F, -1.5F, -1.5F};
        for (std::size_t position = 0; position < setup.clientIds.size(); ++position) {
            ExternalFullParameterDeltaPayload model;
            model.setupId = setup.setupId;
            model.roundId = roundId;
            model.clientId = setup.clientIds[position];
            model.parameterDelta.assign(setup.parameterNumel, localValues[position]);
            engine.SubmitFullParameterDelta(model);

            AuxiliaryDeltaPayload auxiliary;
            auxiliary.setupId = setup.setupId;
            auxiliary.roundId = roundId;
            auxiliary.clientId = setup.clientIds[position];
            auxiliary.runningMeanDelta = {
                localValues[position], 2.0F * localValues[position]};
            auxiliary.runningVarianceDelta = auxiliary.runningMeanDelta;
            engine.SubmitAuxiliaryDeltas(auxiliary);
        }
        const ExternalEngineRoundResult result = FinalizeAcknowledged(engine);
        const bool fullValuesAreCorrect = std::all_of(
            result.parameterDelta.begin(),
            result.parameterDelta.end(),
            [](float value) { return Near(value, 0.25F); });
        passed &= Check(
            fullValuesAreCorrect &&
                result.acceptedActualChunkIds.empty() &&
                result.cohortSizes.empty() &&
                result.fullModelClientCiphertextCount == setup.clientIds.size() &&
                result.fullModelAggregateCiphertextCount == 1 &&
                result.sparseClientCiphertextCount == 0 &&
                result.r2mCiphertextCount == 0 &&
                result.m2rCiphertextCount == 0,
            "FULL external round uses no sparse routing, R2M, or M2R");
        const auto hasTasks = [&](CryptoStage stage) {
            return result.cryptoStageTimings[
                static_cast<std::size_t>(stage)].taskCount > 0;
        };
        passed &= Check(
            result.cryptoThreadCount == 3 &&
                hasTasks(CryptoStage::ENCRYPTION) &&
                hasTasks(CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH) &&
                hasTasks(CryptoStage::RLWE_AGGREGATION) &&
                hasTasks(CryptoStage::PARTIAL_DECRYPTION) &&
                hasTasks(CryptoStage::DECODING),
            "FULL and auxiliary ciphertext stages use the shared executor");
        passed &= Check(
            result.estimatedParallelCryptoSeconds <=
                result.cryptoSequentialSeconds + 1.0e-9,
            "FULL parallel-client crypto estimate does not exceed wall time");
        const auto fullClientStagesAreIsolated = std::all_of(
            result.fullModelClientStageTimings.begin(),
            result.fullModelClientStageTimings.end(),
            [](const CryptoStageTimings& timings) {
                return timings[static_cast<std::size_t>(
                           CryptoStage::ENCRYPTION)].taskCount == 1 &&
                    timings[static_cast<std::size_t>(
                           CryptoStage::CLIENT_TO_COLLECTIVE_KEY_SWITCH)]
                           .taskCount == 1 &&
                    timings[static_cast<std::size_t>(
                           CryptoStage::PARTIAL_DECRYPTION)].taskCount == 1 &&
                    timings[static_cast<std::size_t>(CryptoStage::R2M)]
                           .taskCount == 0 &&
                    timings[static_cast<std::size_t>(CryptoStage::M2R)]
                           .taskCount == 0;
            });
        passed &= Check(
            result.fullModelClientStageTimings.size() ==
                    setup.clientIds.size() &&
                result.auxiliaryClientStageTimings.size() ==
                    setup.clientIds.size() &&
                fullClientStagesAreIsolated &&
                result.fullModelServerStageTimings[
                    static_cast<std::size_t>(CryptoStage::RLWE_AGGREGATION)]
                    .taskCount == 1 &&
                result.fullModelServerStageTimings[
                    static_cast<std::size_t>(CryptoStage::DECODING)]
                    .taskCount == 1,
            "FULL reports isolated client and server operation breakdowns");
        double expectedMaximumPreAggregation = 0.0;
        double expectedMaximumPartialDecryption = 0.0;
        double expectedMaximumTotal = 0.0;
        for (std::size_t client = 0;
             client < setup.clientIds.size();
             ++client) {
            expectedMaximumPreAggregation = std::max(
                expectedMaximumPreAggregation,
                result.clientEncryptionSeconds[client] +
                    result.clientKeySwitchSeconds[client]);
            expectedMaximumPartialDecryption = std::max(
                expectedMaximumPartialDecryption,
                result.clientPartialDecryptionSeconds[client]);
            expectedMaximumTotal = std::max(
                expectedMaximumTotal,
                result.clientTotalCryptoSeconds[client]);
        }
        passed &= Check(
            result.maximumClientPreAggregationSeconds ==
                    expectedMaximumPreAggregation &&
                result.maximumClientPartialDecryptionSeconds ==
                    expectedMaximumPartialDecryption &&
                result.maximumClientTotalCryptoSeconds ==
                    expectedMaximumTotal,
            "FULL paper timings use maxima of isolated per-client phases");
        engine.CommitRound();
    }
    catch (const std::exception& error) {
        std::cout << "[FAIL] external-input FULL crypto engine exception: "
            << error.what() << '\n';
        passed = false;
    }

    std::cout << (passed
        ? "All external-input crypto engine tests passed.\n"
        : "One or more external-input crypto engine tests failed.\n");
    return passed ? 0 : 1;
}
