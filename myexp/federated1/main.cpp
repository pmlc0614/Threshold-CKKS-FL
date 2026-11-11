#include "coefficient-codec-test.h"
#include "ring-conversion-test.h"
#include "threshold-ckks-test.h"
#include "collective-conversion-keygen-test.h"
#include "crypto-execution-test.h"
#include "federated-simulation-test.h"
#include "protocol-stage1-test.h"
#include "protocol-setup-test.h"
#include "protocol-stage3-test.h"
#include "protocol-stage4-test.h"
#include "protocol-stage5-test.h"
#include "protocol-contract-test.h"
#include "external-input-crypto-engine-test.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "all";
    if (argc > 2) return 1;
    if (mode == "--help" || mode == "-h") {
        std::cout << "FullSD / SparseSD v17 functional regression suites:\n";
        std::cout << "  codec\n";
        std::cout << "  ring\n";
        std::cout << "  threshold\n";
        std::cout << "  collective\n";
        std::cout << "  crypto-execution\n";
        std::cout << "  federated\n";
        std::cout << "  protocol-stage1\n";
        std::cout << "  protocol-setup\n";
        std::cout << "  protocol-stage3\n";
        std::cout << "  protocol-stage4\n";
        std::cout << "  protocol-stage5\n";
        std::cout << "  protocol-contract\n";
        std::cout << "  protocol-external-engine\n";
        return 0;
    }
    int result = 0;
    bool matched = false;
    if (mode == "all" || mode == "codec") { matched = true; result |= RunCoefficientCodecTests(); }
    if (mode == "all" || mode == "ring") { matched = true; result |= RunRingConversionTests(); }
    if (mode == "all" || mode == "threshold") { matched = true; result |= RunThresholdCKKSTests(); }
    if (mode == "all" || mode == "collective") { matched = true; result |= RunCollectiveConversionKeyTests(); }
    if (mode == "all" || mode == "crypto-execution") { matched = true; result |= RunCryptoExecutionTests(); }
    if (mode == "all" || mode == "federated") { matched = true; result |= RunFederatedSimulationTests(); }
    if (mode == "all" || mode == "protocol-stage1") { matched = true; result |= RunProtocolStage1Tests(); }
    if (mode == "all" || mode == "protocol-setup") { matched = true; result |= RunProtocolSetupTests(); }
    if (mode == "all" || mode == "protocol-stage3") { matched = true; result |= RunProtocolStage3Tests(); }
    if (mode == "all" || mode == "protocol-stage4") { matched = true; result |= RunProtocolStage4Tests(); }
    if (mode == "all" || mode == "protocol-stage5") { matched = true; result |= RunProtocolStage5Tests(); }
    if (mode == "all" || mode == "protocol-contract") { matched = true; result |= RunProtocolContractTests(); }
    if (mode == "all" || mode == "protocol-external-engine") { matched = true; result |= RunExternalInputCryptoEngineTests(); }
    if (!matched) { std::cerr << "Unknown test suite: " << mode << '\n'; return 1; }
    return result;
}
