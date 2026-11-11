#include "openfhe.h"

using namespace lbcrypto;
using namespace std;

int main() {


    uint32_t multDepth = 3;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 16;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetScalingTechnique(ScalingTechnique::FIXEDAUTO);    // FIXEDAUTO

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // 모듈러스 체인 출력
    cout << "------------------------------" << endl;
    auto m_params = cc->GetCryptoParameters();
    std::cout << *m_params << std::endl;

    auto m_paramsP = std::dynamic_pointer_cast<CryptoParametersRNS>(m_params)->GetParamsP();
    std::cout << *m_paramsP << std::endl;

    // 링 차원 출력
    cout << m_params->GetRingDimension() << endl;

    // MaxQ 출력
    cout << StdLatticeParm::FindMaxQ(DistributionType::HEStd_ternary, SecurityLevel::HEStd_128_classic, 65536) << endl;
    cout << StdLatticeParm::FindMaxQ(DistributionType::HEStd_ternary, SecurityLevel::HEStd_128_classic, 32768) << endl;
    cout << StdLatticeParm::FindMaxQ(DistributionType::HEStd_ternary, SecurityLevel::HEStd_128_classic, 16384) << endl;
    cout << StdLatticeParm::FindMaxQ(DistributionType::HEStd_ternary, SecurityLevel::HEStd_128_classic, 8192) << endl;

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);


    std::vector<double> x = {1.0, 2.0, 3.0};
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    cout << ctxt->GetScalingFactor() << endl;
    cout << ctxt->GetLevel() << endl;

    cout << *(ctxt->GetElements()[0].GetParams()) << endl;

    cout << "-------------------------" << endl;
    
    auto ctxt_y = cc->EvalMult(ctxt, ctxt);

    cout << ctxt_y->GetScalingFactor() << endl;
    cout << ctxt_y->GetLevel() << endl;
    cout << *(ctxt_y->GetElements()[0].GetParams()) << endl;

    Plaintext result;
    cc->Decrypt(keys.secretKey, ctxt_y, &result);

    std::cout.precision(8);
    std::cout << result << std::endl;


    return 0;
}