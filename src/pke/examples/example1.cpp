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
    parameters.SetScalingTechnique(ScalingTechnique::FIXEDMANUAL);    // FIXEDMANUAL

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    auto m_params = cc->GetCryptoParameters();
    std::cout << *m_params << std::endl;

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);


    // modulus print

    cout << cc->GetModulus() << endl;
    m_params = cc->GetCryptoParameters();
    std::cout << *m_params << std::endl;
    auto m_paramsPK = m_params->GetParamsPK();
    std::cout << *m_paramsPK << std::endl;
    auto m_paramsP = std::dynamic_pointer_cast<CryptoParametersRNS>(m_params)->GetParamsP();
    std::cout << *m_paramsP << std::endl;

    // 암호화
    std::vector<double> x = {1.0, 2.0, 3.0};
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    cout << ctxt->GetScalingFactor() << endl;
    cout << ctxt->GetLevel() << endl;
    cout << *(ctxt->GetElements()[0].GetParams()) << endl;

    // 동형 곱셈
    cout << "-------------------------" << endl;
    auto ctxt_y = cc->EvalMult(ctxt, ctxt);
    cout << ctxt_y->GetScalingFactor() << endl;
    cout << ctxt_y->GetLevel() << endl;
    cout << *(ctxt_y->GetElements()[0].GetParams()) << endl;

    // rescaling
    ctxt_y = cc->ModReduce(ctxt_y);
    cout << ctxt_y->GetScalingFactor() << endl;
    cout << ctxt_y->GetLevel() << endl;
    cout << *(ctxt_y->GetElements()[0].GetParams()) << endl;

    // Plaintext result;
    // cc->Decrypt(keys.secretKey, ctxt_y, &result);

    // std::cout.precision(8);
    // std::cout << result << std::endl;


    return 0;
}