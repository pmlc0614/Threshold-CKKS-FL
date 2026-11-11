#include "openfhe.h"

using namespace lbcrypto;
using namespace std;

using DCRTPoly = DCRTPolyImpl<BigVector>;
using TugType = TernaryUniformGeneratorImpl<NativeVector>;

int main() {

    // 컨텍스트 생성
    uint32_t multDepth = 3;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 16;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetScalingTechnique(ScalingTechnique::FIXEDAUTO);    // FIXEDAUTO
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    auto m_params = cc->GetCryptoParameters();
    auto paramsQ = m_params->GetElementParams();

    // 다항식 생성
    cout << "============================================" << endl;
    TugType tug;    
    DCRTPoly s(tug, paramsQ, Format::COEFFICIENT);
    auto m_vectors = s.GetAllElements();

    cout << "vector size: " << m_vectors.size() << endl;

    // q0 성분만 출력
    for (int i = 0; i < 10; i++) {
        cout << "q0 idx " << i << ": " << m_vectors[0].at(i) << endl;
    }
    cout << endl;

    // 첫 번째 성분만 출력
    for (size_t i=0; i<m_vectors.size(); i++) {
        cout << "q" << i << " idx 0: " << m_vectors[i].at(0) << endl;
    }



    return 0;
}