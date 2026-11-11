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

    // 인코딩
    double x = 3.1415926535;
    double scalingFactor = pow(2.0,40);
    BigInteger Q = BigInteger(1) << 60;
    long double scaled = static_cast<long double>(x) * static_cast<long double>(scalingFactor);
    long long rounded = llround(scaled);
    BigInteger absValue = BigInteger(llabs(rounded));
    BigInteger result;
    if(rounded >= 0) {
        result = absValue.Mod(Q);
    } else {
        BigInteger negValue = Q.ModSub(absValue, Q);
        result = negValue;
    }

    cout << "x: " << setprecision(10) << x << endl;
    cout << "scaling factor: " << scalingFactor << endl;
    cout << "result: " << result << endl;

    // 디코딩
    BigInteger halfQ = Q>>1;
    BigInteger signedY;
    if(result > halfQ) {
        signedY = result - Q;
    } else {
        signedY = result;
    }
    long double value = static_cast<long double>(signedY.ConvertToInt());
    double decoded = static_cast<double>(value / scalingFactor);

    cout << "decoded result: " << setprecision(10) << decoded << endl;


    return 0;
}