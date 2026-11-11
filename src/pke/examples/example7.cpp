#include "openfhe.h"

using namespace lbcrypto;
using namespace std;

using DCRTPoly = DCRTPolyImpl<BigVector>;
using TugType = TernaryUniformGeneratorImpl<NativeVector>;

DCRTPoly EncodeCoeffs(const CryptoContext<DCRTPoly>& cc, const vector<double>& vecx, double scalingFactor) {

    auto m_params = cc->GetCryptoParameters();
    auto paramsQ = m_params->GetElementParams();

    // 모듈러스 집합
    auto N = cc->GetRingDimension();
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
        cout << "i: " << qi[i] << endl;
    }

    // 인코딩
    DCRTPoly poly(paramsQ, Format::COEFFICIENT, true);
    auto& towers = poly.GetAllElements();

    const size_t t = qi.size();
    vector<NativeVector> valuesPerTower;
    valuesPerTower.reserve(t);
    for (size_t j=0; j<t; j++){
        valuesPerTower.emplace_back(NativeVector(N, qi[j]));
    }

    for(usint i=0; i<N; i++) {
        double x = vecx[i];
        long double scaled = static_cast<long double>(x) * static_cast<long double>(scalingFactor);
        long long rounded = llroundl(scaled);
        NativeInteger abs_rounded(llabs(rounded));

        // tower에 대한 for문
        for(size_t j=0; j<qi.size(); j++) {
            NativeInteger r = abs_rounded.Mod(qi[j]);
            if(rounded<0 && r!=NativeInteger(0)) {
                r = qi[j].ModSub(r, qi[j]);
            }
            valuesPerTower[j][i] = r;
        }
    }
    for (size_t j = 0; j < t; ++j) {
        towers[j].SetValues(std::move(valuesPerTower[j]), Format::COEFFICIENT);
    }

    return poly;
}
void DecodeCoeffs(const CryptoContext<DCRTPoly>& cc, DCRTPoly& pt, double scalingFactor, vector<double>& decoded) {
    
    // 디코딩
    // 1) CRT 합성: DCRT → 큰 모듈러스 Q의 단일 Poly
    auto N = cc->GetRingDimension();
    auto bigPoly = pt.CRTInterpolate();          // PolyLargeType
    auto V       = bigPoly.GetValues();            // VecType with modulus Q
    BigInteger Q    = bigPoly.GetModulus();
    BigInteger half = Q >> 1;                          // Q/2

    decoded.resize(N);
    long double invScale = 1.0L / scalingFactor;  

    // 2) 각 계수마다 center-lift 후 스케일 복원
    for (usint i = 0; i < N; ++i) {
        BigInteger vi = V[i];              // vi ∈ [0, Q)
        bool neg = false;

        // center-lift: (-Q/2, Q/2] 구간의 대표로 변환
        if (vi > half) {
            // 음수 계수: -(Q - vi)
            vi  = Q - vi;
            neg = true;
        }
        // 이제 vi는 계수의 |정수값| (절댓값) 크기

        // 3) 스케일 복원: (정수) / 2^scaleBits
        //    (vi가 64비트 범위 내라고 가정. 더 클 수 있으면 ToString()→stold 등으로 변환)
        long double val = static_cast<long double>(vi.ConvertToInt());
        if (neg) val = -val;
        val *= invScale;                          // 나누기 2^scaleBits

        decoded[i] = static_cast<double>(val);
    }
}
void DecodeCoeffsNativePoly(const CryptoContext<DCRTPoly>& cc, NativePoly& nativePoly, double scalingFactor, vector<double>& decoded) {
    
    // 디코딩
    // 1) CRT 합성: DCRT → 큰 모듈러스 Q의 단일 Poly
    auto N = cc->GetRingDimension();
    auto V       = nativePoly.GetValues();            // VecType with modulus Q
    NativeInteger Q    = nativePoly.GetModulus();
    NativeInteger half = Q >> 1;                          // Q/2

    decoded.resize(N);
    long double invScale = 1.0L / scalingFactor;  

    // 2) 각 계수마다 center-lift 후 스케일 복원
    for (usint i = 0; i < N; ++i) {
        NativeInteger vi = V[i];              // vi ∈ [0, Q)
        bool neg = false;

        // center-lift: (-Q/2, Q/2] 구간의 대표로 변환
        if (vi > half) {
            // 음수 계수: -(Q - vi)
            vi  = Q - vi;
            neg = true;
        }
        // 이제 vi는 계수의 |정수값| (절댓값) 크기

        // 3) 스케일 복원: (정수) / 2^scaleBits
        //    (vi가 64비트 범위 내라고 가정. 더 클 수 있으면 ToString()→stold 등으로 변환)
        long double val = static_cast<long double>(vi.ConvertToInt());
        if (neg) val = -val;
        val *= invScale;                          // 나누기 2^scaleBits

        decoded[i] = static_cast<double>(val);
    }
}

int main() {

    // 컨텍스트 생성
    uint32_t multDepth = 1;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 16;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetScalingTechnique(ScalingTechnique::FIXEDMANUAL);    // FIXEDMANUAL
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    auto m_params = cc->GetCryptoParameters();
    auto N = cc->GetRingDimension();
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    // 키생성
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // 입력
    vector<double> vecx = {3.1415926535, 2.142423424, 1.14243424244};
    double scalingFactor = pow(2.0,40);
    NativeInteger scalingFactorInt = NativeInteger(1) << 40;
    
    // 인코딩 (PlaintextImpl 객체 생성)
    DCRTPoly poly = EncodeCoeffs(cc, vecx, scalingFactor);

    auto paramsQ = cc->GetElementParams();
    auto pt = make_shared<CKKSPackedEncoding>(paramsQ, cc->GetEncodingParams(), REAL);
    pt->SetLevel(0);
    pt->SetNoiseScaleDeg(1);
    pt->SetScalingFactor(scalingFactor);
    pt->SetScalingFactorInt(scalingFactorInt);

    auto& el_dcrtpoly = pt->GetElement<DCRTPoly>();
    el_dcrtpoly = poly;
    // auto el_nativepoly = pt->GetElement<NativePoly>();
    // el_nativepoly = NativePoly(paramsQ->GetParams()[0], COEFFICIENT, true);
    // auto el_poly = pt->GetElement<Poly>();
    // el_poly = Poly(paramsQ, COEFFICIENT, true);

    // cout << "debugging " << pt->GetElement<Poly>().GetValues();


    // 암호화
    auto ctxt = cc->Encrypt(keys.publicKey, pt);
    cout << "#ciphertext modulus: " << ctxt->GetElements()[0].GetAllElements().size() << endl;
    // ctxt = ctxt+ctxt;

    // level down
    cc->LevelReduceInPlace(ctxt, nullptr, 1);
    cout << "#ciphertext modulus: " << ctxt->GetElements()[0].GetAllElements().size() << endl;

    // 복호화
    auto privateKey = keys.secretKey;
    if (ctxt == nullptr)
        OPENFHE_THROW("ciphertext is empty");

    Plaintext decrypted = CryptoContextImpl<DCRTPoly>::GetPlaintextForDecrypt(
        ctxt->GetEncodingType(), ctxt->GetElements()[0].GetParams(), cc->GetEncodingParams(),
        cc->GetCKKSDataType());

    DecryptResult result;
    result = cc->GetScheme()->Decrypt(ctxt, privateKey, &decrypted->GetElement<NativePoly>());
    decrypted->SetScalingFactorInt(result.scalingFactorInt);

    auto poly2 = decrypted->GetElement<NativePoly>();
    vector<double> decoded2(N);
    DecodeCoeffsNativePoly(cc, poly2, scalingFactor, decoded2);

    for(int i=0; i<5; i++) {
        cout << "decoded[" << i << "]: " << setprecision(10) << decoded2[i] << endl;
    }



    return 0;
}