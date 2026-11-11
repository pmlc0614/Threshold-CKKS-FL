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
void DecryptCoeff(const CryptoContext<DCRTPoly>& cc, const Ciphertext<DCRTPoly>& ctxt, PrivateKey<DCRTPoly>& privateKey, double scalingFactor, vector<double>& decoded) {

    if (ctxt == nullptr)
        OPENFHE_THROW("ciphertext is empty");

    Plaintext decrypted = CryptoContextImpl<DCRTPoly>::GetPlaintextForDecrypt(
        ctxt->GetEncodingType(), ctxt->GetElements()[0].GetParams(), cc->GetEncodingParams(),
        cc->GetCKKSDataType());

    DecryptResult result;
    result = cc->GetScheme()->Decrypt(ctxt, privateKey, &decrypted->GetElement<NativePoly>());
    decrypted->SetScalingFactorInt(result.scalingFactorInt);

    auto poly2 = decrypted->GetElement<NativePoly>();
    DecodeCoeffsNativePoly(cc, poly2, scalingFactor, decoded);

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

    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
        cout << "i: " << qi[i] << endl;
    }

    // N/k 차원 ILDCRTParams<BigInteger> 객체 생성
    uint32_t k = 2;
    vector<shared_ptr<ILParamsImpl<NativeInteger>>> vec_ILparams;
    auto cycloOrder = 2*N/k;

    for(size_t i=0; i<vec_params.size(); i++) {
        vec_ILparams.emplace_back(make_shared<ILParamsImpl<NativeInteger>>(cycloOrder, qi[i]));
    }
    shared_ptr<ILDCRTParams<BigInteger>> dcrtparams = make_shared<ILDCRTParams<BigInteger>>(cycloOrder, vec_ILparams);


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


    // 암호화
    auto ctxt = cc->Encrypt(keys.publicKey, pt);
    cout << "#ciphertext modulus: " << ctxt->GetElements()[0].GetAllElements().size() << endl;

    // 포맷 확인
    cout << "format" << endl;
    cout << ctxt->GetElements()[0].GetFormat() << endl;

    // Split 알고리즘
    auto b = ctxt->GetElements()[0];
    auto a = ctxt->GetElements()[1];

    b.SetFormat(COEFFICIENT);
    a.SetFormat(COEFFICIENT);


    // b_nativevec 행렬 생성
    auto vec_polyimpl = b.GetAllElements(); 
    auto num_modulus = ctxt->GetElements()[0].GetAllElements().size();
    vector<NativeVector> b_nativevec;
    b_nativevec.reserve(num_modulus);
    for(size_t i=0; i<num_modulus; i++) {
        b_nativevec.emplace_back(vec_polyimpl[i].GetValues());
    }

    // bi_nativevec 3차원 구조 생성
    vector<vector<NativeVector>> bi_nativevec(k, vector<NativeVector>(num_modulus));
    for(size_t l=0; l<k; l++) {
        for(size_t i=0; i<num_modulus; i++) {
            bi_nativevec[l][i] = NativeVector(N/k, qi[i]);
        }
    }
    for(size_t l=0; l<k; l++) {
        for(size_t i=0; i<num_modulus; i++){
            for(size_t j=0; j<N/k; j++) {
                bi_nativevec[l][i][j] = b_nativevec[i][l+k*j];
            }
        }
    }

    // b_split 설정
    vector<DCRTPoly> b_split;
    for(size_t l=0; l<k; l++) {
        DCRTPoly tmp(dcrtparams, COEFFICIENT, true);
        for(size_t i=0; i<num_modulus; i++) {

            PolyImpl<NativeVector> polyimpl_tmp(b.GetAllElements()[i].GetParams(), COEFFICIENT, true);
            auto tmp_values = polyimpl_tmp.GetValues();
            tmp_values = bi_nativevec[l][i];
            tmp.SetElementAtIndex(i, polyimpl_tmp);
        }
        b_split.emplace_back(tmp);
    }



    // vector<DCRTPoly> b_split;
    // auto vec_polyimpl = b.GetAllElements();     // b를 구성하는 작은 다항식들을 불러옴. vector<PolyImpl<NativeVector>> 타입을 가짐.

    // // i=0인 경우만 간단히 구현
    // auto polyimpl = vec_polyimpl[0];        // PolyImpl<NativeVector> 타입을 가짐
    // NativeVector nativevec = polyimpl.GetValues();

    // NativeVector nativevec_tmp(N/k, qi[0]);
    // for(size_t j=0; j<N/k; j++) {
    //     nativevec_tmp[j] = nativevec[k*j];
    // }

    // PolyImpl<NativeVector> polyimpl_tmp(polyimpl.GetParams(), COEFFICIENT, true);
    // auto temp = polyimpl_tmp.GetValues();
    // temp = nativevec_tmp;

    // b_split[0] = DCRTPoly(dcrtparams, COEFFICIENT, true);
    // b_split[0].SetElementAtIndex(0, polyimpl_tmp);


    // level down
    cc->LevelReduceInPlace(ctxt, nullptr, 1);
    cout << "#ciphertext modulus: " << ctxt->GetElements()[0].GetAllElements().size() << endl;

    // 복호화
    vector<double> decoded(N);
    DecryptCoeff(cc, ctxt, keys.secretKey, scalingFactor, decoded);


    for(int i=0; i<5; i++) {
        cout << "decoded[" << i << "]: " << setprecision(10) << decoded[i] << endl;
    }



    return 0;
}