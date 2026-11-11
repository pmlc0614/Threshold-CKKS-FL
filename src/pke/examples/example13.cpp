#include "openfhe.h"

using namespace lbcrypto;
using namespace std;

using DCRTPoly = DCRTPolyImpl<BigVector>;
using TugType = TernaryUniformGeneratorImpl<NativeVector>;

DCRTPoly EncodeCoeffs(const CryptoContext<DCRTPoly>& cc, const vector<double>& vecx, double scalingFactor) {

    // 파라미터
    auto m_params = cc->GetCryptoParameters();
    auto paramsQ = m_params->GetElementParams();
    auto N = cc->GetRingDimension();
    // if(N != vecx.size()) OPENFHE_THROW("N != vecx.size()");

    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
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

    for(usint j=0; j<N; j++) {
        double x = vecx[j];
        long double scaled = static_cast<long double>(x) * static_cast<long double>(scalingFactor);
        long long rounded = llroundl(scaled);
        NativeInteger abs_rounded(llabs(rounded));

        // tower에 대한 for문
        for(size_t i=0; i<qi.size(); i++) {
            NativeInteger r = abs_rounded.Mod(qi[i]);
            if(rounded<0 && r!=NativeInteger(0)) {
                r = qi[i].ModSub(r, qi[i]);
            }
            valuesPerTower[i][j] = r;
        }
    }
    for (size_t j = 0; j < t; ++j) {
        towers[j].SetValues(std::move(valuesPerTower[j]), Format::COEFFICIENT);
    }

    return poly;
}
void DecodeCoeffs(const CryptoContext<DCRTPoly>& cc, DCRTPoly& pt, double scalingFactor, vector<double>& decoded) {
    
    if(pt.GetParams()->GetParams().size() != 1) OPENFHE_THROW("number of modulus should be one");

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
void DecodeCoeffsNativePoly(const CryptoContext<DCRTPoly>& cc, NativePoly& nativePoly, double scalingFactor, size_t N, size_t k, vector<double>& decoded) {
    
    // 디코딩
    // 1) CRT 합성: DCRT → 큰 모듈러스 Q의 단일 Poly
    auto V       = nativePoly.GetValues();            // VecType with modulus Q
    NativeInteger Q    = nativePoly.GetModulus();
    NativeInteger half = Q >> 1;                          // Q/2

    decoded.resize(N/k);
    long double invScale = 1.0L / scalingFactor;  

    // 2) 각 계수마다 center-lift 후 스케일 복원
    for (usint i = 0; i < N/k; ++i) {
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

void Split(const CryptoContext<DCRTPoly>& cc, DCRTPoly& b, vector<DCRTPoly>& b_split, size_t k, size_t N) {

    if(b.GetFormat() == EVALUATION) OPENFHE_THROW("Format should be COEFFICIENT");


    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
    }

    // num_modulus
    size_t num_modulus = b.GetParams()->GetParams().size();

    // N/k 차원 ILDCRTParams<BigInteger> 객체 생성
    vector<shared_ptr<ILParamsImpl<NativeInteger>>> vec_ILparams;
    auto cycloOrder = 2*N/k;
    for(size_t i=0; i<num_modulus; i++) {
        vec_ILparams.emplace_back(make_shared<ILParamsImpl<NativeInteger>>(cycloOrder, qi[i]));
    }
    shared_ptr<ILDCRTParams<BigInteger>> dcrtparams = make_shared<ILDCRTParams<BigInteger>>(cycloOrder, vec_ILparams);

    // b_nativevec 행렬 생성
    const auto vec_polyimpl = b.GetAllElements(); 
    // auto num_modulus = ctxt->GetElements()[0].GetAllElements().size();
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
    b_split.clear();
    for(size_t l=0; l<k; l++) {
        DCRTPoly tmp(dcrtparams, COEFFICIENT, true);
        for(size_t i=0; i<num_modulus; i++) {

            PolyImpl<NativeVector> polyimpl_tmp(dcrtparams->GetParams()[i], COEFFICIENT, true);
            // PolyImpl<NativeVector> polyimpl_tmp(b.GetAllElements()[i].GetParams(), COEFFICIENT, true);
            polyimpl_tmp.SetValues(bi_nativevec[l][i], COEFFICIENT);

            // auto tmp_values = polyimpl_tmp.GetValues();
            // tmp_values = bi_nativevec[l][i];
            tmp.SetElementAtIndex(i, polyimpl_tmp);
        }

        b_split.emplace_back(tmp);
    }

}
void Combine(const CryptoContext<DCRTPoly>& cc, const vector<DCRTPoly>& a, DCRTPoly& res, size_t k, size_t N) {

    for(size_t i=0; i<k; i++) {
        if(a[i].GetFormat() != COEFFICIENT) OPENFHE_THROW("Combine: Format should be COEFFICIENT");
    }

    // num_modulus
    size_t num_modulus = a[0].GetParams()->GetParams().size();

    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(num_modulus);
    for(size_t i=0; i<num_modulus; i++) {
        qi.push_back(vec_params[i]->GetModulus());
    }

    // k_dcrtpoly_a 3차원 구조 생성
    vector<vector<NativeVector>> k_dcrtpoly_a(k, vector<NativeVector>(num_modulus));
    for(size_t l=0; l<k; l++) {
        const auto vec_polyimpl = a[l].GetAllElements();
        for(size_t i=0; i<num_modulus; i++) {
            k_dcrtpoly_a[l][i] = NativeVector(N/k, qi[i]);
            k_dcrtpoly_a[l][i] = vec_polyimpl[i].GetValues();
        }
    }

    // dcrtpoly 행렬 생성 
    vector<NativeVector> dcrtpoly_a(num_modulus);
    for(size_t i=0; i<num_modulus; i++) {
        dcrtpoly_a[i] = NativeVector(N, qi[i]);
    }
    for(size_t i=0; i<num_modulus; i++) {
        for(size_t j=0; j<N; j++) {
            dcrtpoly_a[i][j] = k_dcrtpoly_a[j%k][i][j/k];
        }
    }

    // ILDCRTParams 생성
    vector<shared_ptr<ILParamsImpl<NativeInteger>>> vec_ILparams;
    auto cycloOrder = 2*N;
    for(size_t i=0; i<num_modulus; i++) {
        vec_ILparams.emplace_back(make_shared<ILParamsImpl<NativeInteger>>(cycloOrder, qi[i]));
    }
    shared_ptr<ILDCRTParams<BigInteger>> dcrtparams = make_shared<ILDCRTParams<BigInteger>>(cycloOrder, vec_ILparams);

    // a consturction
    res = DCRTPoly(dcrtparams, COEFFICIENT, true);
    auto& m_vectors = res.GetAllElements();
    for(size_t i=0; i<num_modulus; i++) {
        m_vectors[i].SetValues(dcrtpoly_a[i], Format::COEFFICIENT);
    }


}
void MultX(const CryptoContext<DCRTPoly>& cc, DCRTPoly& a, DCRTPoly& res, size_t k, size_t N, size_t num_modulus) {

    if(a.GetFormat() == EVALUATION) OPENFHE_THROW("Format should be COEFFICIENT");

    if(a.GetAllElements()[0].GetValues().GetLength() != N/k) OPENFHE_THROW("vector length of a is not N/k");

    // 추가
    res = DCRTPoly(a.GetParams(), COEFFICIENT, true);

    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
    }
    const vector<NativePoly> vec_polyimpl = a.GetAllElements();
    
    // vector<NativeVector> mat_a;
    // mat_a.reserve(num_modulus);
    for(size_t i=0; i<num_modulus; i++) {
        NativeVector temp = vec_polyimpl[i].GetValues();
        NativeVector temp2(N/k, qi[i]);

        temp2[0] = qi[i]-temp[N/k-1].Mod(qi[i]);
        // temp2[0] = (-temp[N/k-1]).Mod(qi[i]);
        for(size_t j=1; j<N/k; j++) {
            temp2[j] = temp[j-1];
        }

        PolyImpl<NativeVector> poly(vec_polyimpl[i].GetParams(), COEFFICIENT, true);
        poly.SetValues(temp2, COEFFICIENT);
        res.SetElementAtIndex(i, poly);
        
    }

}
void RingEmbedding(const CryptoContext<DCRTPoly>& cc, const DCRTPoly& a, DCRTPoly& res, size_t k, size_t N) {

    if(a.GetFormat() == EVALUATION) OPENFHE_THROW("Format should be COEFFICIENT");

    // num_modulus
    size_t num_modulus = a.GetParams()->GetParams().size();

    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
    }

    // double_CRT 행렬 생성 
    vector<NativeVector> dcrt_a(num_modulus);
    for(size_t i=0; i<num_modulus; i++) {
        dcrt_a[i] = NativeVector(N/k, qi[i]);
    }

    vector<NativeVector> dcrt_embed_a(num_modulus);
    for(size_t i=0; i<num_modulus; i++) {
        dcrt_embed_a[i] = NativeVector(N, qi[i]);
        for(size_t j=0; j<N; j++) {
            if(j%(N/k) == 0) {
                dcrt_embed_a[i][j] = dcrt_a[i][j/(N/k)];
            } else {
                dcrt_embed_a[i][j] = 0;
            }
        }
    }

    // ILDCRTParams 생성
    vector<shared_ptr<ILParamsImpl<NativeInteger>>> vec_ILparams;
    auto cycloOrder = 2*N;
    for(size_t i=0; i<num_modulus; i++) {
        vec_ILparams.emplace_back(make_shared<ILParamsImpl<NativeInteger>>(cycloOrder, qi[i]));
    }
    shared_ptr<ILDCRTParams<BigInteger>> dcrtparams = make_shared<ILDCRTParams<BigInteger>>(cycloOrder, vec_ILparams);

    // ring_embedding 생성
    res = DCRTPoly(dcrtparams, COEFFICIENT, true);
    auto& m_vectors = res.GetAllElements();
    for(size_t i=0; i<num_modulus; i++) {
        m_vectors[i].SetValues(dcrt_embed_a[i], Format::COEFFICIENT);
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
    uint32_t k = 2;
    cout << "ring dimension: " << N << endl;

    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    // 모듈러스 집합
    vector<NativeInteger> qi;
    auto vec_params = cc->GetElementParams()->GetParams();
    qi.reserve(vec_params.size());
    for(size_t i=0; i<vec_params.size(); i++) {
        qi.push_back(vec_params[i]->GetModulus());
        cout << i << ": " << qi[i] << endl;
    }

    // 키생성
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // 입력
    vector<double> vecx = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
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
    auto num_modulus = ctxt->GetElements()[0].GetAllElements().size();

    // Split 알고리즘
    auto b = ctxt->GetElements()[0];
    auto a = ctxt->GetElements()[1];
    auto s = keys.secretKey->GetPrivateElement();

    b.SetFormat(COEFFICIENT);
    a.SetFormat(COEFFICIENT);
    s.SetFormat(COEFFICIENT);

    vector<DCRTPoly> a_split, b_split, s_split; 
    Split(cc, a, a_split, k, N);
    Split(cc, b, b_split, k, N);
    Split(cc, s, s_split, k, N);


    // (k,k) 차원 구조 DCRTPoly 객체 생성
    vector<vector<DCRTPoly>> mat_mlwe_c(k, vector<DCRTPoly>(k));
    for(size_t i=0; i<k; i++) {
        for(size_t j=0; j<k; j++) {
            mat_mlwe_c[i][j] = a_split[(i-j+k)%k];

            if((i-j+k)%k + j >= k) {
                DCRTPoly tmp;
                MultX(cc, mat_mlwe_c[i][j], tmp, k, N, num_modulus);
                mat_mlwe_c[i][j] = tmp;
            }
        }
    }

    // // format 변경
    // for(size_t i=0; i<k; i++) {
    //     for(size_t j=0; j<k; j++) {
    //         mat_mlwe_c[i][j].SetFormat(EVALUATION);
    //     }
    //     a_split[i].SetFormat(EVALUATION);
    //     b_split[i].SetFormat(EVALUATION);
    //     s_split[i].SetFormat(EVALUATION);
    // }


    // MLWE -> RLWE 변환
    // 결합
    vector<DCRTPoly> C(k+1);
    for(size_t j=0; j<k; j++) {
        vector<DCRTPoly> vecpoly_tmp(k);
        for(size_t l=0; l<k; l++) {
            vecpoly_tmp[l] = mat_mlwe_c[l][j];
        }
        Combine(cc, vecpoly_tmp, C[j], k, N);
    }
    for(size_t l=0; l<k; l++) {
        Combine(cc, b_split, C[k], k, N);
    }



    // DCRTPoly m0 = b_split[0];
    // for(size_t i=0; i<k; i++) {
    //     m0 += mat_mlwe_c[0][i] * s_split[i];
    // }
    // m0.SetFormat(COEFFICIENT);

    // DCRTPoly m1 = b_split[1];
    // for(size_t i=0; i<k; i++) {
    //     m1 += mat_mlwe_c[1][i] * s_split[i];
    // }
    // m1.SetFormat(COEFFICIENT);

    
    // // level down
    // cc->LevelReduceInPlace(ctxt, nullptr, 1);
    // cout << "number of modulis: " << ctxt->GetElements()[0].GetAllElements().size() << endl;

    // vector<double> decoded(N/k);
    // DecodeCoeffsNativePoly(cc, m0.GetAllElements()[0], scalingFactor, N, k, decoded);
    // for(int i=0; i<5; i++) {
    //     cout << "decoded[" << i << "]: " << setprecision(10) << decoded[i] << endl;
    // }

    // DecodeCoeffsNativePoly(cc, m1.GetAllElements()[0], scalingFactor, N, k, decoded);
    // for(int i=0; i<5; i++) {
    //     cout << "decoded[" << i << "]: " << setprecision(10) << decoded[i] << endl;
    // }



    return 0;
}