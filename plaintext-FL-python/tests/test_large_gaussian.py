"""Sampler smoke checks and actual Full/Sparse rounds; not security certificates."""
from dataclasses import replace
import math
import unittest
import numpy as np
from plaintext_fl.native_engine import NativeCryptoEngine, native_binding_available
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator
from tests.test_large_scaling import options_for
from tests.test_native_engine import make_setup, make_full_setup, make_candidates, make_full_candidates

@unittest.skipUnless(native_binding_available(), 'native binding unavailable')
class LargeGaussianTests(unittest.TestCase):
    def test_small_rational_gaussian_histogram(self):
        import ciphertext_fl_cpp as native
        samples=list(map(int,native._gaussian_samples(3.2,32768)))
        # Independent finite PMF for sigma=16/5, support [-77,77].
        weights={x:math.exp(-x*x*25/512) for x in range(-77,78)}
        total=sum(weights.values())
        self.assertTrue(all(-77<=x<=77 for x in samples))
        for x in range(-8,9):
            p=weights[x]/total
            self.assertLess(abs(samples.count(x)/len(samples)-p),8*math.sqrt(p*(1-p)/len(samples))+1/len(samples))

    def test_large_integer_support_and_moments(self):
        import ciphertext_fl_cpp as native
        for sigma in (2**55,2**56,2**71,2**79,2**128):
            with self.subTest(sigma=sigma):
                samples=list(map(int,native._gaussian_samples(float(sigma),4096)))
                self.assertTrue(all(abs(x)<=24*sigma for x in samples))
                values=np.array([x/sigma for x in samples])
                self.assertLess(abs(values.mean()),0.15)
                self.assertGreater(np.mean(values*values),0.75)
                self.assertLess(np.mean(values*values),1.25)
                self.assertTrue(any(x<0 for x in samples) and any(x>0 for x in samples))
                if sigma>2**64:
                    self.assertTrue(any(abs(x)>2**64 for x in samples))
                    # A float-rounded sample would lose all these low bits.
                    self.assertTrue(any(x%256!=0 for x in samples))

    def test_invalid_sigma_rejected(self):
        import ciphertext_fl_cpp as native
        for sigma in (0,-1,0.5,3.21,math.inf,math.nan,float(2**129)):
            with self.subTest(sigma=sigma),self.assertRaises((ValueError,RuntimeError)):
                native._gaussian_samples(sigma,1)

    def test_rns_lifting_preserves_large_signed_samples(self):
        import ciphertext_fl_cpp as native
        for sigma in (2**71,2**79):
            result=native._gaussian_polynomial(options_for().to_binding_dict(),float(sigma))
            q=int(result['modulus'])
            samples=[int(v) if int(v)<=q//2 else int(v)-q for v in result['residues']]
            self.assertTrue(all(abs(x)<=24*sigma for x in samples))
            self.assertTrue(any(x>2**64 for x in samples))
            self.assertTrue(any(x<-(2**64) for x in samples))
            self.assertLess(abs(sum(samples)/len(samples)/sigma),0.2)

    def test_full_sparse_rounds_with_representative_noise(self):
        for mode in ('FULL','SPARSE'):
            for compressed in (False,True):
                with self.subTest(mode=mode,compressed=compressed):
                    options=options_for(mode,smudging_sigma=float(2**79),aux_smudging_sigma=float(2**71),
                        use_sparse_compression=compressed,use_auxiliary_compression=compressed,
                        compression_c1=40,compression_ca=55,compression_c2=40,
                        aux_compression_c1=40,aux_compression_ca=55,aux_compression_c2=40)
                    engine=NativeCryptoEngine(options);runner=FederatedRoundOrchestrator(engine)
                    runner.setup(make_full_setup() if mode=='FULL' else make_setup())
                    self.assertEqual(engine.effective_protocol['setup_noise_sigmas'],[3.2]*4)
                    self.assertEqual(engine.effective_protocol['sampler_id'],'exact-rational-truncated-dg-v3')
                    candidates=make_full_candidates() if mode=='FULL' else make_candidates()
                    result=(runner.run_full_round(0,candidates) if mode=='FULL' else runner.run_round(0,candidates)).decrypted
                    start,end=(0,600) if mode=='FULL' else (256,512)
                    # Four truncated samples / (128 * 2^87), plus compression allowance.
                    np.testing.assert_allclose(result.parameter_delta[start:end],0.25,atol=0.0032,rtol=0)
                    self.assertTrue(np.all(np.isfinite(result.parameter_delta)))
                    for name in ('running_mean_delta','running_variance_delta'):
                        expected=sum(np.asarray(getattr(c,name),dtype=np.float64) for c in candidates)/4
                        np.testing.assert_allclose(getattr(result,name),expected,atol=0.001,rtol=0)

if __name__=='__main__':unittest.main()
