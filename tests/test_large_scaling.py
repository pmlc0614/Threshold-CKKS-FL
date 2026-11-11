"""Large scales checked against exact Python Fraction arithmetic and real RLWE rounds."""
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from plaintext_fl.native_engine import NativeCryptoEngine, native_binding_available
from plaintext_fl.wire_contract import canonical_parameter_descriptor, parse_public_setup
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator
from tests.test_native_engine import make_options, make_setup, make_full_setup, make_candidates, make_full_candidates


def nearest(value, scale):
    x=Fraction(float(value))*scale+Fraction(1,2)
    return x.numerator//x.denominator


def options_for(mode='SPARSE', **kwargs):
    values=dict(q_modulus_bits=(48,48), p_modulus_bits=(55,), maximum_qp_bits=160,
                model_scaling_factor=2**87, aux_scaling_factor=2**81,
                measure_pre_smudging_noise=True, protocol_mode=mode)
    if mode=='FULL': values.update(mlwe_rank=1,selection_ratio=1.0,selected_chunks_per_client=None)
    values.update(kwargs)
    return replace(make_options(), **values)


class ScaleBoundaryTests(unittest.TestCase):
    def test_reject_noninteger_nonpositive_and_oversized_scales(self):
        for scale in (True,1.5,0,-1,2**256,'123'):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                options_for(model_scaling_factor=scale).validate()
        options_for(model_scaling_factor=2**255+1).validate()


@unittest.skipUnless(native_binding_available(), 'native binding unavailable')
class NativeLargeScaleTests(unittest.TestCase):
    def test_exact_coefficients_and_negative_half_ties(self):
        import ciphertext_fl_cpp as native
        for scale in (1,3,2**64-1,2**81,2**87,2**87+1):
            values=[0.0,-0.0,0.5,-0.5,1.5,-1.5,0.1,-0.1,
                    2**-88,-2**-88,math.nextafter(2**-88,math.inf),
                    math.nextafter(-2**-88,-math.inf),2**-1074,-2**-1074]
            with self.subTest(scale=scale):
                result=native._coefficient_scale_vectors(options_for(model_scaling_factor=scale).to_binding_dict(),values)
                expected=[nearest(x,scale) for x in values];q=int(result['modulus'])
                self.assertEqual(list(map(int,result['rounded'])),expected)
                self.assertEqual(list(map(int,result['residues'])),[x%q for x in expected])
                for got,x in zip(result['decoded'],expected):
                    self.assertAlmostEqual(got,float(Fraction(x,scale)),places=14)

    def test_centered_range_and_nonfinite_rejected(self):
        import ciphertext_fl_cpp as native
        options=options_for(model_scaling_factor=2**100).to_binding_dict()
        for values in ([1.0],[-1.0],[math.inf],[math.nan]):
            with self.subTest(values=values),self.assertRaises((ValueError,RuntimeError,OverflowError)):
                native._coefficient_scale_vectors(options,values)

    def test_public_descriptor_keeps_adjacent_large_integer_scales_distinct(self):
        descriptors=[]
        for scale in (2**87,2**87+1):
            engine=NativeCryptoEngine(options_for(model_scaling_factor=scale));engine.setup(make_setup())
            effective=engine.effective_protocol
            self.assertEqual(effective['model_scaling_factor'],scale)
            self.assertEqual(effective['aux_scaling_factor'],2**81)
            descriptor=parse_public_setup(engine.public_setup_binding)['descriptor']
            self.assertEqual(descriptor,canonical_parameter_descriptor(effective))
            self.assertIn(str(scale).encode('ascii'),descriptor);descriptors.append(descriptor)
        self.assertNotEqual(*descriptors)

    def test_full_sparse_and_auxiliary_large_scale_rounds(self):
        for mode in ('FULL','SPARSE'):
            for compressed in (False,True):
                with self.subTest(mode=mode,compressed=compressed):
                    model_scale=2**87+(0 if compressed else 1)
                    aux_scale=2**81+(0 if compressed else 1)
                    options=options_for(mode,model_scaling_factor=model_scale,aux_scaling_factor=aux_scale,
                        use_sparse_compression=compressed,use_auxiliary_compression=compressed,
                        compression_c1=40,compression_ca=55,compression_c2=40,
                        aux_compression_c1=40,aux_compression_ca=55,aux_compression_c2=40)
                    engine=NativeCryptoEngine(options);runner=FederatedRoundOrchestrator(engine)
                    runner.setup(make_full_setup() if mode=='FULL' else make_setup())
                    candidates=make_full_candidates() if mode=='FULL' else make_candidates()
                    result=(runner.run_full_round(0,candidates) if mode=='FULL' else runner.run_round(0,candidates)).decrypted
                    audit=result.phase_audit
                    start,end=(0,600) if mode=='FULL' else (256,512)
                    vals=[c.parameter_delta[0] if mode=='FULL' else c.selected_values[0,0] for c in candidates]
                    expected=sum(32*nearest(value,model_scale) for value in vals)
                    self.assertGreater(abs(expected),2**64)
                    self.assertTrue(all(int(v)==expected for v in audit['model']['integer_references'][start:end]))
                    np.testing.assert_allclose(result.parameter_delta[start:end],0.25,atol=2e-4,rtol=0)
                    expected_aux=[sum(32*nearest(c.running_mean_delta[i],aux_scale) for c in candidates) for i in range(2)]
                    self.assertEqual(list(map(int,audit['aux']['integer_references'][:2])),expected_aux)

if __name__=='__main__':unittest.main()
