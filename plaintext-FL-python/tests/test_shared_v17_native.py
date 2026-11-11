"""Opt-in native integration at v17 dimensions; synthetic updates, not FL training."""
import json
import os
from pathlib import Path
import time
import unittest
import numpy as np
from plaintext_fl.crypto_parameter_profile import load_crypto_parameter_profile
from plaintext_fl.native_engine import NativeCryptoEngine, native_binding_available
from plaintext_fl.crypto_engine import CryptoEngineSetup
from plaintext_fl.protocol_contract import LayoutKind, RegisteredClientWeight, RoundPhase
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator, ClientFullRoundCandidate, ClientRoundCandidate
from tests.test_native_engine import make_layout

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'interop/SHARED_V17_PARAMETERS.json').read_text())

@unittest.skipUnless(os.environ.get('FL_V17_NATIVE_TEST') == '1' and native_binding_available(),
                     'set FL_V17_NATIVE_TEST=1 with ABI 22 native binding')
class SharedV17NativeTest(unittest.TestCase):
    def test_representative_full_and_sparse_round(self):
        records = []
        for mode in ('full', 'sp'):
            with self.subTest(mode=mode):
                start = time.monotonic()
                p = load_crypto_parameter_profile(ROOT / f'plaintext_fl/profiles/resnet32-cifar100-{mode}-shared-v17.json')
                e = NativeCryptoEngine(p.native_options(master_seed=bytes(32)))
                runner = FederatedRoundOrchestrator(e)
                setup = CryptoEngineSetup(setup_id=bytes(16), protocol_mode=p.mode,
                    parameter_layout=make_layout(LayoutKind.PARAMETER, 'weight', p.parameter_numel),
                    running_mean_layout=make_layout(LayoutKind.RUNNING_MEAN, 'bn.running_mean', p.running_mean_numel),
                    running_variance_layout=make_layout(LayoutKind.RUNNING_VARIANCE, 'bn.running_var', p.running_variance_numel),
                    clients=tuple(RegisteredClientWeight(client_id=i, example_count=w) for i,w in enumerate(MANIFEST['integer_weights'])),
                    chunk_size=p.chunk_size, minimum_cohort_size=p.minimum_cohort_size)
                runner.setup(setup)
                self.assertEqual(e.effective_protocol['q_primes'], MANIFEST['q_primes'])
                self.assertEqual(e.effective_protocol['p_primes'], MANIFEST['p_primes'])
                candidates=[]; model_ref=np.zeros(p.parameter_numel);aux_ref=np.zeros(p.running_mean_numel)
                selected=tuple(range(p.selected_chunk_count()))
                active=len(selected)*p.chunk_size if mode=='sp' else p.parameter_numel
                for i,w in enumerate(MANIFEST['integer_weights']):
                    x=(.3*np.sin(np.arange(active)*.031)+(i-3.5)*.02).astype(np.float32)
                    a=(20*np.cos(np.arange(p.running_mean_numel)*.07)+i*.1).astype(np.float32)
                    model_ref[:active]+=w*x/128.;aux_ref+=w*a/128.
                    args=dict(client_id=i,running_mean_delta=a,running_variance_delta=-a)
                    candidates.append(ClientFullRoundCandidate(parameter_delta=x,**args) if mode=='full' else
                        ClientRoundCandidate(selected_actual_chunk_ids=selected,selected_values=x.reshape(len(selected),p.chunk_size),**args))
                r=(runner.run_full_round(0,tuple(candidates)) if mode=='full' else runner.run_round(0,tuple(candidates)))
                self.assertEqual(e.phase,RoundPhase.COMMITTED)
                model_rmse=float(np.sqrt(np.mean((r.decrypted.parameter_delta[:active]-model_ref[:active])**2)))
                aux_rmse=float(np.sqrt(np.mean(np.concatenate((r.decrypted.running_mean_delta-aux_ref,r.decrypted.running_variance_delta+aux_ref))**2)))
                self.assertTrue(np.isfinite(model_rmse) and np.isfinite(aux_rmse))
                # Finite-sample integration tolerance, not a statistical certificate.
                self.assertLess(model_rmse,2*p.aggregate_update_bound/2**p.precision_bits)
                self.assertLess(aux_rmse,2*p.aggregate_aux_update_bound/2**p.aux_precision_bits)
                if mode=='sp':
                    self.assertEqual(r.admission.accepted_actual_chunk_ids,selected)
                    np.testing.assert_array_equal(r.decrypted.parameter_delta[active:],0)
                record=dict(mode=p.mode,model_rmse=model_rmse,aux_rmse=aux_rmse,seconds=time.monotonic()-start,
                            q_primes=e.effective_protocol['q_primes'],p_primes=e.effective_protocol['p_primes'])
                records.append(record);print(json.dumps(record),flush=True)
        if os.environ.get('FL_V17_RESULT_PATH'):
            Path(os.environ['FL_V17_RESULT_PATH']).write_text(json.dumps(records,indent=2)+'\n')
