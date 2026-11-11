"""Synthetic CIFAR-10-sized FullSD runtime probe; pass a result JSON path.
Setup and the independent correctness assertion are outside the timed round.
"""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import replace
from plaintext_fl.crypto_parameter_profile import load_crypto_parameter_profile
from plaintext_fl.native_engine import NativeCryptoEngine
from plaintext_fl.crypto_engine import CryptoEngineSetup
from plaintext_fl.protocol_contract import RegisteredClientWeight
from plaintext_fl.model import resnet20
from plaintext_fl.model_adapter import build_model_protocol_layouts
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator,ClientFullRoundCandidate
root=Path(__file__).resolve().parents[1]
p=load_crypto_parameter_profile(root/'plaintext_fl/profiles/resnet20-cifar10-full-shared-v17.json')
l=build_model_protocol_layouts(resnet20())
e=NativeCryptoEngine(replace(p.native_options(master_seed=bytes(32)),crypto_threads=4,collusion_bound=1))
r=FederatedRoundOrchestrator(e)
r.setup(CryptoEngineSetup(setup_id=bytes(16),protocol_mode='FULL',parameter_layout=l.parameter,running_mean_layout=l.running_mean,running_variance_layout=l.running_variance,clients=tuple(RegisteredClientWeight(i,w) for i,w in enumerate([15,11,23,19,15,20,11,14])),chunk_size=8192,minimum_cohort_size=3))
x=(np.sin(np.arange(272474)*.031)*.1).astype(np.float32)
a=np.full(784,.1,dtype=np.float32)
c=tuple(ClientFullRoundCandidate(client_id=i,parameter_delta=x.copy(),running_mean_delta=a.copy(),running_variance_delta=a.copy()) for i in range(8))
start=time.perf_counter();out=r.run_full_round(0,c);elapsed=time.perf_counter()-start
np.testing.assert_allclose(out.decrypted.parameter_delta,x,atol=.003,rtol=0)
d=e.last_diagnostics
result=dict(wall_seconds=elapsed,**{k:d[k] for k in ['crypto_sequential_seconds','crypto_server_seconds','estimated_parallel_crypto_seconds','maximum_client_pre_aggregation_seconds','maximum_client_partial_decryption_seconds']})
Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
