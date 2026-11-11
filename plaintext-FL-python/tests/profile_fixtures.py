"""Small arithmetic fixtures; not deployment profiles."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from plaintext_fl.crypto_parameter_profile import load_crypto_parameter_profile

def communication_profile():
    document = {'schema': 'fl-v17-crypto-parameter-profile-v3', 'name': 'communication-test-fixture', 'protocol_version': 'v17', 'model_name': 'resnet20-cifar10', 'experiment': {'mode': 'SPARSE', 'clients': 8, 'minimum_cohort_size': 3, 'selection_ratio': 0.3, 'safety_factor': 1.0, 'quantization_enabled': True}, 'dimensions': {'parameter_numel': 272474, 'running_mean_numel': 784, 'running_variance_numel': 784, 'auxiliary_numel': 1568, 'parameter_layout_sha256': 'f84d49b0d411463939494d4646f03f72fd1a8174f7193469952bed2aef483d1b', 'running_mean_layout_sha256': 'ac5043f25309e52246c8877b31d35ce9e84194693287b1b802374995140b7432', 'running_variance_layout_sha256': 'f9680e6e83481731d095e4b3972283440736bbee46c5bf430f127e3cde0bd5ac'}, 'ring': {'ring_dimension': 8192, 'mlwe_rank': 4, 'q_modulus_bits': [30, 30], 'p_modulus_bits': [60], 'hybrid_large_digits': 2, 'maximum_qp_bits': 218}, 'scaling': {'model_factor': 2251799813685248, 'aux_factor': 70368744177664}, 'precision': {'parameter_bits': 14, 'auxiliary_bits': 14}, 'aggregate_bounds': {'parameter': 0.494, 'auxiliary': 17.21}, 'compression': {'c1': 22, 'ca': 40, 'c2': 18, 'cd': 15, 'aux_c1': 22, 'aux_ca': 40, 'aux_c2': 18, 'aux_cd': 15}, 'smudging': {'parameter_sigma': 549755813888, 'auxiliary_sigma': 549755813888}, 'delivery': {'parameter_scale': 32768, 'auxiliary_scale': 900}}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return load_crypto_parameter_profile(path)
