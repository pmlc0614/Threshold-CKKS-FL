"""Verify saved preflight outputs (not a security or accuracy certificate)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plaintext_fl.encrypted_training import _build_encrypted_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path, required=True)
parser.add_argument("--dropout", action="store_true")
args = parser.parse_args()
root = args.work_dir.resolve()
manifest = json.loads((Path(__file__).resolve().parents[1] / 'interop/SHARED_V17_PARAMETERS.json').read_text())
results = {}
for mode in ('full', 'sp'):
    folder = root / mode
    summary = json.loads((folder / 'summary.json').read_text())
    rows = [json.loads((folder / 'rounds' / f'round_{i:04d}.json').read_text()) for i in (1, 2)]
    assert summary['completed_rounds'] == 2
    assert summary['aborted_attempts'] == []
    assert rows[1]['round_id'] == rows[0]['round_id'] + 1
    assert rows[0]['setup_id'] == rows[1]['setup_id']
    assert not rows[0]['setup_reused'] and rows[1]['setup_reused']
    for row in rows:
        assert bool(row['experimental_dynamic_cohort']) == args.dropout
        if args.dropout:
            assert row['native_effective_protocol']['dropout_attempt_cap'] == 1
            assert row['native_effective_protocol']['dropout_target_cap'] == 8
        assert row['integer_alphas'] == manifest['integer_weights']
        assert row['client_sample_counts'] == [2*w for w in manifest['integer_weights']]
        assert row['native_effective_protocol']['q_primes'] == manifest['q_primes']
        assert row['native_effective_protocol']['p_primes'] == manifest['p_primes']
        for key in ('weighted_train_loss', 'test_loss', 'test_accuracy_percent'):
            assert np.isfinite(row[key])
    if args.dropout:
        assert rows[0]['dropout']['survivor_client_ids'] == list(range(1,8))
        assert rows[1]['dropout']['survivor_client_ids'] == list(range(8))
        assert rows[0]['dropout']['dropped_client_ids'] == [0]
        assert not rows[1]['dropout']['applied']
    bounds = json.loads((folder / 'local-input-bounds.json').read_text())
    assert len(bounds) == (48 if mode == 'full' else 16)
    assert all(np.isfinite(r['maximum']) and r['maximum'] <= r['bound'] for r in bounds)
    aux_maxima = [client[key] for row in rows for client in row['clients_detail']
                  for key in ('running_mean_max_abs', 'running_variance_max_abs')]
    assert len(aux_maxima) == 32
    assert all(np.isfinite(value) and value <= 73.02073288 for value in aux_maxima)
    models = [torch.load(folder / 'checkpoints' / f'model_round_{i:04d}.pt', weights_only=False) for i in (1, 2)]
    for checkpoint in models:
        _build_encrypted_model('resnet32').load_state_dict(checkpoint['model_state_dict'], strict=True)
        assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    assert any(not torch.equal(v, models[1]['model_state_dict'][k]) for k, v in models[0]['model_state_dict'].items())
    final = torch.load(folder / 'final_model.pt', weights_only=False)
    assert all(torch.equal(v, final['model_state_dict'][k]) for k,v in models[1]['model_state_dict'].items())
    record = {
        'dropout_then_rejoin_verified': args.dropout, 'completed_rounds': 2, 'round_ids': [r['round_id'] for r in rows],
        'setup_reused_second_round': True, 'checkpoint_reload_passed': True,
        'model_changed_between_rounds': True, 'final_matches_last_checkpoint': True,
        'local_input_check_count': len(bounds),
        'maximum_model_input': max(r['maximum'] for r in bounds if r['bound'] < 2),
        'aux_input_check_count': len(aux_maxima),
        'maximum_aux_input': max(aux_maxima),
        'train_losses': [r['weighted_train_loss'] for r in rows],
        'test_losses': [r['test_loss'] for r in rows],
        'test_accuracy_percent': [r['test_accuracy_percent'] for r in rows],
        'accepted_chunk_counts': [r['accepted_chunk_count'] for r in rows],
        'total_seconds': summary['total_seconds'],
        'final_checkpoint_sha256': hashlib.sha256((folder / 'final_model.pt').read_bytes()).hexdigest(),
    }
    if mode == 'sp':
        inputs = json.loads((folder / 'sparse-input-with-residual-bounds.json').read_text())
        assert len(inputs) == 16
        assert all(np.isfinite(r['maximum']) and r['maximum'] <= r['bound'] for r in inputs)
        states = [torch.load(folder / 'checkpoints' / f'error_feedback_round_{i:04d}.pt', weights_only=False) for i in (1, 2)]
        for state, row in zip(states, rows):
            assert state['setup_id'].hex() == row['setup_id']
            assert state['round_id'] == row['round_id']
            assert len(state['residuals']) == 8
            assert all(v.numel() == 472756 and torch.isfinite(v).all() for v in state['residuals'])
        if args.dropout:
            assert torch.count_nonzero(states[0]['residuals'][0]) == 0
            assert torch.count_nonzero(states[1]['residuals'][0]) > 0
        assert any(not torch.equal(a,b) for a,b in zip(states[0]['residuals'], states[1]['residuals']))
        last = torch.load(folder / 'error_feedback_state.pt', weights_only=False)
        assert all(torch.equal(a,b) for a,b in zip(last['residuals'],states[1]['residuals']))
        record.update(residual_checkpoint_reload_passed=True, residual_changed_between_rounds=True,
                      maximum_input_with_residual=max(r['maximum'] for r in inputs))
    results[mode] = record
report = {'scope': 'Two-round real CIFAR-100 subset smoke test; not an accuracy, performance, or security certificate.',
          'dataset': {'train':256,'test':128,'clients':8,'partition':'label-sorted fixed counts, not Dirichlet'},
          'environment': {'python':sys.version.split()[0], 'torch':torch.__version__,'numpy':np.__version__, 'pybind11_build':'2.13.6', 'device':'cpu', 'crypto_threads':4},
          'results': results}
(root / 'verified-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
