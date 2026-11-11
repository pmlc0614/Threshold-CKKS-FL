"""Small real-CIFAR preflight through the production CLI, with explicit subset injection."""
import sys,json,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('full','sp'))
parser.add_argument('--work-dir', type=Path, required=True,
                    help='Directory containing extracted data/cifar-100-python; outputs are written here')
parser.add_argument("--dropout", action="store_true", help="Drop one client in round 1, then rejoin in round 2")
options=parser.parse_args()
ROOT=options.work_dir.resolve()
ROOT.mkdir(parents=True,exist_ok=True)
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from plaintext_fl import encrypted_training as cli
from plaintext_fl import data

mode=options.mode
assert mode in ('full','sp')
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
weights=[13,15,17,16,14,19,18,16]
manifest=json.loads((REPO/'interop/SHARED_V17_PARAMETERS.json').read_text())
original_loader=cli._load_encrypted_dataset
partition_path=ROOT/'partition.json'
output=ROOT/mode
output.mkdir(exist_ok=True)

def small_dataset(**kwargs):
    train,test=original_loader(**kwargs)
    rng=np.random.default_rng(42)
    train_ids=rng.choice(len(train),256,replace=False)
    test_ids=rng.choice(len(test),128,replace=False)
    train.data=train.data[train_ids];train.targets=[train.targets[int(i)] for i in train_ids]
    test.data=test.data[test_ids];test.targets=[test.targets[int(i)] for i in test_ids]
    # Label-skewed fixed-count split; not the default unconstrained Dirichlet split.
    order=np.argsort(train.targets,kind='stable')
    partitions=list(np.split(order,np.cumsum(np.array(weights)*2)[:-1]))
    doc=data._partition_file_document(dataset_name='cifar100',targets=np.asarray(train.targets),partitions=partitions,partition='noniid',dirichlet_alpha=1.0,partition_seed=42,minimum_client_samples=1)
    doc['preflight_partition_method']='label-sorted fixed counts, not a Dirichlet draw; alpha field is CLI compatibility metadata'
    partition_path.write_text(json.dumps(doc,indent=2))
    selection={'train_original_indices':train_ids.tolist(),'test_original_indices':test_ids.tolist(),'train_count':256,'test_count':128,'counts':[2*w for w in weights],'partition_method':doc['preflight_partition_method'],'scope':'Small real-data smoke test, not accuracy evaluation.'}
    (ROOT/'dataset-subset.json').write_text(json.dumps(selection,indent=2))
    return train,test
cli._load_encrypted_dataset=small_dataset

# Check Full model/aux deltas, Sparse model deltas and complete update+residual
# before encryption. Sparse aux maxima are verified from saved client metrics.
# No clipping or value replacement is performed.
input_records=[]
original_delta=cli.compute_layout_delta

def checked_delta(local,global_model,layout):
    value=original_delta(local,global_model,layout)
    maximum=float(np.max(np.abs(value)))
    model=layout.total_numel==472756
    bound=1.472850219 if model else 73.02073288
    record={'kind':str(layout.kind),'maximum':maximum,'bound':bound}
    input_records.append(record)
    (output/'local-input-bounds.json').write_text(json.dumps(input_records,indent=2))
    if maximum>bound:raise RuntimeError(f'Local input bound exceeded: {record}')
    return value
cli.compute_layout_delta=checked_delta
original_compress=cli.compress_with_error_feedback
residual_records=[]
def checked_compress(**kwargs):
    u=kwargs['update'].clone()
    previous=kwargs.get('previous_residual')
    if previous is not None:u+=kwargs.get('residual_beta',1.0)*previous
    maximum=float(u.abs().max())
    residual_records.append({'maximum':maximum,'bound':1.472850219})
    (output/'sparse-input-with-residual-bounds.json').write_text(json.dumps(residual_records,indent=2))
    if maximum>1.472850219:raise RuntimeError('Sparse complete update+residual exceeds input bound')
    return original_compress(**kwargs)
cli.compress_with_error_feedback=checked_compress
original_setup=cli.NativeCryptoEngine.setup
def checked_setup(self,setup):
    result=original_setup(self,setup)
    effective=self.effective_protocol
    assert effective['q_primes']==manifest['q_primes']
    assert effective['p_primes']==manifest['p_primes']
    assert list(self.setup_config.alphas)==weights
    (output/'effective-setup.json').write_text(json.dumps(effective,indent=2))
    return result
cli.NativeCryptoEngine.setup=checked_setup
sys.argv=['encrypted-sparse','--dataset','cifar100','--model','resnet32','--data-dir',str(ROOT/'data'),'--output-dir',str(output),'--crypto-parameter-profile',str(REPO/f'plaintext_fl/profiles/resnet32-cifar100-{mode}-shared-v17.json'),'--rounds','2','--local-epochs','1','--batch-size','16','--eval-batch-size','64','--split','noniid','--alpha','1','--partition-file',str(partition_path),'--learning-rate','0.1','--schedule','constant','--device','cpu','--crypto-threads','4','--num-workers','0','--no-download']
if options.dropout:
    from plaintext_fl.dropout import DropoutScheduleEntry
    # Deterministic first-round-only dropout; validate actual production CLI.
    original_schedule = cli.build_dropout_schedule
    def first_round_dropout(**kwargs):
        if kwargs['dropout_round_count'] == 0:
            return original_schedule(**kwargs)
        return (DropoutScheduleEntry(round_id=0, dropped_client_id=0,
                                     survivor_client_ids=tuple(range(1,8))),)
    cli.build_dropout_schedule = first_round_dropout
    sys.argv += ['--experimental-dynamic-cohort', '--dropout-round-count', '1',
                 '--dropout-target-cap', '8', '--collusion-bound', '1']
(ROOT/f'{mode}-command.json').write_text(json.dumps(sys.argv,indent=2))
cli.main()
