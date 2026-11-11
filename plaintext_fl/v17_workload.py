"""Capture real CIFAR client training updates for native v17 validation replay."""
import argparse
import copy
import hashlib
import io
import tarfile
import numpy as np
from PIL import Image
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset,Dataset
from .data import load_cifar10,load_cifar100,partition_indices,build_cifar10_transforms
from .model import resnet20,resnet32
from .model_adapter import build_model_protocol_layouts,compute_layout_delta


def digest(data):return hashlib.sha256(data).hexdigest()


class CifarImageArchive(Dataset):
    """Read the public FastAI PNG mirror without extracting archive paths."""
    def __init__(self,path):
        self.transform=build_cifar10_transforms()[0]
        with tarfile.open(path,'r:gz') as archive:
            members=[m for m in archive.getmembers() if m.isfile() and '/train/' in m.name and m.name.endswith('.png')]
            classes=sorted({Path(m.name).parent.name for m in members})
            if len(members)!=50000 or len(classes)!=10:raise ValueError('expected full CIFAR-10 training mirror')
            self.targets=[classes.index(Path(m.name).parent.name) for m in members]
            images=[]
            for member in members:
                image=np.asarray(Image.open(io.BytesIO(archive.extractfile(member).read())).convert('RGB'))
                if image.shape!=(32,32,3):raise ValueError('CIFAR image shape mismatch')
                images.append(image)
            self.data=np.stack(images)
    def __len__(self):return len(self.targets)
    def __getitem__(self,index):return self.transform(Image.fromarray(self.data[index])),self.targets[index]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cifar-image-archive',type=Path)
    parser.add_argument('--data-mode',choices=['IID','nonIID'],required=True)
    parser.add_argument('--dataset',choices=['cifar10','cifar100'],default='cifar10')
    parser.add_argument('--model',choices=['resnet20','resnet32'],default='resnet20')
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--learning-rate',type=float,default=.01)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<1 or args.batch_size<1:parser.error('positive steps/batch-size required')
    cfg=json.loads(args.candidate.read_text(encoding='utf-8'))
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    if args.cifar_image_archive:
        if args.dataset!='cifar10':raise ValueError('PNG archive option supports CIFAR-10 only')
        train=CifarImageArchive(args.cifar_image_archive)
    else:
        train,_=(load_cifar10 if args.dataset=='cifar10' else load_cifar100)(args.data_root)
    partitions=partition_indices(train.targets,cfg['experiment']['clients'],
        partition='iid' if args.data_mode=='IID' else 'noniid',seed=args.seed)
    global_model=(resnet20 if args.model=='resnet20' else resnet32)(num_classes=10 if args.dataset=='cifar10' else 100)
    layouts=build_model_protocol_layouts(global_model,model_name=args.model)
    if layouts.parameter.total_numel!=cfg['experiment']['model_dimension'] or layouts.running_mean.total_numel+layouts.running_variance.total_numel!=cfg['experiment']['aux_dimension']:
        raise ValueError('model dimensions differ from candidate')
    updates={key:[] for key in ('model','running_mean','running_variance')}
    used=[]
    for client,indices in enumerate(partitions):
        torch.manual_seed(args.seed+client+1)
        local=copy.deepcopy(global_model);local.train()
        opt=torch.optim.SGD(local.parameters(),lr=args.learning_rate)
        loader=DataLoader(Subset(train,indices.tolist()),batch_size=args.batch_size,shuffle=False,num_workers=0)
        consumed=0
        for step,(images,labels) in enumerate(loader):
            opt.zero_grad();torch.nn.functional.cross_entropy(local(images),labels).backward();opt.step()
            consumed+=len(labels)
            if step+1>=args.steps:break
        if step+1<args.steps:raise ValueError('client partition too small for requested steps')
        used.append(indices[:consumed].tolist())
        for key,layout in [('model',layouts.parameter),('running_mean',layouts.running_mean),('running_variance',layouts.running_variance)]:
            values=compute_layout_delta(local,global_model,layout)
            # Capture actual updates without silently clipping or rescaling.
            bound=cfg['precision']['u_max' if key=='model' else 'u_max_aux']
            if abs(values).max()>bound:raise ValueError('observed update exceeds candidate bound; candidate/workload must be revised')
            updates[key].append(values.astype(float).tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(updates,allow_nan=False),encoding='utf-8')
    partition_doc=dict(partitions=[x.tolist() for x in partitions],used_indices=used)
    partition_path=args.output.with_suffix('.partition.json')
    partition_path.write_text(json.dumps(partition_doc,sort_keys=True),encoding='utf-8')
    dataset_hash=hashlib.sha256(train.data.tobytes())
    dataset_hash.update(json.dumps(train.targets,separators=(',',':')).encode())
    manifest=dict(description=f'{args.dataset}/{args.model}: {args.steps} local SGD steps; frozen actual client updates, independent crypto replays',
        candidate_sha256=digest(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()),
        data_mode=args.data_mode,dataset_sha256=dataset_hash.hexdigest(),partition_sha256=digest(partition_path.read_bytes()),
        runner_sha256=digest(Path(__file__).read_bytes()),updates_sha256=digest(args.output.read_bytes()),
        partition_file=partition_path.name,updates_file=args.output.name,seed=args.seed,
        local_steps=args.steps,batch_size=args.batch_size,learning_rate=args.learning_rate,
        dataset=args.dataset,model=args.model,
        dataset_format='fastai-png-archive' if args.cifar_image_archive else 'torchvision-cifar')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
