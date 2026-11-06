"""Run three matched continuation pairs; persist the protocol before training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.composition_train import source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/joint_stability/study')
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--stage',choices=('train','evaluate','audit'),default='train')
    a=p.parse_args()
    runs=[dict(parent_seed=20261310+i,seed=20261400+i) for i in range(3)]
    protocol=dict(runs=runs,steps=a.steps,modes=['sum','group_clip'],batch=8,context=16,memory_batch=16,
        validate_every=20,validation_count=512,validation_chunks=8,test_count=512,test_chunks=32,
        primary_conditions=9,core_lr=.0003,readout_lr=.002,producer_lr=.0001,gradient_cap=1.,
        source_hashes=source_hashes(),data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        parents={str(r['parent_seed']):hashlib.sha256((a.parents/str(r['parent_seed'])/'last.pt').read_bytes()).hexdigest() for r in runs},
        endpoints=dict(text='Test NLL < 4 nats/byte',memory='All 27 recall/correction/withheld fields >= .90',
            recurrence='K2 mean shared-right composition exceeds K1 by >= .05',
            state='Exact retained ownership and serialized generation continuation'),
        comparison='Each pair shares starting weights, optimizer, carried text state/cursor and fresh memory episodes. Only task/group clipping differs.',
        prior_exposure='Parent test data were seen in the previous phase. New phase text test uses the second half of the reserved test split; synthetic seed namespaces are new.',
        selection='Validation only; guard on all 27 fields, then lowest text NLL. Preserve fixed-budget final checkpoint separately.')
    a.root.mkdir(parents=True,exist_ok=True)
    lock=a.root/'protocol_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())!=protocol:raise ValueError('Locked protocol changed.')
    else:lock.write_text(json.dumps(protocol,indent=2)+'\n')

    def worker(run,mode):
        out=a.root/str(run['seed'])/mode
        out.mkdir(parents=True,exist_ok=True)
        if a.stage=='train':
            command=[sys.executable,'-m','experiments.text_regions.joint_stability',
                '--parent',str(a.parents/str(run['parent_seed'])/'last.pt'), '--data',str(a.data),
                '--out',str(out),'--seed',str(run['seed']),'--mode',mode,'--steps',str(a.steps)]
            if (out/'last.pt').exists():
                if torch.load(out/'last.pt',weights_only=True,map_location='cpu')['step']==a.steps:return
                command.append('--resume')
        else:
            command=[sys.executable,'scripts/'+a.stage+'_joint_stability.py',str(out),'--data',str(a.data)]
            if a.stage=='audit':
                command.extend(['--parents',str(a.parents)])
        with (out/(a.stage+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(seed=run['seed'],mode=mode,stage=a.stage,complete=True)),flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,r,mode):(r['seed'],mode) for r in runs for mode in ('group_clip','sum')}
        for f in as_completed(futures):
            try:f.result()
            except Exception as error:failures.append(dict(run=futures[f],error=str(error)))
    (a.root/(a.stage+'_status.json')).write_text(json.dumps(dict(complete=not failures,failures=failures),indent=2)+'\n')
    if failures:raise SystemExit(str(failures))


if __name__=='__main__':main()
