"""Run three resumable parent/data pairs through training, evaluation and audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.composition_train import source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/joint_text')
    p.add_argument('--parents',type=Path,default=ROOT/'artifacts/multifact')
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--steps',type=int,default=160);p.add_argument('--memory-weight',type=float,default=1.)
    p.add_argument('--core-lr',type=float,default=.0003);p.add_argument('--readout-lr',type=float,default=.002)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--stage',choices=('all','train','evaluate','audit'),default='all')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    runs=[dict(parent_seed=20261220+i,seed=20261310+i) for i in range(3)]
    protocol=dict(runs=runs,steps=a.steps,batch=8,context=16,memory_batch=16,
        core_lr=a.core_lr,readout_lr=a.readout_lr,producer_lr=.0001,memory_weight=a.memory_weight,
        validation_count=64,validation_chunks=8,validate_every=20,test_count=512,test_text_bytes=4096,
        text_target='Held-out NLL below 4 nats/byte and at least 50% below the unmodified parent',
        memory_target='Every primary recall/correction/withheld field >= 90% on all three parent/data pairs',
        recurrence_target='Two rounds improve the shared-right composition mean by >= 5 percentage points over one round',
        state_target='Exact memory ownership during text and exact resumable byte generation',
        source_hashes=source_hashes(),data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        parents={str(r['parent_seed']):hashlib.sha256((a.parents/str(r['parent_seed'])/'best.pt').read_bytes()).hexdigest() for r in runs})
    lock=a.root/'protocol_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())!=protocol:raise ValueError('The locked protocol differs.')
    else:lock.write_text(json.dumps(protocol,indent=2)+'\n')
    (a.root/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=str(torch.__version__),
        platform=platform.platform(),threads_per_process=1,workers=a.workers),indent=2)+'\n')
    status=dict(state='running',stage=a.stage,runs={str(r['seed']):'pending' for r in runs});mutex=threading.Lock()
    def mark(seed,value):
        with mutex:
            status['runs'][str(seed)]=value
            temporary=a.root/'run_status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n')
            temporary.replace(a.root/'run_status.json')
    def worker(run):
        seed=run['seed'];out=a.root/str(seed);out.mkdir(parents=True,exist_ok=True)
        commands=[]
        if a.stage in ('all','train'):
            last=out/'last.pt';complete=last.exists() and torch.load(last,map_location='cpu',weights_only=True)['step']==a.steps
            if not complete:
                command=[sys.executable,'-m','experiments.text_regions.joint_text','--parent',str(a.parents/str(run['parent_seed'])/'best.pt'),
                    '--data',str(a.data),'--out',str(out),'--seed',str(seed),'--steps',str(a.steps),'--batch','8','--memory-batch','16',
                    '--core-lr',str(a.core_lr),'--readout-lr',str(a.readout_lr),'--memory-weight',str(a.memory_weight)]
                if last.exists():command.append('--resume')
                commands.append(('train',command))
        if a.stage in ('all','evaluate'):
            commands.append(('evaluate',[sys.executable,'scripts/evaluate_joint_text.py',str(out),'--data',str(a.data)]))
        if a.stage in ('all','audit'):
            commands.append(('audit',[sys.executable,'scripts/audit_joint_text.py',str(out),'--data',str(a.data)]))
        for name,command in commands:
            mark(seed,name)
            with (out/(name+'.log')).open('a') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        mark(seed,'complete')
        return seed
    failed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,r):r['seed'] for r in runs}
        for future in as_completed(futures):
            try:print(json.dumps(dict(seed=future.result(),complete=True)),flush=True)
            except Exception as error:
                seed=futures[future];failed.append(seed);mark(seed,'failed');print(json.dumps(dict(seed=seed,error=str(error))),flush=True)
    status['state']='failed' if failed else 'complete'
    (a.root/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
