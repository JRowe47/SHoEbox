"""Recoverable three-parent multi-source retention experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.composition_train import source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'artifacts/multifact')
    p.add_argument('--steps',type=int,default=120);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.root=a.root.resolve();a.root.mkdir(parents=True,exist_ok=True)
    seeds=(20261220,20261221,20261222);parents=(20261130,20261131,20261132)
    parent_paths=[]
    for seed in parents:
        target=a.root/'parents'/f'joint_{seed}.pt'
        if not target.exists():
            original=ROOT/'artifacts/joint_retained'/f'joint_{seed}'/'best.pt'
            if not original.exists():original=ROOT/'artifacts/multifact/parents'/f'joint_{seed}.pt'
            target.parent.mkdir(exist_ok=True);shutil.copyfile(original,target)
        parent_paths.append(target)
    protocol=dict(schema='shoebox-multifact-protocol-v1',seeds=list(seeds),parent_seeds=list(parents),steps=a.steps,
        batch=a.batch,validation_count=64,validate_every=20,test_count=512,primary_gap=16,long_gap=64,
        learning_rates=dict(producer=.0003,edge=.001),gaps=[0,4,8,16],segment=16,
        primary_target='Every seed >=90% on each field for recall, either-source shared correction, either-private correction, and withheld corrections',
        source_target='Each source alone >=90% on common and its own private field; unsupported private <=35% on independent factors',
        live_target='After three field queries, a pure update suffix yields >=90% on each field; no initial-fact re-observation',
        controls='Exact owner preservation and exports; blocked/reset <=35%; K2 exceeds K1 by >=5 points; matched expert counts',
        selection='Validation-only lexicographic minimum field/task accuracy, mean accuracy, lower NLL; earliest exact tie',
        parent_sha256=[hashlib.sha256(x.read_bytes()).hexdigest() for x in parent_paths],
        protocol_document_sha256=hashlib.sha256((ROOT/'docs/SHoEbox_Multifact_Protocol.md').read_bytes()).hexdigest())
    lockpath=a.root/'protocol_lock.json'
    if lockpath.exists():assert json.loads(lockpath.read_text())==protocol,'Locked protocol changed.'
    else:lockpath.write_text(json.dumps(protocol,indent=2)+'\n')
    (a.root/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
        threads_per_process=1,workers=a.workers),indent=2)+'\n')
    statuspath=a.root/'run_status.json';guard=threading.Lock()
    status=json.loads(statuspath.read_text()) if statuspath.exists() else dict(state='running',runs={})
    def record(seed,phase,value):
        with guard:
            status['state']='running';status['runs'].setdefault(str(seed),{})[phase]=value
            temporary=statuspath.with_suffix('.tmp');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(statuspath)
    def worker(seed,parent):
        folder=a.root/str(seed);folder.mkdir(exist_ok=True)
        if (folder/'protocol.json').exists():
            recorded=json.loads((folder/'protocol.json').read_text())
            assert recorded['source_hashes']==source_hashes(),'Recorded production source changed.'
            assert recorded['parent_sha256']==hashlib.sha256(parent.read_bytes()).hexdigest(),'Parent changed.'
        train=[sys.executable,'-m','experiments.text_regions.multifact_train','--parent',str(parent),'--out',str(folder),
            '--seed',str(seed),'--steps',str(a.steps),'--batch',str(a.batch),'--validation-count','64','--validate-every','20']
        phases=(('train',train),('evaluate',[sys.executable,'scripts/evaluate_multifact.py',str(folder)]),
                ('audit',[sys.executable,'scripts/audit_multifact.py',str(folder),'--resume']))
        for phase,command in phases:
            if status['runs'].get(str(seed),{}).get(phase,{}).get('complete'):continue
            record(seed,phase,dict(running=True,started=time.time()))
            with (folder/(phase+'.log')).open('a') as log:
                returncode=0
                if phase=='train':
                    # Short resumable invocations bound allocator growth and
                    # survive execution-worker lifetimes without altering the
                    # optimizer horizon, data cursor, or validation schedule.
                    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True) if (folder/'last.pt').exists() else None
                    step=saved['step'] if saved is not None else 0;del saved
                    while step<a.steps:
                        stop=min(a.steps,((step//20)+1)*20)
                        invocation=command+(['--resume'] if (folder/'last.pt').exists() else [])+['--stop-after',str(stop)]
                        result=subprocess.run(invocation,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        returncode=result.returncode
                        if returncode:break
                        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True);step=saved['step'];del saved
                        assert step==stop
                        record(seed,phase,dict(running=True,checkpoint=step,updated=time.time()))
                        print(json.dumps(dict(seed=seed,checkpoint=step)),flush=True)
                else:returncode=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            record(seed,phase,dict(complete=returncode==0,returncode=returncode,finished=time.time()))
            if returncode:raise RuntimeError(f'{seed} {phase} failed; inspect {folder/(phase+".log")}')
            print(json.dumps(dict(seed=seed,phase=phase,complete=True)),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,s,p) for s,p in zip(seeds,parent_paths)]
        for f in futures:f.result()
    status['state']='complete';statuspath.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(dict(complete=True,root=str(a.root))),flush=True)


if __name__=='__main__':main()
