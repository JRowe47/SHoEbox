"""Recoverable matched runs, with a lock created before training/test access."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(20261130,20261131,20261132)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/joint_retained')
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();out=a.root.resolve();out.mkdir(parents=True,exist_ok=True)
    parents=out/'parents';parents.mkdir(exist_ok=True)
    for seed in SEEDS:
        source=ROOT/f'artifacts/retained_composition/{seed-10}/best.pt';target=parents/f'{seed}.pt'
        if not source.exists():source=ROOT/f'artifacts/joint_retained/parents/{seed}.pt'
        if not target.exists():shutil.copyfile(source,target)
        if source.exists():assert source.read_bytes()==target.read_bytes()
    lock=dict(protocol_sha256=hashlib.sha256((ROOT/'docs/SHoEbox_Joint_Retention_Protocol.md').read_bytes()).hexdigest(),
        parents={str(seed):hashlib.sha256((parents/f'{seed}.pt').read_bytes()).hexdigest() for seed in SEEDS},
        seeds=list(SEEDS),steps=80,batch=32,test_count=512,modes=['edge','joint'])
    lockpath=out/'protocol_lock.json'
    if lockpath.exists():assert json.loads(lockpath.read_text())==lock
    else:lockpath.write_text(json.dumps(lock,indent=2)+'\n')
    statuspath=out/'run_status.json'
    status=json.loads(statuspath.read_text()) if statuspath.exists() else dict(completed=[],state='training')
    def save():
        tmp=statuspath.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(statuspath)
    def run(seed,mode,phase):
        path=out/f'{mode}_{seed}';path.mkdir(exist_ok=True)
        if phase=='training':
            command=[sys.executable,'-m','experiments.text_regions.joint_retained_train','--parent',str(parents/f'{seed}.pt'),
                '--out',str(path),'--seed',str(seed),'--mode',mode,'--steps',str(lock['steps'])]
            if (path/'last.pt').exists():
                import torch
                if torch.load(path/'last.pt',map_location='cpu',weights_only=True)['step']==lock['steps']:
                    return f'{phase}/{mode}/{seed}'
                command.append('--resume')
        else:
            script='evaluate_joint_retained.py' if phase=='evaluation' else 'audit_joint_retained.py'
            command=[sys.executable,str(ROOT/'scripts'/script),str(path)]
        with (path/(phase+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return f'{phase}/{mode}/{seed}'
    try:
        for phase in ('training','evaluation','audit'):
            status['state']=phase;save()
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                jobs=[pool.submit(run,seed,mode,phase) for mode in ('joint','edge') for seed in SEEDS
                      if f'{phase}/{mode}/{seed}' not in status['completed']]
                for job in as_completed(jobs):status['completed'].append(job.result());save()
        status['state']='complete';save()
    except BaseException as error:
        status['state']='failed';status['error']=repr(error);save();raise


if __name__=='__main__':main()
