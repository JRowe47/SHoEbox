"""Recoverable three-seed integration run; each worker is a local process."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/retained_composition'
SEEDS=(20261120,20261121,20261122)


def main():
    OUT.mkdir(parents=True,exist_ok=True);parents=OUT/'parents';parents.mkdir(exist_ok=True)
    dependencies={'composer.pt':ROOT/'artifacts/selective_publication/parent.pt',**{
        f'producer_{seed}.pt':ROOT/f'artifacts/selective_publication/learned_{seed-10}/best.pt' for seed in SEEDS}}
    for name,source in dependencies.items():
        target=parents/name
        if not target.exists():shutil.copyfile(source,target)
        if source.exists():
            assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
    lock=dict(protocol_sha256=hashlib.sha256((ROOT/'docs/SHoEbox_Retained_Composition_Protocol.md').read_bytes()).hexdigest(),
        parents={name:hashlib.sha256((parents/name).read_bytes()).hexdigest() for name in dependencies},
        seeds=list(SEEDS),steps=160,test_count=512)
    lp=OUT/'protocol_lock.json'
    if lp.exists():assert json.loads(lp.read_text())==lock
    else:lp.write_text(json.dumps(lock,indent=2)+'\n')
    statuspath=OUT/'run_status.json'
    status=json.loads(statuspath.read_text()) if statuspath.exists() else dict(completed=[],state='training')
    def save():
        temp=statuspath.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(statuspath)
    def run(seed,phase):
        path=OUT/str(seed);path.mkdir(exist_ok=True)
        if phase=='training':
            command=[sys.executable,'-m','experiments.text_regions.retained_composition_train',
                '--composer',str(parents/'composer.pt'),'--producer',str(parents/f'producer_{seed}.pt'),
                '--out',str(path),'--seed',str(seed)]
            if (path/'last.pt').exists():
                import torch
                if torch.load(path/'last.pt',map_location='cpu',weights_only=True)['step']==160:return f'{phase}/{seed}'
                command.append('--resume')
        else:command=[sys.executable,'scripts/evaluate_retained_composition.py',str(path)]
        with (path/(phase+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return f'{phase}/{seed}'
    try:
        for phase in ('training','evaluation'):
            status['state']=phase;save()
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs=[pool.submit(run,seed,phase) for seed in SEEDS if f'{phase}/{seed}' not in status['completed']]
                for job in as_completed(jobs):status['completed'].append(job.result());save()
        status['state']='awaiting_audit';save()
    except BaseException as error:
        status['state']='failed';status['error']=repr(error);save();raise


if __name__=='__main__':main()
