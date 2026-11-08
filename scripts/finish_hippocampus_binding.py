"""Bounded evaluation jobs for the fixed hippocampus binding comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_memory_selectivity import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=['parent','legacy','event_expert','controls'],
                   default=['parent','legacy','event_expert','controls'])
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.workers<=3: p.error('Use one through three evaluation workers.')
    root=ROOT/'artifacts/hippocampus_binding'; out=root/'evaluation'; out.mkdir(exist_ok=True)
    choices={'parent':ROOT/'artifacts/hippocampus/parent.pt',
             'legacy':ROOT/'artifacts/hippocampus/C/last.pt',
             'event_expert':root/'event_expert/last.pt'}
    common=['--seed','1511','--start','120000','--pair-count','16','--single-count','32']
    jobs=[]
    for name in a.models:
        if name=='controls':
            checkpoint=choices['event_expert']
            controls={'blocked':['--control','blocked'],
                      'round_cue':['--cue-clock','round'],
                      'ingress_read':['--reinstatement','ingress']}
            for label,flags in controls.items():
                jobs.append((label,checkpoint,'pairs',['--gaps','0','32','64',*flags]))
        else:
            checkpoint=choices[name]
            for part in ('pairs','single','correction','language'):
                jobs.append((name+'_'+part,checkpoint,part,
                    ['--gaps','0','32','64'] if part=='pairs' else ['--gaps','64']))
    for checkpoint in {j[1] for j in jobs}:
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
        if checkpoint!=choices['parent'] and saved['step']!=64:
            raise ValueError('Final evaluation requires the declared update-64 endpoint.')
    progress=dict(models=a.models,completed=[],failed=[])
    progress_path=out/('progress_'+'_'.join(a.models)+'.json')
    def work(job):
        label,checkpoint,part,flags=job
        command=[sys.executable,str(ROOT/'scripts/evaluate_hippocampus.py'),
                 '--checkpoint',str(checkpoint),'--out',str(out/(label+'.json')),
                 '--part',part,*common,*flags]
        with (out/(label+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return label,result.returncode
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(work,j) for j in jobs]):
            name,code=future.result()
            progress['completed' if code==0 else 'failed'].append(name)
            write_json(progress_path,progress)
            print(json.dumps(dict(job=name,exit_code=code)),flush=True)
    if progress['failed']: raise SystemExit(1)

if __name__=='__main__': main()
