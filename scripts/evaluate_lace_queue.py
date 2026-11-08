"""Additional evaluation workers; output locks prevent duplicate computation."""
import argparse,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'artifacts/lace_separation'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workers',type=int,default=3);parser.add_argument('--offset',type=int,default=0)
parser.add_argument('--forward',action='store_true');args=parser.parse_args()
while not (root/'training_complete.json').exists():time.sleep(3)
jobs=[]
checkpoints={arm:root/arm/'last.pt' for arm in 'ABCD'}
checkpoints['parent']=ROOT/'artifacts/hippocampus/parent.pt'
for name,checkpoint in checkpoints.items():
    for cohort in ('validation','test'):
        for part in ('pairs','retention','language')+(('unseen',) if cohort=='test' else ()):
            label=f'{name}_{cohort}_{part}'
            jobs.append((label,checkpoint,cohort,part,'intact'))
for control in ('bias_off','cue_permuted','blocked'):
    jobs.append(('D_test_'+control,root/'D/last.pt','test','pairs',control))
def run(job):
    label,checkpoint,cohort,part,control=job
    with (root/(label+'.reverse.log')).open('a') as stream:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_lace_separation.py'),
            '--checkpoint',str(checkpoint),'--cohort',cohort,'--part',part,'--control',control,
            '--out',str(root/'evaluation'/f'{label}.json')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    return label,result.returncode
ordered=jobs[args.offset:]+jobs[:args.offset]
if not args.forward:ordered=list(reversed(ordered))
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    for future in as_completed([pool.submit(run,j) for j in ordered]):
        label,code=future.result();print(json.dumps(dict(job=label,exit_code=code)),flush=True)
        if code:raise RuntimeError(label)
