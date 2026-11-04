"""Fit validation-scaled residual heads on the twelve frozen regional graphs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.residual_adapter import write_json, STRENGTHS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('artifacts/residual_adapter_v1'))
    parser.add_argument('--jobs',type=int,default=3);parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if min(args.jobs,args.threads)<1:parser.error('Positive worker and thread counts required.')
    out=args.out if args.out.is_absolute() else ROOT/args.out
    if out.exists():parser.error('Choose a new study destination.')
    manifest=json.loads((ROOT/'experiments/text_regions/results/emission_adapter/manifest.json').read_text())
    runs=[]
    for row in manifest['runs']:
        name=row['name'];prior=Path(manifest['source_directory'])/name
        if not all((ROOT/p).is_file() for p in (row['checkpoint'],row['reference'],prior/'features.pt',prior/'results.json')):
            parser.error('Restore the complete emission-adapter package first: '+name)
        command=[sys.executable,'-m','experiments.text_regions.residual_adapter',row['checkpoint'],
            '--reference',row['reference'],'--prior-run',str(prior),'--out',str(out/name),'--threads',str(args.threads)]
        runs.append({**row,'prior_run':str(prior),'command':command,'status':'pending'})
    out.mkdir(parents=True)
    status={'schema_version':1,'complete':False,'runs':runs,
        'settings':{'hidden':16,'max_iter':400,'l2':.001,'adapter_seeds':[20261121,20261122,20261123],
                    'representation':'raw','validation_count_per_cell':128,'strengths':list(STRENGTHS),
                    'jobs':args.jobs,'threads':args.threads}}
    lock=threading.Lock()
    def save():write_json(out/'status.json',status)
    save()
    def execute(run):
        started=time.monotonic()
        with lock:run['status']='running';save()
        with (out/(run['name']+'.log')).open('w') as log:
            completed=subprocess.run(run['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with lock:
            run.update(returncode=completed.returncode,seconds=time.monotonic()-started,
                       status='complete' if completed.returncode==0 else 'failed');save()
        print(json.dumps({k:run[k] for k in ('name','status','seconds','returncode')}),flush=True)
        return completed.returncode
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures=sum(f.result()!=0 for f in as_completed([pool.submit(execute,r) for r in runs]))
    status['complete']=not failures;save()
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
