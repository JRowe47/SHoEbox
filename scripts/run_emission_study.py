"""Run matched emission-adapter experiments on the twelve frozen graphs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('artifacts/emission_adapter'))
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if min(args.jobs,args.threads)<1:parser.error('Positive worker and thread counts required.')
    out=args.out if args.out.is_absolute() else ROOT/args.out
    if out.exists():parser.error('Choose a new study destination.')
    prior=ROOT/'experiments/text_regions/results/readout_diagnostic'
    manifest=json.loads((prior/'manifest.json').read_text())
    runs=[]
    for row in manifest['runs']:
        name=row['name'];reference=Path('artifacts/readout_diagnostic')/name/'readouts.pt'
        results=prior/row['directory']/'results.json.gz'
        if not all(p.is_file() for p in (ROOT/row['checkpoint'],ROOT/reference,results)):
            parser.error('Restore the complete frozen-readout package first: '+name)
        command=[sys.executable,'-m','experiments.text_regions.emission_adapter',row['checkpoint'],
            '--reference',str(reference),'--prior-results',str(results.relative_to(ROOT)),
            '--out',str(out/name),'--threads',str(args.threads)]
        runs.append({**row,'reference':str(reference),'prior_results':str(results.relative_to(ROOT)),
                     'command':command,'status':'pending'})
    out.mkdir(parents=True)
    status={'schema_version':1,'complete':False,'runs':runs,
            'settings':{'hidden':64,'max_iter':400,'l2':.001,'adapter_seeds':[20261121,20261122,20261123],
                        'jobs':args.jobs,'threads':args.threads}}
    lock=threading.Lock()
    def save():
        path=out/'status.json.tmp';path.write_text(json.dumps(status,indent=2)+'\n');path.replace(out/'status.json')
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
        failures=sum(f.result()!=0 for f in as_completed([pool.submit(execute,run) for run in runs]))
    status['complete']=not failures;save()
    return int(bool(failures))


if __name__=='__main__':
    raise SystemExit(main())
