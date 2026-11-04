"""Run the twelve frozen-readout comparisons from restored checkpoint paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('artifacts/readout_diagnostic'))
    parser.add_argument('--jobs',type=int,default=3);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--fit-per-cell',type=int,default=64);parser.add_argument('--count-per-query',type=int,default=128)
    parser.add_argument('--max-iter',type=int,default=300);parser.add_argument('--seed',type=int,default=20261120)
    args = parser.parse_args()
    if min(args.jobs,args.threads) < 1:
        parser.error('Positive worker and thread counts required.')
    out = args.out if args.out.is_absolute() else ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():
        parser.error('Study destination already has a status file; use a new directory.')
    original = {('original',1):'artifacts/views/views',('gated',1):'artifacts/evidence/views',
                **{(p,s):f'artifacts/evidence/seed{s}_{p}' for s in (2,3) for p in ('original','gated')}}
    runs = []
    for seed in (1,2,3):
        for policy in ('original','gated'):
            for schedule in ('dropout','balanced'):
                checkpoint = (original[(policy,seed)] if schedule == 'dropout' else
                              f'artifacts/source_schedule/{policy}_seed{seed}')+'/checkpoint.pt'
                if not (ROOT/checkpoint).is_file():
                    parser.error('Missing checkpoint: '+checkpoint)
                name = f'{policy}_{schedule}_seed{seed}'
                command = [sys.executable,'-m','experiments.text_regions.readout_diagnostic',checkpoint,
                    '--out',str(out/name),'--fit-per-cell',str(args.fit_per_cell),'--count-per-query',str(args.count_per_query),
                    '--max-iter',str(args.max_iter),'--seed',str(args.seed),'--threads',str(args.threads)]
                runs.append({'name':name,'publication_policy':policy,'source_schedule':schedule,'training_seed':20261105+seed,
                             'checkpoint':checkpoint,'checkpoint_sha256':hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
                             'command':command,'status':'pending'})
    status = {'schema_version':1,'complete':False,'runs':runs,
              'settings':{'fit_per_cell':args.fit_per_cell,'count_per_query':args.count_per_query,
                          'max_iter':args.max_iter,'seed':args.seed,'threads_per_job':args.threads,'jobs':args.jobs}}
    lock = threading.Lock()
    def save():
        temporary = out/'status.json.tmp'
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(out/'status.json')
    save()
    def execute(run):
        started = time.monotonic()
        with lock:
            run['status'] = 'running';save()
        with (out/(run['name']+'.log')).open('w') as log:
            completed = subprocess.run(run['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with lock:
            run.update(returncode=completed.returncode,seconds=time.monotonic()-started,
                       status='complete' if completed.returncode == 0 else 'failed')
            save()
        print(json.dumps({k:run[k] for k in ('name','status','seconds','returncode')}),flush=True)
        return completed.returncode
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        failures = sum(future.result() != 0 for future in as_completed([workers.submit(execute,run) for run in runs]))
    status['complete'] = failures == 0
    save()
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
