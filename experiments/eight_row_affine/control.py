"""Isolated, write-once, metered correctness command; never used for timing."""
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent;root=here.parents[1]
    for n in ('bin','tmp','events','builds','python_events'): (a.out/n).mkdir()
    wrapper=a.out/'bin/g++';wrapper.write_text('#!/bin/sh\nexec '+sys.executable+' '+str(here/'instrument.py')+' "$@"\n');wrapper.chmod(0o755)
    (a.out/'bin/sitecustomize.py').write_text('import sys\nsys.path.insert(0,'+repr(str(root))+')\nfrom experiments.eight_row_affine.python_events import install\ninstall()\n')
    env={**os.environ,'PATH':str(a.out/'bin')+':'+os.environ['PATH'],'CXX':str(wrapper),
         'TMPDIR':str(a.out/'tmp'),'A11_INSTRUMENT_ROOT':str(a.out/'builds'),
         'A11_EVENT_DIR':str(a.out/'events'),'A11_PYTHON_EVENT_DIR':str(a.out/'python_events'),
         'PYTHONPATH':str(a.out/'bin')+':'+str(root)+':/workspace/scratch/7484c22b7e5e/test_deps',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    started=datetime.now(timezone.utc).isoformat();wall=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (a.out/'output.txt').open('x') as f:r=subprocess.run(a.command,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=root)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(command=a.command,started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,
                wall_seconds=time.perf_counter()-wall,user_cpu_seconds=after.ru_utime-before.ru_utime,system_cpu_seconds=after.ru_stime-before.ru_stime,
                cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,peak_rss_kib=after.ru_maxrss,
                scope='Whole child process tree; compilation, counters, bookkeeping and failures included. No fitting campaign.')
    (a.out/'process.json').write_text(json.dumps(result,indent=2)+'\n')
    print(a.out.name,'exit',r.returncode,'CPU',result['cpu_seconds'],flush=True);sys.exit(r.returncode)
if __name__=='__main__':main()
