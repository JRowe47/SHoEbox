"""Run every reference test module once, in independent subprocess groups.

Fixture libraries have module-specific filenames; runtime compilation already
uses file locks. No model or live runtime state is shared between groups.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.jobs<=8:p.error('--jobs must be between 1 and 8')
    modules=[path.stem for path in sorted((ROOT/'tests').glob('test_*.py'))]
    weights={'test_completion':6,'test_local_credit':4,'test_neural_retention':4,
             'test_reference_defaults':3,'test_contributions':2,'test_relays':2}
    groups=[[] for _ in range(a.jobs)];loads=[0]*a.jobs
    for module in sorted(modules,key=lambda name:(-weights.get(name,1),name)):
        i=min(range(a.jobs),key=loads.__getitem__);groups[i].append(module);loads[i]+=weights.get(module,1)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONPATH=str(ROOT/'tests')+os.pathsep+os.environ.get('PYTHONPATH',''))
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='shoebox-regressions-') as temporary:
        folder=Path(temporary)
        def run(i):
            log=folder/f'group{i}.txt'
            with log.open('w') as stream:
                completed=subprocess.run([sys.executable,'-m','unittest','-v',*groups[i]],
                                         cwd=ROOT.parents[1],env=env,stdout=stream,stderr=subprocess.STDOUT)
            text=log.read_text();counts=re.findall(r'Ran (\d+) tests? in [\d.]+s',text)
            count=int(counts[-1]) if counts else 0
            passed=completed.returncode==0 and count>0 and text.rstrip().endswith('OK')
            print(f'Group {i+1}: {count} tests; '+('passed' if passed else 'FAILED'),flush=True)
            return passed,count,text
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:rows=list(pool.map(run,range(a.jobs)))
    count=sum(row[1] for row in rows);passed=all(row[0] for row in rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Group output contains individual timings; the terminal count is the
    # distinct full-suite total and wall time, in the usual unittest format.
    text=''.join(f'Group {i+1}: '+', '.join(groups[i])+'\n'+row[2]+'\n' for i,row in enumerate(rows))
    text+=f'Ran {count} tests in {time.perf_counter()-start:.3f}s\n\n'+('OK\n' if passed else 'FAILED\n')
    a.output.write_text(text)
    if not passed:raise SystemExit(1)
    print(f'All {count} tests passed across {a.jobs} independent groups.',flush=True)


if __name__=='__main__':main()
