"""Verify a real stage-3 checkpoint retains its request-wise learning behavior."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args();script=ROOT/'src/compare_union_reverse.py'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='shoebox-union-compat-') as temporary:
        directory=Path(temporary);checkpoint=directory/'warmup_stage3.npz'
        common=[sys.executable,str(script),'--worker','--checkpoint',str(checkpoint),
                '--warmup','256','--targets','512','--cpu',str(a.cpu),'--output',str(directory/'row.json')]
        subprocess.run(common+['--runtime',str(a.baseline_runtime),'--initialize'],env=env,check=True)
        rows={}
        for name,runtime in [('stage3',a.baseline_runtime),('current_root',ROOT/'runtime')]:
            subprocess.run(common+['--runtime',str(runtime),'--snapshot',str(directory/(name+'.npz')),'--trace'],env=env,check=True)
            rows[name]=json.loads((directory/'row.json').read_text())
        with np.load(directory/'stage3.npz') as old,np.load(directory/'current_root.npz') as new:
            identical={key:bool(np.array_equal(old[key],new[key])) for key in old.files}
        result=dict(actual_stage3_checkpoint=True,arrays_bitwise=identical,
                    state_fingerprint_equal=rows['stage3']['fingerprint']==rows['current_root']['fingerprint'],
                    nll_equal=rows['stage3']['nll']==rows['current_root']['nll'])
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        if not all(identical.values()) or not result['state_fingerprint_equal'] or not result['nll_equal']:
            raise AssertionError('Historical checkpoint continuation changed: '+str(result))
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
