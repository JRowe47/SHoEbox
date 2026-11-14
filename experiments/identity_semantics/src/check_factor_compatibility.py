"""Stage-2 -> stage-3 training -> stage-2 export -> stage-3 continuation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-runtime',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]/'runtime')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args();script=Path(__file__).with_name('compare_factor_aggregation.py');rows={}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='shoebox-factor-compat-') as temporary:
        folder=Path(temporary);initial=folder/'old.npz';trained=folder/'trained.npz';bridge=folder/'bridge.npz'
        def run(name,runtime,checkpoint,warmup=1024,targets=512,mode='periodic',flags=()):
            output=folder/(name+'.json')
            command=[sys.executable,str(script),'--worker','--runtime',str(runtime.resolve()),
                     '--checkpoint',str(checkpoint),'--output',str(output),'--snapshot',str(folder/(name+'.npz')),
                     '--cpu',str(a.cpu),'--warmup',str(warmup),'--targets',str(targets),'--mode',mode,*flags]
            subprocess.run(command,env=env,check=True)
            if output.exists():rows[name]=json.loads(output.read_text())
            print(name+' passed',flush=True)
        run('initialize',a.baseline_runtime,initial,flags=['--initialize'])
        run('stage3_training',a.runtime,initial,flags=['--save-after',str(trained)])
        run('stage2_frozen',a.baseline_runtime,trained,warmup=1536,targets=256,mode='eval')
        run('stage3_frozen',a.runtime,trained,warmup=1536,targets=256,mode='eval')
        run('stage2_roundtrip',a.baseline_runtime,trained,flags=['--roundtrip','--save-after',str(bridge)])
        run('stage3_direct',a.runtime,trained,warmup=1536,targets=256)
        run('stage3_roundtrip',a.runtime,bridge,warmup=1536,targets=256)
    for left,right in [('stage2_frozen','stage3_frozen'),('stage3_direct','stage3_roundtrip')]:
        for key in ('fingerprint','nll','hashes'):
            if rows[left][key]!=rows[right][key]:raise AssertionError((left,right,key))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(protocol=__doc__,cross_runtime_frozen_bitwise=True,
        checkpoint_roundtrip_bitwise=True,runs=rows),indent=2)+'\n')


if __name__=='__main__':main()
