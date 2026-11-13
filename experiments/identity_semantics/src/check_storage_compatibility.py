"""Old -> compact -> old continuation using actual independently built runtimes."""
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
    p.add_argument('--bridge-mode',choices=('periodic','eval'),default='periodic')
    a=p.parse_args();script=Path(__file__).with_name('compare_optimized_runtime.py')
    rows={}
    with tempfile.TemporaryDirectory(prefix='shoebox-storage-compat-') as temporary:
        folder=Path(temporary);initial=folder/'initial.npz'
        def run(name,runtime,checkpoint,warmup,targets,save=None,initialize=False,mode='periodic'):
            output=folder/(name+'.json')
            command=[sys.executable,str(script),'--worker','--runtime',str(runtime.resolve()),
                     '--checkpoint',str(checkpoint),'--output',str(output),'--cpu',str(a.cpu),
                     '--warmup',str(warmup),'--targets',str(targets),'--mode',mode]
            if initialize:command+=['--initialize']
            if save:command+=['--save-after',str(save)]
            subprocess.run(command,check=True)
            if not initialize:rows[name]=json.loads(output.read_text())
            print(name+' passed',flush=True)
        run('initialize',a.baseline_runtime,initial,1024,1024,initialize=True)
        old=folder/'old.npz';compact=folder/'compact.npz'
        run('old_phase1',a.baseline_runtime,initial,1024,1024,old,mode=a.bridge_mode)
        run('compact_phase1',a.runtime,initial,1024,1024,compact,mode=a.bridge_mode)
        run('old_phase2',a.baseline_runtime,old,2048,513)
        run('compact_phase2',a.runtime,compact,2048,513)
        run('compact_to_old_phase2',a.baseline_runtime,compact,2048,513)
    keys=('nll','logits_sha256','main_sha256','extra_sha256','composition_sha256','patterns_sha256')
    for group in (('old_phase1','compact_phase1'),('old_phase2','compact_phase2','compact_to_old_phase2')):
        for key in keys:
            if len({rows[name][key] for name in group})!=1:raise AssertionError((group,key))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(bitwise_continuation=True,protocol=__doc__,
                                      bridge_mode=a.bridge_mode,runs=rows),indent=2)+'\n')


if __name__=='__main__':main()
