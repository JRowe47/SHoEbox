"""Compare scope union and bank-versioned epochs with a reverse update-scale sweep.

All variants restore one root-policy checkpoint. Epochs freeze shared expert
W/bias/router reads and delay constituent updates; predictors, readout and bottom
state remain online. Timing is single-worker serial and excludes evaluation,
checkpoint IO, generation and compilation. Temporary checkpoints are removed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from runtime import Model,fresh_model,core
from reference_acceptance import collect,check_integrity
from run_reference_acceptance import generate
from compare_optimized_runtime import TRAIN,source_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus',type=Path)
    p.add_argument('--validation',type=Path);p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--targets',type=int,default=2048);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu})
    n=a.warmup+a.targets+1
    raw=a.corpus.read_bytes()[:n] if a.corpus else (TRAIN*(n//len(TRAIN)+1))[:n]
    if len(raw)<n:raise ValueError('Insufficient training bytes')
    data=np.frombuffer(raw,np.uint8)
    # A separate corpus split is required for natural-text held-out scores.
    validation=a.validation.read_bytes()[:1025] if a.validation else (
        b'The bird crossed the valley. The queen walked through the gate. '
        b'The river runs beside the bridge. A king looked over the water. '*10)[:1025]
    val=np.frombuffer(validation,np.uint8)
    result=dict(protocol=__doc__,cpu=a.cpu,experts=core.E,warmup=a.warmup,targets=a.targets,
                train_sha256=hashlib.sha256(raw).hexdigest(),validation_sha256=hashlib.sha256(validation).hexdigest(),
                validation_kind='separate FineWeb split' if a.validation else 'held-out recombinations of repeated fixture',
                source_sha256=source_hash(ROOT/'runtime'),runs=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Alternate shorter and longer epochs instead of a monotonic epoch sweep.
    policies=[dict(policy='root',epoch_bytes=1,update_scale=1.)]
    policies += [dict(policy='union',epoch_bytes=e,update_scale=scale) for scale in (.5,1.,2.) for e in (1,16,4,8)]
    with tempfile.TemporaryDirectory(prefix='shoebox-union-sweep-') as temporary:
        path=Path(temporary)/'warmup.npz';m=fresh_model(seed=7);m.configure_reverse(dict(policy='root'))
        m.run(data[:a.warmup],data[1:a.warmup+1],mode='periodic',lr=.04,lp=.04/3);m.save(path);m.close()
        for policy in policies:
            m=Model.load(path);m.configure_reverse(policy);points=[];seconds=0.;weighted_loss=0.
            for offset in range(0,a.targets,512):
                size=min(512,a.targets-offset);lo=a.warmup+offset;start=time.perf_counter()
                nll=m.run(data[lo:lo+size],data[lo+1:lo+size+1],mode='periodic',lr=.04,lp=.04/3)
                elapsed=time.perf_counter()-start;seconds+=elapsed;weighted_loss+=float(nll)*size
                points.append(dict(targets=offset+size,seconds=seconds,train_nll=float(nll)))
            # Targets divisible by 16 avoid attributing uncommitted tail work
            # to a throughput gain; finish remains part of measured training.
            start=time.perf_counter();m.finish_learning_epoch();seconds+=time.perf_counter()-start
            integrity=check_integrity(collect(m));check=m.evaluation_copy(memory='cold_frozen')
            heldout=float(check.run(val[:-1],val[1:],mode='eval'));check.close()
            row=dict(policy=policy,seconds=seconds,bytes_per_second=a.targets/seconds,
                     train_nll=weighted_loss/a.targets,heldout_nll=heldout,curve=points,
                     reverse_metrics=m.reverse_metrics(),factor_metrics=m.content_factor_metrics(),
                     credit_integrity=True,fingerprint=m.fingerprint(),
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if policy['update_scale']==1.:row['generation']=generate(m,b'The river ',count=64)
            result['runs'].append(row);m.close();a.output.write_text(json.dumps(result,indent=2)+'\n')
            print(policy,round(row['bytes_per_second'],1),'B/s',round(heldout,6),'heldout',flush=True)
    # Select within each policy by held-out NLL, without claiming this short
    # probe establishes a long-run optimum.
    result['best_by_epoch']={str(e):min((r for r in result['runs'] if r['policy']['policy']=='union' and r['policy']['epoch_bytes']==e),
                                      key=lambda r:r['heldout_nll'])['policy'] for e in (1,4,8,16)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
