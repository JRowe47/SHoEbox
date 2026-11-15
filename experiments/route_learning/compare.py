"""Serial matched continuation: corrected C versus route-local LR cycles.

Uses the same trained checkpoint and a reshuffled pass through its training
pool. This is a continuation screen, not a fresh-data or multi-seed claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shoe_multiscale import Model
from experiments.scale_100m.train import DocumentChunks,frozen_score,atomic,file_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--targets',type=int,default=5_000_000)
    p.add_argument('--seed',type=int,default=1007);p.add_argument('--cpu',type=int,default=2)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{a.cpu})
    meta=json.loads((a.data/'manifest.json').read_text())
    train_meta,dev_meta=meta['files']['train'],meta['files']['development']
    train=np.memmap(a.data/train_meta['file'],dtype=np.uint8,mode='r')
    dev=np.memmap(a.data/dev_meta['file'],dtype=np.uint8,mode='r')
    for item in (train_meta,dev_meta):
        assert file_hash(a.data/item['file'])==item['sha256']
    assert not {d['id'] for d in train_meta['documents']} & {d['id'] for d in dev_meta['documents']}
    assert not {d['text_sha256'] for d in train_meta['documents']} & {d['text_sha256'] for d in dev_meta['documents']}
    stream=DocumentChunks(train_meta['documents'],a.seed);assert 0<a.targets<=stream.targets
    result=dict(checkpoint_sha256=file_hash(a.checkpoint),train_sha256=train_meta['sha256'],
        development_sha256=dev_meta['sha256'],segment_sha256=hashlib.sha256(stream.segments.tobytes()).hexdigest(),
        targets=a.targets,shuffle_seed=a.seed,cpu=a.cpu,
        protocol='same 20M-trained seed-7 corrected-C checkpoint; reshuffled training-pool replay; serial alternating 1M blocks; no audit-based selection',
        timing='training plus document resets; excludes load, compilation, evaluation, statistics and checkpoint writes',
        arms={})
    models={}
    try:
        for arm in ('C','route'):
            m=Model.load(a.checkpoint);m.reset_context(clear_memory=True)
            assert m.stats()['routing_arm']=='C' and m.stats()['balance_policy']=='warmup'
            if arm=='route':m.configure_route_lr(lower=.5,upper=1.5,anchor=1.,amplitude=1.,period=256)
            models[arm]=m
            nll,docs=frozen_score(m,dev,dev_meta['documents'])
            result['arms'][arm]=dict(parameters=m.parameter_count,base_rates=m.stats()['learning_rates'],
                route_learning=m.route_lr_stats(),curve=[dict(targets=0,nll=nll,train_seconds=0.)],
                train_seconds=0.,targets=0)
        for turn,boundary in enumerate(range(1_000_000,a.targets+1_000_000,1_000_000)):
            boundary=min(boundary,a.targets)
            for arm in (('C','route') if turn%2==0 else ('route','C')):
                m=models[arm];r=result['arms'][arm];done=r['targets'];weighted=0.;window=0
                while done<boundary:
                    x,y,reset=stream.take(done,boundary-done);t=time.perf_counter()
                    if reset:m.reset_context(clear_memory=True)
                    loss=m.run(train[x:y],train[x+1:y+1]);r['train_seconds']+=time.perf_counter()-t
                    if not np.isfinite(loss):raise RuntimeError('nonfinite training loss')
                    done+=y-x;weighted+=loss*(y-x);window+=y-x
                nll,docs=frozen_score(m,dev,dev_meta['documents']);r['targets']=done
                r['curve'].append(dict(targets=done,nll=nll,train_nll=weighted/window,train_seconds=r['train_seconds']))
                r['development_documents']=docs;r['route_learning']=m.route_lr_stats(include_routes=True)
                atomic(a.out/'comparison.json',result)
                print(json.dumps(dict(arm=arm,**r['curve'][-1])),flush=True)
        # Exact partial-state checkpoint continuation through a moving anchor.
        m=models['route'];m.set_learning_anchor(.8,over_targets=8193)
        with m.clone() as twin,Model.loads(m.dumps()) as resumed:
            twin.run(train[:4097],train[1:4098]);resumed.run(train[:4097],train[1:4098])
            assert twin.dumps()==resumed.dumps()
            result['resume_check_sha256']=hashlib.sha256(twin.dumps()).hexdigest()
        m.set_learning_anchor(1.)
        for arm,m in models.items():
            m.save(a.out/f'{arm}_25M.shoe')
            result['arms'][arm]['checkpoint_sha256']=file_hash(a.out/f'{arm}_25M.shoe')
        result['status']='complete';atomic(a.out/'comparison.json',result)
    finally:
        for m in models.values():m.close()


if __name__=='__main__':main()
