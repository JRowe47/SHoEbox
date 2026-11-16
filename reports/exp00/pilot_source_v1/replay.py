"""Replay an archived development-only pilot; never used for formal v2 results.

Example: python -m reports.exp00.pilot_source_v1.replay initial --out /tmp/pilot.json
"""
import argparse
import json
import os
from pathlib import Path
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from .tasks import Spec, dataset
from .controls import PositiveControl, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['initial','larger_dataset','longer_training'])
    p.add_argument('--out',required=True,type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error('output already exists')
    manifest = json.loads(Path(__file__).parents[1].joinpath('pilots.json').read_text())
    phase = next(x for x in manifest['phases'] if x['name'] == args.phase)
    torch.set_num_threads(1)
    rows = []
    for prior in phase['runs']:
        task = prior['task']
        spec = (Spec('binding') if task == 'binding' else
                Spec('one_hop',vocab=16,items=3,key_pool=16) if task == 'one_hop' else
                Spec(task,items=4,key_pool=4))
        train = dataset(spec,prior['train_cases'],'train')
        dev = dataset(spec,phase['dev_requested_cases'],'dev')
        control = PositiveControl(spec,seed=1,lr=prior['lr'])
        rng = np.random.default_rng(1067)
        seen,next_eval,cpu = 0,phase['eval_every_bytes'],time.process_time()
        curve = []
        while seen < phase['maximum_bytes_per_fit']:
            batch = [train[i] for i in rng.integers(0,len(train),32)]
            control.update(batch)
            seen += sum(len(e.data)-1 for e in batch)
            if seen >= next_eval:
                row = {'task':task,'bytes':seen,'cpu_seconds':time.process_time()-cpu,
                       **evaluate(control,dev)}
                curve.append(row)
                print(json.dumps(row),flush=True)
                next_eval += phase['eval_every_bytes']
                if row['exact_accuracy'] >= phase['stop_dev_exact_at']:
                    break
        rows.append({'task':task,'seed':1,'lr':prior['lr'],
                     'train_cases':prior['train_cases'],'curve':curve})
        args.out.write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    main()
