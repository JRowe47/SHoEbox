"""Run disjoint held-out cohorts in four CPU workers and verify their join."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--single-cases',type=int,default=32)
    p.add_argument('--selective-cases',type=int,default=16)
    p.add_argument('--documents',type=int,default=8)
    p.add_argument('--data-seed',type=int,default=917)
    p.add_argument('--start',type=int,default=20000)
    a = p.parse_args()
    if a.out.exists():
        p.error('Combined output already exists; use its verified result or a new output path.')
    directory = a.out.parent/(a.out.stem+'_parts')
    directory.mkdir(parents=True,exist_ok=True)
    jobs = []
    for part in ('single','pairs','revision','streams'):
        log = (directory/f'{part}.log').open('a')
        arguments=[sys.executable,str(Path(__file__).with_name('evaluate_memory_selectivity.py')),
            '--checkpoint',str(a.checkpoint),'--out',str(directory/f'{part}.json'),'--part',part,
            '--single-cases',str(a.single_cases),'--selective-cases',str(a.selective_cases),
            '--documents',str(a.documents),'--data-seed',str(a.data_seed),'--start',str(a.start)]
        if a.data is not None: arguments.extend(('--data',str(a.data)))
        process = subprocess.Popen(arguments,
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((part,process,log))
    codes = []
    for part,process,log in jobs:
        codes.append((part,process.wait())); log.close()
    if any(code for part,code in codes):
        raise RuntimeError(f'Evaluation workers failed: {codes}')
    identity = None
    measurements = {}
    for part,process,log in jobs:
        result = json.loads((directory/f'{part}.json').read_text())
        own = result['identity']; assert own.pop('part') == part
        if identity is None:
            identity = own
        elif own != identity:
            raise ValueError('Evaluation worker identities differ.')
        if measurements.keys() & result['measurements'].keys():
            raise ValueError('Evaluation cohorts overlap.')
        measurements.update(result['measurements'])
    identity['part'] = 'full'
    for split in ('test','long_delay'):
        source = directory/f'streams_{split}_traces.jsonl'
        target = a.out.parent/f'{a.out.stem}_{split}_traces.jsonl'
        target.write_bytes(source.read_bytes())
    temporary = a.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(identity=identity,measurements=measurements),indent=2,allow_nan=False)+'\n')
    temporary.replace(a.out)
    print(json.dumps(dict(output=str(a.out),measurements=len(measurements))),flush=True)


if __name__ == '__main__':
    main()
