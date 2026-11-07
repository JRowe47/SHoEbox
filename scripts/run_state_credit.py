"""Run or resume both arms of the producer-state credit comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/state_credit')
    p.add_argument('--steps', type=int, default=32)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    parent = a.out/'parent.pt'
    if parent.exists():
        if digest(parent) != digest(a.parent):
            raise ValueError('Existing study has a different parent.')
    else:
        shutil.copy2(a.parent, parent)
    sources = {str(path.relative_to(ROOT)):digest(path) for path in
        sorted((ROOT/'experiments').rglob('*.py'))}
    environment = dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
        parent_sha256=digest(parent),raw_sha256=digest(a.data),source_sha256=sources)
    env_path = a.out/'environment.json'
    if env_path.exists():
        saved = json.loads(env_path.read_text())
        if any(saved[k] != environment[k] for k in ('torch','parent_sha256','raw_sha256','source_sha256')):
            raise ValueError('Training environment/source changed since launch.')
    else:
        env_path.write_text(json.dumps(environment,indent=2)+'\n')
    processes = []
    for mode in ('state','publication'):
        folder = a.out/mode
        command = [sys.executable,'-m','experiments.text_regions.state_credit_train',
            '--data',str(a.data.resolve()),'--parent',str(parent.resolve()),
            '--out',str(folder.resolve()),'--steps',str(a.steps),
            '--state-credit' if mode == 'state' else '--no-state-credit']
        if (folder/'last.pt').exists():
            command.append('--resume')
        handle = (a.out/(mode+'.log')).open('a')
        process = subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        processes.append((mode,process,handle))
    failures=[]
    for mode,process,handle in processes:
        if process.wait():
            failures.append(mode)
        handle.close()
    if failures:
        raise RuntimeError('Failed training arms: '+', '.join(failures))
    (a.out/'run_complete.json').write_text(json.dumps(dict(steps=a.steps,arms=['state','publication']))+'\n')
    print('Both training arms completed.',flush=True)


if __name__ == '__main__':
    main()
