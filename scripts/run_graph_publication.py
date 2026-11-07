"""Matched unified-graph continuation, with and without delayed publication credit."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/graph_publication')
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parent = args.out/'parent.pt'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if parent.exists():
        if digest(parent) != digest(args.parent):
            raise ValueError('Existing study has a different parent.')
    else:
        shutil.copy2(args.parent,parent)
    processes = []
    for arm in ('credit','no_credit'):
        out = args.out/arm
        command = [sys.executable,'-m','experiments.text_regions.graph_mixed_train',
            '--data',str(args.data.resolve()),'--out',str(out.resolve()),'--steps',str(args.steps),'--final-eval']
        if arm == 'no_credit':
            command.append('--no-delayed-credit')
        if (out/'last.pt').exists():
            saved = torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            if saved['raw_sha256'] != digest(args.data) or saved['initialization']['sha256'] != digest(parent):
                raise ValueError('Run corpus or parent mismatch.')
            if saved['protocol']['steps'] != args.steps or saved['protocol']['delayed_credit'] != (arm == 'credit'):
                raise ValueError('Run budget or arm mismatch.')
            if saved['step'] == args.steps and (out/'evaluation.json').exists():
                continue
            command += ['--resume',str((out/'last.pt').resolve())]
            if saved['step'] == args.steps:
                command.append('--evaluate-only')
        else:
            command += ['--init-from',str(parent.resolve())]
        if args.parallel:
            processes.append((arm,subprocess.Popen(command,cwd=ROOT)))
        else:
            subprocess.run(command,cwd=ROOT,check=True)
    failed = [arm for arm,process in processes if process.wait()]
    if failed:
        raise RuntimeError('Failed arms: '+', '.join(failed))


if __name__ == '__main__':
    main()
