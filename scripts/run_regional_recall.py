"""Resume the matched mixed-task continuations from one learned recall parent."""
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
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/regional_recall')
    parser.add_argument('--steps', type=int, default=320)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parent = args.out/'parent.pt'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if parent.exists():
        if digest(parent) != digest(args.parent):
            raise ValueError('Existing study has a different parent.')
    else:
        shutil.copy2(args.parent, parent)
    for layout in ('distinct', 'pooled'):
        out = args.out/layout
        command = [sys.executable, '-m', 'experiments.text_regions.mixed_recall_train',
            '--data', str(args.data.resolve()), '--out', str(out.resolve()),
            '--steps', str(args.steps), '--recall-layout', layout, '--eval-every', '80', '--final-eval']
        if (out/'last.pt').exists():
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
            if (saved['raw_sha256'] != digest(args.data)
                    or saved['initialization']['sha256'] != digest(parent)
                    or saved['config']['recall_layout'] != layout
                    or saved['protocol']['steps'] != args.steps):
                raise ValueError('Existing run does not match the requested study.')
            if saved['step'] == args.steps and (out/'evaluation.json').exists():
                continue
            command += ['--resume', str((out/'last.pt').resolve())]
            if saved['step'] == args.steps:
                command += ['--evaluate-only']
        else:
            command += ['--init-from', str(parent.resolve())]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
