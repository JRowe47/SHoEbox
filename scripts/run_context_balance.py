"""Resume the matched contextual-retention study from its completed pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/context_balance')
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--parallel', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    parent = a.out/'parent.pt'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if parent.exists():
        if digest(parent) != digest(a.parent):
            raise ValueError('Existing study has a different parent.')
    else:
        shutil.copy2(a.parent, parent)
    processes = []
    for mode in ('group_clip', 'sum'):
        folder = a.out/mode
        command = [sys.executable, '-m', 'experiments.text_regions.context_balance',
            '--data', str(a.data.resolve()), '--parent', str(parent.resolve()),
            '--out', str(folder.resolve()), '--mode', mode, '--steps', str(a.steps)]
        if (folder/'last.pt').exists():
            saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
            if (saved['raw_sha256'] != digest(a.data) or
                saved['protocol']['parent_sha256'] != digest(parent) or
                saved['protocol']['steps'] != a.steps or saved['protocol']['gradient_mode'] != mode):
                raise ValueError('Existing arm has a different protocol, corpus or parent.')
            if saved['step'] == a.steps and (folder/'evaluation.json').exists():
                continue
            command.append('--resume')
            if saved['step'] == a.steps:
                command.append('--evaluate-only')
        folder.mkdir(exist_ok=True)
        # Logs live in the study root so each new arm starts with an empty dir.
        log = (a.out/(mode+'.log')).open('a')
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        processes.append((mode, process, log))
        if not a.parallel and process.wait():
            raise RuntimeError('Failed arm: '+mode)
    failed = []
    for mode, process, log in processes:
        if process.wait():
            failed.append(mode)
        log.close()
    if failed:
        raise RuntimeError('Failed arms: '+', '.join(failed))
    subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_context_parent.py'),
        '--root', str(a.out.resolve()), '--data', str(a.data.resolve())], cwd=ROOT, check=True)
    (a.out/'run_complete.json').write_text(json.dumps(dict(steps=a.steps, arms=['group_clip','sum']))+'\n')


if __name__ == '__main__':
    main()
