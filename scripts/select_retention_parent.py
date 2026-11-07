"""Freeze the validation-selected parent before competing-fact training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/memory_selectivity'))
    a = p.parse_args()
    candidates = []
    for name in ('retention','retention_low'):
        directory = a.root/name
        last = torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
        best = torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        if last['step'] != last['protocol']['steps']:
            raise ValueError(f'{name} has not completed its declared budget.')
        initial = json.loads((directory/'initial.json').read_text())
        history = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
        scores = [(0,initial['acquisition']['nll'])]+[(r['step'],r['validation']['acquisition']['nll'])
            for r in history if 'validation' in r]
        selected = min(scores,key=lambda row:row[1])
        if best['step'] != selected[0] or best['best'] != selected[1]:
            raise ValueError(f'{name} selected checkpoint does not match validation evidence.')
        candidates.append(dict(arm=name, selected_update=best['step'], validation_nll=best['best'],
            lr=best['protocol']['lr'], checkpoint_sha256=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()))
    chosen = min(candidates,key=lambda row:row['validation_nll'])
    target = a.root/'retention_parent.pt'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != chosen['checkpoint_sha256']:
        raise ValueError('Existing parent differs; do not overwrite an active lineage.')
    shutil.copyfile(a.root/chosen['arm']/'best.pt', target)
    decision = dict(rule='Lowest selected validation NLL; 0.003 wins exact ties',
        selected=chosen, candidates=candidates)
    (a.root/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
