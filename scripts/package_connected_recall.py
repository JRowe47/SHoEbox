"""Package resumable checkpoints, corpus and measured evidence for download."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/connected_recall')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    a.root = a.root.resolve()
    required = [ROOT/'artifacts/state_credit/state/best.pt', ROOT/'data/tiny_shakespeare.txt',
                ROOT/'experiments/text_regions/results/connected_recall/summary.json']
    for arm in ('connected','cut_16'):
        required += [a.root/arm/name for name in ('best.pt','last.pt','protocol.json','initial.json','history.jsonl')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f'Missing required artifacts: {missing}')
    paths = set(required)
    paths.add(a.root/'activation_comparison.json')
    paths.update(a.root/f'benchmark_{mode}'/'benchmark.json' for mode in ('full','flat'))
    for directory in (a.root, a.root/'eval_connected', a.root/'eval_cut_16'):
        evaluation = directory/'evaluation.json'
        if not evaluation.exists():
            continue
        paths.add(evaluation)
        signatures = {row['signature'] for row in json.loads(evaluation.read_text()).values()}
        paths.update(path for signature in signatures for path in directory.glob(f'{signature}_*_traces.jsonl'))
    entries = sorted((str(path.relative_to(ROOT)),path) for path in paths)
    manifest = dict(commit=a.commit, repository='https://github.com/JRowe47/SHoEbox',
        files=[dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for name,path in entries])
    readme = f'''# SHoEbox connected recall checkpoints

Code: https://github.com/JRowe47/SHoEbox/commit/{a.commit}

Extract into a fresh checkout of this revision. This archive contains the
selected update-32 parent, both selected and final continuation checkpoints,
corpus, protocols, histories, held-out predictions and resource measurements.
MANIFEST.json records every included file's size and SHA-256.

The completed comparison uses 16 updates, four streams, 128-byte optimizer
windows, 16-byte execution chunks and the existing ordinary language/task
grammar. All weights and AdamW moments transfer from the parent. Each arm has
independent saved source states, credit histories, cursors and RNG state.

Reproduce in new output directories:

    python scripts/run_connected_recall.py --arm connected --out artifacts/reproduction/connected
    python scripts/run_connected_recall.py --arm cut_16 --out artifacts/reproduction/cut_16
    python scripts/evaluate_connected_recall.py --root artifacts/reproduction

Use --resume with the same protocol to recover an interrupted run. These
checkpoints have completed their fixed budget; a new training budget is a new
continuation, not an exact resume of the completed comparison.

    python -m experiments.text_regions.content_credit_train \\
      --data data/tiny_shakespeare.txt \\
      --parent artifacts/connected_recall/connected/best.pt \\
      --out artifacts/connected_recall_next --horizon 384 --batch 4 \\
      --steps 16 --checkpoint-segments --audit-coverage --data-seed 683

The optional recomputation flag reduces activation memory while preserving
the connected optimizer window. Details: docs/connected_recall.md.
'''
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name,path in entries:
            archive.write(path,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        archive.writestr('CHECKPOINT_README.md',readme)
    with zipfile.ZipFile(a.out) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Archive CRC failure: {bad}')
        for entry in manifest['files']:
            if hashlib.sha256(archive.read(entry['path'])).hexdigest() != entry['sha256']:
                raise ValueError(f'Archive digest mismatch: {entry["path"]}')
    print(json.dumps(dict(path=str(a.out),bytes=a.out.stat().st_size,files=len(entries)+2)),flush=True)


if __name__=='__main__':
    main()
