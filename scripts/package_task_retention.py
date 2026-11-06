"""Package measured retention runs, full audit traces and resumable checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    entries = {}
    def add(path, name=None):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name or path.relative_to(ROOT).as_posix()] = path
    for cohort, arms in [('pilot', ('legacy', 'conditioned', 'competitive', 'no_credit', 'fresh')),
                         ('continuation', ('conditioned', 'competitive', 'fresh'))]:
        for arm in arms:
            folder = ROOT/'artifacts/task_retention'/cohort/arm
            for name in ('best.pt', 'last.pt', 'invocation.json', 'history.jsonl', 'evaluation.json',
                         'test_traces.jsonl', 'long_delay_traces.jsonl', 'final_test_traces.jsonl',
                         'final_long_delay_traces.jsonl', 'query_audit/summary.json', 'query_audit/queries.jsonl'):
                add(folder/name)
    for name in ('best.pt', 'history.jsonl', 'invocation.json'):
        add(ROOT/'artifacts/task_memory/pilot_v2/integrated'/name)
    for path in (ROOT/'artifacts/task_memory/query_audit_baseline').rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl'):
            add(path)
    for path in (ROOT/'experiments/text_regions/results/task_retention').rglob('*'):
        if path.is_file():
            add(path)
    for name in ('regression.log', 'final_focused.log', 'recovery_checks.log'):
        add(ROOT/'artifacts/task_retention'/name)
    for name in ('SHoEbox_Task_Retention_Report.md', 'SHoEbox_Task_Retention_Report.pdf',
                 'SHoEbox_Task_Retention_Learning_Curves.png'):
        add(args.out/name, 'reports/'+name)
    add(ROOT/'docs/SHoEbox_Task_Retention_Protocol.md')
    manifest = dict(source_commit=commit, source_url='https://github.com/JRowe47/SHoEbox/commit/'+commit,
        files=[dict(path=name, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for name, path in sorted(entries.items())])
    readme = '''# SHoEbox retention artifacts

Extract this archive into the matching source checkout to restore the original
artifacts/ paths. All eight arms include best and final resumable checkpoints,
complete evaluation traces and whole-query interventions. The common parent is
included for reproducing the continuation cohort. Each checkpoint records model
weights, AdamW, stream cursor, recurrent/publication state, eligibility and RNG.

Source commit: COMMIT

Re-evaluate a run:

```bash
python -m experiments.text_regions.task_memory_evaluate --run artifacts/task_retention/continuation/competitive
python -m experiments.text_regions.task_query_audit --checkpoint artifacts/task_retention/continuation/competitive/best.pt --out artifacts/task_retention/continuation/competitive/query_audit --count 16 --fresh-control
python scripts/summarize_task_retention.py --continuation artifacts/task_retention/continuation
```

Continue its final checkpoint with the same training settings:

```bash
python -m experiments.text_regions.task_memory_train --out artifacts/task_retention/continuation/competitive --resume artifacts/task_retention/continuation/competitive/last.pt --steps 1000 --batch 16 --context 16 --seed 29 --data-seed 229 --eval-every 100 --eval-count 8 --stream-preset retention --memory-aware-write --bootstrap-empty --write-mode competitive --final-eval
```

Use --init-from only to start a separate study with fresh optimizer and runtime
state; --resume restores exact continuation. The experiment README contains all
matched-arm commands. Runtime provenance and byte budgets are in invocation.json,
history.jsonl and the report. Manifest hashes cover every included file.
'''.replace('COMMIT', commit)
    archive = args.out/'SHoEbox_Task_Retention_Artifacts.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(entries.items()):
            z.write(path, name)
        z.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')
        z.writestr('README.md', readme)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(entries)+2
    source = args.out/'SHoEbox_Task_Retention_Source.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--output='+str(source.resolve()), commit],
                   cwd=ROOT, check=True)
    print(json.dumps(dict(commit=commit, artifacts=str(archive), bytes=archive.stat().st_size,
                          files=len(entries)+2, source=str(source))))


if __name__ == '__main__':
    main()
