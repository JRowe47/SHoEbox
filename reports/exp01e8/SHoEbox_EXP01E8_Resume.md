# SHoEbox EXP01-E8 reproduction and exact continuation

Extract `SHoEbox_EXP01E8_Code_and_Evidence.zip` into a new directory. Its root
contains `SHoEbox/` source, the five `exp01e8-*` run directories, two pinned
corpora, manifests, and `historical/SHoEbox_EXP01E_Code_and_Evidence.zip`.
`PUBLISHED_COMMIT.txt` identifies the verified main commit. `SHA256SUMS.json`
checks every other archive member. Never overwrite qualified snapshots.

## Environment and policy identity

Run commands from the extracted `SHoEbox/` directory. Use Python 3 with NumPy,
Matplotlib for plots, and a C++17 compiler (`g++`) for native kernels. Exact
versions and flags are in root `e8-environment.json`. Libraries compile locally
into a temporary cache on first use; compiled binaries are not distributed.
The source requires a Linux CPU environment for affinity and process accounting.

E8 is `experiments.joint_cold_gate.adapter.Adapter`; it uses outer magic
`0x38455041514853`, preserving E7's separate magic and all historical policies.
Do not load E8 research snapshots with `shoe_multiscale.Model`. The public
runtime remains v19 pending versioned integration. The checkpoint manifest
names the exact loader for every snapshot.

| Snapshot group | Loader / cursor helper | Pinned corpus |
| --- | --- | --- |
| E8 pilot, local/generic/no-local formal arms | `joint_cold_gate.adapter.Adapter` / `joint_cold_gate.stream.advance` | Synthetic E2-family train worlds |
| Native/frozen-D3 formal controls | `joint_readout.adapter.Adapter` / `joint_readout.stream.advance` | Same synthetic train worlds |
| Any `.retained.qr` | Manifest's loader / `joint_cold_gate.resume.language_advance` | `exp01e-data` |
| Fresh generic language | Manifest's loader / `joint_cold_gate.resume.language_advance` | `exp01e7-data` |
| Fresh generic copy | Manifest's loader / `joint_cold_gate.resume.task_advance` | `fresh_cases` training fold |
| Fresh generic reconstruction | Manifest's loader and saved decoder / same task helper | `fresh_cases` training fold |

The original retention corpus and fresh generic corpus are different. Their
hashes, UTF-8 bytes, document boundaries and original manifests are included.
Do not regenerate either with an inherited preparation helper. Binding labels
come from observed prefixes of training fixtures; generic streams supply none.

## Verify integrity without fitting

```bash
cd SHoEbox
python - <<'PY'
import hashlib, json
from pathlib import Path
root=Path('..')
for row in json.loads((root/'e8-checkpoint-manifest.json').read_text())['checkpoints']:
    assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest()==row['sha256']
    assert hashlib.sha256((root/row['cursor']).read_bytes()).hexdigest()==row['cursor_sha256']
print('77 checkpoint/cursor pairs intact')
PY
```

Native/Python checks and discarded-copy cursor verification:

```bash
python -m unittest experiments.joint_cold_gate.test_adapter experiments.joint_cold_gate.test_request experiments.joint_cold_gate.test_prior -v
python -m experiments.joint_cold_gate.check_native
python -m experiments.joint_cold_gate.check_request
python -m experiments.joint_cold_gate.verify_cursors --workspace .. --out /tmp/e8-cursor-recheck.json
```

The last command uses 1,000 updates versus 371 + serialization +629 per pair
on discarded copies, never extending selected snapshots. It also verifies
separate reconstruction decoder arrays. All 77 pairs passed this check.

## Example: continue the selected binding snapshot

This illustrative 1,000-byte extension is a **new fitting budget**, not part
of the published qualification. It completes the saved partial episode first.
No reset, finish or replay is inserted at the arbitrary resume point.

```python
import json
from pathlib import Path
from experiments.joint_cold_gate.adapter import Adapter
from experiments.joint_cold_gate.stream import advance

root=Path('..')
path=root/'exp01e8-formal/7/local_gate.qr'
cursor=json.loads(path.with_suffix('.cursor.json').read_text())
out=root/'continued-e8'
out.mkdir(exist_ok=False)
with Adapter.loads(path.read_bytes()) as model:
    cursor,loss=advance(model,cursor,1000)
    (out/'binding.qr').write_bytes(model.dumps())
    (out/'binding.cursor.json').write_text(json.dumps(cursor,indent=2)+'\n')
```

At the published selected binding cursor, the 2M continuation has completed
11,936 episodes and is inside episode 11,937: dataset index 2031, spacers `[0,3]`,
boundary offset 29, 48 bytes consumed. Its 250K native warmup is separate in
training accounting; do not replay that warmup. The cursor preserves both RNGs,
remaining shuffle order, current episode and hash chain.

## Example: continue retained language

```python
import json
from pathlib import Path
from experiments.joint_cold_gate.adapter import Adapter
from experiments.joint_cold_gate.resume import language_advance
from experiments.joint_readout.regressions import Corpus

root=Path('..')
path=root/'exp01e8-formal/7/local_gate.retained.qr'
cursor=json.loads(path.with_suffix('.cursor.json').read_text())
out=root/'continued-e8-language'
out.mkdir(exist_ok=False)
with Adapter.loads(path.read_bytes()) as model:
    cursor=language_advance(model,cursor,1000,Corpus(root/'exp01e-data'))
    (out/'retained.qr').write_bytes(model.dumps())
    (out/'retained.cursor.json').write_text(json.dumps(cursor,indent=2)+'\n')
```

The retained cursor has 1M targets, document index 389, 509 targets consumed
in that document, and `at_end=true`. Fresh generic language instead ends at
index 365 /7,441 targets and uses `exp01e7-data`. The helper explicitly begins
the next pinned epoch after an end cursor, preserving document resets and
registered 250K-target finish events. Repeating the corpus is a new epoch and
must be charged; it is not unseen data.

## Copy and reconstruction continuation

Call `resume.load_task(path, loader, seed)` to obtain `(model, decoder)` and
`resume.task_advance(model, decoder, cursor, count, task)`. The selected copy
cursor is at episode index 212, byte 58 of 66; reconstruction is at index 68,
byte 16 of 64. Never discard these partially observed examples. The helper
validates the registered order digest and chains new continuation updates
separately. Use task names `selective_copy` or `compression` exactly.

Copy checkpoints contain only the checked adapter payload. Reconstruction
checkpoints are ZIP containers with `adapter.qr` and `decoder.npz`; save every
array in `decoder.arrays` with `numpy.savez` and preserve both members when
writing the continued checkpoint. Save its matching JSON cursor at the same
time. Do not substitute a fresh decoder. `verify_cursors.py` gives executable
examples of exact interrupted continuation for all three generic tasks.

## Reproduce the fixed fits

The adapter README contains fresh-fit commands for pilot, generic, formal,
1M and 5M campaigns. Use new output directories. The registered research
budget is 106M bytes total including controls and padding. Fixed budgets,
finish events, source hashes and data order are in each `plan.json` and
`train.json`. Model registrations include the initial `diagnose.py` source;
its reporting-only correction is recorded in `diagnostic_correction.json`.
The exact original source is preserved in `source_snapshots/e8_initial_fit.zip`.
No fitting depended on the corrected diagnostic helper.

The E2-family binding and E7 generic audits are **now opened**. Existing
`audit.json` files are immutable qualification evidence. Reproduction of this
fixed result is not a fresh held-out selection opportunity. Any new numerical
revision needs preregistered fresh groups excluding every opened audit group.

Rebuild summaries/plots from the included raw evidence:

```bash
python -m experiments.joint_cold_gate.report --workspace .. --out /tmp/e8-report
python -m experiments.joint_cold_gate.plot --workspace .. --report /tmp/e8-report
```

For historical E1-E7 continuation, extract the nested historical archive
separately and follow its own source and cursor instructions. Never reinterpret
those checkpoints under E8. The qualified next action is public API integration
with numeric/checkpoint parity, as registered in `PUBLIC_INTEGRATION_PLAN.md`.
