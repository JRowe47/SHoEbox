# Resume the completed A8 evidence

A8 development failed. Keep qualified public E8 and all A1–A8 audits closed.
Read the report, `DIAGNOSIS.md` and the complete `A9_REGISTRATION.md` before
new research. The fixed A8 campaign is finished; continuing its checkpoints
would consume new research bytes and is not part of its 20.25M-byte budget.

## Archive contents

The ordinary `SHoEbox_A8_Code_and_Evidence.tar.xz` contains:

- `SHoEbox/`: the published source, tests, registrations and compact reports.
- `source_history.bundle`: complete reachable Git history, including both original local and published histories, with the
  pre-fit implementation/source-data-lock commit `d622eb08e4ab5ee72315c739f5439255665b373f` and final publication.
- `baseline/`: original main `3487578a98198e19de833e118cf0d6c5f6b5c0f6` source.
- `a8_work/`: raw world manifests, source/data/receipt locks, all 18 model/cursor
  pairs, full evaluation traces, process costs and passing/failed logs.
- `PUBLISHED_COMMIT.txt`, `publication.json`: verified publication identity.
- `SHA256SUMS.json`: SHA-256 for every other file in this archive.

No compiled binaries are included. Checkpoints and generated datasets remain
outside the source repository. `a8_work/a8/{1,7,19}/` holds each arm's final
`{e8,a6,a8}.qr` plus matching `.cursor.json`, and each warmup
`{e8,a6,a8}.warmup.qr` plus matching `.warmup.cursor.json`.
Never pair a cursor with another seed, phase or policy.

## Verify and reconstruct

Extract into a new directory. From that directory, verify every member:

```python
import hashlib, json
from pathlib import Path
root = Path('.')
for name, expected in json.loads((root/'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('Every archived member verifies')
```

`a8_work/pre_fit_worlds.json` preserves the
explicit pre-fit world identities, and `frozen_diagnosis.lock.json` covers every
diagnostic checkpoint before and after tracing. The latter supplements the
inherited survivor tracer's narrower A6-only hash glob without changing locked
source or model execution.

The source export is immediately usable. Git-aware historical verification can
use a fresh clone of the included bundle alongside it:

```bash
git clone source_history.bundle SHoEbox_git
cd SHoEbox_git
git rev-parse HEAD
python -m experiments.balanced_record_context.verify_evidence \
  --workspace ../a8_work --out ../verification_recheck.json
```

Compare HEAD with `../PUBLISHED_COMMIT.txt`. Read `PUBLICATION.md` for the
API transport provenance and the preserved original pre-fit commit object. Verification reads saved evidence;
it does not fit or open an audit. Historical checks use `../baseline`.

Recorded environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu for
historical Python tests, g++ 14.2, Linux x86-64. Matplotlib generates figures.
See `a8_work/environment.json` and each process receipt for exact commands.
Native adapters compile their shared objects into temporary directories.
FMA remains disabled. ASan/UBSan passed; leak checking was explicitly disabled.

## Exact cursor continuation

The outer cursor carries explicit model/curriculum policy, episode/order/phase/finish position in addition to the
model's unfinished stream state. Do not reset, finish, replay warmup or rebuild
the curriculum when loading an interrupted pair. This demonstration explicitly
charges 1,000 new bytes and writes to new paths; it was not executed as an extra fit:

```python
import json
from pathlib import Path
from experiments.balanced_record_context.backend import Model
from experiments.balanced_record_context.stream import advance

p = Path('../a8_work/a8/7/a8.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a8') as model:
    cursor, loss = advance(model, cursor, 1000)
    Path('../continued.qr').write_bytes(model.dumps())
    Path('../continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

For matched E8/A6, load that arm's own pair with its explicit policy and this
same A8 campaign driver. Older experiments retain their own original drivers.
All 18 saved pairs passed 1,000 uninterrupted versus 371 + save/load +629 bytes;
their 36,000 discarded correctness bytes are separate from fitting.

## Reproduction and next research

`experiments/balanced_record_context/README.md` lists the completed campaign commands.
Its fitting driver deliberately refuses reused output paths. A full reproduction
costs another 20.25M bytes and must keep separate write-once receipts. Do not
silently count a repeated campaign as free correctness work.

A9 requires new sources, fresh excluded-world folds and passing pre-fit receipts;
it cannot reuse A8's fitted models or warmups. Preserve every original measurement
and threshold. The user explicitly authorizes completed verified work, including
failure, to be committed and pushed directly to main without another permission
request. Preserve unrelated changes and remote history; never force-push.
