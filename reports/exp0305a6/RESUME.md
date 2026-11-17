# Resume the completed A6 evidence

A6 development failed. Keep qualified public E8 and all A1–A6 audits closed.
Read the report, `DIAGNOSIS.md` and the complete `A7_REGISTRATION.md` before
new research. The fixed A6 campaign is finished; continuing its checkpoints
would consume new research bytes and is not part of its 20.25M-byte budget.

## Archive contents

The ordinary `SHoEbox_A6_Code_and_Evidence.tar.xz` contains:

- `SHoEbox/`: the published source, tests, registrations and compact reports.
- `source_history.bundle`: complete reachable Git history, including the
  pre-fit implementation/source-data-lock commit `1ddb724` and final publication.
- `baseline/`: original main `803dc86e563ff048866023bb536830ce0f28ea98` source.
- `a6_work/`: raw world manifests, source/data/receipt locks, all 18 model/cursor
  pairs, full evaluation traces, process costs and passing/failed logs.
- `PUBLISHED_COMMIT.txt`, `publication.json`: verified publication identity.
- `SHA256SUMS.json`: SHA-256 for every other file in this archive.

No compiled binaries are included. Checkpoints and generated datasets remain
outside the source repository. `a6_work/a6/{1,7,19}/` holds each arm's final
`{e8,a5,a6}.qr` plus matching `.cursor.json`, and each warmup
`{e8,a5,a6}.warmup.qr` plus matching `.warmup.cursor.json`.
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

The source export is immediately usable. Git-aware historical verification can
use a fresh clone of the included bundle alongside it:

```bash
git clone source_history.bundle SHoEbox_git
cd SHoEbox_git
git rev-parse HEAD
python -m experiments.record_presence_cost.verify_evidence \
  --workspace ../a6_work --out ../verification_recheck.json
```

Compare HEAD with `../PUBLISHED_COMMIT.txt`. Verification reads saved evidence;
it does not fit or open an audit. Historical checks use `../baseline`.

Recorded environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu for
historical Python tests, g++ 14.2, Linux x86-64. Matplotlib generates figures.
See `a6_work/environment.json` and each process receipt for exact commands.
Native adapters compile their shared objects into temporary directories.
FMA remains disabled. ASan/UBSan passed; leak checking was explicitly disabled.

## Exact cursor continuation

The outer cursor carries episode/order/phase/finish position in addition to the
model's unfinished stream state. Do not reset, finish, replay warmup or rebuild
the curriculum when loading an interrupted pair. This demonstration explicitly
charges 1,000 new bytes and writes to new paths; it was not executed as an extra fit:

```python
import json
from pathlib import Path
from experiments.record_presence_cost.backend import Model
from experiments.record_presence_cost.stream import advance

p = Path('../a6_work/a6/7/a6.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a6') as model:
    cursor, loss = advance(model, cursor, 1000)
    Path('../continued.qr').write_bytes(model.dumps())
    Path('../continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

For matched E8/A5, load that arm's own pair with its explicit policy and this
same A6 campaign driver. Older experiments retain their own original drivers.
All 18 saved pairs passed 1,000 uninterrupted versus 371 + save/load +629 bytes;
their 36,000 discarded correctness bytes are separate from fitting.

## Reproduction and next research

`experiments/record_presence_cost/README.md` lists the completed campaign commands.
Its fitting driver deliberately refuses reused output paths. A full reproduction
costs another 20.25M bytes and must keep separate write-once receipts. Do not
silently count a repeated campaign as free correctness work.

A7 requires new sources, fresh excluded-world folds and passing pre-fit receipts;
it cannot reuse A6's fitted models or warmups. Preserve every original measurement
and threshold. The user explicitly authorizes completed verified work, including
failure, to be committed and pushed directly to main without another permission
request. Preserve unrelated changes and remote history; never force-push.
