# EXP03/05-A2: query selection before truncation

Separate research implementation. Public E8, A1, E7 and historical constructors
and checkpoints are unchanged. Read `reports/exp0305/A2_REGISTRATION.md`,
`A2_FIXTURE_AMENDMENT.md` and `reports/exp0305a2/EXECUTION_LOCK.md` together.
The three-value overwrite fixture is mandatory; the A1 binary fixture remains
historical diagnostic evidence with its original costs.

A2 keeps A1 admission and ranks positively recognized record phases by learned
query/key squared distance before keeping two. Exact distance ties use stored
occurrence, then slot/phase order. The historical presence ranking fills missing
candidates without duplicates. Paired value decoding, sigmoid attention, cold
prior, objectives and state capacity remain unchanged. The implementation adds
zero fitted weights or persistent sequence state; its local automatic stack
scratch is four 24-byte candidate records, one temporary candidate and a source vector. No inference
parser, byte comparison, source oracle or external store is used. Local labels
are optional training-only observed-window supervision. Generic byte objectives
leave those heads frozen; all predictions retain the full 256-byte distribution.

```bash
python -m unittest experiments.query_first_retention.test_research -v
python -m experiments.query_first_retention.check_native
# The driver requires passing pre-fit receipts and identical locked source:
python -m experiments.query_first_retention.prefit --workspace /path/work
python -m experiments.query_first_retention.run --out /path/work/a2 --prefit /path/work/prefit.json
python -m experiments.query_first_retention.extended --runs /path/work/a2 --out /path/work/extended
python -m experiments.query_first_retention.verify_cursors --runs /path/work/a2 --out /path/work/cursors.json
python -m experiments.query_first_retention.update_metrics --runs /path/work/a2 --out /path/work/update_metrics.json
# Stop other local CPU jobs first:
python -m experiments.query_first_retention.benchmark --runs /path/work/a2 --out /path/work/benchmark.json
```

All outputs belong outside git. Fresh fitting charges 20.25M bytes, including
all independently repeated E8/A1/A2 warmups and controls. The artifact bundle
contains data worlds, source locks, exact model/cursor pairs and process ledgers.
A2 magic is `0x3241353050584553`; E8/A1 loaders reject it and vice versa.

```python
import json
from pathlib import Path
from experiments.query_first_retention.backend import Model
from experiments.query_first_retention.stream import advance
path=Path('/path/work/a2/7/a2.qr')
cursor=json.loads(path.with_suffix('.cursor.json').read_text())
with Model.loads(path.read_bytes(),policy='a2') as model:
    cursor,loss=advance(model,cursor,1000)  # explicitly charged new extension
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor,indent=2)+'\n')
```

Never reset, finish or replay a warmup at an arbitrary resume boundary. The
cursor preserves partial episodes, RNGs, shuffle epoch, operations, order,
spacers, phase and finish events. Use the same A2 fixture cursor driver for
this campaign's matched `policy='e8'` and `policy='a1'` snapshots; historical A1
cursors still use their original driver.
