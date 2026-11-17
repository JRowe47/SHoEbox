# EXP03/05-A6: fixed record-presence negative cost

Read `reports/exp0305a5/A6_REGISTRATION.md` in full and this campaign's
`reports/exp0305a6/EXECUTION_LOCK.md` / `DIAGNOSIS_PLAN.md`.
Only four record-phase negative costs change 1→4; positive costs remain 4.
The backend inherits A5 and changes one checked line plus checkpoint magic
`0x3641353050584553`. No arithmetic reuse, projection cache, parameter, slot,
optimizer state or inference change is introduced. FMA is disabled.
Generic/local separation and every historical checkpoint remain protected.

All outputs below belong outside git. Meter commands with
`python -m experiments.record_presence_cost.meter PREFIX COMMAND ...`.

```bash
python -m unittest experiments.record_presence_cost.test_research experiments.record_presence_cost.test_objective -v
python -m experiments.record_presence_cost.check_native
python -m experiments.record_presence_cost.check_policy_isolation
python -m experiments.record_presence_cost.disclose /path/work
# Complete runtime/public/historical suites and passing process receipts first.
python -m experiments.record_presence_cost.prefit --workspace /path/work
# Commit the passing source/data lock before fitting.
python -m experiments.record_presence_cost.run --out /path/work/a6 --prefit /path/work/prefit.json
python -m experiments.record_presence_cost.extended --runs /path/work/a6 --out /path/work/extended
python -m experiments.record_presence_cost.verify_cursors --runs /path/work/a6 --out /path/work/cursors.json
python -m experiments.record_presence_cost.update_metrics --runs /path/work/a6 --out /path/work/update_metrics.json
# All other CPU jobs must finish before timing.
python -m experiments.record_presence_cost.benchmark --runs /path/work/a6 --out /path/work/benchmark.json
```

Fresh matched E8/A5/A6, seeds 1/7/19, exactly 20.25M fitting bytes. Runs are
write-once; no pilot, inherited warmup or replay. Checkpoints retain exact outer
cursors. The completed A6 result and next preregistration are in the report.
To resume a saved pair for separately charged new research:

```python
import json
from pathlib import Path
from experiments.record_presence_cost.backend import Model
from experiments.record_presence_cost.stream import advance
p = Path('/path/work/a6/7/a6.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a6') as model:
    cursor, loss = advance(model, cursor, 1000)
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

Use this same campaign cursor for its matched E8/A5, each with its own policy;
never reset/finish/replay at an arbitrary interruption. All older campaigns
retain their original drivers. Numerical tests use test-only envelope conversion
for identical-weight checks; production cross-policy loaders reject conversion.

Elkan, IJCAI 2001, https://cseweb.ucsd.edu/~elkan/rescale.pdf, grounds distinct
error costs. Four is the project's preregistered factor, not a paper result.
A5's fixed-code hybrid and A4 paired chronological selection remain exact.
