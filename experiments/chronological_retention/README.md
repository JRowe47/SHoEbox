# EXP03/05-A3: continuous chronological matching

Isolated research policy implementing the exact fixed rule in
`reports/exp0305a2/A3_REGISTRATION.md`; execution details are in
`reports/exp0305a3/EXECUTION_LOCK.md`. Public E8 and historical sources are unchanged.

For every recognized record phase, A3 uses float32 query/key squared distance,
then float64 `log sigmoid(4 * (7.84 - distance))` and chronological remaining
mass. Newest occurrences are visited first, with stable slot/phase ties. The
two largest log weights are normalized for the shared paired-value mixture.
One recognized candidate gets all mass; none uses the complete historical A2
presence fallback. Only the registered strict remaining-log-mass bound stops
older projections. The optimized scan equals the exhaustive scan exactly.

No fitted weights, persistent state, optimizer moments, parser or byte equality
are added. `A3Scratch` contains bounded per-call arrays and cached projections;
native checks print its measured byte size. All local labels remain optional,
observed-prefix training supervision; generic byte objectives freeze the local
heads. Inference always produces a normalized full 256-byte distribution.

A3 checkpoint magic is `0x3341353050584553`, nested v19. E8/A1/A2/E7 loaders
reject A3; no implicit conversion. The matched E8/A2 controls use their exact
historical numerical source. Every fit output belongs outside git.

```bash
python -m unittest experiments.chronological_retention.test_research -v
python -m experiments.chronological_retention.check_native
# Run required public, historical and sanitizer checks with process receipts.
python -m experiments.chronological_retention.prefit --workspace /path/work
python -m experiments.chronological_retention.run --out /path/work/a3 --prefit /path/work/prefit.json
python -m experiments.chronological_retention.extended --runs /path/work/a3 --out /path/work/extended
python -m experiments.chronological_retention.verify_cursors --runs /path/work/a3 --out /path/work/cursors.json
python -m experiments.chronological_retention.update_metrics --runs /path/work/a3 --out /path/work/update_metrics.json
# Stop other local CPU work before isolated timing.
python -m experiments.chronological_retention.benchmark --runs /path/work/a3 --out /path/work/benchmark.json
```

Reproduction charges another 20.25M research bytes. The runner refuses existing
destinations, requires passing receipts and unchanged source/data locks, and
never evaluates the reserved audit. All 18 warmup/final snapshot-cursor pairs
are retained. Resume an actual pair without replay/reset/finish at interruption:

```python
import json
from pathlib import Path
from experiments.chronological_retention.backend import Model
from experiments.chronological_retention.stream import advance
p = Path('/path/work/a3/7/a3.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a3') as model:
    cursor, loss = advance(model, cursor, 1000)  # new, explicitly charged research
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

For this campaign's matched E8/A2 snapshots use this same A3 cursor driver and
their own saved policy. Earlier A2 cursors retain their original driver. The
new schema records shuffle epoch, both RNGs, changed key, partial episode and
finish schedule. No completed failed candidate is silently extended for tuning.
