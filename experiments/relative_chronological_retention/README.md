# EXP03/05-A4: query-relative chronological matching

Isolated research policy implementing the exact fixed rule in
`reports/exp0305a3/A4_REGISTRATION.md`; execution details are in
`reports/exp0305a4/EXECUTION_LOCK.md`. Public E8 and historical sources are unchanged.

For every positively recognized record phase A4 computes its key and float32
query distance, then finds the minimum across all ages. Float64
`4 * (double(d_min) + 1.96 - double(d_i))` feeds stable log-sigmoid chronological
weights. Newest born first, stable slot/phase ties; normalize the top two weights
for the common paired-value mixture. One candidate gets all mass; none uses the
complete historical A2/A3 fallback. All keys are projected before any bound can
skip remaining scalar weights. Independent exhaustive native and Python
references verify the optimized implementation before fitting.

No fitted weights, persistent state, optimizer moments, parser or byte equality
are added. `A4Scratch` contains bounded per-call arrays and cached projections;
native checks print its measured byte size. All local labels remain optional,
observed-prefix training supervision; generic byte objectives freeze the local
heads. Inference always produces a normalized full 256-byte distribution.

A4 checkpoint magic is `0x3441353050584553`, nested v19. E8/A1/A2/A3/E7 loaders
reject A4; no implicit conversion. The matched E8/A3 controls use their exact
historical numerical source. Every fit output belongs outside git.

```bash
python -m unittest experiments.relative_chronological_retention.test_research -v
python -m experiments.relative_chronological_retention.check_native
# Run required public, historical and sanitizer checks with process receipts.
python -m experiments.relative_chronological_retention.prefit --workspace /path/work
python -m experiments.relative_chronological_retention.run --out /path/work/a4 --prefit /path/work/prefit.json
python -m experiments.relative_chronological_retention.extended --runs /path/work/a4 --out /path/work/extended
python -m experiments.relative_chronological_retention.verify_cursors --runs /path/work/a4 --out /path/work/cursors.json
python -m experiments.relative_chronological_retention.update_metrics --runs /path/work/a4 --out /path/work/update_metrics.json
# Stop other local CPU work before isolated timing.
python -m experiments.relative_chronological_retention.benchmark --runs /path/work/a4 --out /path/work/benchmark.json
```

Reproduction charges another 20.25M research bytes. The runner refuses existing
destinations, requires passing receipts and unchanged source/data locks, and
never evaluates the reserved audit. All 18 warmup/final snapshot-cursor pairs
are retained. Resume an actual pair without replay/reset/finish at interruption:

```python
import json
from pathlib import Path
from experiments.relative_chronological_retention.backend import Model
from experiments.relative_chronological_retention.stream import advance
p = Path('/path/work/a4/7/a4.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a4') as model:
    cursor, loss = advance(model, cursor, 1000)  # new, explicitly charged research
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

For this campaign's matched E8/A3 snapshots use this same A4 cursor driver and
their own saved policy. Earlier A2/A3 cursors retain their original drivers. The
new schema records shuffle epoch, both RNGs, changed key, partial episode and
finish schedule. No completed failed candidate is silently extended for tuning.
