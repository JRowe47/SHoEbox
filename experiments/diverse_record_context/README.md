# EXP03/05-A7: diverse record-context curriculum

Read the full `reports/exp0305a6/A7_REGISTRATION.md` and this campaign's
`reports/exp0305a7/EXECUTION_LOCK.md` and `DIAGNOSIS_PLAN.md`.
Only the two original record spacers in A7 continuation receive SHA-256-derived
bytes in [160,255]. Length/order/phase/RNG/masks/labels are preserved. Warmup,
E8/A6 controls and evaluation retain PAD. Native arithmetic is exactly A6;
only A7 save/load magic differs. No arithmetic optimization is introduced.

All generated outputs/checkpoints belong outside git. Every command is metered
with `python -m experiments.diverse_record_context.meter PREFIX COMMAND ...`.
The registration requires passing controls, stream hashes and a committed
pre-fit lock before the write-once 20.25M-byte matched E8/A6/A7 campaign.

```bash
python -m experiments.diverse_record_context.check_python /path/work/python_counts.json
python -m experiments.diverse_record_context.check_native
python -m experiments.diverse_record_context.check_policy_isolation
python -m experiments.diverse_record_context.check_streams --out /path/work/streams.json
python -m experiments.diverse_record_context.disclose /path/work
# Also complete public/runtime/historical Python/native suites and receipts.
python -m experiments.diverse_record_context.prefit --workspace /path/work
# Commit passing pre-fit source/data/receipt lock here, before any fit.
python -m experiments.diverse_record_context.run --out /path/work/a7 --prefit /path/work/prefit.json
python -m experiments.diverse_record_context.extended --runs /path/work/a7 --out /path/work/extended
python -m experiments.diverse_record_context.verify_cursors --runs /path/work/a7 --out /path/work/cursors.json
python -m experiments.diverse_record_context.update_metrics --runs /path/work/a7 --out /path/work/update_metrics.json
# Lock final outcomes before the registered frozen diagnostics.
# Isolated throughput must run only after every other CPU job finishes.
python -m experiments.diverse_record_context.benchmark --runs /path/work/a7 --out /path/work/benchmark.json
```

Exact cursor schema `exp0305-a7-cursor-1` records model policy, curriculum,
episode/order/phase/finish position, partial episode, both RNGs and actual
PAD/nuisance/replaced-padding counts. Resume each pair with its own policy:

```python
import json
from pathlib import Path
from experiments.diverse_record_context.backend import Model
from experiments.diverse_record_context.stream import advance
p = Path('/path/work/a7/7/a7.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a7') as model:
    cursor, loss = advance(model, cursor, 1000)  # 1,000 newly charged bytes
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

Do not reset, finish or replay at an arbitrary interruption. Every arm/seed
uses its own independent warmup; the controls use the same A7 campaign driver
with their original model policies and PAD curriculum. Earlier campaigns keep
their original sources and cursors. Cross-policy imports are rejected.

Paired addressing: Miller et al., EMNLP 2016,
https://aclanthology.org/D16-1147/. Multi-query recall measurement: Arora et al.,
ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html.
The exact spacer curriculum is this project's registered hypothesis.
