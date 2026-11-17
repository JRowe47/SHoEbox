# EXP03/05-A8: phase-balanced PAD/diverse curriculum

Completed: A8 fails development; E8 stays public and all A1–A8 audits stay
closed. Read the [measured report](../../reports/exp0305a8/SHoEbox_EXP0305_A8_Report.md),
[diagnosis](../../reports/exp0305a8/DIAGNOSIS.md),
[exact resume](../../reports/exp0305a8/RESUME.md) and
[next A9 registration](../../reports/exp0305a8/A9_REGISTRATION.md).
A9 is registered only, not implemented or fitted. All commands below describe
the completed A8 campaign; a repetition consumes a new separately charged budget.

Read the full `reports/exp0305a7/A8_REGISTRATION.md` and this campaign's
`reports/exp0305a8/EXECUTION_LOCK.md` and `DIAGNOSIS_PLAN.md`.
In alternating 64-episode blocks beginning with PAD, the two original spacers
in diverse A8 continuation episodes receive the exact A7 v7 SHA-256-derived
bytes in [160,255]. Length/order/phase/RNG/masks/labels are preserved. Warmup,
E8/A6 controls and evaluation retain PAD. Native arithmetic is exactly A6;
only A8 save/load magic differs. No arithmetic optimization is introduced.

All generated outputs/checkpoints belong outside git. Every command is metered
with `python -m experiments.balanced_record_context.meter PREFIX COMMAND ...`.
The registration requires passing controls, stream hashes and a committed
pre-fit lock before the write-once 20.25M-byte matched E8/A6/A8 campaign.

```bash
python -m experiments.balanced_record_context.check_python /path/work/python_counts.json
python -m experiments.balanced_record_context.check_native
python -m experiments.balanced_record_context.check_policy_isolation
python -m experiments.balanced_record_context.check_streams --out /path/work/streams.json
python -m experiments.balanced_record_context.disclose /path/work
# Also complete public/runtime/historical Python/native suites and receipts.
python -m experiments.balanced_record_context.prefit --workspace /path/work
# Commit passing pre-fit source/data/receipt lock here, before any fit.
python -m experiments.balanced_record_context.run --out /path/work/a8 --prefit /path/work/prefit.json
python -m experiments.balanced_record_context.extended --runs /path/work/a8 --out /path/work/extended
python -m experiments.balanced_record_context.verify_cursors --runs /path/work/a8 --out /path/work/cursors.json
python -m experiments.balanced_record_context.update_metrics --runs /path/work/a8 --out /path/work/update_metrics.json
# Lock final outcomes before the registered frozen diagnostics.
# Isolated throughput must run only after every other CPU job finishes.
python -m experiments.balanced_record_context.benchmark --runs /path/work/a8 --out /path/work/benchmark.json
# With all required process receipts and the prior final-fit lock present:
python reports/exp0305a8/freeze_evaluation.py /path/work
# Only then execute the registered frozen diagnostics, with no refit.
```

Exact cursor schema `exp0305-a8-cursor-1` records model policy, curriculum,
episode/order/phase/finish position, partial episode, both RNGs and actual
PAD/nuisance/replaced-padding counts. Resume each pair with its own policy:

```python
import json
from pathlib import Path
from experiments.balanced_record_context.backend import Model
from experiments.balanced_record_context.stream import advance
p = Path('/path/work/a8/7/a8.qr')
cursor = json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(), policy='a8') as model:
    cursor, loss = advance(model, cursor, 1000)  # 1,000 newly charged bytes
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

Do not reset, finish or replay at an arbitrary interruption. Every arm/seed
uses its own independent warmup; the controls use the same A8 campaign driver
with their original model policies and PAD curriculum. Earlier campaigns keep
their original sources and cursors. Cross-policy imports are rejected.

Paired addressing: Miller et al., EMNLP 2016,
https://aclanthology.org/D16-1147/. Multi-query recall measurement: Arora et al.,
ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html.
The exact spacer curriculum is this project's registered hypothesis.
