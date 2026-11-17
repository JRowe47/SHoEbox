# EXP03/05-A5: fixed local discriminative key reconstruction

Completed, development failed: 90.23% competing, 95.70% three-value overwrite,
63.80% diverse256 and 67.57% native speed. Public E8 remains unchanged and all
A1–A5 audits stay closed. See `reports/exp0305a5/SHoEbox_EXP0305_A5_Report.md`,
`DIAGNOSIS.md`, `RESUME.md` and the unfitted `A6_REGISTRATION.md` there.

Read `reports/exp0305a4/A5_REGISTRATION.md` and
`reports/exp0305a5/EXECUTION_LOCK.md` in full. The isolated adapter changes only
positive query/key local reconstruction. It keeps all 256 fixed codes, float64
softmax/gradients from float32 pre-update projections, and the exact registered
hybrid coefficients. Value/presence/applicability/native updates, A4 inference,
chronology/ties/fallback, paired values, admission, state and E8 prior remain.
Generic byte objectives freeze the local heads. Full 256-byte inference never
uses the codebook. No new fitted parameter, optimizer moment, slot or persistent
projection cache. A5 has its own magic `0x3541353050584553`, nested v19.

```bash
python -m unittest experiments.discriminative_keys.test_research experiments.discriminative_keys.test_objective -v
python -m experiments.discriminative_keys.check_native
python -m experiments.discriminative_keys.check_policy_isolation
# Run public/runtime/historical tests with the required process receipts first.
python -m experiments.discriminative_keys.prefit --workspace /path/work
python -m experiments.discriminative_keys.run --out /path/work/a5 --prefit /path/work/prefit.json
python -m experiments.discriminative_keys.extended --runs /path/work/a5 --out /path/work/extended
python -m experiments.discriminative_keys.verify_cursors --runs /path/work/a5 --out /path/work/cursors.json
python -m experiments.discriminative_keys.update_metrics --runs /path/work/a5 --out /path/work/update_metrics.json
# Stop all other local CPU work for isolated timing:
python -m experiments.discriminative_keys.benchmark --runs /path/work/a5 --out /path/work/benchmark.json
```

Reproduction charges another 20.25M bytes. Outputs are write-once, outside git.
The prefit receipt locks passing independent references, finite differences,
codebook disclosure, exhaustive selection, correctness/causality/lifecycle,
protected policies, source dependencies and fresh excluded-world folds. A5 uses
fresh 512/128/256 train/dev/closed-audit worlds and 128 fresh binary dev worlds;
all original thresholds/sample sizes apply. No audit is opened on failure.

Use a snapshot together with its own cursor, without reset/replay/extra finish:

```python
import json
from pathlib import Path
from experiments.discriminative_keys.backend import Model
from experiments.discriminative_keys.stream import advance
p=Path('/path/work/a5/7/a5.qr')
cursor=json.loads(p.with_suffix('.cursor.json').read_text())
with Model.loads(p.read_bytes(),policy='a5') as model:
    cursor,loss=advance(model,cursor,1000)  # explicitly charged new research
    Path('/path/continued.qr').write_bytes(model.dumps())
    Path('/path/continued.cursor.json').write_text(json.dumps(cursor,indent=2)+'\n')
```

Use this same campaign cursor driver for matched E8/A4, with the saved policy.
Earlier experiments retain their own cursors/loaders. All 18 actual pairs are
checked at interrupted boundaries. Codebook/objective probes are diagnostic
interfaces; production loaders reject any implicit cross-policy conversion.

Primary grounding: [Movshovitz-Attias et al., ICCV 2017](https://arxiv.org/abs/1703.07464)
uses learned proxies for distance objectives. A5 uses existing **fixed** codes
and its own preregistered hybrid, not that paper's learned-proxy implementation.
[Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/) motivates preserved
paired addressing. [Arora et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)
motivates multi-query recall measurement. No citation substitutes for the gates.
