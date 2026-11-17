# Resume A9 pre-fit development

A9 is implemented but unfit. Its original 128-world binary development sample
cannot be formed: only 96 remain under the registered salt/exclusions. There
is no passing pre-fit receipt, no research checkpoint and no started fitting
campaign. Never reinterpret this as a completed capability failure or resume
from correctness-test weights.

The resumable archive contains:

- `SHoEbox/`: current source, original historical reports and the A9 blocked report.
- `source_history.bundle`: complete reachable Git history including publication.
- `baseline/`: source at starting main `3c24b6623e8f03ae39f2d7d6266609bfcc9344ba`.
- `a9_work/`: all six data-only stream/cursor/credit manifests, the exhaustive
  world-availability proof, source locks, failed/passing controls and CPU receipts.
- `PUBLISHED_COMMIT.txt`, `publication.json`, and per-member `SHA256SUMS.json`.

Compiled binaries are omitted. No trained checkpoints exist. Raw generated
world data stays outside the Git repository. The 18MB stream manifest contains
complete per-block/condition/key/phase counts and partial terminal cursors;
its bytes do not represent model fitting.

After extraction, verify every archive member:

```python
import hashlib, json
from pathlib import Path
root = Path('.')
for name, expected in json.loads((root/'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
```

The source snapshot can be used directly. For Git-aware verification:

```bash
git clone source_history.bundle SHoEbox_git
cd SHoEbox_git
python reports/exp0305a9/verify_delivery.py ../a9_work
```

The checker validates the blocked state and all locks without running a model.
Read the original `reports/exp0305a8/A9_REGISTRATION.md`, the blocked report,
and `A9_F1_REGISTRATION.md` in full. F1 is the sole prospective amendment,
changing only binary development allocation. It is not implemented.

Before fitting, implement that amendment in a new authorized execution, verify
its complete 128-world sample and all exclusions, rerun affected controls and
commit a passing pre-fit source/world/stream/credit lock. Preserve the original
failure and every cost. Keep all primary world identities and A8/A9 training
streams unchanged. The original exact 20.25M-byte fitting budget is entirely
unused; every arm/seed still needs its own native-only 250K warmup and 2M
continuation. No inherited warmup, pilot, sweep, replay or extra fit.

All original development gates remain, including >=80% isolated native
throughput; lock outcomes before diagnosis. Public E8 and A1–A8 audits remain
unchanged, and no A9 audit has been scored. Passing development still requires
separately registered fresh generic/capacity qualification.

Completed verified work, including unsuccessful research, is explicitly
authorized for direct main commit/push without another confirmation. Preserve
unrelated changes and remote history; never force-push. Deliver ordinary
Markdown, primary PNGs, resumable archive and another copy/paste prompt.
No Pages or Spaces; export and attach without confirmation.
