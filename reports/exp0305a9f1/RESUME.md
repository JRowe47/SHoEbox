# Resume EXP03/05-A9-F1

The original A9 allocation blocker is repaired and the fixed E8/A8/A9 campaign
is complete. The 20.25M research-fitting-byte budget is exhausted. Do not rerun,
extend, replay or tune any fitted arm. Read the completed report, diagnosis and
the sole next registration before new research. Qualified public E8 remains.
All A1–A9 audit identities are reserved and closed.

## Sources and pre-fit boundary

Starting main: `fa54cbd1c8c90be756ea4da8cfef2ccb735ed183`.
Local pre-fit commit: `00d5cb393c075710925a6fd9add8606915ca3a5a`.
Identical published pre-fit commit: `7dc8b5c275d00b469f068d788064b63025833120`.
Both trees: `3bcf7f5428ae12cb6295dc11228bba7fefdc07b8`.
The local commit preceded fitting. Terminal push lacked credentials, so the
connected GitHub service published the identical tree without replacing remote
history. The archive preserves both commits and the mapping receipt.

The original blocked A9 report, RESUME, F1 registration and receipts are retained
unchanged. Their zero-fit statement describes that earlier blocked campaign.
The original failed 96-world check is still reproducible with `salted_worlds`.
F1 uses the complete eligible binary pool exactly as registered, selecting 128
from 336, with all 896 primary worlds and original stream manifests unchanged.

## Archive layout

- `SHoEbox/`: complete tracked published source and compact evidence.
- `repository.bundle`: full available Git history, including the local pre-fit
  commit and published main. `PUBLISHED_COMMIT.txt` records the final main SHA.
- `baseline/`: starting-main source used for historical parity checks.
- `a9_f1_work/`: all 18 warmup/final models and exact cursors, generated world
  file, full stream/credit manifests, raw outcomes and traces, locks, accounting,
  failed/passing logs, and `prior_a9/` with original blocked-A9 raw evidence.
- `SHA256SUMS`: SHA-256 for every payload file except the manifest itself.
- `RESTORE.md`: extraction and Git restoration commands. The external delivery
  manifest records final archive hash, compression and archive-check costs.

Checkpoints, datasets and compiled binaries are not Git-tracked. Compiled caches
are excluded from the archive and rebuild from source with FMA disabled.

## Verification without research fitting

After extraction, verify `SHA256SUMS` and restore Git using `RESTORE.md`.
From `SHoEbox/`, the read-only delivery verifier is:

```bash
python reports/exp0305a9f1/verify_delivery.py ../a9_f1_work
```

This checks the pre-fit source closure, original stream identity, complete fit
and evaluation locks, all 18 diagnostic checkpoint inputs, exact fitting and
credit ledgers, historical preservation, and the pre-fit commit/tree mapping.
It creates no model, performs no inference and executes no updates. Its JSON
receipt is deterministic. Raw source and file hashes permit independent review.

The archived `frozen_fit.lock.json` pins 44 fitting outputs. The separate
`evaluation.lock.json` pins every full outcome, cursor receipt and speed result
before diagnosis. `frozen_diagnosis.lock.json` pins all 18 model inputs and
diagnostic source files. Individual trace locks retain their narrower inputs.
Final verification checks the complete lock even where inherited tracers use a
narrower glob. The first-32-world teacher-fed traces are not full autoregressive
scores; `coverage.json` separately reports first32/rest96/full128 endpoints.

## Exact checkpoint cursors

Pairs live at `a9_f1_work/a9/{1,7,19}/{e8,a8,a9}[.warmup].qr` with matching
`.cursor.json`. Warmups have 250,000 independent native-only PAD bytes; each
final pair includes its 2,000,000-byte continuation. Load with the explicit
matching adapter policy and restore the entire outer cursor. Never synthesize
an episode boundary or seed a new order generator from the saved byte count.

A9 uses magic `0x3941353050584553`, nested v19, and outer cursor schema
`exp0305-a9-cursor-1`. The cursor carries phase/order/augmentation state,
incomplete episode and training-only record-key flag derivation. The flag is
not model state or an inference argument. All 18 actual pairs passed identical
1,000-step uninterrupted versus 371/save/load/629 continuation, using 36,000
discarded correctness bytes. These tests do not authorize a new research fit.

## Costs and publication

`accounting.json` separates current research/verification costs from original
blocked A9 and delivery artifacts. All process timers include descendants and
compilation; nested fit timers are not added twice. Historical native scalar
microtrajectory writes are not fully instrumented; complete CPU/logs remain.
The initial report syntax error executed no module/model; its failed source is
preserved. The initial failed terminal push changed no remote reference.

Completed verified work, including unsuccessful research, is authorized for
direct main commit/push without confirmation. Preserve unrelated changes and
remote history; never force-push. Deliver ordinary downloadable Markdown,
primary PNGs, a resumable archive and a next prompt retaining this authorization.
No Pages or Spaces; export and attach without confirmation.
