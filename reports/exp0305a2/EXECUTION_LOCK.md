# A2 execution details, fixed before fitting

Implements the existing A2 registration and mandatory three-value amendment;
all original thresholds and the 20,250,000-byte total remain unchanged.
Starting main: `0cdbabaf12776673dd5c70dfd1382e58557494f9`.

- Separate `experiments/query_first_retention` module; every A1/public/historical
  source stays unchanged. A2 magic `0x3241353050584553`, nested v19, no automatic
  conversion. Query-first routing adds no weights or persistent state.
- Salt `SHoEbox-EXP0305-query-first-v2` with the established NUL separator;
  world RNG 20261124. Enumerate all ordered distinct values among 0/1/2 for
  each sorted pair of 64 keys. Exclude every historical full test fold using
  A1's predicate and all 256 reserved A1 audit initial worlds. Three-value
  worlds use 512/128/256 train/dev/closed-audit counts. Binary regression uses
  128 independently shuffled binary worlds from the same fresh dev fold;
  never an A2 train/audit world or an old audit. Overlap within dev is disclosed.
- Four record/query-order siblings per world. The changed key is
  `(world_index + completed_shuffle_epoch) % 2` in training: exactly balanced
  within every complete epoch, alternating operations on the same initial
  state across epochs. Primary/offset/counterfactual evaluation changes
  `world_index % 2`, balanced independently of both orders. Both changed keys
  are also interpreter/causality checked before fitting. Third value replaces
  the selected key; untouched value stays distinct. No no-op update.
- Existing five-condition shuffled mixture, native clean warmup, 0–7 spacers,
  64 phases, order RNG 4701/4702 and augmentation RNG +1000, finish cadence,
  rates and local target codes remain registered. Cursor schema is explicitly
  `exp0305-a2-cursor-1`, adding the shuffle epoch and changed-key choice so
  interruptions cannot change the stream. All three arms share exact ordering.
- Seeds 1/7/19; E8/A1/A2 each receives its own 250K native warmup and 2M mixed
  continuation. No pilot, reused warmup, optimizer sweep or extra fit.
- Primary and extended sample counts match A1. Fresh binary canonical regression
  repeats primary/per-position, both-orders, all offsets, independent padding
  delays and all lesions at the original sample counts and thresholds.
  The full 256-byte autoregressive argmax is the endpoint; the scorer's legal
  target range is three values solely for validation/restricted diagnostics.
- First/last/constant repeated-value shortcuts and independent interpreter
  agreement are checked before fitting. Additional operation-level diagnostic
  tables distinguish changed and untouched keys; they do not select a fit.
- Isolated speed uses the three fitted seed-7 arms, exact public E8 import, and
  a fresh native v19 500K timing reference (498,367 actual parameters, zero fit
  bytes). Three interleaved 65,536-byte full-distribution repeats plus 1,024-byte
  warmup, one CPU, all other jobs stopped. Preserve the >=80% native threshold.
- All pre-fit tests are discarded correctness trajectories, separately metered.
  `prefit.json` pins passing logs/process receipts, all dependency hashes,
  fixture hashes, interpreter and shortcut checks before `run` accepts fitting.
  Model/cursor snapshots and development outcomes are locked before extended
  diagnostics. Neither A1 nor A2 audit is scored if development fails.
