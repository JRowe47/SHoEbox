# EXP03/05-A9 with F1 allocation repair

A9-F1 is implemented. It selects the registered 128 binary development worlds
from 336 eligible candidates, excludes all 896 unchanged primary groups and
all historical exclusions, and prohibits binary training/audit allocation.
The independent proof reproduces the original 96-world shortage and verifies
all six original stream/credit manifests byte for byte. The original blocked
report, receipts and CPU remain in `reports/exp0305a9/`; completed execution
results will be recorded separately in `reports/exp0305a9f1/`.

Read `reports/exp0305a8/A9_REGISTRATION.md` and
`reports/exp0305a9/A9_F1_REGISTRATION.md` together. Every numerical policy,
training budget and gate remains unchanged.

A9 retains A8's alternating 64-episode PAD/diverse input curriculum and suppresses
only positive record-key hybrid updates in diverse blocks. `allow_record_key_update`
is ephemeral training input, separate from labels/masks; standalone local calls
default to true. Generic calls cannot train local heads. The flag is derived
from the exact saved episode position and never enters inference or model state.
The four presence costs remain 4/4; query/value/applicability/native learning,
A4 chronology/admission, all-256 predictions, codebook and state capacity remain
unchanged. FMA is disabled. Checkpoint magic is `0x3941353050584553`, nested v19;
outer cursor is `exp0305-a9-cursor-1`. Historical policies are untouched.

`credit.py` independently counts eligible, executed and omitted key events/rows.
`check_streams.py` locks input/target/mask/label/credit hashes and RNG/cursor state
for all six arm/phase streams and keeps block/condition/key/phase ledgers.
These are data-only traversals, not research fitting. All outputs belong outside
git and are process-metered with `meter.py`.

```bash
python -m experiments.selective_record_key_credit.check_python /path/work/python_counts.json
python -m experiments.selective_record_key_credit.check_native
python -m experiments.selective_record_key_credit.check_policy_isolation
python -m experiments.selective_record_key_credit.check_streams --out /path/work/streams.json
python -m experiments.selective_record_key_credit.audit_feasibility /path/work
```

The complete 20-test suite keeps the 128-world requirement and tests F1
development-only allocation, primary disjointness and the preserved original
shortage. `check_f1_allocation` independently verifies identities/exclusions
and exact original streams. `prefit.py` refuses any failed prerequisite;
`run.py` requires the exact passing source/data receipt committed at HEAD.

Before fitting, complete every prerequisite and commit the source/world/stream/
credit locks. Then `run --out /path/work/a9 --prefit /path/work/prefit.json`
allocates nine independent fits and refuses an existing output directory.
Correctness trajectories never supply a warmup. Each arm/seed receives its own
250K native-only PAD warmup and 2M continuation: exactly 20.25M bytes.

Full primary, binary, offset, order, padding, repeat/diverse and lesion outcomes,
all 18 actual snapshot/cursor checks and isolated >=80% native timing precede
the complete evaluation lock. Only then run the registered A8/A9 frozen traces.
No diagnostic-driven fit or audit access follows a failed development gate.
All datasets, snapshots and compiled binaries stay outside git.
