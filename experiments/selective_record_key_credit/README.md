# EXP03/05-A9: selective record-key credit

**Implemented; blocked before research fitting.** The original registered binary
sample requires 128 fresh worlds, but only 96 exist in its development bucket
after all exclusions. Do not fit or replace the missing worlds implicitly.
Read `reports/exp0305a9/SHoEbox_EXP0305_A9_Report.md`, the original
`reports/exp0305a8/A9_REGISTRATION.md` and the prospective
`reports/exp0305a9/A9_F1_REGISTRATION.md` together. F1 is not implemented.

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

The unamended Python suite intentionally retains the original 128-world
requirement and currently reports its shortage. Never mark that suite passing.
The numerical tests and optimized/ASan/UBSan references are reported separately.
`prefit.py` refuses a failed prerequisite, and `run.py` requires the exact passing
receipt committed at HEAD. No such fitting authorization receipt exists.

After the separately registered F1 change, all preconditions must pass before
`run --out /path/work/a9 --prefit /path/work/prefit.json` can execute. It allocates
nine independent fits and refuses an existing output directory. Do not use
correctness trajectories as warmups. The complete outcome/diagnostic tools are
retained for that future campaign; no development result is available now.

No A9 trained checkpoints exist. The supplied archive resumes development from
source, receipts and exact data-only stream manifests. It does not represent a
partially completed research fit. Public E8 and all A1–A8 evidence stay fixed.
