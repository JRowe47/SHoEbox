# A4 evidence archive and exact continuation

The delivery archive contains `SHoEbox/` (published source),
`source_history.bundle` (published main plus the local pre-fit/final history),
`baseline/` (the starting source), `a4_work/` (all original raw evidence),
`PUBLISHED_COMMIT.txt`, `publication.json`, and `SHA256SUMS.json`.
The 18 `.qr` snapshots and 18 exact `.cursor.json` partners are under
`a4_work/a4/{1,7,19}/`. Both independently charged warmup and final snapshots
are present for E8/A3/A4. Generated worlds include the reserved audit identities;
the audit was never evaluated. Do not open it for further development.

Verify every extracted member from the archive root:

```python
import hashlib, json
from pathlib import Path
root = Path('.')
for name, expected in json.loads((root/'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('Every archive member verified')
```

Restore a complete Git checkout alongside the source snapshot:

```bash
git clone -b main source_history.bundle restored
cd restored
python -m experiments.relative_chronological_retention.verify_evidence --workspace ../a4_work --out ../reverified.json
```

This integrity check validates original source/fold/result/checkpoint hashes and
unchanged historical sources; it performs no research fitting or audit scoring.
Python, NumPy and a C++17 compiler are needed for model use; Matplotlib is used
only for report figures. The delivered environment versions are recorded in
`a4_work/environment.json`. Native libraries rebuild outside Git.

The exact continuation API is documented in
`experiments/relative_chronological_retention/README.md`. Load a snapshot and its
own cursor together; use this campaign's `stream.advance` for all three saved
policies. Its cursor preserves both RNGs, epoch/shuffle position, changed key,
partial episode, offset and finish cadence. Do not reset state, replay warmup or
insert `finish` at a loaded boundary. All 18 actual pairs have passed uninterrupted
versus split-save-load continuation. Continuing consumes additional research
bytes and does not change this campaign's fixed 20.25M-byte results.

A4 failed development. Keep qualified public E8 and all A1–A4 audits closed.
The next authorized research item is the exact, unfitted A5 registration in
`reports/exp0305a4/A5_REGISTRATION.md`, using fresh excluded-world folds.
The adjacent `NEXT_CHAT_PROMPT.md` preserves the publication authorization.
