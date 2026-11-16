# E8 public integration: reproduce and resume

Extract `SHoEbox_E8_Public_Code_and_Evidence.zip` into a new directory.
`PUBLISHED_COMMIT.txt` identifies the verified main commit. `SHA256SUMS.json`
checks every other archive member. `SHoEbox/` is the current source. Original
research runs/corpora retain their paths. `public-imports/checkpoints/` contains
29 public checkpoints, unchanged exact cursors, and import hash receipts.

Dependencies: Python 3.10+, NumPy, C++17/g++, Matplotlib for plots. The API
compiles native libraries into a temporary cache; no compiled binary is included.
Use the archived source/compiler flags for exact numerical reproduction.

## Verify existing evidence

```bash
cd SHoEbox
python -m unittest discover -s multiscale/tests -v
python -m experiments.e8_public.check_native
python -m multiscale.tests.run_all --no-leak-check
python -m experiments.e8_public.verify_imports --workspace .. --out /tmp/e8-import-check
```

The import check uses 87,000 discarded correctness bytes and does not modify
selected checkpoints. It verifies all 77 original checkpoint/cursor hashes,
converts the 29 minimal joint kind-4 E8 snapshots, and compares 1,000 research
updates against 1,000 public and 371 + save/load +629 public updates.
It checks separate reconstruction arrays and every full-byte prediction.

The original published source is also in
`source_snapshots/E8_Qualified_Source.zip`. Extract it into a separate folder
and pass that folder to `experiments.e8_public.verify_history --old ...` to
reproduce the 14 historical constructor/checkpoint comparisons.

Run timing only after other CPU work stops:

```bash
python -m experiments.e8_public.benchmark --workspace .. --out /tmp/e8-public-speed.json
```

All research audit results in this archive are already OPENED evidence.
These fixed reproduction commands are not permission to select/tune on those
worlds/documents. Any numerical revision needs fresh preregistered folds.

## Continue an exact public binding checkpoint

This example makes a **new, charged 1,000-byte extension**, outside the completed
integration result. It finishes the saved partial episode first, without replay,
reset or finish at the arbitrary interruption point.

```python
import json
from pathlib import Path
from shoe_multiscale import Model
from experiments.e8_public.bridge import PublicCursorModel
from experiments.joint_cold_gate.stream import advance

root = Path('..')
path = root/'public-imports/checkpoints/exp01e8-formal/7/local_gate.shoe'
cursor = json.loads(path.with_suffix('.cursor.json').read_text())
out = root/'continued-public-e8'
out.mkdir(exist_ok=False)
with PublicCursorModel(Model.load(path)) as model:
    cursor, loss_sum = advance(model, cursor, 1000)
    (out/'binding.shoe').write_bytes(model.dumps())
    (out/'binding.cursor.json').write_text(json.dumps(cursor, indent=2)+'\n')
```

The original published cursor is inside episode 11,937, dataset index 2031,
spacers `[0,3]`, offset 29, 48 input bytes consumed. Both RNG states and the
remaining shuffled order are included. The 250K warmup is separately accounted
from the 2M binding continuation. Scheduled original finish events must remain
part of any exact recipe reproduction.

## Language and generic tasks

For `local_gate.retained.shoe`, use `language_advance` from
`experiments.joint_cold_gate.resume` with `Corpus(root/'exp01e-data')` and the
same public cursor bridge. For fresh generic `language-local_gate.shoe`, use
`Corpus(root/'exp01e7-data')`. Do not replace either corpus with new downloads.
The end cursor explicitly continues into the next pinned epoch; repeated bytes
are new training cost, not unseen data.

Selective-copy snapshots use `task_advance(model, None, cursor, count,
'selective_copy')`. Reconstruction `.shoe` files are ZIP containers: load
`model.shoe` with `Model.loads`, and restore every saved `decoder.npz` array
into `CompressionHead(4,16,seed)`. Call `task_advance` with the public bridge,
that decoder and task `'compression'`. Save both members and the matching
cursor. The final partial example and decoder state must not be discarded.

Generic drivers pass no local labels. The binding driver uses explicit causal
fixture labels; that is a separate objective, as in the qualified recipe.
The public `run_objectives` supplies exact loss sums and target/input counts.

## Explicit new conversion

```bash
python -m multiscale.e8.import_checkpoint original.qr converted.shoe
```

The importer refuses existing outputs, incorrect research formats, E7,
non-local-gate controls and nonminimal/nonjoint policies. It writes checkpoint,
cursor and import receipt together without fitting. Normal public loading never
implicitly reinterprets a research or historical checkpoint.
