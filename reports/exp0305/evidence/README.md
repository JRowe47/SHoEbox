# Evidence index

These are compact, non-dataset copies of completed measurements. The downloadable
code/evidence archive contains complete raw records, generated worlds and all
actual model/cursor files; none of those checkpoints or datasets is in git.

- `fit_plan.json`: pre-fit source/data hashes, folds, seeds, budgets and CPU assignment.
- `fit_selection_lock.json`: hashes locked after every fit and primary development run.
- `development.json`, `extended.json`: per-seed native metrics, offsets, padding,
  order changes, lesions and state traces. Per-presentation Boolean arrays are
  omitted here and preserved in the archive.
- `diagnostic_*`: original/corrected source locks and the failed attempt's cost.
- `cursors.json`, `history.json`: exact continuation results and original file hashes.
- `benchmark.json`: isolated interleaved timing, warmup counts and public-import parity.
- `update_metrics.json`: changed/untouched accuracy and the binary overwrite shortcut.
- `*.txt`: complete captured process output, including the failed initial diagnosis.
- `environment.json`: compiler, Python, packages, CPU exposure and numerical flags.

The parent directory's `verification.json` applies registered gates and verifies
source/data/checkpoint locks. `accounting.json` separates fitting, evaluation,
discarded correctness trajectories and process costs. The audit remains closed.
Artifact packaging receipts and a complete archive hash manifest accompany the
download, separately from the fixed research-cost total.
