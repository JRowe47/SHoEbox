# SHoEbox composition and credit repair

The default runtime separates reusable constituent content from neural context,
learns variable-length event boundaries, and follows the retained input state
used by upper regions. The original `native` kernel remains available for the
matched comparison. The source is the tested width-64, uniform-bank repair.

## Build and verify

Use Linux, Python 3.12+, NumPy and a C++17 compiler (`g++`). From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tests/run_sanitizer.py
```

Compilation targets the local CPU; compiled binaries are not checked in. Tests
use synthetic inputs and need no downloaded corpus or trained checkpoints.
The repository's `python scripts/check_all.py` also runs this 19-test suite.
LeakSanitizer is disabled by the sanitizer runner for compatibility with the
ptrace-based execution host; address and undefined-behavior checks remain enabled.

Set `SHOEBOX_EXPERTS` before importing: 12 gives 1,023,744 neural parameters,
64 gives 4,990,720, and 1312 gives 100,198,144. Width is fixed at 64. The 5M and
100M configurations passed smoke checks; the matched language runs use 1M.

## Use

```python
import sys
import numpy as np
sys.path.insert(0, 'runtime')
from runtime import Model, fresh_model

model = fresh_model(seed=1)  # composition kernel
data = np.frombuffer(b'A document with reusable constituents.', np.uint8)
loss = model.run(data[:-1], data[1:], mode='periodic', lr=.012, lp=.004)
model.finish_document(mode='periodic', lr=.012, lp=.004, reset=True)
model.save('my_checkpoint.npz')
model.close()
```

`finish_document()` drains pending constituents without adding input bytes or
synthetic targets. `reset=True` clears working context and eligibility while
preserving weights, learned dictionaries and persistent hippocampal memory.
Use it at document boundaries, not batching boundaries. `frozen_copy()` preserves
fitted dictionaries but starts a fresh stream; `snapshot_copy()` preserves the
complete stream. `content_events()` exposes IDs, spans, child provenance and
recognition status; `composition_metrics()` exposes event and credit counters.

Pattern slots identify exact ordered compositions, independently of expert IDs.
Neural experts learn their predictive meaning in context. Retained-input credit
uses the input-state Jacobian with bounded delayed local teaching, rather than
full backpropagation through time. Ordinary source-eligibility misses remain
explicitly measured.

## Results

Two matched seeds, fresh 1M-parameter models and 1M training byte targets each:

| Measurement | Original | Repaired |
|---|---:|---:|
| Mean held-out NLL, nats/byte | 2.060284 | 2.035142 |
| Median training bytes/s | 22,564 | 17,098 |

The throughput trials alternate on one Intel Xeon Platinum 8573C core. These
measurements show 1.22% lower loss at 24.2% lower throughput. L3 removal hurts
prediction much more after repair; E4/G5 still contribute little after this run.
The added dictionary/event/credit state is 2.96 MiB, independent of expert count.

See the [Markdown report](deliverables/SHoEbox_Composition_Repair_Report.md),
[PDF report](deliverables/SHoEbox_Composition_Repair_Report.pdf),
[results JSON](deliverables/SHoEbox_Composition_Repair_Results.json), and
[curves](deliverables/SHoEbox_Composition_Repair_Curves.png).
`results/` contains the individual learning histories and audits.

## Restore data and checkpoints

Trained weights and corpus bytes stay in the downloadable
`SHoEbox_Composition_Repair_Source_and_Checkpoints.zip` supplied with the report.
`ARTIFACT_MANIFEST.json` records the archive and individual file SHA-256 values.
Restore its four checkpoints and exact corpus into this directory:

```bash
python src/restore_artifacts.py --archive /path/to/SHoEbox_Composition_Repair_Source_and_Checkpoints.zip
```

The script verifies the full archive and each selected file before writing;
existing different files are not overwritten. Data can also be restored from
a local Tiny Shakespeare `input.txt` with its recorded SHA-256, or fetched from
the recorded public source and verified:

```bash
python src/restore_artifacts.py --data-file /path/to/input.txt
python src/restore_artifacts.py --download-data
```

`DATA_MANIFEST.json` describes the 1M-target comparison and exact train/held-out
split. The archive's older source manifest describes the source experiment's
10M-target protocol; it is retained as provenance, not this run's target budget.

With data restored, rerun either matched arm (choose an available CPU index):

```bash
python src/benchmark.py --kernel native --seed 1 --core 0 --tag rerun
python src/benchmark.py --kernel composition --seed 1 --core 0 --tag rerun
```

Use seed 2 for the second pair. With the recorded checkpoints restored:

```bash
python src/verify_performance.py --core 0
python src/lesions.py --core 0
```

The latter commands replace their measurement JSON outputs. Optional report
regeneration uses `requirements-reports.txt` and `python src/make_report.py`.

## Checkpoint conversion

`Model.load('results/delivery_composition_seed1/last.npz')` loads a repaired
checkpoint. Version 28 preserves partial compositions, dictionaries, actual
neural provenance, temporal credit and supported policies; content-state ABI is 2.
The included kernels are `composition` and its `native` comparison baseline.

To reuse legacy uniform-width-64 weights under the new message semantics:

```bash
python src/migrate_checkpoint.py old_checkpoint.npz new_composition_checkpoint.npz
SHOEBOX_EXPERTS=1312 python src/migrate_checkpoint.py old_100m.npz new_100m.npz
```

Conversion preserves neural weights and fixed transports. Constituent
dictionaries, working context, eligibility and hippocampal associations start
fresh because their representations have changed. It writes a new file and is
not exact continuation of the old runtime.

## Source provenance

`SOURCE_MANIFEST.json` records every published runtime source hash and its match
to the delivered implementation. `src/integrate.py` and `src/integrate_python.py`
reproduce the native and wrapper integration from the preserved `native.cpp`
and `model_legacy.py`. The earlier width-scaling and online-CPU experiments remain
in their own directories with their original reproduction protocols.
