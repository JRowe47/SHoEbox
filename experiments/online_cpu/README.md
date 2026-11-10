# SHoEbox online CPU runtime

The width-aware successor and matched 25M comparison are in
[`../scaling_cpu/`](../scaling_cpu/README.md). This directory retains the original
fixed-width implementation and its profiling history.

This is the current 18-region, 64-coordinate CPU implementation, with the selected
centered-routing and centered-Y/split-normalization policy. It includes learned
byte state, temporal lace credit, the EC/hippocampus interface, and the optimized
native training engine. The earlier PyTorch implementation remains in `shoebox/`.

The CPU profile kept topology, parameter counts, learning policy, update order
and version-16 checkpoints fixed. The selected native kernel matches the frozen
reference bit for bit over the recorded 100,000-update checks for both sizes.

| Shape | Parameters | Previous bytes/s | Optimized bytes/s | Improvement |
|---|---:|---:|---:|---:|
| 1M, 12 experts/region | 1,023,744 | 14,238 | 19,721 | 38.5% |
| 5M, 64 experts/region | 4,990,720 | 12,606 | 19,180 | 52.1% |

These are training-only medians on the recorded AMD EPYC host. A second core
for immutable checkpoint/validation snapshots reduced the measured end-to-end
elapsed time by 32.9% and 36.6%. Parallel rows within one expert were slower;
the default training kernel uses one worker. Independent trainers can use
additional cores for seeds and comparisons.

## Build and check

Requirements: Linux x86-64, GCC with C++17, Python 3.12 and NumPy. From the repository
root:

```bash
python -m pip install -r experiments/online_cpu/requirements.txt
SHOEBOX_EXPERTS=12 python experiments/online_cpu/check_packaging.py
SHOEBOX_EXPERTS=64 python experiments/online_cpu/check_packaging.py
```

The first model construction builds a CPU-local shared object with
`-O3 -march=native -std=c++17`. AVX512, AVX2 and scalar mask paths are included.
The checks compare the default engine to `baseline.cpp`, then exercise exact
save/resume and the asynchronous snapshot pipeline. They need no trained weights
or external dataset. `--checkpoint /path/to/last.npz` tests a version-16 trained
checkpoint instead of freshly initialized parameters.

## Use the selected model

Set `SHOEBOX_EXPERTS=12` or `64` before importing. In a Python process started from
the repository root:

```python
import sys
import numpy as np
sys.path.insert(0, 'experiments/online_cpu')
from runtime import Model, fresh_model, SnapshotPipeline

m = fresh_model(seed=1)
# To continue existing training instead:
# m = Model.load('/path/to/version16/last.npz')

data = np.fromfile('your_training_stream.bin', dtype=np.uint8)
x, y = data[:4096], data[1:4097]
loss = m.run(x, y, mode='periodic', lr=.008, lp=.008/3, route=.002)
m.save('last.npz')
m.close()
```

The example uses the existing continuation rates. A fresh scientific training run
should explicitly select its schedule. `fresh_model()` retains the recorded fixed
transports and calibrated address centers while initializing trainable parameters
from the requested seed. The fixed fixtures in `configs/` are not trained weights.

## Overlap validation and checkpointing

```python
with SnapshotPipeline(validation_bytes, 'checkpoints', worker_cpu=None) as pipeline:
    for x, y, total_targets in training_chunks:
        m.run(x, y, mode='periodic', lr=.008, lp=.008/3, route=.002)
        if total_targets % 100_000 == 0:
            pipeline.submit(m, total_targets)
results = pipeline.results
```

Submit after the trainer's native call returns. Each job owns an independent
snapshot; at most one is pending. Closing the context waits for completion and
propagates worker failures. Set `worker_cpu` to an available CPU to pin the worker.

## Source and results

- `hierarchy_lab/throughput_probe/model.py`: compatible version-16 model wrapper.
- `native.cpp`, `mask_kernels.inc`: selected runtime and exact SIMD feature masks.
- `baseline.cpp`: frozen numerical reference; `cached.cpp` is identical to `native.cpp`.
- `build_all.py`: reproducible candidate generation from the reference.
- `pipeline.py`: bounded snapshot validation/checkpoint overlap.
- `results/`: recorded profiles, timing comparisons and correctness checks.
- `reports/SHoEbox_CPU_Profile_Report.md` and `.pdf`: full profile report.
- `reports/SHoEbox_CPU_Profile_Results.json` and `_Comparison.csv`: machine-readable results.

Source paths in the middle six entries are relative to `hierarchy_lab/throughput_probe/`.
The earlier model layers and experiment runners are retained as dependencies and
research provenance. `runtime.py` selects the latest optimized implementation.

The original downloadable `SHoEbox_CPU_Optimized_Runtime.zip` also contains the
3M-trained 1M/5M checkpoints and exact benchmark byte streams. Large training
artifacts and host-specific compiled binaries are omitted from git. To reproduce
the recorded benchmarks, copy these directories from that archive's
`SHoEbox_scale_probe/capacity_comparison/` into this directory:

```text
hierarchy_lab/scale_probe/data/
hierarchy_lab/utility_probe/results/1m/seed1/center_split/
hierarchy_lab/utility_probe/results/5m/seed1/center_split/
```

Checkpoint hashes are recorded in `configs/checkpoint_manifest.json`; data
provenance is under `hierarchy_lab/scale_probe/data/`. Benchmark CPU lists must be
adapted to the host. Run one timing process at a time. For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SHOEBOX_EXPERTS=64 python experiments/online_cpu/hierarchy_lab/throughput_probe/paired_bench.py
```

Recorded benchmark continuations replay an existing segment. They measure runtime
performance and equivalence; they are not additional unique-data training progress.
Original report paths identify the downloadable archive layout.

