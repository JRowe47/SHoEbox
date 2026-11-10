# SHoEbox scaling comparison runtime

Prepared November 10, 2026. This directory updates the CPU runtime and prepares a matched, fresh-initialization comparison. The six full 25M-byte runs have **not** been executed. Calibration and verification records are included.

## Run the comparison

Linux x86-64, Python 3.10+, NumPy and a C++17 compiler (`g++` by default). The native kernel builds automatically for each architecture and CPU. No PyTorch, CUDA, optimizer moments or dense gradient buffers are required.

```bash
cd experiments/scaling_cpu
python3 -m pip install -r requirements-data.txt
python3 restore_data.py
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 comparison.py launch --dry-run
python3 comparison.py launch --workers 3
```

The supplied calibration is already selected. `restore_data.py` reconstructs the exact byte streams from the compressed, pinned FineWeb manifest and verifies every document and file hash. Alternatively, copy `data/comparison/` from the downloadable `SHoEbox_Scaling_Comparison_Runtime.zip` into this directory. The launcher verifies source, configuration and data hashes, queues six runs, pins each worker to a separate allowed CPU, and writes results beneath `runs/`. It does not terminate a run because validation briefly worsens. On an interruption, use:

```bash
python3 comparison.py launch --workers 3 --resume
python3 comparison.py summary
```

Resume uses the latest atomic checkpoint, including recurrent state, hippocampal contents, lace eligibility, learned centering, adaptation policy, byte offset and update-observer counters. Interrupted work after the last checkpoint is replayed from that checkpoint. Changed runtime, rates, fixed transforms, data, schedule or seed are rejected. Existing runs are never silently restarted.

Run one arm, or pause it at a controlled boundary:

```bash
python3 comparison.py run --arm width_20m --seed 1 --output runs/width_20m_seed1 --max-targets 1000000
python3 comparison.py run --arm width_20m --seed 1 --output runs/width_20m_seed1 --resume
```

`--max-targets` must be a multiple of 50,000. It pauses the original 25M schedule; it does not compress that schedule. Completed runs are left complete by `--resume`.

## Matched arms

| Arm | Regional width | Experts per region | Parameters | Seeds | Training targets per seed |
|---|---:|---:|---:|---|---:|
| `control_5m` | 64 | 64 | 4,990,720 | 1, 2 | 25,000,000 |
| `experts_20m` | 64 | 256 | 19,638,016 | 1, 2 | 25,000,000 |
| `width_20m` | 128 | 64 | 19,565,824 | 1, 2 | 25,000,000 |

The 18-region topology, top-1 expert execution, cadence, input byte stream, local/lace learner and schedule structure are shared. Expert matrices scale as `D x D`; publications, recurrent state, EC keys/values, predictors and the A1/Y decoder widen with D. Routing and temporal coordinate budgets retain their existing fractions. Expert-bank growth preserves common initial expert prefixes and decoder initialization at a fixed width. Wider initialization uses fan-in scaling.

The control is a **fresh 5M model under the new matched policy**, not the old 150.25M checkpoint. The test therefore separates width growth from bank growth at approximately equal parameter counts. Two seeds and the same bytes are used for all arms.

## Training policy and calibration

The main/local rate warms up through 1M bytes, holds through 20M, and cools linearly to 0.1 of peak over 20M-25M. Local predictor LR is main LR / 3. Competitive prototype and unsupervised route adaptation mature gradually from full strength at 5M to 0.1 at 20M, then follow cooldown. Fitted centering follows the same decay. A1 prototype and decoder controls receive a further 1-to-0.1 consolidation taper. Task-driven routing is compensated for the final main-rate cooldown. Regional teaching multipliers and output gains stay at unity.

Every applied rate is logged. These schedules are the comparison policy; they are not asserted to be an established optimum.

`calibration/selection.json` contains the selected peak rate per arm. Each arm tested 0.002, 0.004 and 0.008 with a separate seed 0, a separate 500K-target calibration stream and a compressed pilot schedule. Selection uses the average shared-validation NLL at 450K and 500K, on 131,072 validation targets. Fixed transport centering uses the first 100K calibration bytes with a frozen two-expert reference and never uses validation/test bytes. All same-width arms share the same fixed transforms and centering.

To redo calibration on another machine with the same source and data, use a fresh work directory and supply it to the launcher:

```bash
python3 comparison.py calibrate --work calibration_new --workers 3
python3 comparison.py launch --work calibration_new --output runs_new --workers 3
```

## Data and evaluation

Restored FineWeb splits: 1M calibration targets, 25M training targets, 2M validation targets and 4M reserved test targets. Each binary has one extra byte for next-byte targets. All arms consume identical byte sequences. Splits contain whole shuffled documents except for the final clipped document, whose entire source identity is excluded from the other splits. Streaming state continues across document separators, matching the existing runtime convention.

The compressed manifest in `datasets/comparison_manifest.json.gz` records the pinned dataset revision, row groups, document IDs, full-text hashes, segment hashes, byte spans and the counts/hashes of prior-data exclusion inputs. The full prior exclusion list remains in the downloadable archive. Files and cross-split identity/hash disjointness are checked on every command. Previous training/evaluation documents are excluded by known ID and exact text hash. Near-duplicate semantic detection is not part of this protocol.

The reserved test split is not evaluated during preparation, calibration or these six runs. Integrity hashing does not compute model test loss. Use validation for arm and checkpoint selection; retain the test for the later selected-model evaluation.

Every 1M targets, the runner evaluates all 2M validation bytes and a fixed 131,072-byte seen-training probe using independent frozen model copies. It retains initial, best-validation, latest and pre-cooldown 20M checkpoints. Every 5M targets it also saves a fixed 32,768-target diagnostic probe:

- Effective expert count, source/input/output covariance rank and tanh saturation by region.
- Routing selections sampled every 64 bytes and their change since the preceding probe.
- Actual expert-update magnitudes separated by language/lace and local teaching, sampled every 256 calls.
- Regional executions and usage; temporal/lace credit, missed-parent and overwritten-credit counters.
- Mean-Y and blocked-recall changes in language NLL on the same diagnostic bytes.

These short probes diagnose mechanisms; full validation NLL determines the comparison. Local prediction MSE is not labeled as regional language loss. Evaluation is checked to leave the training fingerprint unchanged.

Primary comparison: average validation NLL at 23M, 24M and 25M for each run, then the two-seed mean. `summary.json` also reports endpoints, best checkpoints, paired deltas against the 5M control, training compute time and wall time. Negative deltas favor the larger arm. Compare byte exposure and measured training cost together. The first 25M is a shakedown screen, not the intended final 100M training budget.

## Runtime changes

- Width is model-local: 64, 128 and 256 coexist in one process.
- Feature masks use one, two or four 64-bit words throughout routing, sparse updates, temporal records and lace credit; no coordinate truncation above 63.
- Packed regional offsets permit independent expert counts without padded trainable banks. Counts are validated in the range 2-256.
- Legacy width-64 optimized selection is preserved. Version 16-19 checkpoint loading and version 20 architecture-aware saving are supported.
- Exact int64 byte counts avoid float32 counter rounding beyond 16,777,216 targets.
- Native compilation is keyed by source, architecture, compiler options/version and CPU features; concurrent builds use a lock.
- Atomic checkpoints and a protocol fingerprint protect interrupted or resumed experiments.

The proposed 100M allocation is included in `configs/proposed_100m.json`: width 256 and the 18 bank counts `[128,128,96,96,96,96,80,80,80,80,64,64,64,64,48,64,64,96]`, totaling **99,727,616** parameters. It has been instantiated and smoke-trained; it is not one of the three 25M comparison arms.

```python
from runtime import Model, Architecture
model = Model(Architecture(width=128, experts=64), seed=1)
checkpoint = Model.load("runs/width_20m_seed1/last.npz")
checkpoint.close()
model.close()
```

Checkpoints contain raw native recurrent-state layouts and are intended for this Linux x86-64 runtime family. Compiled `.so` files are excluded from the distribution and rebuilt locally. Native parameter counts exclude fixed transports, memory entries and diagnostic buffers, following the previous counting convention.

## Verification and provenance

```bash
python3 -m unittest discover -s tests -v
SHOEBOX_SANITIZE=1 python3 -m unittest discover -s tests -p test_runtime.py -v
python3 tools/verify_100m.py
```

Verification includes coordinate boundaries 63/64, 127/128 and 191/192 through 255; heterogeneous banks; upper-coordinate learning; frozen evaluation; checkpoint save/load; exact interrupted versus uninterrupted training; schedule limits; data tamper and protocol-change rejection. The 1M and 5M historical checkpoints were each replayed for 50,000 targets against their original runtime: predictions, parameters and all five serialized state blobs matched exactly. Reports are in `reports/`.

`reports/source_provenance.json` identifies the archived source and records the native-source hash. The baseline checkpoint archives are not duplicated in this package. `tools/verify_legacy.py` can repeat parity checks given the original source archive and checkpoints.

Files under `tests/` are small synthetic verification runs. Files under `calibration/` are LR pilots. Neither is presented as a completed 25M result. The main `runs/` directory starts empty.

## Throughput

`reports/throughput.json` records the matched single-core measurements on the AMD EPYC 9V74:

| Model | Parameters | Training bytes/s | Projected 25M training time |
|---|---:|---:|---:|
| 5M control (D=64, E=64) | 4,990,720 | 20,526 | 20.3 min |
| 20M bank (D=64, E=256) | 19,638,016 | 18,756 | 22.2 min |
| 20M width (D=128, E=64) | 19,565,824 | 6,426 | 64.8 min |
| 99.73M (D=256, regional banks) | 99,727,616 | 1,887 | 220.8 min |

These projections cover training only. Validation and checkpointing add elapsed time.

 The earlier 500K-rate pilots measured 20.1k bytes/s for the 5M control, 17.5k for the 20M expert bank and 6.5k for width 128. The original 8,192-byte 100M smoke test measured 2.0k bytes/s. These are CPU training byte targets per second, excluding validation and checkpointing.

To repeat the longer matched benchmark after restoring the data:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/benchmark.py --output reports/local_throughput.json
```

Each architecture receives 16,384 warmup targets and 250,000 timed targets, sequentially on one pinned CPU. The script reports per-chunk rates and 25M training-time projections; it writes no trained checkpoints and does not advance the six comparison runs.

