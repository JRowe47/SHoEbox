# SHoEbox scaling runtime and 25M comparison

November 10, 2026

**Built and verified. The package is ready for six matched 25M-byte runs.** Preparation includes new data, rate calibration and runtime checks. The full 25M runs have not started.

## Comparison ready to launch

| Arm | Width | Experts / region | Parameters | Peak LR |
|---|---:|---:|---:|---:|
| 5M control | 64 | 64 | 4,990,720 | 0.008 |
| 20M expert bank | 64 | 256 | 19,638,016 | 0.008 |
| 20M width | 128 | 64 | 19,565,824 | 0.008 |

Each arm starts fresh with seeds 1 and 2, consuming the same 25M-byte stream. The two larger arms closely match total parameters while separating larger expert banks from wider active computation. The fresh 5M arm is the reference for the new training policy.

## Runtime delivered

- Model-local widths 64, 128 and 256; packed, independently sized banks for all 18 regions.
- Wider expert matrices, messages, recurrent state, EC recall, predictors, byte decoder and credit paths. Feature masks span all coordinates through 255.
- Existing top-1 sparse execution and local/lace learning preserved; no dense gradient or optimizer-state allocation.
- Versioned checkpoints, exact byte counters, atomic saves, CPU-specific native builds and guarded resume.
- Validation, seen-training loss, expert usage, routing changes, representation rank, tanh saturation, actual update sizes, credit expiry and Y/recall contribution probes.

The proposed **99,727,616-parameter** configuration is also implemented: width 256 with the recommended regional bank allocation. It passed an 8,192-byte training smoke run with finite weights, learning above coordinate 192 and read-only evaluation. It is ready for later experiments, outside this three-arm comparison.

## Verification completed

- Nine automated tests pass, including exact resumed versus uninterrupted training, heterogeneous banks, schedule bounds and data/protocol mismatch rejection.
- Five native-runtime tests pass with undefined-behavior sanitization at widths 64, 128 and 256.
- The historical 1M and 5M models each match their original runtime exactly after 50,000 further training targets: logits, learned parameters and all five serialized state buffers.
- Data files pass hashes, document disjointness and exclusion checks against 70,325 known prior document IDs. The reserved test split has not been evaluated.

<!-- pagebreak -->

## Calibration and training schedule

Each arm tested main rates 0.002, 0.004 and 0.008 on 500K separate calibration targets, with seed 0. Scores below average validation at 450K and 500K on 131,072 fixed held-out targets; lower is better.

| Arm | LR 0.002 | LR 0.004 | LR 0.008 |
|---|---:|---:|---:|
| 5M control | 2.739142 | 2.619596 | 2.528296 |
| 20M expert bank | 2.744793 | 2.629946 | 2.533218 |
| 20M width | 2.762850 | 2.645940 | 2.541627 |

The selected rate is the best of this screen, not a final scaling result. These pilots choose the launch rate; the matched 25M runs will judge the architecture.

- **0-1M:** warm up the main and local learning rates.
- **1-20M:** hold peak learning; gradually slow competitive adaptation after 5M.
- **20-25M:** cool main/local learning to 0.1 of peak, with coordinated prototype, centering and A1 decoder slowdown; preserve useful task-driven routing strength.

The local rate is main / 3. Regional teaching and output-gain multipliers start at unity. All effective schedule controls are recorded. No validation-based early stopping is used.

## Data, checkpoints and decision rule

Included splits contain 1M calibration, 25M training, 2M validation and 4M reserved test targets. All are document- and exact-text-disjoint. The pinned FineWeb revision, document spans, source hashes and prior exclusions are supplied. A target is one next-byte prediction, not a BPE token.

Validation runs every 1M targets. Initial, best, latest and pre-cooldown 20M checkpoints are retained. Compare the mean validation loss at 23M, 24M and 25M, averaged across both seeds, alongside measured training time. The summary also reports paired differences from the 5M control. Test data stays reserved during arm selection.

## Launch and resume

Extract the ZIP, enter `shoebox_scaling`, and use Linux x86-64 with Python, NumPy and g++. The selected calibration and prepared data are included.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 comparison.py launch --dry-run
python3 comparison.py launch --workers 3
```

After an interruption:

```bash
python3 comparison.py launch --workers 3 --resume
python3 comparison.py summary
```

The package README documents single-arm runs, controlled pauses, dependencies, recalibration and the public runtime API. The `reports/` directory contains parity, smoke, data-integrity and test evidence. Main training starts in a clean `runs/` directory.
