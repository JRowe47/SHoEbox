# SHoEbox - full-rank refits and compiled output heads

November 4, 2026 (America/Denver)

The next phase is complete: **72 full-rank byte-decoder fits across 12 frozen graphs**, compared with all 432 retained identity/GELU corrections. A full-matrix refit does not provide a consistent accuracy improvement over the compact identity correction. The useful engineering result is a compiled output head: linear refits and identity corrections fold into one matrix multiplication, with unchanged predicted bytes on every audited case.

## Main comparison

Values are held-out private / shared accuracy in percent on the 384-case all-source test set. Each condition averages three graph seeds; identity and GELU additionally average three correction initializations per graph. All updated heads use 3,840 training cases and the existing validation selection of preservation penalty and strength.

| Graph condition | Frozen base | Full-rank refit | Identity correction | GELU correction |
|---|---:|---:|---:|---:|
| original / dropout | 73.70 / 86.98 | 74.35 / 87.76 | 77.26 / 88.11 | 75.09 / 83.42 |
| original / balanced | 55.99 / 80.47 | 55.47 / 82.29 | 57.99 / 82.81 | 58.98 / 78.04 |
| gated / dropout | 60.94 / 91.15 | 60.03 / 90.10 | 59.81 / 90.10 | 62.28 / 90.71 |
| gated / balanced | 54.95 / 87.50 | 56.12 / 89.06 | 56.64 / 88.98 | 56.64 / 86.28 |

The original/dropout condition remains strongest with the retained identity correction: 77.26 / 88.11, versus 74.35 / 87.76 for the full-rank refit. The default graph and decoder remain unchanged. The compiled module and exported heads are available explicitly.

## Matched protocol

| Item | Setting |
|---|---|
| Frozen graphs | Original/gated publication x dropout/balanced source exposure x three graph seeds |
| Training cohorts | 960 and 3,840 cases; 64 and 256 per supported source/query cell |
| Cohort relationship | Exact nested prefixes of the already captured training features |
| Input / output | 64 raw P/Q features to one common 256-byte decoder |
| Normalization | The retained base training mean and scale; no refitting of normalization |
| Validation | 1,920 fixed cases; query-balanced NLL with all/shared and B-only/shared guards |
| Selection grid | Strength 0, 0.125, 0.25, 0.5, 1; preservation penalty 1 or 10 |
| Control | Penalty zero, with the same validation strength selection |
| Evaluations | 384 test/all cases and 3,456 independent intervention cases |
| Optimizer | Full-batch float64 L-BFGS, strong Wolfe; maximum 1,000 iterations |
| Initialization | Frozen baseline matrix; deterministic with no RNG use |
| Regularization | 0.001 coefficient; entire refitted weight and bias matrix |
| Additional expert executions | Zero; all fitting and comparisons use retained feature caches |

The full-rank head has 16,640 fitted scalars. Each retained correction has 5,392 fitted scalars around the fixed 16,640-scalar base. The full-rank objective regularizes the entire new matrix; the earlier residual objectives regularize their factor matrices and biases. This compares practical decoder alternatives, rather than isolating rank under an identical regularizer. The residual comparisons retain their original 400-iteration fits.

The preservation anchors are training cases whose shared answer the old head already gets right. A squared margin penalty discourages losing those answers, with target margins capped at one. Prediction consumes only the P/Q feature vector; query and source labels never select an output head.

---

## Cohort size and preservation controls

Full-rank results below use validation-selected preservation and strength. All numbers are private / shared percent.

| Graph condition | 960 cases | 3,840 cases | Change, percentage points |
|---|---:|---:|---:|
| original / dropout | 73.70 / 86.98 | 74.35 / 87.76 | +0.65 / +0.78 |
| original / balanced | 55.99 / 80.47 | 55.47 / 82.29 | -0.52 / +1.82 |
| gated / dropout | 60.94 / 91.15 | 60.03 / 90.10 | -0.91 / -1.04 |
| gated / balanced | 54.95 / 87.50 | 56.12 / 89.06 | +1.17 / +1.56 |

| Nonzero full-rank update accepted | 960 cases | 3,840 cases |
|---|---:|---:|
| Penalty zero | 1 / 12 | 11 / 12 |
| Preservation selected | 1 / 12 | 9 / 12 |

| Graph condition | Penalty-zero refit, full strength | Penalty-zero refit, selected strength | Preservation selected |
|---|---:|---:|---:|
| original / dropout | 75.39 / 85.16 | 76.04 / 85.16 | 74.35 / 87.76 |
| original / balanced | 55.21 / 83.85 | 55.21 / 83.85 | 55.47 / 82.29 |
| gated / dropout | 59.90 / 89.84 | 61.33 / 90.36 | 60.03 / 90.10 |
| gated / balanced | 54.04 / 88.02 | 54.04 / 88.02 | 56.12 / 89.06 |

The accuracy guards are enforced on validation. The test results above show where shared-answer loss still occurs on different cases; the full-rank refit does not remove that tradeoff.

## Independent B-only and missing-information controls

| Graph condition | Base B-only | Full-rank B-only | Identity B-only | GELU B-only |
|---|---:|---:|---:|---:|
| original / dropout | 85.68% | 86.20% | 90.36% | 90.10% |
| original / balanced | 100.00% | 100.00% | 100.00% | 100.00% |
| gated / dropout | 95.83% | 96.35% | 95.57% | 96.09% |
| gated / balanced | 98.70% | 99.22% | 99.22% | 98.70% |

Selected full-rank accuracy with the required information missing averages **25.207%**, against 25% chance. All 256 output bytes remain eligible; the task answers are the four bytes A-D.

---

## Compiled inference cost

Compilation absorbs the frozen normalization into the weights and bias. Identity corrections are multiplied out and added to the base matrix. Full-rank interpolation is likewise folded offline. GELU keeps its two correction matrices and nonlinearity, while normalization, strength and output bias are folded. The resulting modules contain frozen buffers and no trainable parameters.

| Compiled architecture | Stored scalars | Float64 bytes | Matrix MACs / case | Matrix multiplies |
|---|---:|---:|---:|---:|
| Base, full-rank or identity | 16,640 | 133,120 | 16,384 | 1 |
| Active GELU correction plus base | 21,776 | 174,208 | 21,504 | 3 |

The GELU row also evaluates 16 GELU elements. A zero-strength correction compiles to the base alone. Stored-scalar counts exclude Python object overhead; MACs count matrix multiply-accumulates, not biases, checks, argmax or allocation. Folding the identity correction reduces matrix MACs by 23.8% and removes the separately stored correction and normalization arrays.

**Measured CPU prediction latency.** Values are medians across twelve graphs, in microseconds per batch. Each graph uses three timing repetitions with alternating reference/compiled order. The identity and GELU rows below force strength one, so neither benefits from fallback to the base.

| Head | Reference, batch 1 | Compiled, batch 1 | Reference, batch 64 | Compiled, batch 64 |
|---|---:|---:|---:|---:|
| Base | 27.84 | 25.22 | 138.78 | 130.58 |
| Full-rank, selected | 66.23 | 24.92 | 283.24 | 129.69 |
| Identity, strength 1 | 52.96 | 25.30 | 253.77 | 134.66 |
| GELU, strength 1 | 55.53 | 37.66 | 255.72 | 176.20 |

The paired median identity speedup is **2.06x at batch 1** and **1.86x at batch 64**. These timings cover the output decoder including input conversion, finite checks and argmax; they exclude the seven-node graph. They are not end-to-end model speedups.

Timing environment: INTEL(R) XEON(R) PLATINUM 8573C; PyTorch 2.15.0+cpu; one CPU thread; torch.float32 inputs and float64 decoder arithmetic. No fitting or test suite was running alongside the recorded timing pass. Selected-head timings and all repetitions are retained in benchmark.json.

---

## Verification and delivery

All **317 automated tests pass** across five suites. The artifact audit reconstructs all 72 new objectives and gradients, every validation decision, and 720 new prediction groups. It also re-audits the 432 inherited corrections and 3600 retained prediction groups, including their legacy baselines.

All **204 compiled heads** are restored from their exported buffers and checked on both training cohorts, validation and both evaluation cohorts: **2,154,240 exact predicted-byte comparisons**. The largest logit difference from floating-point reassociation is 3.55e-14. These comparisons include repeated cases across heads and the nested training cohorts; they are not a count of unique episodes.

None of the 72 new fits reached the 1,000-iteration cap. Iterations range from 1 to 817; final maximum gradient magnitudes range from 2.92e-07 to 0.00121. Stopping below the cap includes the optimizer's objective-change criterion; it is not a claim that every fit met the gradient tolerance.

Source hashes, cached inputs, graph checkpoints, frozen normalization, split separation and exact nested feature ownership are verified. Every completed fit is saved atomically; resuming with changed code, settings, inputs or baseline is rejected. Public outputs contain code, reports and aggregate results. Raw synthetic predictions and tensors are included in the downloadable artifact package.

## Reproduce and use

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m experiments.text_regions.full_rank_compare \
  --out artifacts/reproduced_full_rank_comparison.json
```

For new fits, use scripts/run_full_rank_study.py with a new --out directory. Add --resume only to continue that run. Use --export-compiled with full_rank_compare for a new export. scripts/benchmark_full_rank_readout.py repeats the decoder timings. The artifact ZIP includes all dependencies required for the audit; the source ZIP contains code, reports and aggregates.

## Next development step

Retain the compact identity correction as the leading low-cost decoder candidate, and use its compiled affine form in the next integration experiment. Train/select on the existing train/validation split, lock the choice, then evaluate once on a new template and fact-combination cohort. That will test whether the current gains carry beyond the fixed evaluation set used in these decoder phases. The full-rank phase supplies the comparison, export and timing tools needed for that step.
