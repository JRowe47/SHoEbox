# SHoEbox - cohort experiment reproduced

Date: 2026-11-05 UTC

The missing cohort-diversity experiment has been rebuilt and rerun: **12 frozen graphs and 432 fitted corrections**. Validation chooses a nonzero identity correction **3 of 36 times with the small cohort, and 33 of 36 times with the large cohort**. Expanding from 960 to 3,840 training cases improved both private and shared accuracy in 3 of four conditions with the selected identity correction, and 1 of four with GELU.

The repetition matches 80 of 80 published accuracy values at their reported precision. The largest difference from a printed value is 0.004583 percentage points, within its original rounding interval.

## Matched protocol

| Training coverage | Small | Large |
|---|---:|---:|
| Cases per supported source/query cell | 64 | 256 |
| Total cases | 960 | 3,840 |
| Distinct episode IDs | 192 | 768 |
| Traces excluding seven-byte ID headers | 960 | 3,840 |
| Observed template signatures | 869 | 2,360 |
| Shared / left / right examples | 448 / 256 / 256 | 1,792 / 1,024 / 1,024 |
| Full fact combinations | 32 | 32 |

All twelve original checkpoints are retained: original/gated publication, dropout/balanced source schedules, and graph seeds 20261106-20261108. The small training cohort is the exact first 64 feature rows of each large-cohort cell. The retained source templates define the template signature, together with source subset, query and the full fact tuple.

Each graph receives two cohort sizes, penalties 0/1/10, identity/GELU activations and initialization seeds 20261121-20261123. Each correction has hidden width 16 and 5,392 parameters. The 16,640-parameter linear base, its normalization, validation (1,920 cases), test/all (384 cases), and independent evaluation (3,456 cases) remain fixed.

Full-batch L-BFGS uses strong Wolfe line search, a 400-iteration cap and L2 coefficient 0.001. Training-only shared-answer margins are capped at 1. Validation chooses strength 0, 0.125, 0.25, 0.5 or 1 by query-balanced NLL, while requiring shared correct counts on all/shared and b_only/shared to meet the frozen base. The same rule chooses between positive penalties 1 and 10; ties prefer smaller strength. Test scores do not select settings.

---

## Selected preservation results

Values are private / shared accuracy in percent, averaged over three graph seeds and three correction initializations. Change is large minus small, in percentage points.

**Identity correction**

| Condition | 960 cases | 3,840 cases | Change, pp |
|---|---:|---:|---:|
| Original / dropout | 73.177 / 87.240 | 77.257 / 88.108 | +4.080 / +0.868 |
| Original / balanced | 55.990 / 80.469 | 57.986 / 82.812 | +1.997 / +2.344 |
| Gated / dropout | 60.938 / 91.146 | 59.809 / 90.104 | -1.128 / -1.042 |
| Gated / balanced | 54.948 / 87.500 | 56.641 / 88.976 | +1.693 / +1.476 |

**GELU correction**

| Condition | 960 cases | 3,840 cases | Change, pp |
|---|---:|---:|---:|
| Original / dropout | 74.045 / 86.372 | 75.087 / 83.420 | +1.042 / -2.951 |
| Original / balanced | 57.248 / 79.080 | 58.984 / 78.038 | +1.736 / -1.042 |
| Gated / dropout | 61.024 / 90.625 | 62.283 / 90.712 | +1.259 / +0.087 |
| Gated / balanced | 56.033 / 86.372 | 56.641 / 86.285 | +0.608 / -0.087 |

| Nonzero correction selected | 960 cases | 3,840 cases |
|---|---:|---:|
| Identity | 3 / 36 | 33 / 36 |
| GELU | 18 / 36 | 22 / 36 |

**B-only shared accuracy**

| Condition | Identity: small / large | GELU: small / large |
|---|---:|---:|
| Original / dropout | 86.72 / 90.36 | 91.32 / 90.10 |
| Original / balanced | 100.00 / 100.00 | 100.00 / 100.00 |
| Gated / dropout | 95.83 / 95.57 | 96.79 / 96.09 |
| Gated / balanced | 98.70 / 99.22 | 98.96 / 98.70 |

Large-cohort selected GELU missing-information accuracy averages 25.049%, against 25% chance. The default model/readout remains unchanged.

---

## Penalty-zero controls

These use the same fitting, fixed baseline and validation scale selection, with preservation penalty zero.

**Identity correction**

| Condition | 960 cases | 3,840 cases | Change, pp |
|---|---:|---:|---:|
| Original / dropout | 72.917 / 87.500 | 74.479 / 86.979 | +1.562 / -0.521 |
| Original / balanced | 55.990 / 80.469 | 59.071 / 82.031 | +3.082 / +1.562 |
| Gated / dropout | 60.938 / 91.146 | 61.719 / 90.104 | +0.781 / -1.042 |
| Gated / balanced | 54.948 / 87.500 | 56.380 / 86.198 | +1.432 / -1.302 |

**GELU correction**

| Condition | 960 cases | 3,840 cases | Change, pp |
|---|---:|---:|---:|
| Original / dropout | 73.611 / 87.153 | 75.000 / 85.590 | +1.389 / -1.562 |
| Original / balanced | 55.990 / 80.469 | 59.983 / 78.472 | +3.993 / -1.997 |
| Gated / dropout | 61.068 / 90.625 | 62.934 / 90.625 | +1.866 / +0.000 |
| Gated / balanced | 55.122 / 87.413 | 55.599 / 87.587 | +0.477 / +0.174 |

## Agreement with the surviving report

| Published metric group | Values matching precision | Largest difference, pp |
|---|---:|---:|
| selected / identity | 16 / 16 | 0.000500 |
| selected / gelu | 16 / 16 | 0.000472 |
| penalty_0 / identity | 16 / 16 | 0.000500 |
| penalty_0 / gelu | 16 / 16 | 0.000472 |
| B-only shared | 16 / 16 | 0.004583 |

The reference contains 64 private/shared means rounded to three decimals and 16 B-only means rounded to two. Agreement uses half of the last displayed decimal place. `reported_comparison.json` records every original value, repeated value and difference; `reported_metrics.json` preserves the values transcribed from the historical report.

This rerun uses the preserved protocol and inputs without tuning to the published scores. The unavailable original implementation prevents a byte-level comparison with that original study. The new package supplies a complete, auditable replacement.

---

## Verification and reproducibility

All **307 automated tests pass** across five suites. Nine dedicated cohort tests cover coverage counts, exact nested feature ownership, split separation, frozen normalization, checkpoint restoration and resume rejection after incompatible changes.

The independent artifact audit reconstructs all 432 training objectives, their validation scale/penalty selections, and all 2,880 new prediction groups. It also exactly reproduces 720 prediction groups from the retained residual and emission studies. Saved predictions and discrete decisions compare exactly; named floating loss fields and normalization reconstruction use absolute/relative tolerance 1e-12. Loaded frozen base tensors remain identical.

All input/source hashes, checkpoint lineages, synthetic cases, coverage counts and expert budgets are checked. Training IDs are disjoint from validation and evaluation. The audit checks predictions from saved tensors rather than trusting the reported summary.

432 of 432 fits reached the 400-iteration cap. Final maximum gradient magnitudes range from 2.394007e-06 to 0.02225517. The largest feature difference from the legacy small training cache is 0.0003931746; both newly fitted cohort sizes use the same fresh capture, while retaining the original base normalization.

Training capture used **96,768,000 expert executions**, with K=1 and 84 per event. Fitting used cached features. Runtime: PyTorch 2.15.0+cpu, two threads per worker and four fitting workers. Every fitted correction is checkpointed atomically. Resume verifies the exact source, configuration, inputs, capture and base before continuing.

## Downloadable package

`SHoEbox_cohort_reproduction_source.zip` contains the source, reports, aggregate audit and verification logs. `SHoEbox_cohort_reproduction_artifacts.zip` additionally contains every required graph, inherited readout, feature/validation cache, all 432 newly fitted corrections, selections and predictions. Each archive has a file-by-file SHA-256 manifest, its source commit and publication status.

From the extracted artifact package:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m experiments.text_regions.diversity_compare \
  --source artifacts/cohort_reproduction_v1 \
  --out artifacts/reproduced_cohort_comparison.json
```

For a fresh fit, run `scripts/run_diversity_study.py` with a new output directory, `--jobs 4 --threads 2`. To continue an interrupted fit, add `--resume` with its existing output directory.

The next development step remains a full-rank linear decoder refit on the expanded cohort, compared on the same fixed evaluations and measured for inference cost. That follow-up is separate from this completed reproduction.
