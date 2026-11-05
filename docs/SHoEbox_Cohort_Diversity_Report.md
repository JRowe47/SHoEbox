# SHoEbox — decoder cohort diversity
Date: 2026-11-05 UTC

Subsequent reproduction: the missing implementation has now been rebuilt and all 432 fits rerun. See [the reproduction report](SHoEbox_Cohort_Reproduction_Report.md) for fresh results, agreement with the figures below, and complete downloadable artifacts. This document preserves the original study and its historical recovery status.

Expanding the decoder fitting cohort from 960 to 3,840 cases substantially increased validation acceptance of the linear residual correction. Its selected preservation variant improved both private and shared accuracy in three of four conditions. The nonlinear GELU variant improved private accuracy in all four conditions, but reduced shared accuracy in three.

The default model and readout remain unchanged.

## Experiment
Twelve frozen graphs cover original/gated publication policies, dropout/balanced source schedules, and graph seeds 20261106, 20261107, and 20261108. Both cohort sizes were captured and fitted afresh in the same runtime. The small cohort is an exact per-cell feature prefix of the large cohort.

| Training coverage | Small | Large |
|---|---:|---:|
| Cases per supported source/query cell | 64 | 256 |
| Total cases | 960 | 3,840 |
| Distinct episode IDs | 192 | 768 |
| Distinct traces, excluding seven-byte ID headers | 960 | 3,840 |
| Source/query/full-fact/template signatures | 869 | 2,360 |
| Shared / left / right cases | 448 / 256 / 256 | 1,792 / 1,024 / 1,024 |

Both cohorts retain the same 32 training fact combinations. Validation remains fixed at 1,920 cases (384 IDs), test/all at 384 cases, and the independent evaluation at 3,456 cases.

Each graph received two cohort sizes × three preservation penalties (0, 1, 10) × two activations (identity, GELU) × three initializations (20261121–20261123): 432 new fits overall. Each correction has hidden width 16 and 5,392 trainable parameters; the 16,640-parameter linear base and its normalization remain frozen.

Training used full-batch LBFGS with a 400-iteration cap and L2 coefficient 0.001. All 432 fits reached the iteration cap; maximum gradient magnitudes ranged from approximately 0.0000024 to 0.022. Positive penalties protect shared training examples correctly answered by the frozen base, with their protected margins capped at 1.

Validation selected a residual strength from 0, 0.125, 0.25, 0.5, and 1 by query-balanced negative log-likelihood, requiring shared correct counts on both all/shared and b_only/shared to meet the frozen base. Ties prefer the smaller strength. Positive penalties 1 and 10 were selected with the same criteria. Strength zero reproduces the frozen base exactly.

## Results
Values below are mean private / shared accuracy in percent.

| Selected preservation — identity | 960 cases | 3,840 cases |
|---|---:|---:|
| Original / dropout | 73.177 / 87.240 | 77.257 / 88.108 |
| Original / balanced | 55.990 / 80.469 | 57.986 / 82.812 |
| Gated / dropout | 60.938 / 91.146 | 59.809 / 90.104 |
| Gated / balanced | 54.948 / 87.500 | 56.641 / 88.976 |

Nonzero identity corrections selected by validation increased from **3 of 36 to 33 of 36**. The gated/dropout condition declined on both metrics.

| Selected preservation — GELU | 960 cases | 3,840 cases |
|---|---:|---:|
| Original / dropout | 74.045 / 86.372 | 75.087 / 83.420 |
| Original / balanced | 57.248 / 79.080 | 58.984 / 78.038 |
| Gated / dropout | 61.024 / 90.625 | 62.283 / 90.712 |
| Gated / balanced | 56.033 / 86.372 | 56.641 / 86.285 |

Nonzero GELU corrections increased from 18 of 36 to 22 of 36. Only 2 of the 12 individual graphs improved both metrics. Missing-information accuracy averaged approximately 25.1%, against 25% chance.

| Ordinary penalty-zero control | 960 cases | 3,840 cases |
|---|---:|---:|
| Identity, original / dropout | 72.917 / 87.500 | 74.479 / 86.979 |
| Identity, original / balanced | 55.990 / 80.469 | 59.071 / 82.031 |
| Identity, gated / dropout | 60.938 / 91.146 | 61.719 / 90.104 |
| Identity, gated / balanced | 54.948 / 87.500 | 56.380 / 86.198 |
| GELU, original / dropout | 73.611 / 87.153 | 75.000 / 85.590 |
| GELU, original / balanced | 55.990 / 80.469 | 59.983 / 78.472 |
| GELU, gated / dropout | 61.068 / 90.625 | 62.934 / 90.625 |
| GELU, gated / balanced | 55.122 / 87.413 | 55.599 / 87.587 |

| B-only shared accuracy, selected preservation | Identity: small → large | GELU: small → large |
|---|---:|---:|
| Original / dropout | 86.72 → 90.36 | 91.32 → 90.10 |
| Original / balanced | 100.00 → 100.00 | 100.00 → 100.00 |
| Gated / dropout | 95.83 → 95.57 | 96.79 → 96.09 |
| Gated / balanced | 98.70 → 99.22 | 98.96 → 98.70 |

## Verification and implementation
The completed local implementation added cohort capture, nested feature selection, fresh fitting, validation selection, comparison/audit, packaging, and ten dedicated tests. All **305 distinct automated tests passed**; the ten cohort tests were repeated after a normalization reconstruction fix. The live output-slot example also passed.

The audit covered all 12 graphs and 432 objectives, training/evaluation identity separation, fixed base and cache hashes, selected tensors, validation decisions, predictions, and losses. An additional comparison reproduced all 720 legacy prediction groups exactly.

Capturing training features required 96,768,000 expert executions (K=1, 84 per event). Fitting used cached features. Runtime was PyTorch 2.15.0+cpu with two threads per worker and four workers. Legacy recapture feature drift reached 0.00039, motivating fresh captures for both cohort sizes. Normalization reconstruction uses absolute and relative tolerance 1e-12; the base tensors and normalization actually loaded by both arms are identical.

## Publication and recovery status
The original four-page PDF was rendered and verified before the workspace disconnected. This Markdown report reconstructs the completed findings from the retained execution record; it is not a byte-for-byte recovery of the original Markdown.

The development source was committed locally as `c4be984e247f738a63433c7a84d4aa7a2c5918ef` on parent `faed544b37000fcdb11b5584396c7856b85a4be5`. Its publication did not complete. The new code, fitted heads, cached features, full comparison JSON, and ZIPs remain unavailable through the disconnected workspace. This publication preserves the report and recovery metadata only. It does not publish the local development commit or make the complete study reproducible from GitHub alone.

Both ZIPs were built and their entries verified before disconnection: source 4,330,845 bytes / 497 files; complete 117,445,411 bytes / 925 files. Their exact hashes and expected paths are recorded in `experiments/text_regions/results/cohort_diversity/recovery.json`. A fresh-extraction portable audit was not completed.

## Next development step
Refit the single full-rank linear byte decoder on the expanded cohort, compare it with identity and GELU residuals on the same fixed evaluations, and measure inference cost. This tests whether the gains can be retained with one emission path and without the correction branch.
