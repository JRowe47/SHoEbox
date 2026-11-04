# SHoEbox: shared-answer preservation during training

November 4, 2026 | Twelve frozen graphs, 144 matched residual fits

**Training-time preservation makes a small improvement, but the shared/private tradeoff remains.** The selected GELU correction raises private accuracy from 56.0% to 57.7% on original/balanced graphs and from 55.2% to 55.9% on gated/balanced graphs relative to the preceding residual study. Shared accuracy falls by 1.0 percentage point in the first case and returns to the frozen baseline's 87.5% in the second. The default decoder remains unchanged.

The new loss protects shared answers that the frozen linear decoder already answers correctly on the training cohort. Validation then chooses both penalty strength and residual scale. This permits three GELU heads to retain a scale of 0.5, where the previous study selected none. However, 19 of 36 GELU heads still select zero, versus 22 previously.

## Held-out private answers

| Frozen graph | Frozen linear | Previous selected GELU | New selected GELU | New selected identity |
|---|---:|---:|---:|---:|
| Original / dropout | 73.7% | 73.7% | 73.8% | 73.0% |
| Original / balanced | 56.0% | 56.0% | 57.7% | 56.0% |
| Gated / dropout | 60.9% | 60.9% | 61.1% | 60.9% |
| Gated / balanced | 54.9% | 55.2% | 55.9% | 54.9% |

Each row averages three graph seeds. Each new decoder result first averages three initialization seeds within each graph. No initialization is selected as a winner. Private accuracy averages the left and right queries on all-source held-out combinations.

## Shared answers and overall accuracy

| Frozen graph | Previous shared | New shared | Previous overall | New overall |
|---|---:|---:|---:|---:|
| Original / dropout | 87.2% | 87.2% | 78.2% | 78.2% |
| Original / balanced | 80.5% | 79.4% | 64.1% | 64.9% |
| Gated / dropout | 90.5% | 90.5% | 70.8% | 70.9% |
| Gated / balanced | 87.4% | 87.5% | 65.9% | 66.4% |

These columns compare validation-selected GELU corrections. The exact original/balanced shared change is -1.04 percentage points; rounded column values differ by 1.1 points. The validation guards preserve correct-answer counts on validation, while the table reports previously unseen test combinations.

The original/balanced private gain appears in graph seeds 1 and 3; seed 2 stays unchanged. The gated/balanced gain appears in seeds 1 and 2; seed 3 stays unchanged. The largest observed original/balanced gain is 4.17 points in graph seed 1, accompanied by a 1.56-point shared reduction.

---

## Recovery from B alone

| Frozen graph | Frozen linear | Previous selected GELU | New selected GELU |
|---|---:|---:|---:|
| Original / dropout | 85.7% | 90.1% | 91.0% |
| Original / balanced | 100.0% | 100.0% | 100.0% |
| Gated / dropout | 95.8% | 97.3% | 96.9% |
| Gated / balanced | 98.7% | 98.6% | 98.9% |

This independent control removes A and C and covers all 64 fact combinations. It uses the same frozen graphs as the held-out all-source test. Missing-information control accuracy averages approximately 25% in every graph family, against the balanced four-answer reference rate of 25%.

## What the loss preserves

For each correctly answered shared training example, define its margin as the true-byte logit minus the largest of the other 255 logits. Its protected target is the smaller of the frozen baseline margin and 1 nat. The added loss is the mean squared positive shortfall below this target, normalized over protected examples.

**Objective = query-balanced byte cross entropy + correction L2 + lambda x shared-margin shortfall.**

Only training examples define the protected set. Incorrect baseline answers and private-query examples receive the ordinary supervised objective without a preservation anchor. The method does not copy a teacher's output distribution. All targets come from the existing synthetic fact generator.

Across the 36 GELU fits at lambda=1, only 5 of 13,389 protected training answers become incorrect. At lambda=10, only 1 does. These counts pool repeated fitting conditions rather than distinct episodes. The margin floor can still be violated without changing the predicted byte: 315 and 195 protected margins fall below their targets, respectively.

The strong training preservation does not transfer fully to unseen combinations. At full residual scale, private gains still accompany substantial shared-answer losses:

| Frozen graph | Lambda 1 private | Lambda 1 shared | Lambda 10 private | Lambda 10 shared |
|---|---:|---:|---:|---:|
| Original / dropout | 71.6% | 83.5% | 72.0% | 83.9% |
| Original / balanced | 62.1% | 66.8% | 63.5% | 64.5% |
| Gated / dropout | 63.2% | 85.9% | 61.9% | 85.4% |
| Gated / balanced | 61.5% | 79.2% | 60.0% | 77.7% |

These full-scale results are diagnostics. They do not replace validation-selected results.

---

## Fitting and selection protocol

| Setting | Value |
|---|---|
| Frozen graphs | 2 publication policies x 2 source schedules x 3 graph seeds |
| Graph seeds | 20261106, 20261107, 20261108 |
| Correction seeds | 20261121, 20261122, 20261123 |
| Main fits | 144: identity/GELU x 2 penalties x 3 initializations x 12 graphs |
| Prior controls | 72 previously fitted unpenalized residual heads, fully reaudited |
| Input / output | Observed 64-dimensional raw P/Q features / 256 byte logits |
| Correction | 64-to-16-to-256, identity or GELU, zero output at initialization |
| Parameters | 5,392 trainable correction; 16,640 frozen linear; 22,032 total |
| Training | 960 supported cases; 192 distinct episode identities per graph |
| Validation | 1,920 cases; 384 distinct identities per graph |
| Evaluation | 384 held-out all-source cases; 3,456 independent controls per graph |
| Optimizer | Full-batch L-BFGS, strong Wolfe, 400 iterations, correction L2=0.001 |
| Penalty / margin cap | Lambda in {1, 10}; cap of 1 nat |
| Residual scale grid | 0, 0.125, 0.25, 0.5, 1 |
| Additional graph executions | Zero; reuse verified frozen feature and validation caches; K=1 |

Every identity/GELU pair shares initial tensors, parameter count, examples, and optimizer settings. The base decoder and all graph weights stay frozen. The prediction interface consumes features only; query labels and source-control labels never enter it.

For each candidate penalty, selection minimizes query-balanced validation cross entropy among scales whose all-source/shared and B-only/shared exact-correct counts are at least the frozen baseline's counts. It then compares the two eligible penalty candidates using the same objective. Ties within 1e-12 prefer the smaller residual scale, then the smaller penalty. Zero remains an exact fallback. Selection locks are saved before test scoring.

| Selected scale | Previous GELU | New GELU | New identity |
|---|---:|---:|---:|
| 0 | 22 | 19 | 33 |
| 0.125 | 10 | 11 | 1 |
| 0.25 | 4 | 3 | 2 |
| 0.5 | 0 | 3 | 0 |
| 1 | 0 | 0 | 0 |

Of the 17 nonzero GELU selections, 11 use lambda=1 and 6 use lambda=10. Zero selections use lambda=1 by the tie rule; that is not evidence favoring that penalty when no correction is used.

---

## Longer-fit diagnostic

All 144 main fits reach the 400-iteration cap. Because the largest coefficient gradient is 0.02549, a separate convergence check selects the largest training-gradient fit in each penalty/activation cell and refits it from the same initialization with a 1,600-iteration cap. No test score is used to choose these four cases, and the main results above stay fixed.

| Penalty / activation | Gradient at 400 | Gradient after extension | Final iterations | Selected scale before / after |
|---|---:|---:|---:|---:|
| 1 / identity | 0.007361 | 0.001373 | 948 | 0 / 0 |
| 1 / GELU | 0.004347 | 0.000378 | 1,600 | 0 / 0 |
| 10 / identity | 0.009343 | 0.000648 | 1,600 | 0 / 0 |
| 10 / GELU | 0.025489 | 0.001613 | 1,600 | 0.125 / 0 |

All four training objectives decrease. Three selections remain zero; the fourth also becomes zero. Longer optimization does not rescue these four corrections under the unchanged validation rule. Three extended fits still reach their iteration cap; the identity/lambda=1 fit stops at 948 iterations with a nonzero gradient.

## Verification and delivery

All **295 automated tests pass**, including 11 new preservation tests. The live output-slot example also passes. Zero penalty reproduces the previous residual optimizer's tensors and objective exactly. A deliberately conflicting training problem verifies that the new objective can prevent a shared answer from being overwritten.

Independent audits reconstruct every main and extended objective, protected training set, validation choice, selected tensor, saved prediction, and evaluation loss. The preceding 72-head residual study is reaudited against its original graph checkpoints and cached features. Train, validation, test, and independent episode identities remain disjoint. Prior raw-linear and residual predictions reproduce exactly.

The source package includes implementation, regression tests, aggregate results, and reports. The complete downloadable ZIP additionally includes all twelve frozen graph checkpoints, upstream features and heads, the 144 new corrections, four longer-fit diagnostics, selection locks, and full synthetic prediction rows. Every packaged file has a size and SHA-256 entry.

```bash
python scripts/run_preservation_study.py \
  --out artifacts/preservation_repeat --jobs 3 --threads 2

python -m experiments.text_regions.preservation_compare \
  --source artifacts/shared_preservation_v1 \
  --convergence artifacts/preservation_convergence_v1 \
  --out artifacts/reproduced_preservation_comparison.json

python scripts/check_all.py
```

## Next development step

Increase the diversity of the decoder's fitting cohort while keeping held-out fact combinations, validation, and tests fixed. Compare ordinary and preservation losses with the same frozen graph features, correction width, and fitting budget. The current result isolates the problem: shared training answers can be protected, but that protection generalizes poorly to unseen combinations. More distinct training descriptions is the next controlled test before changing the regional architecture.
