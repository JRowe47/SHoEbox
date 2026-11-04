# SHoEbox: residual byte decoders

2026-11-04 | Frozen linear path, 16-unit corrections, validation-only selection

**The residual preserves most baseline behavior but does not solve the shared/private tradeoff.** Validation chooses zero strength for 22 of 36 GELU heads and 33 of 36 matched identity heads. Selected GELU improves B-only recovery on dropout-trained graphs, while private-answer accuracy remains nearly unchanged. The default decoder stays unchanged.

## Private answers on held-out combinations

| Frozen graph | Frozen linear | Identity selected | GELU full | GELU selected |
|---|---:|---:|---:|---:|
| Original / dropout | 73.7% | 72.9% | 72.6% | 73.7% |
| Original / balanced | 56.0% | 56.0% | 61.2% | 56.0% |
| Gated / dropout | 60.9% | 60.9% | 62.5% | 60.9% |
| Gated / balanced | 54.9% | 54.9% | 62.4% | 55.2% |

## Shared answers on held-out combinations

| Frozen graph | Frozen linear | Identity selected | GELU full | GELU selected |
|---|---:|---:|---:|---:|
| Original / dropout | 87.0% | 87.5% | 82.0% | 87.2% |
| Original / balanced | 80.5% | 80.5% | 66.3% | 80.5% |
| Gated / dropout | 91.1% | 91.1% | 85.3% | 90.5% |
| Gated / balanced | 87.5% | 87.5% | 74.7% | 87.4% |

At full strength, the balanced-source GELU corrections improve private accuracy from **56.0% to 61.2%** under original publication and **54.9% to 62.4%** under evidence gating. Shared accuracy falls by **14.1 and 12.8 percentage points**. Validation therefore backs off most corrections, also removing their private-answer gains.

## Overall held-out accuracy

| Frozen graph | Frozen linear | Identity selected | GELU selected |
|---|---:|---:|---:|
| Original / dropout | 78.1% | 77.8% | 78.2% |
| Original / balanced | 64.1% | 64.1% | 64.1% |
| Gated / dropout | 71.0% | 71.0% | 70.8% |
| Gated / balanced | 65.8% | 65.8% | 65.9% |

Each row averages three frozen graph seeds. New decoder results first average three initialization seeds within each graph. Every head emits one of 256 bytes from the same observed raw feature vector; query labels never select a decoder or enter its prediction interface. Full-strength results are diagnostics, not deployment selections.

---

## B-only shared recovery

| Frozen graph | Frozen linear | Prior GELU | Selected residual GELU |
|---|---:|---:|---:|
| Original / dropout | 85.7% | 97.6% | 90.1% |
| Original / balanced | 100.0% | 99.8% | 100.0% |
| Gated / dropout | 95.8% | 99.1% | 97.3% |
| Gated / balanced | 98.7% | 99.9% | 98.6% |

B-only evaluation removes A and C and covers all 64 fact combinations. The all-source held-out task above uses 16 combinations excluded from fitting. These conditions share the same frozen models but test different information paths.

## What validation selected

| Strength | Identity heads | GELU heads |
|---|---:|---:|
| 0 | 33 | 22 |
| 0.125 | 0 | 10 |
| 0.25 | 3 | 4 |
| 0.5 | 0 | 0 |
| 1 | 0 | 0 |

One scalar per head is chosen from the fixed grid **0, 0.125, 0.25, 0.5, 1**. Selection minimizes query-balanced validation cross entropy, subject to both all-source/shared and B-only/shared exact-correct counts being at least their frozen-linear baseline. Ties within 1e-12 prefer the smaller strength. Zero is always feasible and reproduces the baseline exactly.

For 14 of the 22 zero-selected GELU heads, every nonzero candidate violates a shared-answer guard. For the other eight, an eligible nonzero candidate exists but does not improve the selection objective enough to beat zero. Identity has 15 guard-only rejections and 18 objective-based zero selections.

The guards apply to validation. Test shared accuracy changes from **91.1% to 90.5%** on gated/dropout graphs and **87.5% to 87.4%** on gated/balanced graphs. They do not guarantee preservation on unseen examples.

## Variation across graphs and initializations

| Frozen graph | Selected GELU private: graph range | Largest within-graph initialization range |
|---|---:|---:|
| Original / dropout | 68.1% to 79.3% | 2.7 pp |
| Original / balanced | 49.2% to 62.1% | 0.0 pp |
| Gated / dropout | 55.1% to 68.4% | 2.0 pp |
| Gated / balanced | 37.5% to 69.8% | 2.0 pp |

Ranges describe observed seed variation, not confidence intervals. All three initializations are reported without choosing a winner. Selected GELU averages **25.1%** on missing-information controls, against a 25% balanced-answer reference rate.

---

## Implementation and protocol

The decoder computes frozen linear logits plus a scaled correction. The correction uses the frozen head's training normalization, a 64-to-16 hidden projection, identity or GELU activation, and a 16-to-256 output projection. Its output weights and bias start at zero, giving exactly the original logits at initialization. Only correction parameters receive gradients.

| Setting | Value |
|---|---|
| Frozen graphs | 12: two publication policies x two source schedules x three graph seeds |
| Graph seeds / correction seeds | 20261106-20261108 / 20261121-20261123 |
| Fitted corrections | 72: identity and GELU, three initializations, twelve graphs |
| Parameters per decoder | 5,392 trainable correction + 16,640 frozen linear = 22,032 weights/biases |
| Training per graph | 960 supported source/query cases; 192 distinct episode identities |
| Validation per graph | 1,920 cases; 384 distinct identities; 16 separate fact combinations |
| Evaluation per graph | 384 all-source held-out cases; 3,456 independent source controls |
| Optimizer | Full-batch L-BFGS, strong Wolfe; 400-iteration cap; L2 0.001 on correction weights/biases |
| Loss weights | Equal total mass per query; uniform supported subsets within query |
| Additional graph work | 4,032,000 expert calls per graph; 48,384,000 total; K=1 |

Training, test and independent features are reused from the previous study. Only validation requires new graph execution. Each identity/GELU pair shares initial tensors, parameter counts and fitting settings. The prior standalone GELU comparison used 64 hidden units and 20,800 trainable parameters; this study's residual correction uses 16 units. Lower trainable count does not imply lower total inference cost because the frozen linear path remains active.

All **72 of 72 fits** reach the 400-iteration cap; the largest final coefficient gradient is **0.001440**. The fit budget is fixed across conditions. A selection-lock file records validation candidate scores and the chosen strength before evaluation runs.

## Verification and reproduction

All **284 automated tests pass**, including 15 new residual checks. The live-output example also passes. Independent artifact audits reconstruct every synthetic cohort, verify frozen graph/base identities and source hashes, reproduce all saved predictions and losses, reconstruct training objectives, and reproduce every validation decision. Prior raw-linear predictions match exactly. Prediction exports are written atomically and verified before a run is marked complete.

```bash
python scripts/run_residual_study.py \
  --out artifacts/residual_reproduction --jobs 3 --threads 2
python -m experiments.text_regions.residual_compare \
  --source artifacts/residual_adapter_v1 \
  --out artifacts/reproduced_residual_comparison.json
python scripts/check_all.py
```

The complete download includes source, reports, original checkpoints, earlier feature caches/heads, new corrections, validation caches, selection locks, and full synthetic prediction rows. The public repository contains code, reports and aggregate results.

## Next development step

Move shared-answer preservation into correction training, so fitting searches for private gains that coexist with shared accuracy. Retain the frozen baseline, parameter-matched identity control and validation-only strength selection. Post-fit scaling alone has now shown its limit: it mostly restores the old answer boundary instead of finding a better shared/private boundary.
