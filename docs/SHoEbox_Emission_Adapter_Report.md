# SHoEbox: shared emission adapters

2026-11-04 | Frozen graphs, matched linear controls, common byte outputs

The shared GELU decoder reaches **62.8% private accuracy** on original/balanced graphs and **62.8%** on gated/balanced graphs. The previous raw linear heads score 56.0% and 54.9%; parameter-matched linear controls score 56.8% and 54.5%.

The experts remain frozen. Each adapter infers the requested answer from one observed vector and selects among all 256 output bytes. Query labels never enter the head or select a task-specific decoder.

## Private answers from full P/Q outputs

| Frozen graph | Existing | Prior linear | Matched linear | GELU |
|---|---:|---:|---:|---:|
| Original / dropout | 76.3% | 73.7% | 72.3% | 73.5% |
| Original / balanced | 49.2% | 56.0% | 56.8% | 62.8% |
| Gated / dropout | 57.2% | 60.9% | 60.5% | 64.3% |
| Gated / balanced | 51.6% | 54.9% | 54.5% | 62.8% |

## Private answers from the compressed latent

| Frozen graph | Existing | Prior linear | Matched linear | GELU |
|---|---:|---:|---:|---:|
| Original / dropout | 76.3% | 73.2% | 70.1% | 73.8% |
| Original / balanced | 49.2% | 55.1% | 56.1% | 61.7% |
| Gated / dropout | 57.2% | 56.5% | 55.9% | 62.5% |
| Gated / balanced | 51.6% | 51.4% | 55.3% | 60.7% |

Each row averages three independently trained graphs. Each new adapter result first averages three initialization seeds within a graph. The identity and GELU heads use the same parameter count, initial tensors, normalization, loss weights, regularization, and optimizer budget.

## Overall held-out accuracy, full outputs

| Frozen graph | Existing | Prior linear | Matched linear | GELU |
|---|---:|---:|---:|---:|
| Original / dropout | 78.5% | 78.1% | 78.2% | 76.1% |
| Original / balanced | 61.5% | 64.1% | 64.7% | 63.9% |
| Gated / dropout | 67.8% | 71.0% | 70.6% | 71.2% |
| Gated / balanced | 65.2% | 65.8% | 64.6% | 67.7% |

Shared accuracy falls from 80.5% to 66.1% on original/balanced graphs and from 87.5% to 77.5% on gated/balanced graphs. Overall accuracy moves from 64.1% to 63.9%, and from 65.8% to 67.7%, respectively.

---

## Shared answers on held-out combinations

| Frozen graph | Prior raw linear | Matched raw linear | Raw GELU | Mixed GELU |
|---|---:|---:|---:|---:|
| Original / dropout | 87.0% | 90.1% | 81.3% | 76.9% |
| Original / balanced | 80.5% | 80.5% | 66.1% | 64.1% |
| Gated / dropout | 91.1% | 90.6% | 84.9% | 89.8% |
| Gated / balanced | 87.5% | 84.6% | 77.5% | 71.1% |

## B-only shared recovery

| Frozen graph | Raw linear | Raw GELU | Mixed linear | Mixed GELU |
|---|---:|---:|---:|---:|
| Original / dropout | 85.7% | 97.6% | 70.1% | 96.4% |
| Original / balanced | 100.0% | 99.8% | 100.0% | 100.0% |
| Gated / dropout | 95.8% | 99.1% | 88.0% | 96.9% |
| Gated / balanced | 98.7% | 99.9% | 96.4% | 99.3% |

These are different conditions: the held-out test supplies all sources with 16 unseen fact combinations; B-only controls remove A and C and cover all 64 fact combinations. The same stored cases and all three earlier baseline predictions reproduce exactly.

## Private-answer variation across seeds

| Frozen graph | Raw GELU graph means | Mixed GELU graph means | Largest raw initialization spread |
|---|---:|---:|---:|
| Original / dropout | 67.1% to 78.6% | 68.5% to 78.8% | 7.0 pp |
| Original / balanced | 51.4% to 69.3% | 46.2% to 72.0% | 5.5 pp |
| Gated / dropout | 63.2% to 65.8% | 61.2% to 64.3% | 5.1 pp |
| Gated / balanced | 51.6% to 68.9% | 52.1% to 65.6% | 5.9 pp |

Graph ranges show the three graph means after averaging adapter initializations. The final column shows the largest range across the three raw-GELU initializations within any one graph; it is not a confidence interval.

Missing-information controls average 25.1% for raw GELU and 25.2% for mixed GELU. Their matched linear controls average 25.0% and 25.1%. Four balanced answers give a 25% reference rate.

All fits use the same fixed 400-iteration L-BFGS cap; **141 of 144** reach that cap. The largest final coefficient gradient is 0.004800. No test score selects a fit, initialization, or setting.

---

## Implementation and experimental protocol

The adapter is a 64-unit hidden layer followed by one 256-byte output layer. The matched control replaces GELU with the identity function, so its two layers collapse to an affine map at inference. Raw and mixed adapters have 20,800 and 18,752 trainable coefficients, respectively, including biases. The earlier single-layer raw/mixed fits have 16,640 and 8,448 coefficients.

| Setting | Value |
|---|---|
| Frozen graphs | 12: two publication policies, two source schedules, three graph seeds |
| Graph seeds | 20261106, 20261107, 20261108 |
| Adapter initialization seeds | 20261121, 20261122, 20261123 |
| Fitted heads | 144: two representations x two activations x three initializations x twelve graphs |
| Training examples per graph | 960 supported source/query cases; 192 distinct episode identities |
| Evaluation per graph | 384 held-out cases and 3,456 independent controls |
| Optimization | Full-batch L-BFGS, strong Wolfe search, at most 400 iterations |
| Regularization | L2 0.001 on both layers and all biases |
| Graph work | 120,000 external events and 10,080,000 expert calls per graph; K=1 |

Training uses generator-provided ground-truth bytes, with equal total loss weight per query and uniform supported subsets within each query. Standardization uses only the weighted training features. The same captured features serve all twelve new heads per graph; adapter fitting makes no additional expert calls. The complete study captures **120,960,000 expert executions**. Cached features support further decoder studies without repeating those executions.

## Verification and reproduction

All **269 automated checks pass**, including 14 new adapter and audit checks. These cover nonlinear XOR learning, matched initialization, identity-map collapse, deterministic fitting, unchanged RNG and input gradients, unrestricted byte output, tensor restoration, data leakage rejection, and altered prediction/metric rejection.

The independent audit restores every head, reproduces every saved prediction from cached features, reconstructs synthetic cases and every fitted training objective, verifies the original linear baselines byte-for-byte, and checks frozen model identities, source hashes, normalization, parameter counts, and expert budgets. One incomplete prediction export was reconstructed from its intact saved heads and features; every original score and the preserved file prefix matched. Recovery provenance is included in the package.

```bash
python scripts/run_emission_study.py \
  --out artifacts/emission_reproduction --jobs 3 --threads 2
python -m experiments.text_regions.emission_compare \
  --source artifacts/emission_adapter_v1 \
  --out artifacts/reproduced_emission_comparison.json
python scripts/check_all.py
```

The complete package restores the original models, previous readouts, all 144 new heads, compact feature caches, and synthetic prediction rows. Raw experiment rows remain in the downloadable package. Development base: 5659be5c103ea3e83f272a8517774991b3d291da.

## Next development step

Test a small nonlinear residual around the existing raw linear head, retaining that linear map and learning only the correction. Use the adapter validation split to set correction strength. The target is the private-answer gain with better preservation of shared-answer behavior.
