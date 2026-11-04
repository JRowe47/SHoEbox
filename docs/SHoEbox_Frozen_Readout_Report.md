# SHoEbox: frozen readout diagnostic

2026-11-04 | https://github.com/JRowe47/SHoEbox

Fitting a common decoder to the full P/Q outputs recovers much of the original model's missing-source capability. Across three seeds, original-policy/dropout B-only shared accuracy rises from **34.4% to 85.7%**, with the graph and expert weights frozen. Held-out overall accuracy is **78.5% versus 78.1%**; private accuracy is **76.3% versus 73.7%**.

Refitting the existing compressed representation already raises B-only accuracy to **70.1%**. Access to the full output adds another **15.6 percentage points**. Decoder fitting and output compression both contribute to the missing-source failure.

## B-only shared recovery

| Frozen graph | Existing decoder | Refit mixed 32-D | Refit raw 64-D |
|---|---:|---:|---:|
| Original policy / dropout | 34.4% | 70.1% | 85.7% |
| Original policy / balanced | 100.0% | 100.0% | 100.0% |
| Gated policy / dropout | 84.9% | 88.0% | 95.8% |
| Gated policy / balanced | 95.6% | 96.4% | 98.7% |

## Held-out private answers, all sources

| Frozen graph | Existing decoder | Refit mixed 32-D | Refit raw 64-D |
|---|---:|---:|---:|
| Original policy / dropout | 76.3% | 73.2% | 73.7% |
| Original policy / balanced | 49.2% | 55.1% | 56.0% |
| Gated policy / dropout | 57.2% | 56.5% | 60.9% |
| Gated policy / balanced | 51.6% | 51.4% | 54.9% |

The gated/dropout graph improves both outcomes with the raw decoder: private accuracy rises from **57.2% to 60.9%**, while B-only recovery rises from **84.9% to 95.8%**. Balanced-source graphs retain excellent B-only recovery but still trail the original/dropout graph on private answers.

## Original-policy/dropout seeds

| Fine-tuning seed | Existing B-only | Raw B-only | Existing private | Raw private |
|---|---:|---:|---:|---:|
| 20261106 | 25.0% | 93.0% | 78.1% | 79.7% |
| 20261107 | 50.0% | 73.4% | 74.6% | 73.4% |
| 20261108 | 28.1% | 90.6% | 76.2% | 68.0% |

Each number above uses a fresh, common diagnostic cohort. Private accuracy pools left and right queries; reported group values are means across three fine-tuning seeds. The earlier report used a different evaluation seed and 512 cases per query; this study uses 128 per query, so its direct-decoder scores are the proper reference for these comparisons.

---

## Where is the information?

Fixed factor readouts can decode substantially more private information than the common answer decoder extracts. This holds for both the raw P/Q concatenation and the current compressed latent.

| Frozen graph | Final raw factor probes | Final mixed factor probes | Common raw answer decoder |
|---|---:|---:|---:|
| Original policy / dropout | 85.0% | 83.2% | 73.7% |
| Original policy / balanced | 80.5% | 79.7% | 56.0% |
| Gated policy / dropout | 79.4% | 78.8% | 60.9% |
| Gated policy / balanced | 86.6% | 82.7% | 54.9% |

Factor probes are separate four-answer linear diagnostics for fixed left/right facts, fitted across all query types and supported source conditions. The common answer decoder is one 256-byte head for every query and source condition. Query labels only select reporting subsets; they never select an answer head or enter its features.

For the balanced original-policy graphs, private facts are **80.5%** decodable from raw outputs and **79.7%** from mixed outputs, while the common raw answer decoder reaches **56.0%**. The private-answer gap is therefore larger than the loss attributable to compression alone. The next target is learning to convert the available regional representation into the requested answer.

## Representation changes across query processing

| Frozen graph | Early head, early features | Final head, final features | Early head, final features |
|---|---:|---:|---:|
| Original policy / dropout | 47.4% | 85.0% | 27.2% |
| Original policy / balanced | 50.4% | 80.5% | 43.9% |
| Gated policy / dropout | 40.6% | 79.4% | 37.6% |
| Gated policy / balanced | 49.9% | 86.6% | 51.4% |

These are raw-output private-factor scores. Early features are captured at the description boundary, before the first query marker. Final features follow the last query. Separate early and final probes are each fitted once on training data and then held fixed.

Final-query factor decodability is much higher than early-boundary decodability. A fixed early probe often transfers poorly to final features. The representation changes during query processing; poor cross-time probe transfer is not evidence that the final representation lacks the facts.

## Held-out overall accuracy

| Frozen graph | Existing decoder | Refit mixed 32-D | Refit raw 64-D |
|---|---:|---:|---:|
| Original policy / dropout | 78.5% | 77.3% | 78.1% |
| Original policy / balanced | 61.5% | 64.5% | 64.1% |
| Gated policy / dropout | 67.8% | 67.9% | 71.0% |
| Gated policy / balanced | 65.2% | 64.0% | 65.8% |

Missing-information controls average **25.2%** for each decoder family (direct 25.2%, mixed 25.2%, raw 25.2%), against a four-answer chance rate of 25%. Every decoder still chooses among all 256 bytes.

---

## Protocol and implementation

Twelve frozen checkpoints cover two publication policies, two original training schedules and three fine-tuning seeds: 20261106, 20261107 and 20261108. Each paired schedule comparison retains the same original initializer and 240-step training protocol. Both baseline interventions remain disabled, view objectives are enabled and utility updates are disabled.

| Per frozen checkpoint | Count |
|---|---:|
| Common decoder fitting cases | 960 |
| Distinct fitting episode identities | 192 |
| Supported query/source fitting cells | 15 |
| Held-out all-source evaluation cases | 384 |
| Independent source/reset evaluation cases | 3,456 |
| Total captured cases | 4,800 |
| External graph events | 120,000 |
| Expert executions, K=1 | 10,080,000 |

All models use diagnostic seed 20261120. Fitting uses 64 cases per supported cell: seven source subsets for shared queries, four retaining A for left-private queries and four retaining C for right-private queries. The same base episodes are reused under the different source interventions. Loss weights give each query type one-third of the total mass, with uniform source-subset weighting within that query.

Evaluation uses 128 cases per query/control. The held-out test covers 16 unseen fact combinations. Independent controls cover all 64 combinations, all seven nonempty source subsets, private-value withholding and reset before the final query. Every model sees identical diagnostic cases, bytes, query markers and source interventions. No fit/evaluation episode identities overlap.

The two answer heads use identical training cases and a deterministic, zero-initialized, full-batch 256-class logistic fit. L-BFGS uses at most 300 iterations, L2 coefficient 0.001 on all weights including the intercept, and training-only weighted feature standardization. Raw and mixed heads have 16,640 and 8,448 fitted coefficients, respectively. All fitted heads finish below 0.00001 maximum gradient. Fixed factor probes use ridge coefficient 0.01.

The graph executes once per batch; its captured features serve both answer heads and every factor probe. The study totals **120,960,000 expert executions**. Captured representations reconstruct the ordinary readout path, and model tensors, publication policy, module modes and RNG remain unchanged during captures.

---

## Verification and reproduction

**All 255 automated checks pass**: 60 preserved baseline checks and 195 regional-model checks. Ten checks cover the new diagnostic and comparator, including byte-unrestricted output, label-free capture, batching-invariant hashes, source coverage, safe tensor restoration, and rejection of altered predictions or evaluation cases.

The portable comparator independently reconstructs the diagnostic inputs and episode identities, recomputes 1,080 answer accuracy records from raw predictions, verifies all twelve model records and six matched training pairs, and checks fixed expert counts. Factor summaries receive arithmetic and class-count checks.

```bash
python scripts/run_readout_study.py \
  --out artifacts/readout_reproduction --jobs 3 --threads 2 \
  --fit-per-cell 64 --count-per-query 128 --max-iter 300

python scripts/package_readout_study.py \
  --source artifacts/readout_reproduction \
  --out artifacts/readout_reproduction_results

python -m experiments.text_regions.readout_compare \
  --manifest experiments/text_regions/results/readout_diagnostic/manifest.json \
  --out artifacts/reproduced_readout_comparison.json

python scripts/check_all.py
```

The complete download supplies the repository, original model checkpoints, all twelve fitted readout artifacts, raw answer predictions and losslessly compressed diagnostic results. Individual models can be evaluated with `python -m experiments.text_regions.readout_diagnostic CHECKPOINT --out OUTPUT`. Source hashes, checkpoint identities and fitted-head inventories accompany the results.

## Next development step

Test one small shared nonlinear emission adapter on the same raw and mixed features, retaining frozen experts and the same query/source cases. It must infer the requested answer from the observed representation, with no query metadata or task-specific answer heads. The original/dropout graphs are the strongest starting point for preserving private answers; the gated/dropout graphs provide the clearest simultaneous improvement in private and missing-source performance.

Development base: b5e3d74388db951ca28eafa4eafe3ab2fb7daa7a. This stage adds diagnostic tools and fitted heads; the underlying regional model implementation is unchanged.
