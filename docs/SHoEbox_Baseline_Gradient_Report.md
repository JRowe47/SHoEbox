# SHoEbox: baseline-gradient ablation

2026-11-04 | https://github.com/JRowe47/SHoEbox

Detaching the source-only baseline from the shared encoder raises mean evidence-gated held-out accuracy from **68.2% to 77.6%** and pooled private-fact accuracy from **57.9% to 73.7%**. B-only shared recovery falls from **85.1% to 67.3%**. Baseline gradients materially affect the learned solution; blanket detachment trades shared-source recovery for stronger average private-fact generalization.

## Three matched seeds under evidence-gated publication

| Accuracy | Attached baseline | Detached baseline | Change, pp |
|---|---:|---:|---:|
| Held-out all-view answers | 68.2% | 77.6% | +9.4 |
| Held-out shared | 88.7% | 85.5% | -3.2 |
| Held-out private, pooled | 57.9% | 73.7% | +15.8 |
| Held-out left private | 53.8% | 63.5% | +9.6 |
| Held-out right private | 62.0% | 83.9% | +21.9 |
| B-only shared | 85.1% | 67.3% | -17.8 |
| A-only left private | 65.7% | 72.3% | +6.6 |
| C-only right private | 63.1% | 78.3% | +15.2 |

Values are unweighted means across fine-tuning seeds 20261106, 20261107 and 20261108. Publication policy is identical for training and inference in both arms. Each cell uses 512 fresh episodes per query, or 1,536 per control per seed. Predictions select from all 256 bytes. The private aggregate weights left and right equally.

| Fine-tuning seed | Held-out overall change, pp | Held-out private change, pp | B-only shared change, pp |
|---|---:|---:|---:|
| 20261106 | -4.9 | -6.7 | -25.6 |
| 20261107 | +17.1 | +26.9 | +18.0 |
| 20261108 | +16.1 | +27.1 | -45.7 |

Pooled private accuracy improves in two of the three gated seeds. In seed 20261106, held-out left-private accuracy falls from 63.5% to 35.7%, while right-private accuracy improves. The detach option therefore remains default-off. These results motivate retaining useful source-local supervision while removing targets the baseline cannot observe; this experiment removes gradients for all baseline targets and does not isolate the unsupported targets alone.

## What changed

The new default-off baseline_detach_features option detaches only the source features passed to the A-only baseline head. The head still learns from the same targets, with weight 0.1 and unchanged normalization. Main answer and view-objective losses continue to update the shared encoder. No parameters, input bytes, source masks, examples, utility eligibility or expert execution paths are removed. Optimizer and clipping settings are unchanged.

---

## One-seed publication-policy interaction

| Publication policy / baseline | All-view answers | Left private | Right private | B-only shared |
|---|---:|---:|---:|---:|
| Original / attached | 79.0% | 78.7% | 74.8% | 24.6% |
| Original / detached | 84.8% | 86.1% | 81.6% | 49.4% |
| Evidence-gated / attached | 71.2% | 63.5% | 65.4% | 98.4% |
| Evidence-gated / detached | 66.3% | 35.7% | 79.7% | 72.9% |

For seed 20261106, the detach effect on pooled private accuracy is +7.1 points under original publication and -6.7 points under evidence gating. The difference between those effects is -13.9 points. Only this seed has the complete two-by-two comparison; the other two seeds replicate detachment under evidence gating.

## Matched training and verification

Every run allocates 265,694 parameters, including a 246,142-parameter graph, across seven nodes and three regions. It uses 384 physical experts and 84 expert calls per external event at K=1. Every attached/detached pair starts with exactly matching tensors from the same correction-trained graph initializer, including matched auxiliary heads. The three seeds vary fine-tuning randomness, not the pretrained graph.

Each arm trains for 240 optimizer steps at batch 32 and segment 16: **96,000 aligned external events and 8,064,000 expert executions**. Source observations, query counts, stream cursors, optimizer settings, loss switches, and schedule match within each pair. Graph state occupies 292,128 bytes at batch 32. View objectives are enabled; utility updates are disabled. The private auxiliary loss weight remains 0.05.

Test/all evaluates the 16 excluded fact combinations. The independent control cohort uses all 64 combinations, with balanced answers, so truly missing facts have a 25% prior. Controls include A/B/C alone, private-value withholding, and reset immediately before the final query. Each control executes 3,225,600 experts over 38,400 aligned events. Shared-host wall times are not used as throughput comparisons.

The comparator checks configuration, initialization, source exposure, parameter and execution counts, checkpoint fingerprints, exact case identities, and answer balance. It recomputes scores and paired gains/losses from saved predictions. Old checkpoints without the new flag retain attached behavior; resume rejects a setting mismatch. Tests verify fixed-weight forward and baseline-head gradient equality, unchanged utility packets, blocked baseline-only encoder gradients, continued language gradients, and exact next-step checkpoint continuation. **204 automated tests passed.**

## Next development step

First test a causal source-observability mask on baseline training targets: retain encoder gradients for facts A has actually observed and exclude right-private answers available only from C. Preserve the original loss denominator and utility eligibility to isolate that change. Then balance supported whole-source subsets at fixed event and expert budgets, tracking changed source observations and auxiliary-label exposure. The retention schedule must rotate independently of answer values. Query-only or identifier-only sources come later, after shared-view losses receive causal shared-fact eligibility masks. The model still falls short of 90% across all supported difficulties; whole-region composition remains a subsequent milestone.

---

## Reproduction and files

Extract SHoEbox_baseline_gradient_checkpoints.zip at the repository root. It restores the common initializer, all eight compared trained models, and eight source readouts under their original artifacts paths. Install requirements.txt, then run:

```bash
python -m experiments.text_regions.view_train --steps 240 \
  --batch 32 --segment 16 --gap 8 --dropout 0.25 \
  --seed 20261106 --threads 2 --view-objective \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --baseline-detach-features --evidence-gated-ports \
  --out artifacts/reproduction/gated_seed1

python -m experiments.text_regions.view_score \
  artifacts/reproduction/gated_seed1/checkpoint.pt \
  --publication-policy evidence --count-per-query 512 \
  --batch 64 --threads 2 --predictions \
  --out artifacts/reproduction/gated_seed1/evaluation

python -m experiments.text_regions.baseline_compare \
  --manifest experiments/text_regions/results/baseline_gradient/manifest.json \
  --out artifacts/reproduction/baseline_comparison.json

python scripts/check_all.py
```

Repeat with seeds 20261107 and 20261108. For the original-policy control, omit --evidence-gated-ports and score with --publication-policy original. To retrain an attached comparator, also omit --baseline-detach-features. Use separate output directories for every arm.

The raw result package is experiments/text_regions/results/baseline_gradient. Its portable manifest references the attached arms already recorded under results/evidence_policy. The source archive contains both result packages, exact protocols, training histories, raw predictions, tests, and this report. No learned tensors are stored in Git.

Checkpoint archive SHA-256:

```text
ea52dcc3b52dcb52e6d1d03c56bef9d300f8a485afe1bfa2de4a4588c0472fb1
```

Development starts from SHoEbox revision ce185d06517524f439959780dbd776d8e9c8ae32. The inherited SHoE revision is b4bb7cfea18c3833819f6ed08973fd014f15f107.
