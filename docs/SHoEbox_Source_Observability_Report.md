# SHoEbox: source-observed baseline targets

2026-11-04 | https://github.com/JRowe47/SHoEbox

Source-observed baseline supervision leaves mean evidence-gated held-out accuracy nearly unchanged: **68.2% to 68.6%**. Private-fact accuracy rises from **57.9% to 61.0%**, while B-only shared recovery falls from **85.1% to 70.3%**. The source mask and count audit match the intended intervention, but removing unsupported baseline targets alone does not resolve the accuracy tradeoff. Defaults remain unchanged.

## Three-seed evidence-gated comparison

| Accuracy | Original | Detached | Observed-only | Change vs original, pp |
|---|---:|---:|---:|---:|
| Held-out overall | 68.2% | 77.6% | 68.6% | +0.4 |
| Held-out shared | 88.7% | 85.5% | 83.7% | -5.1 |
| Held-out private, pooled | 57.9% | 73.7% | 61.0% | +3.1 |
| Held-out left private | 53.8% | 63.5% | 56.8% | +3.0 |
| Held-out right private | 62.0% | 83.9% | 65.2% | +3.3 |
| B-only shared | 85.1% | 67.3% | 70.3% | -14.8 |
| A-only left private | 65.7% | 72.3% | 69.5% | +3.8 |
| C-only right private | 63.1% | 78.3% | 73.7% | +10.6 |

The primary contrast is observed-only versus original baseline supervision, with encoder gradients attached in both. Detached results preserve the previous, separately matched intervention for context. Values are unweighted means across seeds 20261106, 20261107 and 20261108. Each evaluation uses 512 fresh episodes per query, or 1,536 per control, and predictions select from all 256 bytes.

| Fine-tuning seed | Overall change, pp | Private change, pp | B-only shared change, pp |
|---|---:|---:|---:|
| 20261106 | -9.2 | -10.0 | -47.5 |
| 20261107 | +6.4 | +9.0 | +6.2 |
| 20261108 | +4.0 | +10.4 | -3.1 |

Private accuracy improves in two of the three gated seeds. In seed 20261106, right-private accuracy falls from 65.4% to 39.3% and B-only recovery from 98.4% to 51.0%. Blanket detachment retains the strongest mean private accuracy of these three variants, with its own shared-recovery cost. The new observed-only option remains available for controlled experiments; it is not enabled by default.

## Intervention

The baseline predicts from A's source ingress. The new loss mask retains a target only after that source has observed the requested factor. It follows factor identity, availability and resets; a coincidentally equal value in another field does not count. The mask is consumed after the graph pass and never enters model inputs, publications, routing or memory. Supported targets continue to train both the baseline head and encoder.

---

## Exact supervision audit

| Seed | Original eligible targets | Retained targets | Excluded targets |
|---|---:|---:|---:|
| 20261106 | 17,515 | 12,080 | 5,435 |
| 20261107 | 17,425 | 12,030 | 5,395 |
| 20261108 | 17,175 | 11,875 | 5,300 |

Every excluded target asks A for a C-only right-private fact. All eligible shared and left-private targets remain, including 6,400 left targets per seed. The 0.1 loss weight, original per-segment denominator, utility reference losses and utility eligibility are unchanged.

The comparator reconstructs causal knowledge independently from fact positions, source events and reset order, then checks total and per-query counts. All 240 segments have recorded audits. Old checkpoints retain false defaults for both independent baseline switches; resumed legacy runs explicitly mark earlier unaudited segments.

## Original-policy control

| Original publication policy, seed 20261106 | Overall | Left private | Right private | B-only shared |
|---|---:|---:|---:|---:|
| Original baseline | 79.0% | 78.7% | 74.8% | 24.6% |
| Detached baseline | 84.8% | 86.1% | 81.6% | 49.4% |
| Observed-only baseline | 71.2% | 58.4% | 69.5% | 25.4% |

Only seed 20261106 has this original-publication control. The three-seed aggregate above measures the observability intervention under evidence gating, not a three-seed policy interaction. Both publication and observability options remain explicit configuration choices.

## Matched budget and evaluation

All arms share the same correction-trained graph initializer, matched auxiliary-head tensors within seed, source stream, optimizer, loss weights and schedule. Each trains for **240 optimizer steps, 96,000 external events and 8,064,000 expert executions**, with batch 32 and segment 16. The model allocates 265,694 parameters, including a 246,142-parameter seven-node graph, three regions, 384 physical experts and 84 expert calls per event at K=1. View objectives are enabled; utility updates are disabled.

Test/all holds out 16 fact combinations. Independent controls use all 64 combinations with balanced answers, covering all views, each source alone, private-value withholding and reset-before-query. Each control executes 3,225,600 experts over 38,400 events. Source observations and all compute counts match within each comparison. Host-contention timings are not throughput comparisons.

## Next development step

Move to balanced supported whole-source exposure, keeping the original baseline loss as the reference. Compare both publication policies at the same 240-step, event and expert budgets, while preserving episode facts, identifiers and renderings. Rotate source subsets independently of answer values; report the changed source-observation and auxiliary-label counts. Excluded sources remain fully absent in the first test. Query-only or identifier-only sources follow only after shared-view losses receive causal shared-fact eligibility masks. Whole-region composition remains a later milestone.

---

## Reproduce and inspect

Extract SHoEbox_source_observability_checkpoints.zip at the repository root. It restores the common initializer, twelve compared trained models and twelve source readouts under their original artifact paths. Install requirements.txt, then run:

```bash
python -m experiments.text_regions.view_train --steps 240 \
  --batch 32 --segment 16 --gap 8 --dropout 0.25 \
  --seed 20261106 --threads 2 --view-objective \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --baseline-observed-only --evidence-gated-ports \
  --out artifacts/reproduction/observed_gated_seed1

python -m experiments.text_regions.view_score \
  artifacts/reproduction/observed_gated_seed1/checkpoint.pt \
  --publication-policy evidence --count-per-query 512 \
  --batch 64 --threads 2 --predictions \
  --out artifacts/reproduction/observed_gated_seed1/evaluation

python -m experiments.text_regions.observability_compare \
  --manifest experiments/text_regions/results/baseline_observed/manifest.json \
  --out artifacts/reproduction/observability_comparison.json

python scripts/check_all.py
```

Repeat with seeds 20261107 and 20261108. For the original-policy control, omit --evidence-gated-ports and score with --publication-policy original. Omit --baseline-observed-only for the original baseline. The previous detached arm instead uses --baseline-detach-features. Give each arm a separate output directory.

**All 223 automated tests passed.** Tests cover causal source masking, equal-valued unsupported facts, unchanged forward outputs and utility packets, original loss normalization, zero unsupported gradients, legacy configuration, exact checkpoint continuation and strict comparison rejection cases. Both earlier published comparison packages still reproduce exactly.

Raw protocols, histories, audits and predictions are under experiments/text_regions/results/baseline_observed. Its portable manifest references prior attached and detached results without duplicating them. Exact launch-source snapshots are included: a later compatibility repair permits all-ignored targets without a mask, while bitwise parity across two real optimizer steps verifies the measured supplied-mask path is unchanged. The source archive includes every referenced result package and this report; learned tensors remain in the separate checkpoint bundle.

Checkpoint archive SHA-256:

```text
15d3ef1877aae34f033ef8d0aaf1dd19179ba0f73b78f9da56bd5886196ed16e
```

Development starts from SHoEbox revision 662c6e87783e4ee48c3439488c0f847741fab921. The inherited SHoE revision is b4bb7cfea18c3833819f6ed08973fd014f15f107.
