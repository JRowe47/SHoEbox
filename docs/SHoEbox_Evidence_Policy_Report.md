# SHoEbox: evidence-policy development

2026-11-04 | https://github.com/JRowe47/SHoEbox

Three matched fine-tuning seeds show a clear tradeoff: evidence-gated training raises mean B-only shared-fact recovery from **35.1% to 85.1%**, while reducing held-out all-view accuracy from **78.2% to 68.2%**. Applying the gate only at inference gives **43.5%** mean B-only accuracy. The gate remains optional; private-fact recovery regresses in all three trained pairs.

The additional B-only effect attributable to training at the same gated inference policy is **+41.6 percentage points**. The table reports unweighted seed means with minimum-maximum ranges. Every cell uses 512 fresh episodes per query, or 1,536 episodes per control and seed.

## Three-seed capability comparison

| Accuracy | Original / original | Original / gated | Gated / gated |
| --- | --- | --- | --- |
| Held-out all-view answers | 78.2% [76.8%-79.0%] | 78.2% [76.8%-79.0%] | 68.2% [66.3%-71.2%] |
| Held-out left private | 76.8% [70.9%-80.9%] | 76.8% [70.9%-80.9%] | 53.8% [47.5%-63.5%] |
| Held-out right private | 74.3% [68.0%-80.3%] | 74.3% [68.0%-80.3%] | 62.0% [57.0%-65.4%] |
| A-only shared | 57.4% [53.1%-61.9%] | 56.7% [54.1%-59.2%] | 59.6% [57.6%-61.5%] |
| B-only shared | 35.1% [24.6%-50.0%] | 43.5% [23.0%-64.8%] | 85.1% [57.2%-99.6%] |
| C-only shared | 60.5% [59.2%-61.9%] | 59.8% [56.6%-62.1%] | 60.4% [58.0%-62.3%] |
| A-only left private | 72.7% [68.4%-77.3%] | 68.1% [65.0%-71.7%] | 65.7% [62.5%-68.8%] |
| C-only right private | 72.7% [68.9%-78.7%] | 68.3% [65.2%-69.9%] | 63.1% [61.9%-63.9%] |

Column labels are training policy / inference policy. The first-to-second contrast changes only publication validity. The second-to-third contrast changes the training policy while keeping inference gated. All-view held-out tests use 16 excluded fact triples; missing-view tests use all 64 triples so an absent factor has a genuine conditional 25% prior. Exact predictions select from all 256 bytes.

## B-only shared recovery by seed

| Fine-tuning seed | Original / original | Original / gated | Gated / gated |
| --- | --- | --- | --- |
| 20261106 | 24.6% | 64.8% | 98.4% |
| 20261107 | 50.0% | 42.6% | 57.2% |
| 20261108 | 30.7% | 23.0% | 99.6% |

The three runs start from the same correction-trained graph. Seeds vary auxiliary-head initialization and the generated stream, including identifiers, templates, and dropout. Each original/gated pair begins with identical tensor values. These are three paired fine-tuning seeds from one common graph initializer.

---

## Matched budgets and training exposure

Each run allocates 265,694 parameters, including a 246,142-parameter graph, across seven nodes and three regions. The graph has 384 physical experts and executes 84 expert calls per aligned event at K=1. Every run trains for 240 optimizer steps, batch 32, segment 16: **96,000 aligned external events and 8,064,000 expert executions**. Within each seed, available source observations also match exactly. Carried graph state occupies 292,128 bytes at batch 32.

All variants train the same source-only baseline. The view-trained variants enable agreement, prediction, variance, covariance, private-byte prediction, and separation, with private loss weight 0.05. Utility updates are disabled in this comparison. AdamW starts at LR 0.002, warms up, holds its rate through 80% of the horizon, then tapers. The source-dropout probability is 0.25. The inference gate is still optional and does not reduce node or expert execution.

The table counts complete training episodes, not supervised token occurrences. Each seed has 1,280 episodes per query type. Coverage is reconstructed from the saved deterministic stream and verified against recorded source observations.

| Seed | Shared, all sources | Shared, B only | Left, A only | Right, C only |
| --- | --- | --- | --- | --- |
| 20261106 | 988 | 44 | 130 | 103 |
| 20261107 | 959 | 63 | 94 | 103 |
| 20261108 | 925 | 56 | 85 | 114 |

Single-source cases receive far less exposure than all-view cases. Training retains a source containing the requested fact. Unsupported private queries, private-value withholding, and reset-before-query are evaluation controls rather than supervised recovery cases. Source observations and aligned external events remain separate counters.

## Comparison without the view objectives

| Seed 20261106, views disabled | Original / original | Original / gated | Gated / gated |
| --- | --- | --- | --- |
| All-view held-out | 70.4% | 70.4% | 60.8% |
| B-only shared | 0.6% | 34.8% | 50.0% |

This additional matched pair isolates publication-policy behavior without the six view losses. It is kept separate from the three-seed view-objective aggregate.

## Integrity checks

The comparison tool verifies initial tensors, model dimensions, full training protocols except publication policy, source streams, optimizer-step counts, expert counts, saved checkpoint fingerprints, evaluation identities, answer balance, and exact paired predictions. It recomputes scores from the saved predictions and reports both corrected and regressed cases. Every evaluation control executes **3,225,600 expert calls** over **38,400 aligned events**. Wall times include shared-host contention and are not used as throughput comparisons.

---

## Where the information survives

Frozen linear probes now inspect actual A/B/C/U/P/V/Q publications before queries and after the final query. They fit only training combinations, independently for each information control, and never modify model weights. Publication validity is explicit; invalid payloads are excluded. Separate pooled and query-matched training fits distinguish general factor retention from task-conditioned codes. The serialized probes bind to both model tensors and the complete inference configuration.

The table uses seed 20261106, evidence-gated inference, and 192 independent B-only shared queries. Each query-matched probe fits 64 training descriptions; the overall fit cohort contains 192 descriptions per control. These diagnostic rows measure recoverable information, not extra heads used by the deployed model.

| Final representation | Original-trained | Gated-trained |
| --- | --- | --- |
| @B | 100.0% | 100.0% |
| B | 100.0% | 100.0% |
| U | 99.0% | 100.0% |
| V | 99.0% | 100.0% |
| P | 93.2% | 100.0% |
| Q | 97.9% | 100.0% |
| Actual model answer | 66.7% | 99.5% |

B's source ingress and node output retain the fact. High probe accuracy downstream alongside weaker direct answers identifies task-conditioned use of the representation as a key remaining issue. A separate decoder per factor can recover information even when the model's common answer path does not select it correctly; the probes do not establish that replacing only the last linear layer would solve the task.

## Query-only interference

An additional controlled test starts with facts available only from B. A and C receive either no observations, only the final shared-query marker, or all query markers. Their fact values remain unavailable. Every condition uses the same 512 shared episodes, identical byte tensors and **1,075,200 expert calls**; only availability changes.

| Original trained weights | B only | Final query also to A/C | All queries also to A/C |
| --- | --- | --- | --- |
| Original publication policy | 24.61% | 25.00% | 25.00% |
| Evidence-gated policy | 64.84% | 25.00% | 25.00% |

Both query-only interventions force every final prediction to D, across all three B rendering templates. With the final-query intervention, every earlier query prediction is unchanged. Thus the final drop is immediate interference, not failure to deliver the query through B. Query markers are legitimate observations, so this result motivates learned content selection rather than a hard-coded rule about which bytes count as facts.

---

## Reproduce and inspect

Extract the checkpoint bundle at the repository root. It includes the common initializer and all compared trained models. Install requirements.txt, then use the following commands; omit the training gate flag for the original policy and repeat the matched pair with seeds 20261107 and 20261108 in separate output directories.

```bash
python -m experiments.text_regions.view_train --steps 240 \
  --batch 32 --segment 16 --gap 8 --dropout 0.25 \
  --seed 20261106 --threads 2 --view-objective \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --evidence-gated-ports --out artifacts/evidence/views

python -m experiments.text_regions.view_score \
  artifacts/evidence/views/checkpoint.pt --publication-policy evidence \
  --count-per-query 512 --batch 64 --predictions \
  --out artifacts/evidence/views/policy_evaluation/gated

python -m experiments.text_regions.transfer_eval \
  artifacts/evidence/views/checkpoint.pt --publication-policy evidence \
  --fit-count 192 --count-per-query 192 --batch 64 \
  --out artifacts/evidence/transfer_trained_gated

python -m experiments.text_regions.query_interference \
  artifacts/views/views/checkpoint.pt --policy both --count 512 --batch 128 \
  --out artifacts/evidence/query_interference

python -m experiments.text_regions.evidence_compare \
  --manifest experiments/text_regions/results/evidence_policy/manifest.json \
  --out artifacts/evidence/recomputed_comparison.json

python scripts/check_all.py
```

**All 189 automated checks passed**, including policy overrides, exact checkpoint continuation, source masking, strict readout serialization, inference-policy identity, query-only interventions, and matched-comparison rejection cases. Raw protocols, histories, evaluations, predictions, coverage, and the portable comparison manifest are under experiments/text_regions/results/evidence_policy. The source archive includes this report and the reproduction tools.

## Next development target

First isolate the baseline gradient path. The A-only baseline currently backpropagates right-private targets into the shared HAT encoder whenever A is available, even though only C observes those answers. The training split restricts the right value to two possibilities given A's facts; held-out combinations use the other values. This supplies an incentive for a split-specific shortcut. Its contribution to the regression has not been measured. Detaching source features for baseline-head training preserves targets, head learning and utility eligibility while isolating that gradient path.

Next balance visibility exposure and include partially informative sources that retain identifiers/query markers while withholding factual values, always preserving an actual answer source. Compare one common answer decoder with the factor probes to separate task selection from output compression. The model does not yet meet the 90% target across all supported difficulties; recurrent composition and natural text remain separate milestones.

Checkpoint archive: SHoEbox_evidence_policy_checkpoints.zip. SHA-256:

```text
b34bdd94b11bfc1dc77a4587cb4ee778024c222452f4829ba7e6ea6cc3384174
```

Development starts from SHoEbox revision 7bace241fe5ffbd3ca525f8ce841f9e2780eef2d. The inherited SHoE revision is b4bb7cfea18c3833819f6ed08973fd014f15f107.
