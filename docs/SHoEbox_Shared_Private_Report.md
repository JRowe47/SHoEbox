# SHoEbox: shared/private development

2026-11-04 | https://github.com/JRowe47/SHoEbox

View objectives improved exact accuracy on unseen combinations of shared and private facts from **69.8% to 79.2%**, a gain of **9.4 percentage points**. All four variants used identical initial tensors, the same examples, and the same expert execution budget. Private-fact recovery produced the largest gains.

## Matched learning ablation

| Training variant | Shared | Left private | Right private | Overall |
| --- | --- | --- | --- | --- |
| Control | 80.2% | 65.6% | 63.5% | 69.8% |
| View objectives | 83.3% | 78.6% | 75.5% | 79.2% |
| Utility | 91.7% | 65.6% | 59.4% | 72.2% |
| Views + utility | 90.6% | 67.7% | 78.1% | 78.8% |

Each row scores 576 fresh test episodes: 192 per query, with 48 occurrences of each answer in each query. The test set contains 16 factor combinations excluded from training. Predictions select from all 256 output bytes; the four possible answers are balanced. Overall accuracy is also the macro average over query types.

The view objective improved left-private recovery by **13.0 points** and right-private recovery by **12.0 points**. Utility alone improved shared answers, but reduced right-private recovery. Combining utility with views gave 78.8% overall, close to the view-only run; the effects were not additive in this run.

## Information flow finding

All four models encoded B's shared fact with **100% frozen linear-readout accuracy** on held-out descriptions. Yet the regional output often failed when B was the only available source. Cold nodes were publishing learned prior outputs as valid messages despite having received no evidence.

An optional publication policy now requires a node to consume valid evidence before its publication becomes valid. Applying it to the already-trained view model raised B-only shared accuracy from **24.5% to 66.7%** on the full-factorial evaluation cohort, with unchanged weights and expert counts. This identifies cold publications as one contributor to the missing-view failure.

## Development status

The shared/private task, matched training harness, independent information controls, frozen source probes, and exact-resume checkpoints are implemented and measured. The 90% private-fact and missing-view engineering targets remain open. The next step is to train with evidence-gated publications, then improve the transfer from source representations to regional answers before testing additional recurrent rounds.

---

## Benchmark design

Every episode has three four-way facts: a common value observed by A/B/C, a left-private value observed only by A, and a right-private value observed only by C. Each source independently chooses its rendering and field order. Opaque six-hex identifiers are unique across cohorts and allocated independently of the facts.

Explicit masks distinguish missing inputs and padding from literal zero bytes. Evidence arrives in the first 14 aligned events. Query markers at events 14 and 15 provide a local training signal; eight subsequent events contain filler and periodic queries, followed by the final query at event 24. Answers enter only the loss, never the input stream. Private auxiliary labels become available after observation and are cleared by reset.

The 64 possible fact triples are partitioned into 32 training, 16 validation, and 16 test combinations. All individual values and source-local pairs occur in training. An additional independent cohort includes all 64 combinations. Information-removal controls use this cohort because withholding a factor in the restricted holdout could make it inferable from the split rule. In the independent cohort, unsupported private facts have a genuine conditional 25% prior.

Training uses 25% source dropout, retaining a source that contains the requested answer. Every variant receives the identical stream. Evaluation uses fresh identifiers and resets each episode; the training cursor is not carried into inference.

## Matched training conditions

| Quantity | Per variant |
| --- | --- |
| Graph / total allocated parameters | 246,142 / 265,694 |
| Regions / nodes / physical experts | 3 / 7 / 384 |
| Regional rounds / expert calls per event | K=1 / 84 |
| Optimizer steps / batch / segment length | 240 / 32 / 16 |
| Aligned external events | 96,000 |
| Available source observations | 238,681 |
| Expert executions | 8,064,000 |
| Carried state at batch 32 | 292,128 bytes |
| Utility commits: control, views, utility, both | 0, 0, 240, 240 |

All variants start from the same previously trained correction graph, with fresh optimizer, runtime memory, and streams. All allocate the same auxiliary modules and train the A-only baseline with weight 0.1. The view switch enables agreement, cross-view prediction, variance, covariance, private-byte prediction, and separation; private loss weight is 0.05. Utility commits detached signed edge/lace gains explicitly after optimizer updates. No replay mechanism is introduced.

Training uses AdamW, initial LR 0.002 with the repository schedule, clipping at 1, seed 20261106, and two CPU threads per process. Logged wall times range from 731 to 747 seconds on the shared host. The common initial tensor SHA-256 is recorded in every protocol; exact parameter allocation and initial tensor equality are tested.

---

## Independent information controls

The table below uses the view-trained model with the original publication policy. Each control scores 576 episodes over all 64 fact combinations, including 192 examples per query. Every control executes exactly **1,209,600 expert calls** across **14,400 aligned events**.

| Available information / state | Shared | Left private | Right private |
| --- | --- | --- | --- |
| All sources | 84.4% | 78.6% | 80.7% |
| A only | 51.6% | 76.0% | 24.5% |
| B only | 24.5% | 25.0% | 27.1% |
| C only | 59.9% | 29.2% | 79.2% |
| Private values withheld | 100.0% | 20.8% | 27.1% |
| Reset before final query | 25.0% | 25.0% | 25.0% |

Resetting memory returns all three answers to 25%. Withholding private values preserves perfect shared-fact recovery while private answers stay near chance. Single-source controls expose a regional selection problem: the source can contain useful information that does not survive to the final answer.

## Evidence-gated publication intervention

| Trained variant | B-only shared: original | B-only shared: gated |
| --- | --- | --- |
| Control | 0.5% | 35.9% |
| View objectives | 24.5% | 66.7% |
| Utility | 24.5% | 33.9% |
| Views + utility | 50.0% | 50.0% |

This comparison changes only inference publication validity. It uses the same checkpoints, sample identities, and expert counts as the corresponding original B-only control. The graph flag remains off by default. Training with the flag is supported but was not part of these four measured runs.

A node becomes valid after consuming a valid dependency. Once established, its publication stays valid through input absence because the node retains memory; selective reset clears that history. Delayed feedback can provide evidence to a source whose own input is absent. The policy preserves node execution, clocks, state updates, and expert counts. Invalid publications receive no lace utility eligibility. Tests verify exact parity when all views are present, retained evidence, reset, delayed feedback, inactive nodes, and available zero bytes.

## Source diagnostics

Frozen ridge readouts fit 128 training descriptions and evaluate fresh held-out descriptions at the common pre-query boundary, using the last available independent ingress. They decode source activations and shared/private projections without changing model weights. Fit identities are disjoint from evaluation identities. B's common value is fully decodable in all four variants, narrowing the next investigation to regional transfer and selection.

Direct trained private heads are also reported in the raw output. Their diagnostic snapshot is at the pre-query boundary, earlier than the usual training pooling point; metadata records this timing difference. Those scores are not used as a matched measure of the trained head's target-time accuracy.

---

## Reproduction and verification

Extract the checkpoint bundle into the repository root. It restores the common correction initializer, all four trained models, and frozen source-readout weights. Run from the repository root after installing requirements.txt.

```bash
python -m experiments.text_regions.view_train --steps 240 \
  --batch 32 --segment 16 --gap 8 --dropout 0.25 \
  --seed 20261106 --threads 2 \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --view-objective --out artifacts/views/views

python -m experiments.text_regions.view_score \
  artifacts/views/views/checkpoint.pt --count-per-query 192 \
  --batch 64 --predictions --out artifacts/views/views/evaluation

python -m experiments.text_regions.view_score \
  artifacts/views/views/checkpoint.pt --cohorts independent \
  --controls b_only --evidence-gated-ports --skip-readouts \
  --count-per-query 192 --batch 64 --predictions \
  --out artifacts/views/views/evidence_gated

python scripts/check_all.py
```

Use no learning flags for control, --view-objective for views, --utility for utility, and both for the combined run. Use a separate output directory for each. Adding --evidence-gated-ports to training enables the new policy for a future run.

Checkpoints preserve model and auxiliary weights, optimizer slots, utility buffers, graph state, stream cursor, RNG, counters, and schedule. An initial checkpoint is written before training, followed by atomic checkpoints every 20 segments. --resume continues the same horizon; --stop-after allows an intentional checkpoint boundary without changing that schedule.

**All 168 automated tests passed: 60 preserved experiment checks and 108 multi-region checks.** New coverage verifies exact continuation, identical variant initialization, causal private labels, disjoint identities, balanced information controls, fixed expert budgets, tensor immutability during evaluation, source-readout serialization, and evidence-backed publication validity.

## Included evidence

Raw protocols, histories, run summaries, evaluation results, and individual predictions are stored under experiments/text_regions/results/shared_private. The aggregate summary links the matched budgets and before/after results. Model weights are in the separate downloadable checkpoint bundle, with a manifest and restore instructions.

Checkpoint archive: SHoEbox_shared_private_checkpoints.zip. SHA-256:

```text
a3081bd3f7ac2797a74c992ba3815005d5e07ecd4630b01c8fd869c44905674d
```

This work extends SHoEbox revision 82d45dbf27783ae10db28443383854c937ed09a3, based on SHoE revision b4bb7cfea18c3833819f6ed08973fd014f15f107. The comparison is one matched training seed. Further work will test the new publication policy during learning, replicate the strongest configuration across seeds, and then compare relation-chain composition at K=1/2/4.
