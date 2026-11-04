# SHoEbox: probe and learning development

2026-11-04 | Public repository: https://github.com/JRowe47/SHoEbox

The multi-region engine now has runnable memory probes, independent source-view objectives, and explicit post-optimizer edge/lace utility updates. The final correction model achieved **98.6% exact accuracy at an 18-byte delay** and **99.6% at a 34-byte delay** on 512 fresh test episodes per difficulty. Training used 16-byte truncated-backpropagation segments.

## Correction results

| Model / split / value-to-query distance | Carried | Reset before query | Update withheld: new answer | Update withheld: old answer |
| --- | --- | --- | --- | --- |
| Initial / validation / 10 bytes | 49.0% | 24.8% | 16.2% | 52.5% |
| Initial / validation / 34 bytes | 40.6% | 24.8% | 23.6% | 27.9% |
| Final / test / 18 bytes | 98.6% | 26.6% | 0.0% | 100.0% |
| Final / test / 34 bytes | 99.6% | 26.6% | 0.0% | 100.0% |

The reset control clears all carried memory immediately before the final query. The update-withheld control replaces the new fact with neutral filler and retains the original fact. Every control uses the same query schedule and expert execution count. Chance for the four possible answers is 25%; scoring still selects from all 256 byte outputs.

| Final test difficulty | Expert calls per control, 512 episodes | Carried nats/answer byte | Carried bits/answer byte |
| --- | --- | --- | --- |
| 18 bytes | 1,935,360 | 0.0494 | 0.0713 |
| 34 bytes | 2,967,552 | 0.0112 | 0.0162 |

The model retains all seven nodes and three regions with reduced internal capacity: 246,142 trainable parameters, 384 physical experts, and 84 expert executions per external event at K=1. The default 2.56M engine is unchanged in size. Auxiliary heads and utility were disabled in the measured correction runs.

## Training lineage and protocol

Phase 1 used 100 optimizer segments, gap 8, seed 20261104, and 8,840 external events. Phase 2 loaded those weights, started fresh optimizer/state/streams, and trained 150 segments with gap 16, seed 20261105, and 18,000 additional events. Both used batch 8, segment 16, AdamW, initial LR 0.002 with the inherited schedule, and gradient clipping at 1. Phase training wall times were 167.4s and 351.4s on the shared CPU host; these are run logs, not isolated throughput benchmarks. An interrupted intermediate run did not contribute weights to this lineage.

Episodes contain one named fact with a four-way byte value, a distinct corrected value, filler, and query markers. Intermediate queries provide local training signals for writes and retention; their answers are never fed back into observations. Only the final answer is scored. Identifiers and random facts are generated from disjoint train/validation/test streams. A fresh test set was evaluated after the final training phase. This compact shaping task is distinct from the richer natural-language distractor fixture retained in probes.py.

---

## Implemented learning and runtime changes

- Independent ingress taps expose source activations already computed by the forward pass, before peer feedback or graph fusion.
- Shared/private projection heads offer independent agreement, cross-view prediction, variance, covariance, private-byte, and separation switches. Unequal descriptions pool their own available occurrences before comparison.
- A source-only baseline trains with the regional predictor. Detached signed gains produce bounded eligibility statistics for consumed edges and actually selected laces. Substituted messages receive no edge eligibility.
- Explicit commit_feedback runs after the optimizer. Future passes may use bounded gate and lace-score biases. Forward and evaluation calls never commit buffers; no replay store is introduced.
- Probe checkpoints retain model/head/utility weights, optimizer, graph state, cursor, RNG, schedule, and configuration. Periodic atomic checkpoints default to every 25 segments. Exact resume and weights-only warm start are separate operations.
- Generation now accepts an existing state without repeating the reset during token generation. Per-row substitution identity survives mixed active/inactive publication updates and serialization.

## Missing-view gradient fix

An absent B view exposed nested normalization of zero vectors in inherited symbol and minKron queries. Their arbitrary 1/epsilon derivatives produced a gradient norm of 3.035 x 10^25. Zero-direction handling in the graph gives missing directions zero value and zero derivative, reducing the same norm to 173.94 with identical logits. Directions above the normalization epsilon preserve outputs, state, and gradients bit-for-bit. Original HAT/minKron experiments keep their previous defaults; the new graph enables the behavior through zero_safe_norm=True.

## Verification

**135 automated tests passed.** Coverage includes preserved baseline behavior, graph invariants, source independence, matched execution counts, query/reset controls, selected-path eligibility, signed updates, exact checkpoint continuation, warm starts, and normalization parity.

| Group | Checks |
| --- | --- |
| Preserved experiment suites | 60 |
| Multi-region engine and new learning/probe suites | 75 |

The combined shared/private CLI smoke trained with view objectives and utility, saved, reloaded, and continued through two optimizer steps. It logged finite auxiliary terms and 2 explicit utility commits. All six information controls used the same expert count. This run validates integration; shared/private capability accuracy has not yet been established.

The full default-capacity K=2, 128-byte gradient regression remains part of the suite. The learned correction result here uses K=1 and does not claim a benefit from additional regional rounds.

---

## Reproduce the measured training lineage

Run from the repository root after installing requirements.txt.

```bash
python -m experiments.text_regions.probe_train --kind correction \
  --steps 100 --batch 8 --segment 16 --gap 8 --eval-gap 32 \
  --eval-episodes 512 --eval-batch 64 --seed 20261104 \
  --out artifacts/probes/correction

python -m experiments.text_regions.probe_train --kind correction \
  --steps 150 --batch 8 --segment 16 --gap 16 --eval-gap 32 \
  --eval-episodes 512 --eval-batch 64 --seed 20261105 \
  --warm-start artifacts/probes/correction/checkpoint.pt \
  --out artifacts/probes/correction_warm150

python -m experiments.text_regions.probe_eval \
  artifacts/probes/correction_warm150/checkpoint.pt \
  --gaps 16 32 --split test --episodes 512 \
  --out artifacts/probes/correction_warm150/test_evaluation.json

python scripts/check_all.py
```

To enable the new learning paths, use probe_train --kind shared_private --view-objective --utility. Each auxiliary term is logged separately. For interrupted training, pass --resume with the same training protocol and schedule horizon; a warm start deliberately begins a new phase.

## Milestone status and next experiments

| Area | Current status |
| --- | --- |
| M4 persistent streaming | Runtime and exact continuation implemented; compact correction probe now measured beyond the TBPTT segment |
| M5 regional recurrence | Engine and transport/gradient checks complete; learned composition comparison remains |
| M6 representations and utility | Implemented, independently switchable, integrated with training and checkpointing; capability ablations remain |
| M7 integrated demonstration | In progress; this report covers one correction-model lineage and fresh evaluation sets |

Next: train the shared/private task to measure common-fact recovery and private-detail retention, compare view/utility switches independently, then run relation-chain queries at K=1/2/4 from the same prefix state. Expand correction to multiple facts, randomized presentation order, richer distractors, and three independent training seeds. The original specification's 90% training-range engineering target applies to each declared probe difficulty; it does not replace the other milestone criteria.

Raw protocols, histories, predictions, and summaries are under experiments/text_regions/results. The downloadable checkpoint bundle contains both correction phases and the auxiliary integration checkpoint. Source lineage remains anchored to SHoE revision b4bb7cfea18c3833819f6ed08973fd014f15f107.
