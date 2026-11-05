# SHoEbox: joint producer and connection learning

Completed November 5, 2026. Three published parents, six matched continuations, one eight-node architecture.

## Result

Joint local learning reaches **99.80% recall**, **99.87% correction**, and **99.74% withheld-correction accuracy** at gap 16. The stronger 95% per-task threshold passes in **3/3 seeds**. All locked phase targets: **passed**.

The weakest parent's minimum task score moves from **96.09% unchanged** to **96.09% with edge-only continuation** and **99.22% with joint learning**, on the same fresh test episodes.

| Data seed | Arm | Selected update | Recall | Correction | Withheld: old |
| --- | --- | --- | --- | --- | --- |
| 20261130 | Unchanged parent | 0 | 100.00% | 100.00% | 100.00% |
| 20261130 | edge | 0 | 100.00% | 100.00% | 100.00% |
| 20261130 | joint | 80 | 100.00% | 100.00% | 100.00% |
| 20261131 | Unchanged parent | 0 | 98.44% | 96.09% | 97.85% |
| 20261131 | edge | 0 | 98.44% | 96.09% | 97.85% |
| 20261131 | joint | 80 | 99.41% | 99.61% | 99.22% |
| 20261132 | Unchanged parent | 0 | 99.61% | 99.80% | 99.61% |
| 20261132 | edge | 60 | 99.61% | 99.80% | 99.61% |
| 20261132 | joint | 60 | 100.00% | 100.00% | 100.00% |

Each accuracy uses 512 fresh fact episodes with held-out relation-table pairs. The new data seeds map in order to the three published integration parents, 20261120/21/22. Parent and continuation scores use identical cases; these parent scores therefore differ from the previous report's cohorts. No parent is excluded. Validation selects checkpoints, including update zero; test scores do not select weights.

## What changed

The fact producer F and its F-to-A value projection now learn together. Local supervision reaches F's byte contextualizer, ingress projection, sparse expert banks, recurrent processing and write/hold gate. The existing seven-node composer and common 256-byte output head remain exactly frozen. The architecture, inference algorithm, parameter count and expert execution count are unchanged.

The edge-only control sees the same observations, tables, losses and number of updates. Only its 1,024 edge weights can change. This comparison isolates permission to update the producer; it does not equate backward computation or elapsed time.

| Loss | Learning signal |
| --- | --- |
| Retained-payload alignment | Keep the published fact useful to the receiving region |
| Write-candidate alignment | Encode a newly observed fact before it is published |
| Balanced write/hold classification | Distinguish valid updates from distractors |
| Final byte cross-entropy | Make the composed query answer correct |

Alignment uses this model's own frozen A ingress representation of the observed fact byte. It adds no readout or inference head. The ordinary hard publication decision stays in use. Each chunk backpropagates its local losses before the 16-event detach boundary; the final chunk also receives query loss. Earlier activations are discarded without replay. Training labels never enter forward payloads or carried inference state.

## Delays and controls

| Condition | Edge only | Joint |
| --- | --- | --- |
| Recall, gap 16 | 99.35% | 99.80% |
| Correction, gap 16 / K=2 | 98.63% | 99.87% |
| Withheld correction, gap 16 | 99.15% | 99.74% |
| Recall, gap 64 | 99.28% | 99.67% |
| Correction, gap 64 | 99.02% | 99.87% |
| Withheld correction, gap 64 | 98.96% | 99.61% |
| K=1 | 25.52% | 25.52% |
| K=4 | 41.34% | 40.76% |
| No delayed exchange | 25.52% | 25.52% |
| Blocked F-to-A edge | 23.70% | 23.70% |
| Reset memory | 23.70% | 23.70% |
| Shuffled memory: original target | 24.67% | 24.61% |
| Unconditional publication, same weights | 29.56% | 28.12% |
| Four consecutive queries | 98.54% | 99.87% |
| Final endpoint, correction | 98.70% | 99.87% |

Shuffled memory answers the substituted fact's query with **99.87%** accuracy. The payload and complete semantic header move together. Blocked-edge, reset, shuffle, disabled-exchange and unconditional-publication controls match the intact K=2 expert count. Four-value chance accuracy is 25%.

Repeated queries preserve F's contextual state, expert state, payload, header, version and provenance exactly. The producer executes no experts during queries. Two rounds remain the useful learned schedule; K=4 scores **40.76%**.

## Publication and continued acquisition

| Data seed | Arm | Gap 16 write precision | Gap 16 write recall | Gap 64 precision | Gap 64 recall |
| --- | --- | --- | --- | --- | --- |
| 20261130 | edge | 100.00% | 100.00% | 100.00% | 100.00% |
| 20261130 | joint | 100.00% | 100.00% | 100.00% | 100.00% |
| 20261131 | edge | 81.21% | 100.00% | 83.12% | 100.00% |
| 20261131 | joint | 97.15% | 100.00% | 97.25% | 100.00% |
| 20261132 | edge | 99.90% | 100.00% | 100.00% | 100.00% |
| 20261132 | joint | 100.00% | 100.00% | 100.00% | 99.90% |

All **270,278** measured joint-model hold transitions retain the payload bit-for-bit. These gate statistics cover correction episodes and are separate from the task acceptance threshold.

| Live sequence | Edge only | Joint |
| --- | --- | --- |
| Three queries before update | 99.28% | 99.83% |
| Query after correction | 98.18% | 99.74% |
| Query with correction withheld | 99.15% | 99.93% |

This continued-acquisition test keeps the prior protocol: after three queries, both branches observe the original fact again before an optional correction. It measures continued acquisition following live queries.

## Training and verification

Each continuation receives 80 updates and 2,560 new fact episodes. Recall and correction alternate over gaps 0,4,8,16. The two arms have identical data digests, table choices and exposure/execution counts. AdamW uses producer learning rate 0.0003, edge rate 0.001, beta=(0.9,0.95), weight decay 0.01 and gradient clipping at 1. Validation occurs at updates 0,20,40,60,80.

| Seed | Arm | Expert executions | External events | Training minutes |
| --- | --- | --- | --- | --- |
| 20261130 | edge | 1,950,720 | 119,040 | 2.81 |
| 20261130 | joint | 1,950,720 | 119,040 | 6.66 |
| 20261131 | edge | 1,950,720 | 119,040 | 2.71 |
| 20261131 | joint | 1,950,720 | 119,040 | 5.91 |
| 20261132 | edge | 1,950,720 | 119,040 | 3.07 |
| 20261132 | joint | 1,950,720 | 119,040 | 4.14 |

Times are concurrent CPU wall time, excluding validation and audits. Each run has 131,840 source occurrences. The graph still has **317,098 parameters** and **480 physical experts**. Joint training permits **64,443 parameters** to update. A fact byte executes 12 experts; a table pair and two-round query execute 264.

All **389 regression tests pass**. Independent audits reconstruct **480 update batches**, verify checkpoint selection and frozen composer tensors, regenerate **126 prediction groups**, and reproduce exported logits and complete state exactly. Tests verify local gradient reachability, detach boundaries, label isolation, inference parity, parameter ownership and bit-exact optimizer continuation. Full histories, predictions, optimizer checkpoints and the separate pilot are included in the artifact archive.

## Next target

Extend the retained-state interface to multiple independently owned facts, including a shared fact available from either source and source-private details, while preserving these composition and correction controls. This is the remaining functional integration work toward M7; natural-text evaluation follows that integration.

## Reproduce

Extract source and artifact ZIPs into the same directory. Both use the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/check_all.py
python scripts/audit_joint_retained.py artifacts/joint_retained/joint_20261131
python scripts/report_joint_retained.py
```

Run the audit command for each of the six edge/joint folders to regenerate the full comparison. For a fresh complete run, use `python scripts/run_joint_retained.py --root artifacts/joint_retained_fresh`; the runner reuses the three parent checkpoints in the packaged artifacts/joint_retained/parents directory. The run is resumable by repeating the same command. The source ZIP is the exact published Git tree; artifact hashes and source commit are recorded in its companion manifest.
