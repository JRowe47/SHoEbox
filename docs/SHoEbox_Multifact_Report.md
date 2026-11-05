# SHoEbox: simultaneous shared and private retention

Completed November 5, 2026. Three parent seeds; two source descriptions; three retained logical facts in one eleven-node graph.

## Result

All locked phase targets: **passed**. The primary target passes in **3/3 seeds**. Each primary condition queries all three fields in 512 fresh episodes per seed. Relation-table pairs are held out throughout. Intact final fact triples use held-out combinations; withheld-update controls retain their original triple, which need not be held out.

| Seed | Update | Initial five-task min | Trained five-task min | All primary min |
| --- | --- | --- | --- | --- |
| 20261220 | 60 | 26.17% | 99.80% | 99.61% |
| 20261221 | 120 | 41.41% | 97.66% | 97.66% |
| 20261222 | 80 | 30.27% | 99.80% | 99.80% |

The five-task columns compare the same recall/correction tasks and episodes before and after training. Across those tasks and all three seeds, mean accuracy rises from **66.45% to 99.73%**. The all-primary column additionally includes four withheld-correction conditions. The initial model already inherits the published producer and composer weights. Checkpoint selection uses validation only, with update zero eligible.

| Composed-query condition | Shared | Left private | Right private |
| --- | --- | --- | --- |
| Recall all fields | 99.93% | 99.80% | 99.87% |
| Common correction from left | 99.74% | 99.48% | 99.80% |
| Common correction from right | 99.87% | 99.74% | 99.22% |
| Left-private correction | 99.67% | 99.61% | 99.87% |
| Right-private correction | 99.61% | 99.87% | 99.80% |
| Withheld: common correction from left | 99.67% | 99.67% | 99.80% |
| Withheld: common correction from right | 99.80% | 99.74% | 99.80% |
| Withheld: left-private correction | 99.67% | 99.54% | 99.87% |
| Withheld: right-private correction | 99.74% | 99.87% | 99.93% |

An answer is `g(f(x))` for the requested retained fact x and newly supplied relation tables. The model never decodes x into an intermediate byte. Correcting one field must preserve the other two answers.

## What changed

F and L read the complete left-source byte stream; G and R read the complete right-source stream. Each source has its own common-field and private-field producer. Every owner has independent contextual state, sparse expert state, and a retained publication. Their write gates distinguish common `=value` from private `:value` observations among identifiers and distractors.

The field query is an explicit address: shared, left-private, or right-private. A shared read chooses the latest valid F/G publication using occurrence metadata; a private read chooses its dedicated owner. Addressing and freshness arbitration are explicit interface mechanics. The producer experts learn which observed bytes to publish. Natural-language query interpretation remains part of the next integration work.

All four producers begin from the matching joint-retention parent. Private producers exchange the learned input-embedding rows for colon and equals at initialization. Runtime bytes are unchanged. Both fields still reach both local producers, so private/shared separation must survive actual interleaved observations.

The original seven-node composer and 256-byte decoder remain exactly frozen. Only the four producers and their continuous edge value projections learn. Queries run the ordinary two-round composer with the four producers inactive.

## Missing sources and continued updates

| Available source information | Shared | Left private | Right private |
| --- | --- | --- | --- |
| Left source only | 99.93% | 99.87% | 28.26% |
| Right source only | 99.93% | 26.43% | 99.93% |
| Left facts erased, equal compute | 99.80% | 26.37% | 99.93% |
| Right facts erased, equal compute | 99.93% | 99.87% | 28.32% |

These controls use all 64 independent fact triples. An unsupported private value has a conditional 25% prior. True source absence skips its two producers; erased-source controls keep the same execution count as intact acquisition.

| Update after three live queries | Shared | Left private | Right private |
| --- | --- | --- | --- |
| Common correction from left | 99.74% | 99.48% | 99.80% |
| Common correction from right | 99.87% | 99.74% | 99.22% |
| Left-private correction | 99.67% | 99.61% | 99.87% |
| Right-private correction | 99.61% | 99.87% | 99.80% |
| Common correction withheld | 99.80% | 99.74% | 99.80% |

These episodes present the original facts once, ask three live field queries, then append only the update suffix. They do not re-observe the initial facts. Repeated queries preserve all four owners' contextual state, expert state, payload, semantic header, version and provenance exactly. All **391,619** measured hold transitions preserve their publication payload bit-for-bit.

Across the twelve owner/seed pairs, observed write precision ranges from **95.93% to 100.00%**, and write recall ranges from **99.53% to 100.00%**.

## Controls and longer delay

| Condition | Shared | Left private | Right private |
| --- | --- | --- | --- |
| Recall, gap 64 | 99.54% | 99.74% | 99.80% |
| Common correction, gap 64 | 99.80% | 99.54% | 98.96% |
| One regional round | 24.02% | 23.05% | 24.80% |
| Four regional rounds | 39.19% | 39.39% | 42.51% |
| No delayed exchange | 24.02% | 23.05% | 24.80% |
| Memory edges blocked | 24.54% | 25.65% | 23.37% |
| Memory reset | 24.54% | 25.65% | 23.37% |
| Shuffled memory: original target | 20.96% | 18.68% | 16.73% |
| Unconditional publication | 22.01% | 26.17% | 29.82% |
| Three repeated query sets | 99.87% | 99.74% | 99.26% |

Shuffling the complete memory publications yields counterfactual accuracies of **99.87%, 99.74%, 99.28%** for shared, left-private and right-private queries. Blocked, reset, shuffle, no-exchange and unconditional-publication conditions match the intact two-round expert count.

Forcing a stale left common replica after a right-source correction follows the old value with **99.74%** accuracy; the reverse direction follows the old value with **99.87%**. The default latest-publication rule is evaluated separately from the learned write gates.

## Learning and verification

The first development pilot exposed destructive writes to the opposite field. The final gate loss gives those events a separate denominator: 0.5 own-field writes, 0.25 opposite-field writes, and 0.25 ordinary holds. Retained and candidate latents align with this model's frozen A ingress representation; the final objective also includes three byte-query losses and the original routing regularizers. Each 16-event chunk backpropagates before its history is detached. There is no replay store.

The confirmatory protocol uses 120 updates, batch 32, producer LR 0.0003, edge LR 0.001, AdamW beta=(0.9,0.95), weight decay 0.01 and clipping at 1. Gaps cycle through 0, 4, 8, 16. Validation at 20-update intervals selects the minimum task/field accuracy, then mean accuracy, then lower NLL. Test results never select weights.

| Seed | Expert executions | Encoded source occurrences | Training minutes |
| --- | --- | --- | --- |
| 20261220 | 12,718,080 | 921,600 | 45.46 |
| 20261221 | 12,718,080 | 921,600 | 45.53 |
| 20261222 | 12,718,080 | 921,600 | 45.87 |

Times and training counters sum the completed checkpointed updates. Discarded partial work after an execution-worker interruption, validation and audits are excluded. The graph has **510,958 parameters**, **768 physical experts**, and **257,772 trainable parameters**. One aligned pair of source bytes executes 48 experts across four producers; one table pair and two-round query execute 264. Two raw source streams are encoded four times, and the counters include that duplicated work.

All **401 regression tests pass**. Independent audits reconstruct **360 training batches**, verify the selected checkpoint and frozen composer tensors, reproduce **360 field-prediction groups**, and reproduce inference-export logits and complete state exactly. Source isolation, causal labels, held-out factor combinations, unique episode IDs, query preservation and exact optimizer continuation are tested. The pilots are separate from confirmatory results and include their exact source snapshots.

## Next target and reproduction

The next stage is joint natural-text training and a generative byte-stream interface, retaining these memory, source-ownership and composition checks. M7 remains open until that text experiment is measured.

Extract the source and artifact ZIPs into the same directory; both use the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/check_all.py
python scripts/audit_multifact.py artifacts/multifact/20261220
python scripts/report_multifact.py
```

Audit each of the three seed directories to reproduce the full comparison. A fresh run uses `python scripts/run_multifact.py --root artifacts/multifact_fresh`; the runner reuses packaged parent checkpoints and resumes interrupted training when repeated. Initial, selected and final optimizer checkpoints, inference states, per-example predictions, protocols and full audits are in the artifact ZIP. The source ZIP is the exact published Git tree; the companion manifest records its commit and every artifact hash.
