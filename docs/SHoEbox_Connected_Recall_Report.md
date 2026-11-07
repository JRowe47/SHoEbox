# SHoEbox connected recall continuation

November 7, 2026 | Parent code: 614033e | 312,440 trainable parameters

**The connected continuation does not yet establish reliable value-conditioned recall.** This phase completes 16 paired-source updates in each of two matched arms and adds direct coverage and four-value causal checks.

## Held-out behavior

| Checkpoint | Update | Language NLL | Answer NLL | Immediate accuracy | All four correct |
| --- | --- | --- | --- | --- | --- |
| Parent | parent | 3.1215 | 1.7924 | 25.00% | 0.00% |
| Connected selected | 16 | 3.1089 | 1.6659 | 25.00% | 0.00% |
| 16-byte cutoff selected | 16 | 3.1065 | 1.6524 | 25.00% | 0.00% |
| Connected final | 16 | 3.1089 | 1.6659 | 25.00% | 0.00% |
| 16-byte cutoff final | 16 | 3.1065 | 1.6524 | 25.00% | 0.00% |

Language uses 1,024 new natural-byte targets. Ordinary memory uses 144 answer targets across eight documents. The immediate causal test uses 32 query prefixes and 128 answers. Each prefix is evaluated with all four possible earlier values; only that source byte changes. Accuracy uses the full 256-byte vocabulary. A constant-answer strategy gets 25% accuracy and no complete four-value sets.

Selected checkpoints minimize mean validation language/answer NLL at updates 0, 8 and 16. Final checkpoints retain the fixed-budget result even when an earlier checkpoint wins. Test results do not select checkpoints.

For selected connected/control checkpoints, exact accuracy beyond 96 bytes is 22.22%/24.07%. Complete delayed four-value sets score 0.00%/0.00%. Immediate balanced logit margins are 0.000125/0.001880; a source-insensitive answer preference cancels in this measure.

## What the repaired path reaches

| Training answers with exact source credit | Connected | 16-byte cutoff |
| --- | --- | --- |
| All exact queries | 209/291 (71.8%) | 64/291 (22.0%) |
| Native immediate queries | 86/90 (95.6%) | 51/90 (56.7%) |
| Exact queries beyond 96 bytes | 16/71 (22.5%) | 0/71 (0.0%) |

Both arms process 8,192 task bytes and the same number of natural-text bytes, with 374 task answer targets. They share initialization, AdamW moments, byte streams, parameter count and expert budget. Four independent streams use 128-byte optimizer windows and 16-byte execution chunks. Each update sums language/task gradients at identical weights, clips the combined norm to 1.0 and steps AdamW. The existing task grammar and natural distractors are unchanged.

Coverage checks actual source positions, document identity and resets. These are available gradient paths, not a claim that each path learned useful content. The repair trainer's default remains 384 bytes; this first comparison explicitly uses 128-byte windows.

## Memory and verification

Optional segment recomputation reduces measured process peak memory from 1634.1 to 685.2 MiB (**58.1% lower**) on one matched 64-byte, four-stream paired update. Runtime changes from 38.59 to 55.37 seconds (43.5% higher). Runtime state, credit traces, cursors and totals match exactly; the largest post-update weight difference is 2.98e-08. The paired recall study uses ordinary activation storage.

All 43 targeted checks pass, including full parameter-gradient comparison, source/reset isolation, large-bank and batched-stage arithmetic, hook cleanup after failure, balanced value probes and exact resume within each execution mode. Independent audits reproduce paired exposure, checkpoint selection and finite final weights.

The next target remains reliable value-conditioned recall with broader ordinary stream exposure. The optional recomputation path permits larger connected windows within a smaller memory budget. Existing architectural controls remain available for later ablations.

Protocol and commands: docs/connected_recall.md in https://github.com/JRowe47/SHoEbox
