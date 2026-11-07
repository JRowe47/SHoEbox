# SHoEbox: delayed payload credit

November 6, 2026 | Implementation: `a7a14ab` | Two paired seeds, six completed runs

The delayed payload-credit mechanism is implemented, verified, and published. Later reads can now train a small part of the encoding used to write memories across detach boundaries. **This adapter did not produce a consistent retention improvement, so the feature remains optional.**

## Held-out results

Exact-answer NLL, in nats per answer byte; lower is better. Values below are equal-weight means across initialization/data pairs 41/241 and 43/243. Long-delay evaluation triples the configured delays.

| Arm | Selected test | Final test | Selected long | Final long |
| --- | --- | --- | --- | --- |
| Gate only | 1.2079 | 1.2063 | 1.3481 | 1.2554 |
| Payload credit | 1.2177 | 1.2166 | 1.2330 | 1.2729 |
| Trained fresh | 1.2195 | 1.2186 | 1.2677 | 1.3004 |

Payload credit is 0.0097 NLL worse than gate-only credit on selected test checkpoints and 0.0103 worse at the final budget. Its selected long-delay advantage of 0.1151 reverses to a 0.0175 disadvantage at the final budget. It narrowly beats trained fresh reads on mean test NLL, without a consistent advantage from older stored content.

| Seed | Arm | Selected step | Test NLL | Test accuracy | Long NLL |
| --- | --- | --- | --- | --- | --- |
| 41 | Gate only | 500 | 1.1690 | 38.6% | 1.3375 |
| 41 | Payload credit | 600 | 1.2384 | 37.2% | 1.2516 |
| 41 | Trained fresh | 500 | 1.2273 | 34.4% | 1.3019 |
| 43 | Gate only | 400 | 1.2469 | 41.1% | 1.3587 |
| 43 | Payload credit | 500 | 1.1969 | 34.4% | 1.2143 |
| 43 | Trained fresh | 500 | 1.2117 | 36.8% | 1.2335 |

Selected checkpoints minimize all-answer NLL on the same target validation distribution. Final checkpoints are all at update 600. Each test uses 32 documents: 215 exact questions for seed 41 and 209 for seed 43; long-delay counts are 222 and 216.

| Arm | Semantic NLL | Context NLL | Corrected-question NLL |
| --- | --- | --- | --- |
| Gate only | 0.7161 | 0.0014 | 1.0442 |
| Payload credit | 0.7476 | 0.0018 | 1.0488 |
| Trained fresh | 0.7625 | 0.0094 | 1.0536 |

These are selected-checkpoint means. Corrected-question metrics include all question types following corrections; exact-only correction results remain in the audit evidence.

---

## Did older publications help?

Each intervention starts before the entire query, preserves pre-query recurrence and ownership metadata, and allows normal query-time writes. Positive NLL delta means shuffled content hurt prediction. The table uses publications older than 64 bytes, beyond the full live gradient window, at selected checkpoints. Counts refer to actually changed queries and publication cells.

| Seed / arm | Test delta | Queries / cells | Long delta | Queries / cells |
| --- | --- | --- | --- | --- |
| 41 / Gate only | +0.008356 | 10 / 10 | +0.000696 | 115 / 153 |
| 41 / Payload credit | -0.000455 | 21 / 23 | +0.002693 | 121 / 170 |
| 43 / Gate only | -0.000090 | 25 / 30 | +0.007581 | 70 / 76 |
| 43 / Payload credit | +0.000164 | 49 / 52 | -0.001184 | 173 / 307 |

For payload credit, final-test deltas are -0.000455 over 21 changed queries and +0.000137 over only two. Final long-delay deltas are +0.002693 and -0.000026. Whole-query shuffling of all pre-query content also has mixed effects: -0.028260 and +0.001557 at the selected payload checkpoints. The older-content effects do not establish a reproducible retention gain. Audits at 16 bytes are also preserved.

## What changed

Four identity-initialized coefficients per owner adapt producer payloads along four fixed orthonormal directions. A publication retains four bounded sensitivity values tied to its owner and version. Holding preserves that credit, replacement discards the predecessor, and reset clears it. A later memory gradient updates the responsible coefficients once, before clipping and AdamW. Live-window autograd continues training the ordinary producer.

The estimator is a conditional derivative with historical producer features and write decisions held fixed; it does not reconstruct the historical producer graph or differentiate through past optimizer steps. No replay store is added. Rank defaults to zero, preserving the legacy model.

All arms have 549,075 parameters, including the same 16 new coefficients. The adapter adds 1,280 bytes of fixed directions and 768 bytes of persistent credit state at batch eight. Both traces together occupy 480,512 bytes in every arm. Selected evaluation-state storage ranges from 157,824 to 206,720 bytes at batch 32, reflecting the actual retained tensor storage.

Each arm receives 307,200 training bytes and 18,432,000 expert executions over 600 updates: 200 acquisition, 200 bridge, and 200 target. Training plus validation took 1,522-1,580 seconds per arm with six concurrent CPU jobs. Both payload runs received nonzero delayed credit on 597 updates and recorded actual AdamW coefficient changes.

## Verification and next step

All **502 checks across five suites passed**, as did the live-output example and GitHub CI. New checks match factored gradients and AdamW updates to full-history autograd, verify held-publication gradients in the regional model, and check replacement, reset, version ownership, and exact resume across stages. All 32 whole-query audits completed; no unmodified fork changed the predicted answer. The test runner now isolates regional modules to release accumulated tensor/JIT storage.

Keep the adapter available for controlled experiments. The next comparison should apply Raven-style holds to internal owner state at matched candidate compute, testing whether protecting the producer state improves retention under interference. Multiple read slots and the persistent short-delay mixture remain separate comparisons.

Full configs, selected/final/stage checkpoints, optimizer and runtime state, raw histories, query audits, validation logs, and file hashes are in the artifact archive. The source archive includes the protocol and reproduction scripts.

Reproduce: `python scripts/run_payload_credit.py --out artifacts/payload_credit --workers 3`
