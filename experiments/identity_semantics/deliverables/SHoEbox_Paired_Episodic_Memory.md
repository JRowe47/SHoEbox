# SHoEbox: paired content/context episodic memory

13 November 2026 | Fifth recommended fix | Base: 3d17b4e

Fix 5 is implemented and enabled in fresh reference models. EC-C and EC-X now share complete content/context episodes and one retrieval decision. Both learned return ports receive members of the same immutable episode. The neural parameter count remains 1,031,952; episodic records are additional memory state.

## Asynchronous observations, explicit pairing

After a regional pass, the latest completed EC-C observation pairs with the latest completed EC-X observation. Both must exist in the current document stream. A new occurrence tuple is considered once; a held member retains its original sequence, tick and primary boundaries. Stored values are detached observed EC inputs, excluding recurrent state and recalled returns.

## One causal retrieval transaction

The first EC request in a byte or drain step queries with its current observed input and the latest completed opposite-side observation. It locks one episode ID, similarity gate and both return vectors for that step. Later requests reuse this pair. If the first request lacks the opposite side, both returns remain empty. Admission happens after the regional pass, before target scoring.

## Joint addresses and whole-episode replacement

Joint similarity averages the two key cosine scores; a signed joint projection supplies sparse bucket probes. Defaults allow 8,192 complete pairs. Novelty, outcome surprise, salience-weighted reservoir priority and shared utility govern admission through stable/adaptive bucket namespaces. Each episode occupies one row. Reinforcement updates metadata without blending values. Replacement assigns a new ID to both members; there is no FIFO, sliding window or age expiry.

## Credit and document boundaries

The existing EC return matrices remain separately learned. Either port can update shared episode utility. Delayed credit checks the saved slot and immutable ID; retired episodes still train the return matrices from their saved values but cannot teach a replacement’s utility. Clearing memory preserves the serial allocator. Stream reset clears pending observations and advances the epoch while preserving stored episodes.

## Checkpoints and evaluation

Checkpoint v39 adds a versioned paired-memory policy and sidecar; composition ABI 10 and existing raw neural/state layouts remain unchanged. Snapshots and fingerprints include episodes, asynchronous observations and cached read identity. Imports validate the sidecar before replacing live memory. Warm evaluation copies episodes into a fresh pairing stream. Frozen memory blocks admission, reinforcement, competition and utility changes.

## Historical continuation and explicit upgrade

Versions 35-38 retain their independent transition stores. An actual v38 checkpoint from base commit 3d17b4e reproduced 129 continuation bytes, logits, weights, composition, neural ownership, content eligibility and pattern state bit-for-bit. To upgrade explicitly: reset_context(), clear_memory(), configure_paired_memory(True). Policy changes require a fresh stream and empty memory.

## Verification

210 distinct regression tests passed, including 15 new paired-memory tests; all 15 also passed at 64 experts. ASan/UBSan passed training, draining, memory import/export, resets and a live semantic DAG above 16,384 entries. Current checkpoints, snapshots and chunked training continue exactly; production and diagnostic logits match.

Controlled cases cover content-first/context-first writes, held-member provenance, same-step coalescing, read-before-write causality, context disambiguation at identical content, transaction locking, immutable reinforcement, replacement and stale-credit rejection, learned port updates, document isolation, frozen memory, target independence and malformed-state rejection.

## Measurements

Two fresh 1M reference models, seed 7, one pinned CPU, 1,024 warmup targets. Each of 4 timed rounds trains 512 targets from an exact copy of the same warmed state, with alternating order. Only the paired-memory policy differs. Compilation, snapshot copying and warmup are excluded. LR is 0.04; local LR is 0.04/3.

| Measure | Independent transitions | Paired episodes |
| --- | --- | --- |
| Median bytes/s | 141.11 | 140.46 |
| Median bytes/CPU-second | 141.10 | 140.50 |
| Training NLL after 3,072 targets | 2.424809 | 2.424571 |
| Held-out NLL after 3,072 targets | 2.882531 | 2.882532 |

Median throughput changed -0.5% in the repeated-state measurement. The separate 3,072-target mechanism run uses repeated sentences and separate synthetic held-out prose. An initial evolving-state timing had substantial round-to-round variance; its raw measurements remain in the evidence file. All missing neural, content and feedback producers, overwritten credit, expired semantic handles and capacity-pruning counters remained zero.

The mechanism run admitted 2,099 episodes, retained 1,993, replaced 106, and made 1,159 joint recalls. It delivered 13,267 utility updates and rejected 424 retired utility handles without redirecting credit. Content/context publications were 2,220/506, exercising asynchronous pairing. The occupied paired sidecar occupied 2,473,824 uncompressed bytes.

The familiar-prefix frozen probe made 25 joint recalls. Suppressing both returns changed NLL by +0.000003685 nats/byte and logits by RMS 0.000049929. The separate held-out probe made no accepted recalls and had identical logits with returns suppressed. The measured downstream effect in this short run is confined to the familiar prefix.

## Reproduce

```bash
python -m unittest discover -s experiments/identity_semantics/tests -v
SHOEBOX_EXPERTS=64 python -m unittest discover -s experiments/identity_semantics/tests -p test_paired_memory.py -v
python experiments/identity_semantics/tests/run_completion_sanitizer.py
python experiments/identity_semantics/src/validate_paired_memory.py
python experiments/identity_semantics/src/validate_paired_memory.py --repeatable-timing --targets 512 --repeats 4 --output experiments/identity_semantics/results/paired_memory_timing.json
```

Evidence: results/paired_memory_validation.json, paired_memory_timing.json, paired_memory_checks.json, paired_memory_legacy38.json and paired_memory_sanitizer.txt.
