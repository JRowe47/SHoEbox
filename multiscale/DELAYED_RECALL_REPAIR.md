# SHoEbox delayed-recall repair

14 November 2026 | Base: `51030d6` | Native policy: version 4

## Result

The original delayed AB/BA answer failure is repaired: full 256-byte-vocabulary answer accuracy improves from 50% to 100% across seeds 7, 17 and 29 at the same 6,000-example training budget. The repaired model passes all eight prefix shifts and repeated-distractor gaps through 4,096 bytes. Removing either retained retrieval or learned query selection returns this task to chance.

This is a repair of retained-unit selection and delivery. Varied changing distractors remain unresolved at this budget, and original-writer credit still stops at the local derivative boundary. The measured language cost is +0.01703 nats/byte and -29.7% training throughput.

## What was missing and what changed

**Retained identity.** Each scale now keeps at most 16 ordered pairs of consecutive completed contents. The byte-scale pairs overlap, preserving a cue that crosses a four-byte grouping boundary. Fixed order-sensitive sketches provide addresses; values contain both learned content features and fixed features. Consecutive identical pairs do not overwrite retained units or advance the occurrence clock. A returning pair refreshes its occurrence; a new pair evicts the least recent slot when full.

**Present-question selection.** After every observed byte, a learned query from that byte and its fast contextual state reads every scale's retained units. Slow memory can therefore respond before its next expert/composition event. Attention combines fixed-key similarity with an occurrence-recency bias. All records contain observed input only; the next-byte target is unavailable to admission and querying.

**Direct delivery and local learning.** Retrieved values contribute directly to the existing shared binary byte decoder. This avoids forcing their effect through stale cached slow states and another tanh. The answer error teaches query selection and retained-value decoder coefficients online. Its query-state derivative enters the existing fast local block. The stored pair content is detached: this does not backpropagate into a retired writer or across parent/child boundaries.

There are no special cases for AB, BA, x, question marks, answer digits or episode boundaries in the runtime. Training still uses ordinary next-byte targets. Existing compact MinKron/addressed matrices, sparse expert routing, native content composition and local auxiliary objectives remain active. Retained recall is enabled by default, separately from the optional exact-payload association API.

<!-- pagebreak -->

## Matched delayed-answer evidence

Both versions train on the same continuous 6,000-example stream per seed, with gaps 16/48/96 and explicit prefix lengths 0-3. Frozen evaluation uses balanced counterfactual AB/BA pairs with identical suffixes, prefix lengths 0-7, and the gaps below. Each cell aggregates 48 answer trials (16 per seed). Explicit prefix lengths 4-7 are absent from training, although concatenation covers all native grouping phases.

| Distractor gap | Previous v3 | Repaired v4 |
| --- | --- | --- |
| 16 | 50% | 100% |
| 48 | 50% | 100% |
| 96 | 50% | 100% |
| 192 | 50% | 100% |
| 768 | 50% | 100% |
| 4096 | 50% | 100% |

These are full-vocabulary argmax answers, not only a forced choice between 0 and 1. A separate continuous frozen evaluation on 256 shuffled cases per seed also reaches 100%; training/evaluation reset mismatch does not explain the result.

| Frozen read control | Answer accuracy (gap 192) |
| --- | --- |
| no_read | 50.0% |
| recency_only | 50.0% |
| no_fixed_values | 100.0% |
| no_learned_values | 100.0% |

The learned and fixed payload halves each support the simple task when the other half is removed. Selection is essential: recency alone does not recover the answer. This separates useful stored information from successfully selecting and delivering it.

## Language and throughput cost

Fresh nominal-1M models train on the same first 1,000,000 Tiny Shakespeare targets. The final contiguous 10% supplies 111,539 held-out targets. A single worker is pinned to CPU 0; execution order alternates by seed. Compilation, evaluation and checkpoint I/O are excluded from training timing; terminal local updates are included.

| Seed | v3 test NLL | v4 test NLL | v3 bytes/s | v4 bytes/s |
| --- | --- | --- | --- | --- |
| 7 | 1.96827 | 2.00020 | 54,301 | 38,037 |
| 17 | 1.97932 | 2.00275 | 54,535 | 38,479 |
| 29 | 1.99792 | 1.99365 | 54,073 | 38,053 |
| Mean | 1.98183 | 1.99887 | 54,303 | 38,190 |

Mean held-out loss changes by +0.01703 nats/byte. Training retains 70.3% of v3 throughput. Teacher-forced inference averages 59,358 versus 90,455 bytes/s; sampled generation averages 13,321 versus 17,369 bytes/s. The repair establishes delayed-answer capability, not a language-quality improvement.

<!-- pagebreak -->

## Broader boundary and remaining work

Two additional fresh-model tasks use 6,000 examples per seed. The two-cue task presents AB/BA and CD/DC and uses either ? or ! to request the corresponding answer. The varied-distractor task replaces repeated x with random bytes from x/y/z. These are development diagnostics, with 64 and 16 trials per cell respectively; they are not claimed as passed gates.

| Task / seed | Gap 16 | Gap 192 | Gap 768 |
| --- | --- | --- | --- |
| Two cues / 7 | 93.8% | 93.8% | 93.8% |
| Varied distractors / 7 | 75.0% | 43.8% | 50.0% |
| Two cues / 17 | 75.0% | 75.0% | 75.0% |
| Varied distractors / 17 | 62.5% | 50.0% | 75.0% |
| Two cues / 29 | 95.3% | 93.8% | 93.8% |
| Varied distractors / 29 | 56.2% | 50.0% | 50.0% |

Pair changes advance the recency clock even when the alphabet is small; enough distinct content can also evict an earlier cue. Thus a very long repeated gap and a long changing sequence are different retention demands. Selective protection under interference, richer reusable composition, and local reinforcement of retired originating representations remain outstanding. This repair does not change the fact that upward publication carries current content rather than recalled context or forecasts.

## Validation and lifecycle

Passed 2,060 native checks and 17 Python tests, including independent numerical derivatives of query selection and decoder delivery, read causality, bounded slots, pair order, off-boundary slow-memory querying, a learned delayed-answer regression, and exact partial-block checkpoint continuation. AddressSanitizer and UndefinedBehaviorSanitizer pass. LeakSanitizer is explicitly disabled on this ptrace-managed host.

New models use checkpoint version 4. Version-3 native files are copied byte-for-byte from the base revision; versions 1 and 2 also remain unchanged. Loading old checkpoints selects the saved historical backend. Cold context clears causal retained pairs while preserving explicit persistent bindings; warm copies and checkpoints retain pair records and current read caches. Frozen inference admits newly observed pairs but never changes fitted weights.

The four-scale v4 model has 999,615 fitted scalars and expert banks [98,49,29,20], versus v3's 997,183 scalars and [104,52,31,21]. Retained queries/decoder coefficients cost 53,120 fitted scalars; the budget allocator compensates with fewer experts. The four retained streams occupy 32,352 bytes, excluding read caches. Reads scan at most 16 slots per scale and do not execute another expert. The minimum four-scale budget is 205,503 scalars; use 250,000 for small tests.

## Reproduction

```bash
python multiscale/tests/run_all.py --no-leak-check
python -m multiscale.compare_delayed_recall --data input.txt --out /tmp/recall --extra-tasks
python -m multiscale.build_delayed_report
```

Measurements: `results/delayed_recall.json`. Verification: `results/delayed_recall_verification.txt`. Core changes: `recall.hpp`, `native.cpp`, `native.hpp`, `model.py`. Historical controls: `addressed_v3/`. Source, tests and compact evidence are committed; datasets, trained checkpoints, binaries and temporary experiments are excluded.

Hardware: AMD EPYC 9V74; one CPU worker. Python 3.12.14, NumPy 2.3.5; C++17, -O3, -march=native, -ffp-contract=off. Dataset SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
