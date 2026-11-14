# SHoEbox memory and recall repair

14 November 2026 | Base: `4e52bee` | Native policy: version 3

## Result

Two concrete memory defects are repaired: coherent writes no longer accumulate large read magnitudes, and cold inference no longer discards persistent associations. The end-to-end delayed AB/BA answer task remains at 50% accuracy. This is a partial repair, not a completed long-dependency solution.

## 1. Repeated writes overwhelm the recalled signal

Version 2 sums key-value outer products and divides reads by the square root of accumulated key energy. Repeated, aligned writes add coherently; that energy normalizer does not keep their contribution at a constant scale. The fixed decays eventually limit growth, but at levels large enough to saturate downstream tanh states. A 10,000-event constant-input probe reproduces the amplification.

| Repeated-input probe | Previous v2 | Repaired v3 |
| --- | --- | --- |
| Peak matrix norm | 203.513 | 0.613 |
| Peak read norm | 21.458 | 0.348 |
| Cold clone retains binding | No | Yes |

The trained-state probe uses the same seed-7 delayed-cue training stream, then a warm AB + 192 x + ? prefix. A coordinate is counted as saturated when |state| > 0.95. Level-1 read norm falls from 621.54 to 2.14.

| Slow scale | Saturated coordinates: v2 | Saturated coordinates: v3 |
| --- | --- | --- |
| 1 | 64/64 | 4/64 |
| 2 | 59/64 | 0/64 |
| 3 | 50/64 | 0/64 |

## 2. Starting inference erases stored bindings

The native cold-copy path called reset(true), clearing the association store together with recurrent context. Therefore an explicit binding present in a saved or trained model disappeared from the normal evaluation/generation copy. Cold copies now preserve records and handles. clear_memory=True explicitly requests an empty store. The Python lifecycle repair applies to all three checkpoint backends; automatic neural admission remains frozen.

## Repair mechanics

At each slow event, read the existing matrix first. Decay it, predict the observed value from the fixed write key, and write only the residual with rate 0.5 / (1 + ||key||^2). The local backward pass includes this residual dependence exactly. Fixed sketches, rotary clocks, retention half-lives, learned queries and the eight-event training blocks are retained. The old energy normalizer is removed. No extra expert pass or all-history search is introduced.

## Remaining recall failure

Across seeds 7, 17 and 29, both versions remain at 50% on all tested gaps: 16, 48, 96 and 192 distractor bytes, with shifts 0-7. Training uses 6,000 examples, gaps 16/48/96 and shifts 0-3. Preserving a binding or keeping a matrix stable does not establish that the neural decoder learns the delayed answer. The unchanged acceptance test remains failed; neural episodic recall is still disabled by default.

## Matched language results

| Seed | Previous v2 NLL | Repaired v3 NLL |
| --- | --- | --- |
| 7 | 1.96396 | 1.96827 |
| 17 | 1.98008 | 1.97932 |
| 29 | 1.98721 | 1.99792 |
| Mean | 1.97708 | 1.98183 |

Fresh models train on the same first 1,000,000 Tiny Shakespeare targets; the final contiguous 10% supplies 111,539 held-out targets. Execution order alternates by seed on CPU 0. Mean held-out NLL changes by +0.00475 nats/byte. Mean training throughput is 51,417 versus 53,542 bytes/s (-4.0%). This repairs memory behavior with a small measured language-loss regression; it does not demonstrate a language-quality gain.

## Validation and continuation

Passed 1,897 native checks, 14 Python tests, AddressSanitizer and UndefinedBehaviorSanitizer. Coverage includes independent numerical derivatives of the residual write, a repeated-content regression, read-before-write causality, channel interventions, retained/cleared bindings, and exact partial-block checkpoint continuation. LeakSanitizer was explicitly disabled on this ptrace-managed host. A checkpoint created before the repair reloads byte-for-byte through the preserved version-2 backend.

## Checkpoint policy

New models write native version 3. Version-1 and version-2 native sources remain unchanged in legacy/ and addressed_v2/. Old matrices are never silently reinterpreted as residual-write matrices. Existing checkpoints continue their original training arithmetic; a fresh version-3 model is required to use the new write rule. Both current and previous four-scale models contain 997,183 fitted scalars.

## Reproduction

```bash
python multiscale/tests/run_all.py --no-leak-check
python -m multiscale.compare_recall_repair --data input.txt --out /tmp/recall
python -m multiscale.build_recall_report
```

Recorded measurements: `results/memory_recall_repair.json`; verification: `results/memory_recall_verification.txt`. Native diagnostics remain in a separate executable, outside the production loop.

Hardware: AMD EPYC 9V74; one worker pinned to CPU 0. Python 3.12.14, NumPy 2.3.5. C++17, -O3, -march=native, -ffp-contract=off. Compilation, evaluation and checkpoint I/O are excluded from training timing. Dataset SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
