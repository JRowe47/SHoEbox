# SHoEbox: structural optimization stage 2

13 November 2026 (America/Denver) | Baseline: `86a4204862425182c3bd30cb0b14aaaa14060441`

Stage 2 is implemented in the default reference runtime: shared publication payloads, cached role contributions and sparse pending gradients. Archived record payload falls 24.1% in the 1M repeated-text comparison. All 38 timed continuations preserve logits, loss, fitted parameters, routing and recurrent/content/memory state bit-for-bit.

## Implementation

- Publication copies retain occurrence identity, roles, freshness, primary clocks and neural origin alongside shared content/context handles. Histories, pending frames, consumers and relays reuse the payload. Explicit vector edits detach it before writing.

- The bounded role cache reuses permuted structural values, semantic residuals and context values. It keys on immutable payload handles, including changed learned codes and intervention copies. Receiver gains, signs, weights and summation order remain at their original arithmetic boundaries.

- Neural and retained-input gradient arrays allocate only for pending work and release when empty. The collector obtains pending roots from the existing indices instead of scanning every archived gradient. All live derivative records and depth policies remain available.

- Checkpoint v39, composition ABI 10 (including the ABI 9 import), hot lace state and neural archive ABI 1 retain their wire formats through explicit conversion. Shared handles and role caches are runtime-only. Historical kernels keep their original storage.

## Memory and copying

| Measure | Stage 1 | Stage 2 | Reduction |
| --- | --- | --- | --- |
| Copied event / handle | 1960 B | 224 B | 88.6% |
| Input record, before pending arrays | 1552 B | 792 B | 49.0% |
| Upper archive record, before pending arrays | 5400 B | 3880 B | 28.1% |
| Archive after 2,048 total targets | 52.39 MiB | 39.77 MiB | 24.1% |
| Archive after 5,120 total targets | 107.14 MiB | 82.12 MiB | 23.4% |

The 2,048-target endpoint holds 31 neural and 41 retained-input gradient blocks (54.0 KiB total). It records 17,942 content-cache hits and 4,950 context-cache hits during the continuation. Archive figures include allocated pending gradient payloads and bottom temporal records; they exclude map/allocator overhead. Handle size measures copy volume, not the entire shared publication allocation.

## Matched throughput

| Workload | Before B/s | After B/s | Change | Pairs |
| --- | --- | --- | --- | --- |
| 1M repeated training, 1,024 targets | 166.3 | 166.7 | +0.3% | 5 |
| 1M natural training, 1,024 targets | 226.6 | 232.1 | +2.4% | 5 |
| 1M frozen evaluation, 4,096 targets | 2,359.6 | 2,444.1 | +3.6% | 5 |
| 5M repeated training, 1,024 targets | 98.5 | 104.8 | +6.4% | 3 |
| 1M repeated training, 4,096 targets | 52.1 | 53.4 | +2.3% | 1 |

Matched actual stage-1 checkpoint after 1,024 warmup targets; width 64, seed 7, all reference features, periodic training at LR 0.04 and local LR 0.04/3. One pinned CPU on an Intel Xeon Platinum 8573C; implementations run serially in alternating order after the regression/sanitizer work finishes. Timings include full logit export and exclude compilation, loading, saving and hashing. The table reports median bytes/s and pairs per row. The natural-text sample uses the same available FineWeb stream as the earlier review; its data hash is recorded.

Small median changes overlap the observed timing variation. Frozen evaluation ranged from 1,703 to 2,651 B/s before and 588 to 2,722 B/s after; all samples, including the slow outlier, remain in the results. The 4,096-target training row is one paired observation.

This stage changes storage and reuse. The full ancestry learning algorithm remains in place; the larger reductions in repeated gradient work are the next updates.

## Verification

237 regression tests passed, including the three new publication/sparse-storage checks. Six storage tests also passed with 64 experts per region (5M). ASan/UBSan passed training, drain, reset, sparse ownership, cache detach, hot/archive round trips, and a live semantic DAG beyond 16,384 nodes. Leak scanning was disabled because the host blocks LeakSanitizer. Old-to-new-to-old checkpoint continuation passed both after training and after frozen evaluation followed by training. The complete serialized state fingerprint also matches in all timed continuations.

## Reproduction

Use src/compare_optimized_runtime.py against an unchanged checkout of baseline 86a4204. Results are in results/stage2_*.json; validation logs are stored beside them. Use src/check_storage_compatibility.py with and without --bridge-mode eval. Rebuild this report with python experiments/identity_semantics/src/report_publication_storage.py. No trained checkpoint, dataset or compiled binary is included in the change bundle.

## Remaining updates, in review order

| Stage | Update | Work remaining |
| --- | --- | --- |
| 3 | Aggregate clipped update factors | Combine final per-request factors for the same immutable input and bank within each update scope, before forming dense outer products. Preserve clipping, masks, rates and mutation boundaries. |
| 4 | Share reverse traversals | Compare a union reverse pass inside each fixed-weight credit scope; then test short, explicitly versioned update epochs. |
| 5 | Small native MinKron context and local event learning | Keep exact content/occurrence identities and paired hippocampal episodes. Add compact contextual memory, an active learning frontier and directional delayed credit with targeted replay. |
| 6 | Versioned forward/learning workers | After reducing work, overlap previous-event learning and current-event forward computation with explicit weight versions; compare against the same serial schedule. |
