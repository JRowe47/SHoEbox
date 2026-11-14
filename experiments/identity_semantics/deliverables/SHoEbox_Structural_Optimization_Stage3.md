# SHoEbox: structural optimization stage 3

13 November 2026 (America/Denver) | Baseline: `69ca74021e2e46b9b50ef6140592cbb28e102ff2`

Stage 3 is implemented in the default reference runtime. Repeated requests for the same saved content input now share a final update factor before the dense outer product. Router contributions are combined after their individual clipping. The learning objectives, per-request policy and existing update boundaries are preserved.

## Matched throughput

| Workload | Stage 2 B/s | Stage 3 B/s | Change | Pairs |
| --- | --- | --- | --- | --- |
| 1M repeated training, 1,024 bytes | 169.9 | 193.8 | +14.0% | 5 |
| 1M natural training, 1,024 bytes | 254.7 | 273.8 | +7.5% | 5 |
| 5M repeated training, 1,024 bytes | 126.3 | 127.6 | +1.1% | 3 |
| 1M repeated training, 4,096 bytes | 79.1 | 95.6 | +20.9% | 1 |
| 1M frozen evaluation, 4,096 bytes | 1,212.5 | 1,493.0 | +23.1% | 3 |

All measurements start from an actual stage-2 checkpoint after 1,024 warmup byte targets: seed 7, width 64, all reference features, periodic training at LR 0.04 and local LR 0.04/3. One pinned CPU on the Xeon Platinum 8573C; alternating serial implementations. Timings include full logit export and exclude compilation, loading, saving, hashes and the separate route-tracing replay. The natural fixture uses the same FineWeb byte stream as the prior review. Raw samples and data/source hashes are included.

The 5M samples overlap: 112.2-126.9 B/s before and 121.7-134.4 B/s after. Its small median change is effectively flat in this measurement. The 4,096-byte training row contains one pair. Frozen timing varied from 829-1413 B/s before to 1272-2763 B/s after. No factor work runs in that mode, so this control does not establish an aggregation speedup for inference.

## Dense product reduction

| Workload | Requests | Products | Reduction |
| --- | --- | --- | --- |
| 1M repeated training, 1,024 bytes | 2,564,555 | 1,380,418 | 46.2% |
| 1M natural training, 1,024 bytes | 1,219,385 | 613,078 | 49.7% |
| 5M repeated training, 1,024 bytes | 2,565,382 | 1,380,849 | 46.2% |
| 1M repeated training, 4,096 bytes | 26,238,359 | 11,668,839 | 55.5% |

## Numerical validation

| Workload | Max logit drift | NLL delta | Route differences | Top-byte differences |
| --- | --- | --- | --- | --- |
| 1M repeated training, 1,024 bytes | 3.81e-06 | -1e-09 | 0 | 0 |
| 1M natural training, 1,024 bytes | 2.86e-06 | +4.21e-09 | 0 | 0 |
| 5M repeated training, 1,024 bytes | 2.86e-06 | -2.53e-09 | 0 | 0 |
| 1M repeated training, 4,096 bytes | 4.77e-06 | +6.7e-10 | 0 | 0 |
| 1M frozen evaluation, 4,096 bytes | 0 | +0 | 0 | 0 |

These updates reassociate floating-point sums; training is not bitwise identical to stage 2. The maximum main-parameter drift relative to the stage-2 continuation update was 4.54e-06. Untimed replays compared 202,752 regional expert/mask snapshots, including held decisions: 0 expert and 0 mask disagreements. Final retained records compared 78,036 common content handles with 0 expert disagreements. The raw comparison also reports runner choices, unmatched handles, RMS errors and complete per-variant hashes.

## Implementation

- Weight grouping keys include destination bank, full immutable input handle (serial, region and index) and feature mask. Each request applies its own clipping, learning rate, input normalization, gate and regional teaching scale before summation. Bias updates and every saved-input derivative retain their original order.

- Router steps are individually clipped before summation by destination bank, input handle and routing mask. Winner and runner contributions remain signed. Even exactly cancelled steps retain the original normalization of both touched banks.

- Nested scopes use their outer owner. Generation-tagged indices in the saved-input cache avoid hashing and per-request allocation. Factors materialize at the existing boundary in first-occurrence order, using 64-request tiles and disjoint worker rows. Workspace grows with unique inputs, rather than repeated requests.

- Checkpoint v39, composition ABI 10, content ABI 1 and neural ABI 1 are unchanged. Grouping scratch and product counters are excluded from serialized state. Explicit historical kernels retain their existing code paths.

## Verification

241 regression tests passed. Four focused aggregation tests also passed at 5M. The scalar oracle exercises large positive/negative gradients, request-level clipping, mixed rates, sparse/dense masks, nested scopes, frozen banks, cancelling router steps and recycled handles; relative update drift is required below 1e-6. Serial and parallel grouped rows agree bit-for-bit. Both factor-kernel and full completion ASan/UBSan harnesses passed, including the live DAG beyond 16,384 records. Leak scanning is disabled on this host.

Stage-2 checkpoints load into stage 3. A checkpoint trained in stage 3 produces bitwise-identical frozen results in both runtimes. Exporting it through stage 2 and returning to stage 3 preserves bitwise training continuation. Same-runtime split calls, save/load, evaluation and resumed training also remain exact. All 34 timed continuations passed credit-integrity checks and repeated runs were deterministic.

## Reproduction

Run src/compare_factor_aggregation.py against baseline 69ca740; add --corpus for natural text and SHOEBOX_EXPERTS=64 for 5M. Run src/check_factor_compatibility.py for cross-runtime checkpoint checks. Results are in results/stage3_*.json and validation logs beside them. Rebuild this report with src/report_factor_aggregation.py. The change bundle excludes models, datasets and compiled binaries.

## Remaining updates

| Stage | Update | Remaining work |
| --- | --- | --- |
| 4 | Share reverse traversals | Compare a union reverse pass within each fixed-weight credit scope; then short, explicitly versioned update epochs. |
| 5 | Native small-head MinKron and local event learning | Add compact context memory, an exact active learning frontier and directional delayed credit with targeted replay. Keep exact identities and paired hippocampal episodes. |
| 6 | Versioned forward/learning workers | Overlap previous-event learning with current forward work and fuse bank reads/updates; compare against the same serial update schedule. |

Stage 3 reduces repeated matrix-update work. Deep input-derivative traversals remain; stage 4 is next, and stage 5 remains the main change aimed at the 15,000+ bytes/s target.
