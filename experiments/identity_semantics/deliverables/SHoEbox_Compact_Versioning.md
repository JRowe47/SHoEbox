# SHoEbox: compact versioning and external inspection

13 November 2026 (America/Denver) | Baseline: `fbf7c816446289bcd987a85028d7cab59389b7ed`

Implemented compact bank versions and removed inline observation work from production completion builds. The pipeline reuses two job arenas, shares immutable input copies by index, follows indexed per-bank scope lists, and avoids unnecessary weight installation and router-cache refreshes. All 18 regions and the one-byte-delayed learning schedule remain.

Serial single-stream throughput improved by 3.9% to 24.5% over the matched Stage-6 baseline across these workloads.

## Throughput

| Workload / bytes | Stage 6 B/s | Compact serial B/s | Gain | Compact 2 workers B/s |
| --- | --- | --- | --- | --- |
| 1M repeated / 2,048 | 723.1 | 818.5 | +13.2% | 817.3 |
| 1M repository prose / 2,048 | 646.8 | 776.3 | +20.0% | 832.5 |
| 5M repeated / 2,048 | 641.2 | 798.1 | +24.5% | 771.2 |
| 1M repeated / 8,192 | 569.3 | 591.5 | +3.9% | 646.4 |

Intel Xeon Platinum 8573C; caller pinned to CPU 0, learning worker to CPU 1; width 64, 12 or 64 experts per region; seed 7. Each run uses 1,024 warmup bytes, LR 0.04 and local LR 0.04/3. Timings include full logit export and API quiescence in 512-byte chunks. Compilation, warmup, snapshots, inspection, evaluation and file IO are excluded. Short workloads use three repetitions in rotated order; the 8,192-byte continuation uses one run per variant. No validation process overlaps timing. Repository prose uses the baseline README with a separate baseline documentation file for held-out evaluation; this is not the earlier FineWeb benchmark.

## Implementation

| Change | Result |
| --- | --- |
| Router-only results | 528 bytes per serialized bank instead of 16,904 bytes (96.9% smaller). A dense weight tile exists only for banks with weight factors. |
| Reusable storage | Two job objects retain scope, factor, input, index and tile capacity. No new batch allocation or shared-pointer reference counting per byte. |
| Shared input snapshots | Each immutable occurrence input is copied once per job. Factors retain an input index; the worker owns all referenced storage. |
| Indexed scope traversal | Each bank follows only its own ordered scope chain. The worker no longer scans every scope for every bank. |
| Cheaper preparation/commit | Skip unused dense-gradient clearing, avoid zeroing arrays that will be overwritten, install only necessary visible weights, and refresh changed routers only. |
| External observation | No per-job timers, observer branches, covariance collection, norm/histogram accounting or diagnostic-only decoder walk in the production completion loop. |

## Inspection

capture_memory() copies parameters and continuation buffers at a quiescent API boundary; compare_memory() compares those detached arrays. read_memory_index() scans current retained publications, constituents, classes and unresolved records. inspect_pipeline() reports active sparse bank entries, input reuse and reserved buffer capacity. Inspection does not advance the committed version or train the model. Current inventories do not reconstruct historical event counts; old counter fields are retained only for checkpoint compatibility. Execution counts, routing usage, credit reuse, clipping and identity guards remain operational.

```python
from shoebox_reference import fresh_model, capture_memory, compare_memory
m = fresh_model(seed=7, profile='pipeline')
before = capture_memory(m, include_stream=False)
m.run(inputs, targets, mode='periodic', lr=.04, lp=.04/3)
after = capture_memory(m, include_stream=False)
changes = compare_memory(before, after)
queue = m.inspect_pipeline()
```

## Correctness and continuation

286 regressions passed in the explicit historical observation build; 24 production pipeline tests passed at 1M and 11 at 5M. Both the full native lifecycle and production compact pipeline passed ASan/UBSan with leak scanning disabled. An 8,192-byte stream retained bitwise worker/chunk/restart equality across consolidation. Tests also cover router-only payloads, two-buffer reuse, detached inspection, no inspection side effects, malformed import rejection, all upper regions receiving task learning, frozen banks, document drains, evaluation causality and worker changes.

The compact storage/queue changes with historical diagnostics enabled preserve bitwise Stage-6 continuation, including operational state. Removing observation code changes compiler rounding: the largest baseline-versus-production logit difference across the measured continuations is 5.7220459e-06. Production serial and threaded execution remain bitwise identical. Held-out scores are reported below, without a new learning-rule or hyperparameter change.

New pipeline saves use checkpoint v43 / extension 2. The loader accepts real v42 / extension-1 pending jobs; initial imported state and expanded pending banks match bitwise. A resumed 1,024-byte audit continuation is bitwise identical to Stage 6. Production continuation differs by at most 1.90734863e-06 in logits and 2.98023224e-08 in base parameters. Save, snapshots and ordinary chunk ends park the completed pending result. finish_pipeline(), document drain and reset remain explicit commit boundaries.

| Workload / bytes | Stage 6 held-out NLL | Compact held-out NLL |
| --- | --- | --- |
| 1M repeated / 2,048 | 2.326266 | 2.326266 |
| 1M repository prose / 2,048 | 3.726236 | 3.726236 |
| 5M repeated / 2,048 | 2.390591 | 2.390591 |
| 1M repeated / 8,192 | 2.204048 | 2.204048 |

## Pending result size

| Workload / bytes | Stage 6 KiB | Compact KiB | Reduction |
| --- | --- | --- | --- |
| 1M repeated / 2,048 | 462.3 | 254.5 | 44.9% |
| 1M repository prose / 2,048 | 1,023.6 | 608.1 | 40.6% |
| 5M repeated / 2,048 | 594.4 | 322.7 | 45.7% |
| 1M repeated / 8,192 | 478.8 | 271.0 | 43.4% |

These are pending-result sizes at the measured endpoint, not peak RSS. Reuse intentionally retains high-water buffer capacity; inspect_pipeline() reports reserved_bytes separately from active_payload_bytes. No per-byte peak/copy counter is maintained in production. The captured scopes and input arenas are transient and are not serialized.

## Reproduction

Run tests/test_compact_pipeline.py and tests/test_bank_pipeline.py in the production build. Run the complete historical coverage suite with SHOEBOX_INLINE_DIAGNOSTICS=1. Run tests/run_compact_sanitizer.py and tests/run_completion_sanitizer.py. src/measure_compact_pipeline.py compares a supplied Stage-6 runtime; src/check_compact_long.py and src/check_compact_compatibility.py exercise restart and v42 continuation. src/profile_bank_pipeline.py now performs boundary inspection rather than source-level timer injection. Raw measurements are results/compact_*. The code bundle contains source, patch, tests and compact evidence; no models, datasets or binaries.

## Remaining updates

| Order | Update | Work |
| --- | --- | --- |
| 1 | Reduce caller-side learning work | Optimize reverse traversal and local adjoints; retest the benefit of the learning worker after each change. |
| 2 | Reduce record maintenance | Use the live memory indexes to identify expensive collection and retention paths while preserving eligible producers. |
| 3 | Qualify scaling and the 15k target | Longer paired 1M/5M runs, then 100M; compare held-out loss per second, memory and single-stream throughput. |
