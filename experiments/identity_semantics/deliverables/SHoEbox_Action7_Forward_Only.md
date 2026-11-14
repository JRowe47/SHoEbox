# SHoEbox Action 7: dedicated forward-only runtime

14 November 2026 (America/Denver)

Action 7 is complete. The selected 1M profile processes natural text at 7,451 B/s versus 5,083 B/s with frozen training storage: 46.6% faster. Process RSS drops from 160.0 to 76.8 MiB (52.0% less). All matched output hashes and held-out losses are identical.

Baseline: 085e0a60662aa8419bab7ea7f5db3e7d5c1f79c1. Runtime source SHA-256: ae0f0034108e837aad62ddb35bdd3355af9a12a49957d74f26762ec3a936d34b.

## 1M comparison

| Profile | Workload | Frozen B/s | Forward B/s | Speedup | RSS MiB old / new |
| --- | --- | --- | --- | --- | --- |
| Reference | Natural | 4,977 | 6,266 | 1.26x | 151.5 / 74.6 |
| Reference | Repeated | 5,644 | 14,002 | 2.48x | 429.1 / 68.8 |
| Reference | Generation | 4,646 | 6,850 | 1.47x | 147.7 / 73.6 |
| Selected | Natural | 5,083 | 7,451 | 1.47x | 160.0 / 76.8 |
| Selected | Repeated | 9,055 | 16,154 | 1.78x | 277.6 / 69.3 |
| Selected | Generation | 4,480 | 6,116 | 1.36x | 154.7 / 76.0 |

## 5M natural-text cross-check

| Profile | Workload | Frozen B/s | Forward B/s | Speedup | RSS MiB old / new |
| --- | --- | --- | --- | --- | --- |
| Selected | Natural | 4,881 | 6,823 | 1.40x | 175.4 / 92.3 |

## Implementation

The new forward-only-v1 profile keeps all 18 regions, exact binding/source/occurrence semantics, learned content, MinKron recurrence, previous-byte feedback and paired episodic memory. Pattern assignments and lightweight-frame role sums move into immutable forward publications. Semantic derivative slabs, content/context eligibility, neural/temporal archives, reverse work and learning jobs are omitted; training-only collection and bottom temporal Jacobians are skipped.

## Measurement protocol

Host: AMD EPYC 9V74 80-Core Processor; g++ (Debian 14.2.0-19) 14.2.0; production -O3 -march=native -std=c++17 -pthread, caller pinned to CPU 0. Three isolated subprocess trials per pair, with alternating runtime order. Each source receives 2,048 training targets at main/local LR .04/.01333, then a matching warm-adaptive frozen copy receives 1,024 warmup targets. Timed evaluation covers 8,192 targets in 1,024-byte chunks with logits export; generation covers 1,024 sampled bytes at temperature .8, including Python draw/run overhead and excluding full-logits export. Training, construction, compilation, inspection, hashing, evaluation copies and file IO are outside the timers. Tests, sanitizers and timing do not overlap.

## Data and profiles

Natural text is the pinned 70ea7b0 README/STATUS fixture; its separate 1,024-target validation fixture is docs/cortical_fineweb.md. Repeated text and its distinct validation string use the Action-1 fixture. This is not a corpus-scale FineWeb qualification. The reference has 1,031,952 fitted parameters; the selected profile has 1,014,800 at 1M and 4,981,776 at 5M, using rank-16 prediction and Y stride 8. Both retain learned frames. The selected source uses four-byte sparse epochs with one-epoch lag; no learning worker runs during inference.

## Memory

RSS is measured after the source model has closed and the inference stream has warmed, then after timed execution. It includes Python, parameters, live state and allocator overhead. Native inventories report zero semantic derivative slab bytes, derivative-vector capacity, neural/context/temporal archives, live neural/context eligibility, reverse roots and pending bank work in forward-only runs. Fixed native compatibility buffers still occupy about 20.5 MiB; they are not a compact GPU layout.

## Quality

For the natural-text 1M source, held-out NLL is 3.822702 for the reference and 3.792820 for the selected profile, unchanged between frozen and forward-only execution. The 5M selected source gives 3.776932 in both modes. Identical generated-byte hashes use matched RNG streams. These comparisons qualify inference values and cost, not a new learning-policy winner.

## Entry transition

enter_inference() commits materialized slow-Y gradients, reverse epochs, directional replay and bank jobs, joins workers, then retires remaining eligibility and pending next-event teachers without inventing future targets. It preserves the observed prefix. Use finish_training() on the ordinary model for the matching frozen oracle. inference_copy(memory='continue') applies the transition to an independent snapshot; ordinary cold/warm evaluation copies retain the source's visible-parameter contract and do not commit original jobs.

## Episodic memory

cold_adaptive and cold_frozen start with empty episodes; warm_adaptive and warm_frozen copy the source store. Adaptive permits causal admission from the observed prefix, while frozen prevents writes. Query/recurrent state and clocks still evolve. All four policies are tested; paired episode contents, identities, publications and recall behavior match the frozen comparator.

## Checkpoint and continuation

Version 47 saves fitted parameters and a dedicated forward-state payload, including current/pending publications, pattern/frame metadata, context/bottom memory, slow-Y values, clocks and memory-write policy. It omits derivative archives and learning jobs. Irregular chunks, snapshots and save/load preserve continuation. The sampler RNG remains caller-owned.

## Return to training

Training calls are rejected in Python and native run/document APIs until resume_training(reset_context=True). That resets the prefix, eligibility and epoch clocks, reconstructs bottom training accumulators from fitted parameters, and preserves learned weights, routing counts, dictionaries and episodic storage. No credit is claimed for earlier inference bytes. reset() and reset_context() alone leave inference active; learning workers can be enabled again after the explicit return.

## Validation

106 tests passed at 1M: eight new inference tests covering multiple policies and lifecycle scenarios, 13 bank-epoch regressions, and 85 existing storage/indexing/scheduler/prediction/frame/paired-memory/default tests. Four additional inference checks passed at 5M. Coverage includes 16,385 forward continuation targets, lightweight frames, one/two context heads, ancestry/event policies, partial worker epochs, all memory write policies, autoregressive restart, no target leakage, frozen parameters and explicit training rejection/reset.

## Historical compatibility

Old v40/v45/v46 training checkpoints import bit-for-bit. Against the previous 085e0a6 build, 1,025-target training continuation is exact for the full reference; the event/lightweight and selected variants have maximum logit drift 3.8147e-6 and fitted-weight drift below 4.8e-7, with exact structural identities and clocks. The arithmetic policy is unchanged; cross-build floating-point identity is not asserted for those variants. Within the new build, frozen versus forward-only continuation is bitwise exact.

## Sanitizers

ASan and UBSan passed native reference/event/epoch/lightweight-frame runs, 8,193-step continuations, forward export/import, document drain, reset and resumed training, including worker entry. Leak detection was disabled for the host environment. Inspection is read-only at API boundaries; no profiling counters or timers were added to the byte loop.

## Remaining work

Next: Action 8, resident GPU backend and independent-stream batching; Action 9, longer 1M/5M/100M qualification, joint tuning and final profile selection. The selected repeated-text inference result of 16,154 B/s is separate from the still-open 15,000 B/s single-stream training target. The measured Action-6 training profile remains the training recommendation.

## Use

```python
from shoebox_reference import Model
i = model.inference_copy(memory='warm_frozen')
i.run(prefix_bytes)  # NumPy uint8 array; mode='eval'
i.save('inference.npz')
i = Model.load('inference.npz')
i.resume_training(reset_context=True)
```

## Reproduce

See `INFERENCE_POLICY.md`, `src/measure_inference.py`, `tests/test_inference.py`, `tests/check_inference_legacy.py` and `tests/run_inference_sanitizer.py`. Compact raw evidence is in `results/action7_*`. Deliverables exclude datasets, trained checkpoints and compiled binaries.
