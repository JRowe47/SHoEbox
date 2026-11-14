# SHoEbox Action 5: lightweight binding frames

14 November 2026

Source revision: [f7b92fcc6df51c8a73343dd3949043c17a58214e](https://github.com/JRowe47/SHoEbox/commit/f7b92fcc6df51c8a73343dd3949043c17a58214e). Baseline: `b91ca9eb8b31ee506442467bb0d42b958392a70d`.

Action 5 is implemented as lightweight-frames-v1. At the selected rates, the combined variant processes 1,445.5 B/s versus 1,444.5 B/s for Action 4 (+0.1%). Median held-out NLL is 3.08479 versus 3.08863 (-0.00384).

Keep the selected Action-4 policy as the preferred profile; retain lightweight frames as an explicit comparison policy.

## Matched natural-text comparison

| Policy | Main LR | B/s | Held-out NLL | Target seconds | RSS MiB |
| --- | --- | --- | --- | --- | --- |
| Action-3 event | 0.16 | 1,104.5 | 3.13424 | 13.70 (3/3) | 247.1 |
| Lightweight frames | 0.16 | 1,123.4 | 3.10772 | 14.43 (3/3) | 247.1 |
| Selected Action 4 | 0.16 | 1,444.5 | 3.08863 | 9.92 (3/3) | 249.6 |
| Frames + Action 4 | 0.16 | 1,445.5 | 3.08479 | 9.80 (3/3) | 249.0 |

The combined policy reaches the target sooner on 1/3 matched seeds. The median paired time change is 2.41% slower. The selection uses paired changes; ratios of unpaired median target times can give a different impression.

| Seed | Combined speed change | NLL change | Target-time gain |
| --- | --- | --- | --- |
| 7 | -6.45% | +0.00105 | -6.08% |
| 11 | +0.94% | +0.02946 | -2.41% |
| 19 | +0.75% | -0.00803 | +2.07% |

CPU: AMD EPYC 9V74 80-Core Processor; g++ (Debian 14.2.0-19) 14.2.0; production -O3 -march=native, one pinned caller CPU. Nominal 1M uses 18 regions, width 64 and 12 experts per region. Natural-text qualification uses three matched seeds (7, 11, 19), 17,408 training targets and 1,024 separate validation targets. Source text is the pinned repository README/STATUS fixture, with cortical_fineweb documentation held out; this is not a FineWeb corpus run. Native training calls include full logits export; evaluation, inspection, construction, compilation and file IO are excluded. Trials rotate policy order and execute without overlapping tests or profiling. Time to NLL 3.33 is the first observed validation crossing at 2,048-byte intervals. Repeated-text uses 8,192 training targets, separate validation and target NLL 2.62. Each policy is screened at 0.5x/1x/2x of the previous selected LR 0.08; local LR is main LR/3. Selected rates are chosen by final validation NLL on seed 7 before matched-seed comparison.

## Repeated-text cross-check

| Policy | Main LR | B/s | Held-out NLL | Target seconds | RSS MiB |
| --- | --- | --- | --- | --- | --- |
| Action-3 event | 0.16 | 1,153.0 | 2.60958 | 1.49 (3/3) | 336.6 |
| Lightweight frames | 0.16 | 1,116.7 | 2.70161 | 1.53 (3/3) | 392.8 |
| Selected Action 4 | 0.16 | 2,160.5 | 2.87538 | 0.87 (3/3) | 336.8 |
| Frames + Action 4 | 0.16 | 1,964.7 | 2.88233 | 0.85 (3/3) | 394.5 |

On repeated text, the combined variant is 9.1% slower and uses 57.7 MiB more median RSS than Action 4. At LR .16, final losses can rise after the first target crossing. This natural-text screening rate does not replace the earlier general Action-4 setting.

## Implementation

**Intermediate representation.** S2 through G5 keep immutable exact child handles, source roles, occurrence IDs, coverage and primary clocks. Their intermediate code is structure plus the fixed, normalized signed-role sum of learned child residuals. Recognition retains the existing per-role child descriptor, class dictionary, transition evidence and primary-clock boundary rule.

**Completed content.** Every completed unit uses the full regional content expert, adapter and payload gate. A single-frame completion clones the saved binding graph, preserves its exact binding identity and roles, and transforms the saved sum once. Multi-frame completion transforms the ordered sequence of fixed descriptors. Child content therefore remains a learned input at every level.

**Learning and credit.** Frames have no trainable local transform. Saved fixed-role Jacobians carry actual task/local teachers to their child producers; learned updates occur at completed publications. Fixed descriptors are transparent to the local completion frontier and are excluded from synthetic local-update recipients. Single-frame completion links directly to its original children. Frozen recipients still transmit derivatives.

**Version and lifecycle.** Checkpoint v45 requires the explicit frame policy and retains pending frames, producer records, class state, prediction state, and any pending bank job. Older versions keep learned frames exactly. Changing representation requires an empty, unfitted recognition state. Snapshot copies, chunk boundaries, document drains, and worker changes preserve the declared schedule.

**Inspection.** The new inspect_frames() reads live descriptors, expert records, exact edges, admitted-frame clocks, publication clocks and pending frames between API calls. No timers or historical diagnostic counters were added to production. The existing semantic slab is reused; this action does not introduce a smaller descriptor-record allocator.

## Validation

- 349 checks passed in the full 1M regression suite; 35 checks passed in the dedicated frame suite.

- 7 selected 5M checks passed, including numerical credit, continuation, region contribution, and long-stream retention.

- Five real published Action-4 checkpoint configurations continue for 8,192 bytes each with exact logits and logical memory fingerprints, including drain/reset. Coverage includes reference, event, pipeline, and Action-4 event/pipeline combinations.

- ASan and UBSan passed the production descriptor, ownership, prediction and worker lifecycle harness. LeakSanitizer was disabled for this environment.

- Finite differences cover one-, two- and eight-frame completion, frozen-recipient derivative flow, and both reverse policies. Recognition checks cover source/order sensitivity, stable classes, causal boundaries and relay class preservation.

- Natural-text interventions verify all 18 regions publish and verify upper neural/content contributions. Repeated-text long compositions exercise wrap/collection and complete document drains; short repeated streams can legitimately leave G5 pending.

## Memory and boundary behavior

| Policy | Fitted coefficients | Resident descriptors | Resident experts |
| --- | --- | --- | --- |
| Action-3 event | 1,057,552 | 0 | 6,503 |
| Lightweight frames | 1,057,552 | 705 | 5,163 |
| Selected Action 4 | 1,014,800 | 0 | 4,609 |
| Frames + Action 4 | 1,014,800 | 5,410 | 10,958 |

Resident record counts are inventory at the final boundary, not cumulative encoder calls. Frames add no fitted parameters. Existing slab capacity and sidecars remain allocated; RSS includes allocator retention and untimed evaluation copies.

| Region | Action 4 frames/publication | Combined frames/publication |
| --- | --- | --- |
| L3 | 1.04 | 1.04 |
| E4 | 1.00 | 1.00 |
| G5 | 1.00 | 1.00 |

Ratios use admitted frames minus the pending tail, divided by completed publications, then take the seed median. Boundary trajectories may change under the new representation.

The combined 1,445.5 B/s is 9.6% of 15,000 B/s; the remaining single-stream gap is 10.38x.

The inspected natural-text upper regions mostly publish one frame per unit. In those cases, this policy moves the full transform to completion rather than removing it, while also creating an intermediate descriptor record. Larger repeated-text compositions save frame transforms but did not improve end-to-end performance in this comparison.

## Production sampling

| Profile | Thread | CPU seconds | Forward | Reverse | Bank updates |
| --- | --- | --- | --- | --- | --- |
| event | caller | 5.55 | 12.4% | 31.4% | 12.1% |
| compact_serial | caller | 5.24 | 17.0% | 34.9% | 8.3% |
| compact_threaded | caller | 5.26 | 16.1% | 39.5% | 3.6% |
| compact_threaded | learning_worker | 0.61 | 0.0% | 0.0% | 7.9% |

External wall-time instruction samples of the production binary, with 1,024 warmup and 8,192 sampled training targets at LR .04. These categories exclude unresolved/inlined work, allocation, collection and waits from the displayed columns and are not inclusive CPU costs. Thread CPU seconds are measured separately. Sampling does not contribute to throughput numbers.

## Learning-rate screening

| Policy | Main LR | Final validation NLL | B/s |
| --- | --- | --- | --- |
| Action-3 event | 0.04 | 3.31293 | 949.6 |
| Lightweight frames | 0.04 | 3.31591 | 1,102.9 |
| Selected Action 4 | 0.04 | 3.27486 | 1,517.1 |
| Frames + Action 4 | 0.04 | 3.27577 | 1,542.7 |
| Action-3 event | 0.08 | 3.29099 | 1,104.9 |
| Lightweight frames | 0.08 | 3.29954 | 1,095.8 |
| Selected Action 4 | 0.08 | 3.20161 | 1,514.6 |
| Frames + Action 4 | 0.08 | 3.20437 | 1,547.3 |
| Action-3 event | 0.16 | 3.12700 | 1,071.1 |
| Lightweight frames | 0.16 | 3.10772 | 1,071.3 |
| Selected Action 4 | 0.16 | 3.08374 | 1,414.0 |
| Frames + Action 4 | 0.16 | 3.08479 | 1,517.5 |

## Use and reproduction

```python
from shoebox_reference import fresh_model
m = fresh_model(profile="event", frames="lightweight-frames-v1",
                prediction=dict(compact=True, rank=16, slow_stride=8))
```

The original default reference and saved policies remain available. Run the dedicated frame tests, run_factor_regressions.py, check_frame_checkpoint.py and run_frame_sanitizer.py separately from timings. measure_frame_policy.py records all data hashes, runtime source hashes, curves, live inventories and rates; profile_frame_policy.py samples the unmodified production binary externally. These are short controlled screens. Longer data/seed/scale qualification remains Action 9.

## Remaining plan

| Action | Scope |
| --- | --- |
| 6 - next | Sparse bank epochs and broader worker overlap |
| 7 | Dedicated forward-only runtime |
| 8 | Resident GPU backend and independent streams |
| 9 - ongoing | Longer 1M/5M/100M qualification and final profile selection |
