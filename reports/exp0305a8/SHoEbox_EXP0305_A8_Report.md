# SHoEbox EXP03/05-A8: phase-balanced PAD/diverse curriculum

16 November 2026 (America/Denver). Starting main: `3487578a98198e19de833e118cf0d6c5f6b5c0f6`. Pre-fit source/receipt commit: `d622eb08e4ab5ee72315c739f5439255665b373f`.

**A8 development: FAIL. Qualified public E8 remains unchanged.** All A1–A8 audits remain closed. Exactly 20.25M fitting bytes; no pilot, extra fit, sweep, inherited warmup or generic/capacity qualification. Every earlier result, timing and cost remains preserved.

## Full autoregressive results

Native full-256-byte whole-episode exactness. Matched E8/A6/A8, seeds 1/7/19, 128 fresh worlds ×4 record/query-order presentations per condition. Seeds share worlds. These fresh controls do not replace E8’s original qualification.

| Condition | Matched E8 | Matched A6 | A8 |
|---|---:|---:|---:|
| Clean binding | 91.601562% | 92.968750% | 87.890625% |
| Repeat 16 | 94.921875% | 94.921875% | 89.713542% |
| Diverse 16 | 81.640625% | 86.718750% | 86.718750% |
| Competing bindings | 38.932292% | 87.109375% | 80.859375% |
| Three-value overwrite | 25.000000% | 92.187500% | 78.515625% |
| Repeat 64 | 94.921875% | 94.921875% | 89.713542% |
| Repeat 256 | 94.921875% | 94.921875% | 89.713542% |
| Diverse 64 | 3.515625% | 86.328125% | 86.718750% |
| Diverse 256 | 1.562500% | 81.250000% | 83.593750% |

![Primary results](SHoEbox_EXP0305_A8_Results.png)

| Corrected overwrite | Changed key | Untouched key |
|---|---:|---:|
| E8 | 39.843750% | 75.260417% |
| A6 | 95.703125% | 96.484375% |
| A8 | 88.281250% | 90.234375% |

## Exact intervention

Only A8 continuation alternates 64-episode PAD/diverse blocks, beginning with PAD. The rule is `((episode_index // 64) % 2 == 1)`, using the zero-based episode index before increment. Both original record spacers use the same block. Diverse blocks replace their existing 0–7-byte PAD spacers with per-occurrence bytes in [160,255]; PAD blocks change nothing. The exact A7 v7 generator prefix is retained. The cycle never resets at epochs or finish events, and the incomplete final cycle is consumed without rebalancing. The literal registered SHA-256 prefix, raw group-key bytes, little-endian episode/physical-position/block encoding and rejection rule are used. Both existing RNGs, lengths, phases, order, labels, masks, finish schedule and budgets stay unchanged. Every independent warmup and the E8/A6 controls retain PAD. Evaluation fixtures are unchanged.

The generated A8 header is byte-identical to A6 after reversing only its save/load magic `0x3841353050584553`; its native source is identical. Record costs stay 4/4, with A5’s all-256-code hybrid, float64 softmax/gradient and pre-update coupling. Query presence, applicability, values, native credit, A4 inference/chronology/ties/fallback, admission and existing state stay exact. No arithmetic reuse or optimization was introduced; FMA remains disabled.

Every arm has 497,589 fitted parameters, including 7,670 readout weights. There are no new parameters, persistent sequence state, moments, projection caches or slots. The outer cursor `exp0305-a8-cursor-1` records policy, curriculum and exact episode/partial-example position. Generic byte objectives and optional fixture-local supervision remain separate; inference supports all 256 bytes.

The independent spacer/interpreter traversal checks every augmented continuation episode and locks six per-arm/phase streams. Input/target digests change only for A8 continuation; masks, labels, RNG states, schedules and warmup bytes match. Both physical record orders, zero lengths, rejection boundaries, additional hash blocks, future-answer causality and partial-spacer resumes pass.

## Frozen failure localization

First 32 development worlds ×4 orders, both teacher-fed answers, all seeds. These subset measurements localize failure; full autoregressive outcomes set the gates. Final fits and complete evaluation outcomes were locked before these traces.

| Arm | Diverse256 signature survival | Incoming false recognition | True-record phase false negatives | Target removals |
|---|---:|---:|---:|---:|
| A6 | 100.000000% | 3.212851% | 0.000000% | 0 |
| A8 | 100.000000% | 1.857430% | 0.000000% | 0 |

| Arm | Correct given signature survives | Surviving target shortlisted | Recognized filler phases/answer | Filler mass on wrong survivor answers |
|---|---:|---:|---:|---:|
| A6 | 87.500000% | 99.218750% | 2.031250 | 28.663252% |
| A8 | 89.843750% | 99.218750% | 1.187500 | 38.276743% |

| Arm | Competing nearest-key identity | Competing other-key selected mass | Changed-key old-value predictions |
|---|---:|---:|---:|
| A6 | 92.968750% | 10.292267% | 3.125000% |
| A8 | 88.802083% | 16.044485% | 7.812500% |

Admission and exhaustive selection are independently reconstructed. Missing signatures, surviving-target competition and identity/version errors remain separate. Detailed geometry, mass, phase and occurrence records remain in the evidence archive.

| A8 autoregressive episode exactness | First 32 worlds | Remaining 96 | All 128 |
|---|---:|---:|---:|
| Clean binding | 80.468750% | 90.364583% | 87.890625% |
| Diverse 256 | 81.250000% | 84.375000% | 83.593750% |
| Competing bindings | 76.562500% | 82.291667% | 80.859375% |
| Three-value overwrite | 73.437500% | 80.208333% | 78.515625% |

![Frozen diagnosis](SHoEbox_EXP0305_A8_Diagnosis.png)

![Signature survival](SHoEbox_EXP0305_A8_Retention.png)

![Presence and survivors](SHoEbox_EXP0305_A8_Presence.png)

## Measured training-window coverage

Exact data-only traversal of each 2M-byte continuation, with no model calls. Counts are per arm/seed; the generated data schedule is shared across seeds.

| Arm | Positive record-key events | Events containing PAD | Distinct key/phase/windows | Fully diverse negative events | Distinct diverse negatives |
|---|---:|---:|---:|---:|---:|
| E8 | 28,810 | 26,213 | 6,022 | 9,970 | 3,331 |
| A6 | 28,810 | 26,213 | 6,022 | 9,970 | 3,331 |
| A8 | 28,810 | 18,784 | 15,058 | 11,944 | 5,120 |

| A8 block | Consumed bytes | Encountered episodes | Positive record-key events | PAD-containing positives | Distinct key/phase/windows |
|---|---:|---:|---:|---:|---:|
| PAD | 1,003,872 | 5,568 | 14,482 | 13,190 | 4,059 |
| DIVERSE | 996,128 | 5,520 | 14,328 | 5,594 | 11,865 |

The complete continuation encounters 11,088 episodes, including its incomplete last episode: 5,568 PAD and 5,520 diverse. Consumed bytes are 1,003,872 PAD-block and 996,128 diverse-block bytes. Both regimes cover all 64 phases. Exactly 38,797 original PAD input bytes per A8 continuation are replaced; no terminal balancing or appended example occurs. Per-condition/per-phase and block-specific raw-window counts are retained.


## Unchanged extended gates

| A8 condition | All-offset mean | Worst offset, seed mean | Both orders |
|---|---:|---:|---:|
| Clean binding | 87.939453% | 78.125000% | 84.375000% |
| Repeat 16 | 87.288411% | 67.708333% | 88.020833% |
| Diverse 16 | 88.281250% | 78.125000% | 85.937500% |
| Competing bindings | 78.336589% | 65.625000% | 80.468750% |
| Three-value overwrite | 79.687500% | 71.875000% | 77.343750% |

Fresh binary regression: 90.429688% primary; 96.459961% offset mean; 87.500000% worst offset; 85.156250% both orders.

Every primary condition and answer position requires ≥95% mean/90% every seed. Both orders and each padding/repeat/diverse delay require ≥90%/85%. All-offset mean requires ≥95%, worst-offset seed mean ≥90%. No-query drop ≥20 points; reverse must hurt. All state/pairing/gate lesions and every binary gate remain binding.

The same first 64 presentations are tested at all 64 offsets. The next 16 cross all offsets independently at 0/64/128/256 padding delays. Both-order/lesion samples contain 256 presentations. No sample count or threshold was reduced. All per-seed scores/decisions are retained.

## Isolated throughput

| Arm | Bytes/CPU second | Relative native |
|---|---:|---:|
| NATIVE | 21,496.1 | 100.000000% |
| E8 | 18,507.7 | 86.097969% |
| A6 | 14,497.8 | 67.443791% |
| A8 | 14,743.6 | 68.587044% |
| PUBLIC_E8 | 17,792.6 | 82.771077% |

One pinned CPU; three interleaved 65,536-byte repeats plus 1,024 warmup each. Full byte distributions at every input; all other jobs stopped. Compilation/cloning/warmup are outside timed intervals and inside process costs. Native is the fresh untrained v19 reference; public E8 imports matched E8 exactly, with identical prediction checksums. The unchanged speed requirement is ≥80% native.

![Throughput](SHoEbox_EXP0305_A8_Throughput.png)

## Folds, costs and verification

Salt `SHoEbox-EXP0305-balanced-record-context-v8` plus NUL; world RNG 20261130. 512/128/256 train/dev/closed-audit and 128 fresh binary dev worlds. Exclusions include all historical opened binding folds/E2 family, reserved A1–A7 audits and every A2–A7 dev/binary world. Operations/orders remain grouped. No generic data is used; opened E1/E7 generic groups remain excluded later.

Each arm/seed independently receives 250K native warmup +2M continuation. Order RNG 10701/10702, augmentation +1000. No replay, extra fit or inherited warmup. Passing source/data/receipt hashes were committed before fitting; final model/cursor/outcome locks verify.

| Fitting quantity | Total |
|---|---:|
| bytes | 20,250,000 |
| padding bytes | 16,592,550 |
| nuisance bytes | 755,031 |
| replaced padding bytes | 116,391 |
| targets | 18,026,838 |
| answer targets | 226,404 |
| local projection steps | 9,416,016 |
| local presence steps | 35,351,352 |
| gate targets | 17,900,208 |
| local training windows | 22,262,994 |
| Fitting CPU seconds | 647.76 |

PAD/nuisance/replaced-padding counts are included in the 20.25M budget, not added to it. Replaced padding is a subset of nuisance bytes.

| Local cost across three continuations | E8 | A6 | A8 |
|---|---:|---:|---:|
| record positive loss components | 86,430 | 86,430 | 86,430 |
| record negative loss components | 5,730,618 | 5,730,618 | 5,730,618 |
| record positive gradient rows | 86,430 | 86,430 | 86,430 |
| record negative gradient rows | 5,730,618 | 5,730,618 | 5,730,618 |
| hybrid attraction components | 0 | 152,952 | 152,952 |
| hybrid discrimination components | 0 | 152,952 | 152,952 |
| hybrid gradient rows | 0 | 2,447,232 | 2,447,232 |
| hybrid code comparisons | 0 | 39,155,712 | 39,155,712 |

Whole-process research/verification CPU at report generation: **3,142.98 seconds**. Fitting/subphase CPU is included once. Failed metered attempts: prefit_python_initial_process.

A8 Python correctness invocations consumed 105,324 discarded update bytes, including the failed block-boundary coverage assertion and the successful repeated suite. All 18 saved warmup/final pairs pass 1,000 uninterrupted versus 371 + save/load +629 updates: 36,000 separately charged discarded correctness bytes. Historical trajectories consumed 114,716 discarded bytes. Native independent microchecks retain event counts in their logs. Historical unit/native scalar writes remain uninstrumented; complete process CPU and logs are retained. Setup/editing/read-only inspection is not separately CPU-metered; artifact production/packaging has separate receipts.

- 269 distinct Python tests pass: 60 runtime/public, 192 historical research and 17 A8 tests; 21 additional protected-policy checks pass.
- Optimized and ASan/UBSan pass: 824,111 core checks, 723 public E8 checks, 13,863 inherited selection/lifecycle checks, 75,295 hybrid-objective checks and 7,826 record-presence checks per build. Leak checking was explicitly disabled.
- Independent logistic/hybrid gradients, finite differences, all-256-code inclusion and pre-update coupling pass. Codebook has 256 unique codes, no collisions; minimum Hamming distance one. Exact same-weight local/native/inference/admission trajectories pass.
- All 14 instantiated historical constructor/checkpoint trajectories match starting main. v5/v6 loader sources are unchanged but were not independently instantiated. Public/historical numerical sources and earlier reports remain untouched.

## Resume and next work

Read `DIAGNOSIS.md`, `RESUME.md` and the one focused next `A9_REGISTRATION.md`. A8 is complete; no additional diagnostic-driven fit follows its diagnosis. A9 is registered only, not implemented or fitted. The archive contains source/git history, starting-main source, all 18 model/cursor pairs, generated worlds, exact stream/source/outcome locks, raw diagnostics, failed/passing receipts and per-member SHA-256. Checkpoints, datasets and compiled binaries remain outside git. Never reset, finish or replay at an arbitrary cursor boundary.

Failure retains qualified E8 and keeps the A8 audit closed. Passing development would require separately registered fresh generic/capacity qualification at every original margin: language ≤+0.01 nats/byte, copy/reconstruction ≤1-point token loss, post-1M-language retention, all three-seed 1M/5M and speed gates.

Completed verified work, including unsuccessful research, is explicitly authorized for commit/push directly to main without confirmation. Preserve unrelated changes and remote history; never force-push. Deliver ordinary downloadable Markdown, PNGs, resumable archive and copy/paste prompt. No Pages/Spaces.

Primary grounding remains [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/) for paired addressing and [Arora et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html) for multi-query recall measurement. The fixed A8 curriculum is this project’s registered intervention.
