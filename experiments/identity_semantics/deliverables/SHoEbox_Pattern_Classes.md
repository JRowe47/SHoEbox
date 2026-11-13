# SHoEbox: reusable pattern classes

13 November 2026 | Fourth recommended fix | Base: 13aec98

Fix 4 is implemented and enabled in fresh reference models. Regions now learn reusable pattern classes separately from exact binding identity. Completion uses learned class transitions and a next-content forecast. The neural parameter count remains 1,031,952; class prototypes and statistics are additional learned dictionary state.

## Exact provenance remains intact

The occurrence ledger still records every child identity, source role, presence and coverage geometry. Class matching does not replace those IDs or drop held side sources. Class IDs use independent, monotonically increasing serials; expert indices are never semantic labels. Transparent relays retain their original class assignment.

## Role-aware learned classes

A descriptor keeps normalized learned child codes in separate role slots, with source/presence schema, order/overlap relations and bounded geometry relative to the primary clock. Primary and repeated-source roles receive full weight; singleton side roles receive 0.1 weight. Matching radius is 0.22 with a 0.35 guard per core role. Online prototype means adapt over at most 64 observations. Recognition begins at min_count, normally three observations.

## Predictive completion

Adjacent classes update separate transition counts and a per-class prediction of the next role-aware content vector. The target includes all child roles and a smaller contribution from the newly exported expert content. Categorical and continuous prediction confidence each supply half the completion score. Decisions use evidence accumulated before the current transition is learned. Primary gaps or overlaps still split sequences; L2 atom inputs use byte-class transitions.

## Ownership, replacement and checkpoints

Each region has up to 256 classes with least-recently-used replacement. Eviction never reassigns a retired ID. Per-event assignments and forecast targets follow generation-checked semantic ownership and collection. Version 38 saves class policy, prototypes, transitions and live assignments alongside composition ABI 10. Actual version-37 continuation was reproduced exactly. Versions 35-37 keep their historical exact-key recognition policy.

## Evaluation and upgrades

Frozen evaluation and frozen regions do not learn class statistics. Stream resets clear assignments and diagnostics while retaining fitted classes. Snapshots, frozen copies and fingerprints include the appropriate state. Upgrade an older checkpoint with reset_context() followed by configure_pattern_classes(True); changing recognition policy midstream is rejected.

## Verification

195 distinct package regression tests passed; 11 class tests also passed at 64 experts. ASan/UBSan passed training, draining, reset, export/import and a live semantic DAG above 16,384 entries. Current checkpoints, snapshots and chunked training reproduce exact continuation. An actual v37 checkpoint reproduced 269 continuation targets, parameters, composition, neural ownership and content eligibility bit-for-bit.

Controlled cases verify class reuse under changed side context in all 12 completing regions S2-G5, provisional-to-recognized promotion, distinct primary role/order classes, changed learned content at fixed exact identity, predictive evidence before mutation, multi-frame completion, gap rejection, relay preservation, dictionary turnover and non-aliasing, frozen learning and malformed-state rejection.

## Matched measurements

Fresh 1M models, seed 7, one pinned CPU, 1,024 warmup targets followed by 2 rounds of 1,024 targets; LR 0.04 and local LR 0.04/3. The policies differ only in class recognition/completion. Compilation, draining and evaluation are excluded from training timing. Training uses repeated sentences; held-out scoring uses separate synthetic prose.

| Measure | Exact bindings | Pattern classes |
| --- | --- | --- |
| Median bytes/s | 73.15 | 109.32 |
| Median bytes/CPU-second | 73.25 | 109.35 |
| Final training NLL | 2.447432 | 2.424809 |
| Held-out NLL | 2.908183 | 2.882576 |

Throughput increased 49.4% in this matched smoke; changed completion also changes the amount of neural/credit work. All missing neural, local, content and feedback producer counters, overwritten credit and expired semantic handles remained zero. Frozen evaluation/generation preserved fitted weights and dictionary state; production and diagnostic logits matched.

On the same trained checkpoint, disabling class recognition increased held-out NLL by +0.026758 nats/byte and familiar-pattern NLL by +0.085328. Logit RMS differences were 0.214443 and 0.260415, respectively. Positive deltas mean the enabled class policy helped prediction on those probes.

## Regional observations

| Region | Class reuses | Promotions | Multi-frame publications |
| --- | --- | --- | --- |
| L3 | 925 | 86 | 115 |
| S3 | 1129 | 93 | 175 |
| R3 | 199 | 33 | 12 |
| F3 | 189 | 50 | 9 |
| E4 | 13 | 0 | 0 |
| F4 | 34 | 4 | 0 |
| V4 | 92 | 20 | 4 |
| M4 | 109 | 26 | 0 |
| G5 | 0 | 0 | 0 |

This 3,072-target smoke shows reuse and multi-frame formation in several regions. E4 and G5 did not reach recognized classes in that short natural execution; controlled equivalent-pattern tests pass in both. Counts include bound frames and completed sequences. The report does not infer long-run language quality from this short run.

## Generation checks

Prompt: The river | temperature 0.8 | seed 17 | printable ASCII plus newline | 96 generated bytes.

Exact bindings:

```text
te ie orn he zr ri hde iure ithve skle. th tUe iooree re h h heee re hee sre or ih rh hne them g
```

Pattern classes:

```text
te ie nos he zr qo he rkuoe itgw. soe slhe v(e kpllee qe h h heee rie qrhee he Lhe rmchre thehe
```

## Reproduce

```bash
python -m unittest discover -s experiments/identity_semantics/tests -v
SHOEBOX_EXPERTS=64 python -m unittest discover -s experiments/identity_semantics/tests -p test_pattern_classes.py -v
python experiments/identity_semantics/tests/run_completion_sanitizer.py
python experiments/identity_semantics/src/validate_pattern_classes.py
```

Evidence: results/pattern_class_validation.json, pattern_class_checks.json and pattern_class_legacy37.json. No models, datasets or compiled binaries are included.
