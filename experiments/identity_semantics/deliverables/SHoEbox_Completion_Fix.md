# SHoEbox completion and learned-content repair

13 November 2026 | Base: ea878a33392a36f37963d225b6479663b17504e0

Implemented the two requested mechanisms in the explicit completion kernel: learned completion above L2 with multi-source roles intact, and recursively learned constituent exports with saved producer credit. Earlier runtimes remain available for exact comparison.

## Completed event clock

S2 through G5 admit fresh primary-source blocks with distinct, non-overlapping occurrences. Every declared immediate side role remains in the bound frame, including explicit held occurrences. Side updates alone do not publish a new unit. Transition evidence and recognized frames control completion; novel combinations are provisional. Up to eight frames can form a unit. Partial tails drain without reusing a primary occurrence.

## Recursive exported content

A normalized role transform of child.code minus child.structure passes through a bounded learned gate into the parent's semantic adapter output. The result is exported and used by the next level. IDs, structural descriptors and recurrent neural context stay separate. Sixteen scalar gates add 16 parameters: 1,031,952 total at 12 experts.

## Saved credit and teaching

Every encoding saves the actual child dependencies, features and derivatives. Duplicate ancestry paths accumulate before updating their producer. A collected pool of 16,384 semantic records retains live histories, pending frames and eligibility; relays preserve the same handle. Neural eligibility is 64 events in this kernel. Bound frames are encoded on admission; regional experts execute on completion, while Y still executes each byte. Forecasts teach the next completed content once.

## Checkpoint contract

Version 33 / composition ABI 7 saves pending frames, learned gates, semantic dependencies and neural eligibility. The converter accepts version-31 relays and version-32 messages at a fresh stream boundary, preserving fitted weights and dictionaries. Old event state is not reinterpreted.

## Verification

132 existing package tests passed, followed by all 15 completion tests on the final implementation. The native AddressSanitizer/UndefinedBehaviorSanitizer harness passed training, evaluation, draining, reset and state export/import. New tests cover recognition-dependent completion, role/occurrence preservation, odd tails, two-level export sensitivity, repeated-role Jacobians, learned-gate Jacobians, saved ancestry through relays, source freezing, diagnostic parity and exact checkpoint/chunked continuation. Finite-difference tolerances are 2e-5.

## Controlled nested composition

The 4,200-byte repeated-sentence training test, including final drain, produced:

| Region | Mean explicit span (bytes) | Publications |
| --- | --- | --- |
| L2 | 2.27 | 1854 |
| L3 | 21.99 | 192 |
| S3 | 30.22 | 139 |
| F4 | 71.35 | 68 |
| G5 | 239.60 | 35 |

Spans are constituent coverage, including held side sources, not total model memory.

## Matched short training comparison

Two fresh seeds per kernel, 12 experts, 50,000 next-byte training targets from the pinned Tiny Shakespeare stream, identical learning-rate schedule and data. Reserved test: 8,192 targets starting at heldout offset 32,768; validation: the first 8,192 targets. Training-only timing used the same pinned CPU core after warmup. This is a short smoke comparison, not a continuation of the trained 100M model. Raw rows, hashes and timings are in results/completion_study.json.

| Kernel | Mean test NLL | Training bytes/s |
| --- | --- | --- |
| messages | 2.397376 | 2,635 |
| completion | 2.530954 | 1,597 |

The mechanism repair is verified, but this short comparison is not a quality improvement: mean held-out NLL increased by 0.133577 nats/byte and training throughput changed by -39.4%. L3/F4 expert-removal effects are also much smaller. The new event clock needs learning-rate and task-credit tuning before replacing the earlier quality baseline.

| Removed path | Messages delta NLL | Completion delta NLL |
| --- | --- | --- |
| L3 expert | +0.00651161 | -0.00008692 |
| F4 expert | +0.00597655 | +0.00003093 |
| G5 expert | -0.00003741 | +0.00000734 |
| S3 payload | +0.00039413 | +0.00000186 |

Positive intervention delta means removal worsened prediction. These are conditional effects.

Across each 50,000-target completion run, semantic handles lost, semantic work omissions and semantic capacity prunes were all zero; forecast expiry and duplicate scheduling were also zero. The bounded neural path still recorded 122 missed parent edges per run, with zero expired retained-input edges. Semantic credit and neural credit are reported separately.

## Use and reproduce

Select fresh_model(kernel="completion") from experiments/identity_semantics/runtime. The historical fresh_model() default remains composition. To migrate: python src/convert_to_completion.py OLD.npz NEW.npz. Run python -m unittest discover -s tests -v and python tests/run_completion_sanitizer.py. Reproduce measurements with python src/run_completion_study.py --data /path/to/restored/data. No trained models or corpus bytes are included.
