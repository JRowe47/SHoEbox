# SHoEbox: teaching and recognition through relays

13 November 2026 | Third repair from the principled code audit

## Result

Singleton relays now preserve outstanding producer teaching and inherited recognition. The repaired `relays` kernel builds on the published identity/vector runtime; it adds no neural parameters. The 1M configuration remains **1,031,936 parameters**.

All **100 package checks** pass, including the prior 62 checks and the new relay suite. ASan/UBSan and the repository check command pass. Two fresh seeds per arm each train on one million byte targets. Mean reserved-test NLL changes **1.946529 to 1.940301**, a signed change of **-0.006228 nats/byte** (-0.32%). Lower is better.

Same-core median training throughput changes **5,243 to 5,142 bytes/s** (-1.9%).

## The repaired contract

| Object | Behavior |
| --- | --- |
| Pending forecast | Saves the prediction, observed producer input, and the producer's eligibility tick/index. The saved record supplies activation, expert, gate, routing cue and semantic features. |
| Next target | The next observed regional content input teaches exactly once. A completed singleton relay qualifies; idle ticks do not. Y retains its per-byte normalized input and observed prefix-byte cue. |
| Transparent relay | Preserves content ID, descriptor, semantic code, intrinsic arity/kind, children, recognition, confidence and actual producer provenance. It does not execute or schedule a local expert. |
| Local metadata | `local_arity` and `relayed` describe the receiver's action independently of the inherited unit. Relay transport does not increment locally formed-compound counts. |
| Proposed relation | A new relation between familiar units remains provisional until its own dictionary supplies recognition evidence. |

The target is evaluated before replacing the public output. Predictor, expert, router and semantic updates use the saved producer. A nonlearning or frozen target retires the forecast without modifying weights. A missing or overwritten producer is counted and rejected. Document resets clear pending forecasts; chunk boundaries do not. Save/load preserves a pending forecast exactly.

In the current multi-source graph, ordinary singleton traffic occurs at L2. Controlled tests additionally carry recognized and provisional compounds through two transparent hops and verify that inherited familiar units can still form a new provisional relation.

## Verification

Checks cover every declared binding role, identity/vector distinctions, saved-feature gradients, changed public outputs, numerical predictor updates, once-only teaching, multi-hop recognition and provenance, frozen/zero-rate policies, stale-producer rejection, exact checkpoint/chunk continuation, reset behavior and explicit fitted-state conversion. The live-stream and million-byte runs account for every forecast, with **zero expired producers and zero duplicate schedules**.

<!-- pagebreak -->

## Matched language experiment

Width 64, 12 experts per region, unchanged fixed transports, the same Tiny Shakespeare bytes and identical seed-specific initial parameters. Validation-only pilots test relay peak rates 0.02, 0.04 and 0.08 on seed 3 for 100,000 targets. The selected rate is 0.08, matching the prior runtime. Both arms use cosine decay to 0.008 by one million targets, local rate one third of main, and prototype rate 0.002.

The 32,768-target validation segment selects the rate; the disjoint **78,771-target test segment** supplies the endpoint. Evaluation starts with fresh stream/episodic state, frozen neural parameters and dictionaries, and causal memory admission from the observed prefix.

| Seed | Previous semantics | Repaired relays | Change |
| --- | ---: | ---: | ---: |
| 1 | 1.951143 | 1.951268 | +0.000125 |
| 2 | 1.941915 | 1.929333 | -0.012582 |
| Mean | 1.946529 | 1.940301 | -0.006228 |

L2 forecast accounting after training:

| Seed | Scheduled | Neural targets | Relay targets | Pending |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20,474 | 1,458 | 19,016 | 0 |
| 2 | 20,474 | 1,458 | 19,016 | 0 |

## Throughput and state

Six alternating trials per arm use one pinned CPU core, 1,024 warmup targets and 16,384 timed targets after other workers exit. Timing includes forward execution, routing, dictionaries, episodic admission and learning; it excludes compilation, evaluation and checkpoint I/O.

Composition state grows from **3,317,000 to 3,328,808 bytes**, an increase of **11,808 bytes**. No model-size increase accompanies the repair.

## Run and continue

Use `fresh_model(kernel="relays")`. The historical default remains `composition`; `semantics` remains the unchanged comparison kernel. New checkpoints use **version 31 / composition ABI 5** and resume exactly with `Model.load()`.

To start from version-30 learned semantics, `convert_to_relays.py` preserves neural/semantic weights, dictionaries, fixed transports, policies and fitted preprocessing means while resetting stream context, pending forecasts, eligibility, online adaptation state and episodic associations. This is a fresh-stream conversion, not exact continuation of the old event ABI.

```bash
python -m unittest discover -s tests -v
python tests/run_relays_sanitizer.py
python src/convert_to_relays.py OLD.npz NEW.npz
python src/run_relays_study.py pilots
python src/run_relays_study.py full
python src/relays_performance.py
```

Source, tests and compact measurements accompany this repair. Test models and intermediate run artifacts are not published. The next agreed step is regional neural contribution checks with content and relay transport preserved.
