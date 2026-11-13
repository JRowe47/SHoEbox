# SHoEbox: regional contribution checks

13 November 2026 | Fourth step of the composition repair | 1,031,936 parameters

## Result

Contribution checks now distinguish local learned computation from content transport, relay context and retained inputs. They are supported per-model evaluation controls, replacing the need for global selectors or source-text substitutions. Production remains the relays kernel, with unchanged learning behavior and version-31 / ABI-5 checkpoints.

The two existing million-byte-trained models reproduce their published reserved-test losses: **1.951268** and **1.929333**. We completed **226 interventions**, each over all **78,771 test targets**, plus eight baseline comparisons against production. Neural parameters, semantic weights, dictionaries and source checkpoints remain unchanged.

The learned expert transforms in **L3** and **F4** contribute measurably: removing them increases mean NLL by **0.038686** and **0.023036** nats/byte. E4, V4 and G5 remain weak on this test despite substantial realized teaching and credit. This completes the measurement repair; it identifies where prediction still makes little use of the upper regions.

## What each intervention measures

| Control | Contract |
| --- | --- |
| Neural output | Zero only neural publications, after the producer's private memory update. Preserve singleton relays and content. This includes the input skip and EC recall in the removed output. |
| Expert transform | Remove only the learned matrix/bias term. Keep input skip, recall, routing and relay handling. |
| Content descriptor | Zero outgoing descriptor/code, including relays. Preserve exact IDs, recognition, timing, context and provenance. Parent ID anchors still carry identity information. |
| Relay context | Zero context only on singleton relay publications at the receiving region. Preserve code, descriptor and intrinsic structure. |
| Retained input | Suppress previous-input carry; retain the current input coefficient. Bottom learned-memory reads are separate. |
| Semantic residual | Remove the local learned residual when a symbol is encoded; retain the structural descriptor and inherited relay codes. |
| Whole source | Remove source inputs and direct A1/Y decoder features. This intentionally conflates transport with computation and can change event activity. |

Positive delta NLL means the intervention damaged prediction. Effects are conditional and non-additive. The current task measures use of the learned paths on held-out text; it is not a retraining comparison or a claim that every upper region now has a useful abstraction.

<!-- pagebreak -->

## Complete regional comparison

Mean delta NLL across seeds 1 and 2, under cold memory with causal admission. Neural and expert columns retain constituent descriptors and relay context. Very small effects are shown explicitly rather than ranked as improvements.

| Region | Neural output | Expert transform | Descriptor | Retained input |
| --- | ---: | ---: | ---: | ---: |
| A1 | +2.10606 | +0.73743 | +0.37086 | +0.24129 |
| B1 | +0.00010 | +0.00017 | +0.22285 | +1.41e-05 |
| L2 | +0.00019 | +0.00017 | +0.22250 | +2.56e-05 |
| S2 | +0.01337 | +0.01209 | +0.37491 | +0.00424 |
| R2 | +2.37e-05 | +8.88e-06 | +0.04913 | -1.29e-06 |
| F2 | +3.23e-06 | +1.59e-05 | +0.04844 | -6.10e-06 |
| L3 | +0.03142 | +0.03869 | +0.19170 | +0.00282 |
| S3 | +0.00031 | +8.87e-05 | +0.04025 | +5.32e-05 |
| R3 | +0.00044 | +0.00028 | +0.05244 | +6.65e-05 |
| F3 | +0.00053 | +0.00029 | +0.07561 | -1.45e-05 |
| E4 | +8.62e-06 | -1.86e-05 | +0.00049 | +1.04e-05 |
| F4 | +0.02062 | +0.02304 | +0.10212 | +0.00320 |
| V4 | -3.89e-06 | -2.01e-06 | -8.07e-07 | -2.34e-06 |
| M4 | -1.18e-05 | -4.38e-05 | 0 | -1.93e-06 |
| G5 | +8.35e-05 | +7.83e-06 | 0 | -1.06e-05 |
| EC-C | +0.03290 | +0.00845 | +0.06840 | +0.00067 |
| EC-X | +9.82e-05 | +6.07e-05 | 0 | +1.54e-05 |
| Y | +0.59992 | +0.32991 | 0 | +0.06702 |

The distinction changes interpretation. For L3, whole-source removal changes mean NLL by **+0.302074**, compared with **+0.031417** for its neural publication alone. Whole-source removal includes the fixed content transported through L3.

Descriptor effects at G5, EC-X and Y are zero because their outgoing descriptors have no immediate constituent consumers in this graph; their neural context is a different path. These zeros do not diagnose the usefulness of their local input codes. Parent descriptors deliberately use child structural descriptors, not child learned residuals.

The learned local symbol residual is most consequential at Y: removing it changes mean NLL by **+0.063721**. At L3 it changes NLL by **+0.001247**, and at F4 by **+0.000748**. The CSV and JSON include every semantic-residual and whole-source result.

L2 is the only region publishing ordinary singleton relays in this test. Muting only that relay context changes mean NLL by **+0.000884**. The other 34 seed/region relay cases are explicitly marked inactive, not reported as measured zero effects.

<!-- pagebreak -->

## Teaching, credit and event scale

Saved seed-1 training counters, read without additional training. Both seeds account for every forecast with zero expired producers or duplicate schedules. EC-C/EC-X use their memory-port teaching path and intentionally do not schedule this local relay teacher.

| Region | Neural executions | Local targets | Credit edges | Missing parents |
| --- | ---: | ---: | ---: | ---: |
| L2 | 81,897 | 20,474 | 1,523,852 | 0 |
| L3 | 999,998 | 249,999 | 13,244,515 | 0 |
| E4 | 999,997 | 249,999 | 3,999,977 | 0 |
| F4 | 999,997 | 249,999 | 5,999,968 | 0 |
| V4 | 999,998 | 249,999 | 999,994 | 0 |
| G5 | 999,996 | 249,998 | 5,999,956 | 0 |

Credit counts establish delivery, not benefit. E4/V4/G5 are no longer explained by the earlier canceled forecasts or near-total relay execution. Their weak expert-transform effects point to low task use of these learned responses. The next useful architectural investigation is how their learned messages reach prediction and what target makes them distinct from the stronger lower paths.

Completed publication spans in the reserved test, pooled over both seeds:

| Region | Mean span, bytes | Largest span, bytes |
| --- | ---: | ---: |
| L2 | 1.10 | 3 |
| L3 | 3.29 | 7 |
| E4 | 4.03 | 7 |
| F4 | 4.03 | 7 |
| G5 | 5.03 | 8 |
| Y | 4.29 | 8 |

These are explicit event spans, not bounds on recurrent memory or measures of conceptual abstraction. Fixed recipe names and nominal cadences alone cannot establish the realized event scale.

## Explicit memory evaluation

| Memory policy | Seed 1 NLL | Seed 2 NLL | Mean NLL |
| --- | ---: | ---: | ---: |
| cold_adaptive | 1.951268 | 1.929333 | 1.940301 |
| cold_frozen | 1.976604 | 1.954664 | 1.965634 |
| warm_frozen | 1.959043 | 1.936882 | 1.947963 |
| warm_adaptive | 1.948925 | 1.927259 | 1.938092 |

Cold starts with an empty associative store; warm retains the checkpoint store, its coordinate centers and timestamp clock. Frozen blocks store writes; adaptive admits causal associations from the observed test prefix. Every mode starts fresh recurrent/event state and freezes neural parameters and dictionaries. Query normalization still evolves during inference. The historical frozen_copy behavior remains cold_adaptive. EC-C/Y neural interventions are also evaluated under every alternative memory policy in the JSON/CSV.

<!-- pagebreak -->

## Verification and performance

All **115 package tests** pass, including 15 contribution tests. Across the repository and final package checks, **182 unit tests** pass, plus 12/64-expert packaging checks. ASan/UBSan pass for both the ordinary relay runtime and every diagnostic intervention/region. Tests cover independent selectors, invalid modes, unsavable diagnostic models, channel isolation, exact chunking, source preservation, explicit memory lifecycles, document draining and complete disabled-control parity.

The diagnostic hashes every publication's identity, span, source roles, occurrence, intrinsic structure, recognition and producer provenance. Every non-whole-source intervention preserves that full trace and execution counts in all measured runs. Frozen store fingerprints are unchanged; fitted neural, semantic and dictionary states are unchanged for every intervention.

Six alternating same-core trials per condition, after other workers finished. Each uses 1,024 warmup and 16,384 timed targets. Comparison source is the unchanged preceding commit f9eeb8a. Compilation, loading, snapshots and checkpoint I/O are excluded.

| Execution | Before, bytes/s | After, bytes/s | Change |
| --- | ---: | ---: | ---: |
| Production training | 4,967 | 4,792 | -3.5% |
| Production evaluation | 14,098 | 14,279 | +1.3% |
| Diagnostic evaluation vs production | 14,279 | 12,923 | -9.5% |

All timed production losses and disabled-diagnostic losses match exactly within their respective training/evaluation modes. Diagnostic tracing adds evaluation work; production compiles out its controls and counters. No neural parameters or serialized recurrent-state fields were added.

## Reproduce

Use the two relay-repair checkpoints, or regenerate them with the preceding study. The corpus split and checkpoint SHA-256 values are recorded in results/contribution_checks.json. No test models or intermediate run snapshots accompany this publication.

```bash
python -m unittest discover -s tests -v
python tests/run_contribution_sanitizer.py
python tests/run_relays_sanitizer.py
python src/run_contribution_checks.py --workers 6
python src/contribution_credit_audit.py
python src/contribution_performance.py \
  --baseline-runtime /path/to/f9eeb8a/runtime
python src/report_contributions.py
```

Source base: https://github.com/JRowe47/SHoEbox/commit/f9eeb8a2c2fe6eb31a4e36954f2705904a893ecf
