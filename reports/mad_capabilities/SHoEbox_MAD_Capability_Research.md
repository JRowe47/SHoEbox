# SHoEbox MAD capability research

15 November 2026

## Scope and completed work

Completed **435 fits and 131,250,000 observed training bytes**. The campaign calibrates nine task families, reruns selective loops on six task families with three paired seeds and equal optimizer searches, and repeats the earlier intermediate-supervision question at 1M and 5M parameters.

The runtime is corrected C at commit `6384d8ca072075ec58010b4c3d5964e9e60fcd3e`. The interrupted interactive run yielded no recovered results and is excluded. All measurements below come from completed GitHub runner jobs. Production defaults are unchanged.

## Initial capability profile

These are development calibration scores at 500K parameters and 250K input bytes. The alphabet contains eight values, so uniform token chance is 12.5%.

| Task | Development accuracy |
|---|---:|
| composition | 12.76% |
| compression | 94.47% |
| fuzzy_recall | 11.24% |
| memorization | 100.00% |
| noisy_recall | 13.67% |
| pointer_chasing | 19.27% |
| recall | 11.91% |
| selective_copy | 23.89% |
| variable_tracking | 24.22% |

## Rerun of selective loops

Extra passes execute only at completed L2/L3 events. Parameter count and persistent state capacity match across depths. Each depth gets the same six-policy grid: LR multiplier 0.5/1/2 by native auxiliary weight 0/0.1. Every fit uses 250K observed input bytes; the optimizer policy is selected by mean final development NLL across seeds 1, 7 and 19.

Difficulty is selected using corrected-C development scores before the paired rerun. The table reports held-out answer-token accuracy, averaged over seeds. The last column is a simple task-specific shortcut evaluated on the same test cases; it is not a fitted SHoEbox model.

| Task / selected setting | 0 extra | 2 extra | 4 extra | Simple shortcut |
|---|---:|---:|---:|---:|
| composition / one_op | 28.91% | 28.65% | 28.12% | 75.78% |
| compression / vocabulary | 70.77% | 67.06% | 58.33% | 34.38% |
| pointer_chasing / one_hop | 33.85% | 31.51% | 34.11% | 33.33% |
| recall / easy | 55.86% | 57.03% | 60.81% | 53.52% |
| selective_copy / easy | 87.76% | 88.02% | 86.33% | 54.30% |
| variable_tracking / two_ops | 29.95% | 27.60% | 28.65% | 64.84% |

Shortcut definitions: recall/copy/compression use the most common supplied value while ignoring query or position; variable tracking/composition retain the initial queried binding and ignore updates; pointer chasing guesses uniformly among supplied non-start nodes. This last score is an expected accuracy, not a deterministic solver.

### Paired seed differences

Positive values favor loops. Intervals bootstrap the three paired training-seed differences (10,000 resamples).

| Task | Extra passes | Accuracy change, percentage points | 95% seed interval |
|---|---:|---:|---|
| composition | 2 | -0.26 | [-1.56, +0.78] |
| composition | 4 | -0.78 | [-2.34, +0.00] |
| compression | 2 | -3.71 | [-5.66, -0.20] |
| compression | 4 | -12.43 | [-14.84, -9.96] |
| pointer_chasing | 2 | -2.34 | [-7.03, +0.78] |
| pointer_chasing | 4 | +0.26 | [-2.34, +3.91] |
| recall | 2 | +1.17 | [-4.69, +4.30] |
| recall | 4 | +4.95 | [+0.78, +8.20] |
| selective_copy | 2 | +0.26 | [+0.00, +0.39] |
| selective_copy | 4 | -1.43 | [-3.12, +0.78] |
| variable_tracking | 2 | -2.34 | [-3.12, -0.78] |
| variable_tracking | 4 | -1.30 | [-5.47, +1.56] |

### Frozen mechanism checks

Drops in answer accuracy after removing addressed/retained reads or cycling active expert transformations. Positive means the original mechanism helped this fitted control.

| Task, corrected C | Remove memory reads | Permute L2 | Permute L3 |
|---|---:|---:|---:|
| composition | +17.97 pp | +0.00 pp | +0.00 pp |
| compression | +0.00 pp | +2.54 pp | +0.00 pp |
| pointer_chasing | +21.61 pp | +2.60 pp | +0.00 pp |
| recall | +40.10 pp | -1.69 pp | -1.04 pp |
| selective_copy | +75.39 pp | +0.13 pp | +0.13 pp |
| variable_tracking | +11.20 pp | +3.39 pp | +0.52 pp |

Raw evidence also contains frozen 0/1/2/4-depth overrides, longer-delay tests, exact-answer rollout scores, native 256-byte scores, active permutation counts and compression zero-bottleneck tests.

## Rerun of intermediate supervision

This extension retains the earlier verified 64-byte program-record encoding and identical four-register linear head. It applies MAD-style difficulty calibration, finite matched training sets, three seeds, structural train/dev/test separation, measured-time comparisons and frozen interventions. It is separate from the canonical nine byte tasks. Each fit uses 1M bytes with no language mixing; the 4,128 head parameters are now deducted from the total budget.

Final-only, extra-head-label and direct-state-credit arms share one fixed learning policy and identical final queried-register supervision. The intermediate head-only arm isolates extra head training; state_aux adds credit into L2/L3. Difficulty is selected from 1/2/4 operations using the final-only development control. If no difficulty calibrates, the one-operation result remains diagnostic.

| Size | Arm | Held-out accuracy | 8/12-op accuracy | Frozen standardized readout |
|---|---|---:|---:|---:|
| 1,000,000 | final_only | 12.11% | 13.41% | 28.12% |
| 1,000,000 | head_only | 10.42% | 13.93% | 29.56% |
| 1,000,000 | state_aux | 14.71% | 13.41% | 30.73% |
| 5,000,000 | final_only | 11.85% | 11.20% | 32.68% |
| 5,000,000 | head_only | 13.15% | 11.98% | 30.86% |
| 5,000,000 | state_aux | 15.36% | 12.50% | 32.81% |

The standardized probe is trained only on separate training-fold examples. It changes both feature conditioning and the fitting method, so its advantage does not isolate centering alone. The task head receives the queried register index; native byte answers are stored separately in the raw results.

## Interpretation and next decision

**Prioritize query-dependent binding, state readout and retention. The present shared-expert refinement loop does not earn promotion as a general capability improvement.**

On calibrated compression, held-out accuracy declines from 70.77% without loops to 67.06% with two extra passes and 58.33% with four. All three seeds worsen at both depths. On the four-pass-trained compression checkpoints, disabling extra inference passes raises mean accuracy from 58.33% to 63.87%; even that remains below the no-loop-trained control.

Two-binding recall is the positive exception: four-pass training improves held-out token accuracy from 55.86% to 60.81%, a +4.95-point paired mean, with seed interval [+0.78, +8.20]. Evaluating those same four-pass-trained models with no extra passes still scores 60.42%. Thus almost all of the improvement persists without the repeated inference computation; this supports a training effect, not useful additional reasoning at inference.

Whole-episode recall remains weak: the four-pass model answers both queries correctly on 30.99% of episodes, while a query-blind control that simply emits the supplied values in their original order gets 53.12%. The token-accuracy improvement therefore does not establish reliable binding of each value to its query.

Selective copy stays essentially unchanged (87.76% control, 88.02% with two extra passes, 86.33% with four). Composition and pointer chasing show no consistent gain; two-pass variable tracking worsens. The loop policies typically cost a few percent of training throughput.

Retention is a concrete failure: on independently generated 64-byte-delay cases, the no-loop selective-copy control falls from 87.76% to 14.84%, and two-binding recall falls from 55.86% to 18.88%. This measures generalization to longer delays, rather than the ceiling after dedicated delay training. Removing addressed and retained reads also collapses copying, so the model uses these memories but does not retain the task robustly under this distribution shift.

Direct intermediate state credit improves all three paired seeds over final-only supervision at both sizes: +2.60 points at 1M and +3.52 at 5M. The resulting mean accuracies are only 14.71% and 15.36%, versus 71.48% for retaining the initial queried binding and ignoring the update. The positive shift is real in these runs; the procedure is still unsolved. None of the 1/2/4-operation state-control settings entered the registered calibration band.

The frozen standardized readouts recover approximately 31–33% from the direct-state models. There is recoverable information that the online head is failing to use, and additional information is still missing. More generic loop depth is unlikely to resolve either problem by itself.

The shortcut controls also improve the development protocol: being above uniform chance is insufficient to establish the intended skill. Native variable tracking scores 29.95% against a 64.84% ignore-updates baseline; one-operation composition scores 28.91% against 75.78%. One-hop pointer accuracy is approximately the 33.33% node-set guessing baseline. Future calibration should record these controls on development data before any capability promotion.

No natural-language guardrail was run, and no architecture was promoted.

The next experiment should target the weakest measured operation that is still below its simple shortcut baseline: preserve separate key/value bindings, make retrieval depend on the query, and give upper computation a useful state-update target. Retain a readout-only control when testing additional working slots.

## Verification, costs and reproduction

Existing MAD tests passed on each task runner. The structured-state adapter passed its Python checks and its 19 native checks under ASan/UBSan. Selected checkpoint reloads reproduce development predictions exactly. Each paired run checks dataset hashes, input exposure and fitted/state capacity. Generator labels are verified by independent interpreters.

Training time includes context resets, learning and retirement; excludes compilation, initialization, validation and checkpoint I/O. Matched-time comparisons use the last actually measured development checkpoint within the common CPU budget, with byte counts recorded. These finite training horizons are not saturation-floor estimates.

- Base protocol: https://github.com/JRowe47/SHoEbox/blob/6384d8ca072075ec58010b4c3d5964e9e60fcd3e/experiments/mad/README.md
- Calibration: https://github.com/JRowe47/SHoEbox/actions/runs/43787250957
- Loop and state reruns: https://github.com/JRowe47/SHoEbox/actions/runs/43787654696
- Procedural reruns: https://github.com/JRowe47/SHoEbox/actions/runs/43787902832
- MAD research: https://proceedings.mlr.press/v235/poli24a.html

Reproduction uses the registered scripts under experiments/mad_capabilities and experiments/mad_followup on the research/mad-capabilities branch. The source patch is included with the raw JSON evidence. Datasets, trained weights and compiled binaries are excluded.
