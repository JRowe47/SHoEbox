# SHoEbox: intermediate state supervision — Stage A

## Result

**Direct intermediate supervision did not establish a reliable capability gain in this screen.** At 1M, audit accuracy was 13.67% for final-only, 14.06% for extra head labels, and 14.84% for direct state credit. At 5M it was 10.94%, 11.13%, and 12.89%. The 5M comparison with head-only favors state credit (+1.76 percentage points; paired-example interval +0.39 to +3.12). The other three intervals include zero, including both comparisons with final-only. Longer programs also remained near chance.

The useful result is diagnostic: **some binding information exists in L2/L3, but the online head uses little of it**. Standardized frozen probes recover 16.80–24.61% accuracy, while replacing changing L2/L3 features by their training mean alters the online answer NLL by only −0.00196 to +0.00055. This motivates a readout/retention experiment before learned halting. Corrected C remains the selected runtime.

## Matched experiment

Six fresh models: three supervision arms at nominal 1M and 5M, seed 7, **2,000,000 real byte targets per model** (12M total). Each arm has exactly the same parameter count: 1,003,743 at 1M and 5,003,871 at 5M. These counts include an identical 4,128-parameter task head in every arm. No extra thought passes are used in Stage A.

1. **Final answer:** byte language loss plus one final queried-register target.
2. **Extra head labels:** also fit all four register values after initialization and each update, with intermediate gradients stopped at the readout.
3. **Extra state credit:** the same labels and head updates, with their gradient entering L2/L3 states and the existing recurrence/composition learning path.

The auxiliary weight is 0.25, normalized across four register labels and across supervised events per native retirement block. The final-answer channel has weight 1. The shared LR anchor falls linearly from 1 to 0.1 over the real byte budget; route oscillation is disabled. Fast/query rates start at 0.008, slow 0.08, native readout 0.16, task readout 0.2. Neither labels nor validation advance the byte schedule.

Training contains 1,601,280 verified-task bytes and 398,720 Tiny Shakespeare bytes in every arm. Padding is counted. The training episode digest is `6e5a830497fb95bf13d80cb4eaf92569d39ae69c56b946be49ee821a728fc6b0`. The language corpus SHA-256 is `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; the last 10% of bytes is held out. The exact configuration, curves and per-example correctness are attached in JSON.

## Capability results

| Size | Supervision | Audit accuracy | Long (8/12 ops) | Native answer | Language NLL | Train kB/CPU s |
|---|---|---:|---:|---:|---:|---:|
| 1M | Final answer | 13.67% | 11.72% | 10.74% | 2.1792 | 32.26 |
| 1M | Extra head labels | 14.06% | 11.13% | 13.67% | 2.1794 | 32.56 |
| 1M | Extra state credit | 14.84% | 12.11% | 13.87% | 2.1807 | 34.02 |
| 5M | Final answer | 10.94% | 12.11% | 13.28% | 2.2000 | 33.89 |
| 5M | Extra head labels | 11.13% | 11.72% | 14.65% | 2.2201 | 32.85 |
| 5M | Extra state credit | 12.89% | 12.70% | 14.26% | 2.1997 | 32.76 |

Chance is 12.5%; uniform-answer NLL is 2.079442. A symbolic heuristic that merely reads the queried register's **initial** binding and ignores every update gets 49.22% on this audit. It is a memory-task reference, not a trained model with a comparable resource budget.

The main score is a four-group, eight-class task readout over L2/L3. The evaluator supplies the queried register index. The **native answer** column instead scores the actual next-byte generator over all 256 bytes; no register index is supplied to that decoder. The experiment therefore separates state-readout accuracy from byte generation.

- 1M state credit versus final answer: +1.17 percentage points; paired-example 95% interval [-2.15, +4.49].
- 1M state credit versus extra head labels: +0.78 percentage points; paired-example 95% interval [-2.34, +3.91].
- 5M state credit versus final answer: +1.95 percentage points; paired-example 95% interval [-1.37, +5.27].
- 5M state credit versus extra head labels: +1.76 percentage points; paired-example 95% interval [+0.39, +3.12].

Intervals resample paired audit examples for these fixed trained models; they do not measure training-seed variation. This is the planned screening stage. No configuration is promoted from it.

## What the upper states contain

A separate frozen linear diagnostic is fitted on 1,024 independently generated training-fold programs. It standardizes each feature using that training set, fits ridge regression with fixed alpha 1, and evaluates on the same untouched structural audit. No SHoEbox weights change. The probe receives the same queried-register index as the online head.

| Size | Arm | Online after query | Frozen probe before query | Frozen probe after query |
|---|---|---:|---:|---:|
| 1M | Final answer | 13.67% | 22.66% | 21.09% |
| 1M | Extra head labels | 14.06% | 24.61% | 24.61% |
| 1M | Extra state credit | 14.84% | 23.83% | 24.61% |
| 5M | Final answer | 10.94% | 20.70% | 16.80% |
| 5M | Extra head labels | 11.13% | 21.68% | 18.75% |
| 5M | Extra state credit | 12.89% | 22.46% | 22.07% |

At 1M, both extra-head and direct-state training reach 24.61% with the frozen probe; direct state credit provides no additional gain there. At 5M, direct state credit reaches 22.07%, versus 18.75% for head-only and 16.80% for final-only. These are useful diagnostic differences, not a working procedural solver: the initial-binding heuristic still gets 49.22%.

For the direct-state models, the shared feature mean accounts for 89.41% (1M) and 89.89% (5M) of post-query feature energy. The standardized probe's advantage identifies readout conditioning as a concrete follow-up. It does not establish that centering alone will repair the online learner. State information also remains incomplete: the 1M direct-state probe gets 25.78% on tracking and 23.44% on composition; the 5M version gets 28.52% and 15.63%.

The mean-replacement intervention uses a separate training-fold mean of final L2/L3 features. The expert intervention cycles transformations among experts actually selected during training, keeps router keys fixed, and is averaged over three permutations. The chart shows the resulting answer-NLL changes. Frozen probe accuracy measures recoverable information; these interventions measure the fitted online system's dependency on it.

## Resource comparison

| Size | Arm | Matched CPU seconds | Estimated dev NLL at that time | Inference kB/CPU s |
|---|---|---:|---:|---:|
| 1M | Final answer | 58.79 | 2.0945 | 49.98 |
| 1M | Extra head labels | 58.79 | 2.0854 | 51.48 |
| 1M | Extra state credit | 58.79 | 2.0933 | 52.96 |
| 5M | Final answer | 59.01 | 2.1048 | 51.83 |
| 5M | Extra head labels | 59.01 | 2.0960 | 50.62 |
| 5M | Extra state credit | 59.01 | 2.0930 | 52.33 |

Equal-time values interpolate the saved development curves at the fastest arm's completed training time for that model size. They are curve estimates, not additional checkpoints. Training CPU timers include curriculum generation, context resets, labels and retirement, and exclude compilation, validation and checkpoint I/O. Frozen inference timers include clone/reset and answer scoring. Runs are serial and pinned to CPU 0.

The task eligibility tape is bounded at **8,224 bytes**. It adds no production-model parameters beyond the identical task readout. No inference iterations, repeated byte admissions or extra clock ticks occur.

## Verification and reproducibility

- Six Python tests passed, including exact unlabelled equivalence to public corrected C, detached-head control, independent interpreter validation, causal label timing, chunking, epoch retirement and live-checkpoint continuation.
- Nineteen native checks passed, including finite differences through L2/L3 recurrent parameters and exact checkpoint continuation.
- The same native checks passed AddressSanitizer and UndefinedBehaviorSanitizer; host LeakSanitizer was disabled.
- Every trained checkpoint was reloaded and reproduced its development predictions and NLL exactly.
- The production runtime and its checkpoint ABI were unchanged. The research adapter aborts if either of its two source anchors changes.

Source: branch `codex/structured-state-supervision`, based on `600352ce16e89702de82e7b3f61c9b696c34e6f6`. Run commands are in `experiments/structured_state/README.md`. The source/evidence ZIP excludes datasets, trained weights and compiled binaries.

## Research connection and next decision

Keep this supervised task harness as the capability test. The next focused comparison should give the readout centered inputs and give the model four persistent binding slots with explicit retain/overwrite behavior. Include a centered-readout-only control so any gain can be assigned to storage versus readout conditioning. Once those states retain and expose useful bindings, run the fixed-depth K=1/2/4 comparison before learning a stop policy.

The head-only control remains necessary: changing its weights can change later final-answer gradients indirectly, even though intermediate gradients are detached. The direct-state arm measures the additional effect of sending intermediate credit into the upper states.

This follows Stage A of the supplied capability plan: establish useful upper information before expanding recurrent depth. The design is inspired by intermediate supervision in [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) and the broader use of structure-oriented auxiliary objectives in [Token Order Prediction](https://proceedings.mlr.press/v306/zuhri26a.html). This experiment implements verified register-state supervision; it does not reproduce either paper's architecture or loss.
