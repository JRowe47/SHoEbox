# SHoEbox: query-conditioned lace separation

November 7, 2026 (Mountain Time) | Completed four-arm pilot

No arm passes promotion. Keep the retained strong parent as the working model. All four arms complete the locked 32-update continuation from the same bounded-CA1 endpoint. The new mechanism is implemented behind default-off switches, with rank-four query-conditioned bias, helpful-unique reinforcement and selective inhibition of shared routes receiving opposing task credit.

## Fixed endpoint results

Both questions must be correct about the same unchanged history. The fresh final bank has sixteen templates, a full 4x4 value crossing, both source orders and 192 distinct-answer histories per delay. Each cell below is a count, not a switching rate.

| Model | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Retained parent | 0/192 | 0/192 | 0/192 |
| A: bypass | 0/192 | 3/192 | 0/192 |
| B: learned head | 1/192 | 1/192 | 6/192 |
| C: unique credit | 0/192 | 1/192 | 1/192 |
| D: full proposal | 2/192 | 1/192 | 0/192 |

| Model | Single +64 | Correction +64 | Text NLL |
| --- | --- | --- | --- |
| Retained parent | 96.9% | 100.0% | 3.132340 |
| A: bypass | 90.6% | 74.2% | 2.931831 |
| B: learned head | 96.9% | 88.3% | 2.943423 |
| C: unique credit | 79.7% | 69.5% | 2.938370 |
| D: full proposal | 95.3% | 84.4% | 2.936322 |

Retention uses 64 single-fact and 128 correction answers. Text NLL uses 1,024 natural-byte targets. These established cohorts remain unchanged from the preceding CA1 pass.

The head alone is the strongest continuation to inspect next. B improves +64 native paired recall from A's 0/192 to 6/192 and correction accuracy from 74.2% to 88.3%, while matching the retained parent's 96.9% single-fact accuracy. It still fails both paired and correction promotion gates. Unique-only credit C reduces +64 native pairs to 1/192 and lowers retention to 79.7% single-fact / 69.5% correction. Adding overlap inhibition in D recovers retention to 95.3% / 84.4%, but gives 0/192 paired successes at +64. The extra rules do not improve the primary result over the simpler head.

D's local rule is active: the final +32 diagnostic updates 141/168 eligible controller/round pairs, with distinct head inputs and unsaturated bias. Swapping its dedicated cues changes target-versus-contender margins but leaves all 1,536 native answer predictions unchanged. Blocking readbacks also leaves the +64 paired count at zero. Keep the experimental controls available, favor the head-only arm for further diagnosis, and trace where question-specific binding is lost between owner readbacks, expert inputs and answer readout before changing the credit strengths.

## Validation and promotion

| Model | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Retained parent | 0/96 | 0/96 | 0/96 |
| A: bypass | 0/96 | 0/96 | 0/96 |
| B: learned head | 0/96 | 0/96 | 4/96 |
| C: unique credit | 0/96 | 0/96 | 0/96 |
| D: full proposal | 2/96 | 0/96 | 0/96 |

| Arm | Paired | Single | Correction | Language |
| --- | --- | --- | --- | --- |
| A | Fail | Fail | Fail | Pass |
| B | Fail | Pass | Fail | Pass |
| C | Fail | Pass | Fail | Pass |
| D | Fail | Pass | Fail | Pass |

The gates require at least 50% both-correct validation pairs at +0 and +32, no more than two percentage points of single-fact loss, three points of correction loss, or +0.05 natural-text NLL against the retained parent on matching cohorts. Validation retention uses 32 single-fact and 128 correction answers at each of +0/+32; text uses 256 targets.

Initial and final validation are preserved. No intermediate checkpoint was selected. The final test stayed sealed until all four endpoints and source ledgers were complete.

The native short grammar can repeat exact inputs despite fresh seeds and indices. New paired cohorts therefore add 24 held-out natural-text bytes before both writes. Retention gaps remain unchanged. Diagnostic seed/start is 2309/180000, validation 2411/200000 and final 2539/240000. Validation has eight templates and 96 distinct-answer histories per delay; final has sixteen and 192. All models, including the retained parent, see the same locked inputs.

The byte-hash audit compares against 121,312 distinct rendered prior answers from reconstructed evaluation/training banks. It finds zero historical and cross-cohort exact-input intersections. Without the natural lead-ins, 832 new native answers would repeat old inputs. Semantic fact combinations are intentionally reused and balanced. The historical audit correction changed no rendered evaluation input.

## Mechanism and ablations

A versus B measures the conditional head; B versus C measures unique credit; C versus D measures contentious-overlap inhibition. The tables report final weights at update 32 in every arm.

Each node concatenates its existing local lace context with its dedicated regional cue. The bounded bias enters fresh scores before top-k. No question IDs, answer IDs, owner annotations or source positions enter this head. Ordinary gradients still train every upstream component. Extra credit uses the final selected-score gradient from the normal task backward and changes only the small head.

| Initial diagnostic | Selected overlap | Input RMS difference | Updated / eligible |
| --- | --- | --- | --- |
| +0 bytes | 95.4% | 0.04387 | 146/168 |
| +32 bytes | 93.1% | 0.07293 | 141/168 |

Inputs differ and opposing credit is available even with high hard-route overlap. The diagnostic nonzero head also changes answer logits when valid paired cues are swapped. Thus the preflight verifies a live causal path rather than relying on overlap alone.

| Arm, training mean | Selected overlap | Soft L1 | Updated / eligible | Bias RMS |
| --- | --- | --- | --- | --- |
| A | 94.2% | 0.1186 | 0/2688 | 0.0000 |
| B | 94.7% | 0.1046 | 0/2688 | 0.0159 |
| C | 94.7% | 0.1077 | 351/2688 | 0.0164 |
| D | 94.6% | 0.1075 | 2241/2688 | 0.0182 |

Coverage counts aligned controller/round-pair decisions; each includes three lace layers. Same-target decisions receive zero extra correction in every arm. The added gradient never exceeds 25% of the ordinary head-gradient norm. Captures skip inactive rows and no-grad checkpoint forwards, and clear after use. Final diagnostic records separately preserve shared/unique contribution mass, opposing-credit counts, input separation, saturation and gradient norms.

| Final +32 diagnostic | Selected overlap | Unique mass | Updated / eligible | Extra / ordinary |
| --- | --- | --- | --- | --- |
| B | 90.2% | 9.4% | 0/168 | 0.000% |
| C | 97.4% | 2.3% | 15/168 | 0.010% |
| D | 92.5% | 7.0% | 141/168 | 2.235% |

## Fixed-weight controls

| D, distinct targets | Older correct | Newer correct | Query switches |
| --- | --- | --- | --- |
| +0 bytes | 14.1% | 87.0% | 2/192 |
| +32 bytes | 20.3% | 79.7% | 2/192 |
| +64 bytes | 25.5% | 71.4% | 2/192 |

Switching counts are separate from correctness. Equal-answer controls and all per-case predictions are retained in the machine-readable evidence.

| Same D weights | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Intact | 2/192 | 1/192 | 0/192 |
| Bias bypassed | 1/192 | 2/192 | 0/192 |
| Paired cues swapped | 2/192 | 1/192 | 0/192 |
| Readbacks blocked | 5/192 | 1/192 | 0/192 |

The cue swap changes only the head cue input between questions sharing a valid history. C versus D is the training-time inhibition comparison.

| Unseen wording | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Retained parent | 0/192 | 0/192 | 0/192 |
| A: bypass | 1/192 | 1/192 | 0/192 |
| B: learned head | 0/192 | 1/192 | 1/192 |
| C: unique credit | 0/192 | 2/192 | 4/192 |
| D: full proposal | 2/192 | 0/192 | 1/192 |

The separate predeclared wording changes `A x?` to `Value of A x?`; it is never used for training or endpoint selection. There are 192 distinct-answer histories and 64 equal-answer histories per delay in this bank as well.

## Cost and recovery

Each arm consumes 768 answer targets, 23,680 logical task bytes, 20,032 executed batch-byte positions and 4,096 natural-text targets. Executed positions include padded admissions; actual expert execution is counted separately. Source documents advance from 800 through 855, and the full per-update modes, gaps, targets and budgets match across arms.

| Arm | Update time (min) | Peak RSS (MiB) | Bookkeeping (ms/pair update) | Task experts |
| --- | --- | --- | --- | --- |
| A | 17.40 | 1668 | 15.71 | 2,913,792 |
| B | 16.21 | 1590 | 17.61 | 2,913,792 |
| C | 16.62 | 1645 | 16.21 | 2,913,792 |
| D | 16.85 | 1638 | 16.74 | 2,913,792 |

Bookkeeping averages paired updates and includes diagnostic measurement in A/B, whose extra credit is zero.

Installed models have 394,310 unique parameters, including 3,136 new head parameters (448 per node). Arm A installs but bypasses the same heads. Top-k and expert counts remain fixed; changed contribution mass is functional redistribution, not fewer expert executions.

The matched initial timing probe records 26.730 s for bypass, 26.114 s for the head alone, 27.042 s with capture and 27.102 s with capture for the credit arm. Local credit application adds 9.72 ms. Wall times are per process during concurrent CPU runs. Full training time includes language and task forward/backward, clipping and AdamW; evaluation and packaging are additional. Detailed inference timings and memory reads/refinements are recorded for each identical cohort.

| Final +32 inference | Bank time (s) | Posting reads | Readbacks | Refinements |
| --- | --- | --- | --- | --- |
| A | 325.2 | 3,039,801 | 231,936 | 231,936 |
| B | 274.2 | 3,779,724 | 231,936 | 231,936 |
| C | 293.6 | 3,234,129 | 231,936 | 231,936 |
| D | 284.0 | 3,087,642 | 231,936 | 231,936 |

Each inference bank contains 256 histories and both questions, including equal-answer controls; times include shared-history execution on concurrent CPU workers. These are fixed workload wall times. Raw records also preserve expert counts, executed positions, directory/candidate work and peak memory.

## Reproduction and checks

The 52 focused checks pass, with a subsequent ten-check rerun after integration. They cover zero/disabled parity, owner/layer/pair alignment, credit signs, useful-to-both and same-target protection, analytic derivative agreement, inactive rows, runtime/reference equivalence, recomputation and exact resume. A separate production-checkpoint audit compares a complete bypassed update with the unchanged continuation, including inherited optimizer moments and carried language state.

All runs preserve LR .003, AdamW (.9,.95), weight decay .01, language weight .25, syntax weight .1 and global clip 1. They use the same 16-byte recomputation segments above 512 positions. The extra-credit path uses unscaled FP32. There is no second adversarial model, replay store, persistent inhibition trace, altered capsule or global optimizer change.

The source archive includes implementation, tests, protocols, measured summaries and full prediction files. The checkpoint archive includes the strong parent, experimental starting endpoint, initial/8/16/24/32 snapshots, final weights, AdamW/RNG/carried stream state, exact corpus, cohort manifests, timing and checksums. Source and protocol guards remain enforced.

Resume the identical arm with `python scripts/run_lace_separation.py --arm D --out artifacts/lace_separation/D --initial artifacts/lace_separation/initial.json --resume`. For a fresh reproduction, use an empty output directory without `--resume`. The full protocol and commands are in `docs/lace_separation.md`; the current working checkpoint and next development target are in `STATUS.md`.

Training source revision: [827d886](https://github.com/JRowe47/SHoEbox/commit/827d8861c0e821a4e568d58db50fe72123aaeb28). The completed audit checks all 38 evaluation sections against their checkpoint, cohort and evaluator hashes, and verifies that all 223 guarded source files still match the training revision.
