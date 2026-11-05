# SHoEbox: lookup and regional learning

November 4, 2026 (America/Denver)

**Held-out composition accuracy rises to 75.00%, from 33.78% in the previous phase.** A matched continuation without the new regional objective reaches 47.30%. The trained model reaches 24.83% at K=1 and 75.00% at K=2. Disabling delayed regional exchange returns it to 24.83%. The existing source reconstruction head still recovers both complete tables perfectly before and after the query.

## What changed

The first addition is a mastery curriculum: B-side and C-side lookup must each reach 90% validation accuracy on two consecutive checks before the trainer advances. After advancement, fresh lookup examples remain mixed into composition training. The controller, optimizer and random state are checkpointed together.

A 300-step continuation exposed why simply extending the old lookup stage was insufficient. Final training accuracy on its lookup subsets reached 85.00% for B and 89.58% for C, while both validation scores were 43.75%. The gate stayed closed. The identity-table subsets contain only 40 and 48 training cases, and training on those subsets also weakened source retention outside them. That run was stopped at its bounded checkpoint and was not evaluated on the test set.

The follow-up teaches the regional operations directly across the original training pairs. P's first-round publication receives a training target of `f(x)`; Q's second-round publication receives `g(f(x))`. One shared 132-parameter affine auxiliary head supplies these losses. Targets enter only the loss calculation. The complete regional graph learns through those gradients.

The runtime now optionally exposes differentiable P/Q publications after each whole-region round. This adds access to intermediate states without changing graph execution or predictions. Inference uses the existing common 256-byte answer head. Removing both training-only heads preserves its logits exactly and leaves the original 252,478-parameter inference model.

## Matched experiment

Both completed runs start from the preceding phase's selected source-reconstruction model, with identical 737 shared state tensors and identical sampling RNG. Each retains 600 optimizer updates, 38,400 training episodes and 10,137,600 expert executions. The control and regional run use the same example sequence, K=2, batch 64, AdamW, LR 0.001 after a 20-step warmup, gradient clipping at 1 and source reconstruction weight 0.5. The regional run adds its auxiliary loss at weight 1.0.

The follow-up uses all training pairs from the first step. It is a separate learning intervention prompted by the failed mastery curriculum; that curriculum did not pass. Every 100 steps, all 528 validation cases are scored. Lowest answer NLL selects control step 300 and regional step 600 before test evaluation. These are paired continuations of one parent model, rather than independent fresh initializations.

The existing whole-table-pair split is preserved: 296 training, 132 validation and 148 test pairs. All four queries stay with their pair. The 592 original test queries are unique and answer-balanced; individual tables are familiar, while their combinations are held out. Prediction uses argmax across all 256 byte logits; the four-answer chance baseline is 25%.

## Held-out results

| Evaluation | Continuation control | Regional supervision |
| --- | ---: | ---: |
| K=1 | 25.84% (153/592) | 24.83% (147/592) |
| K=2 | 47.30% (280/592) | 75.00% (444/592) |
| K=4 | 34.97% (207/592) | 43.24% (256/592) |
| K=2, delayed exchange off | 25.68% (152/592) | 24.83% (147/592) |
| K=2, memory reset | 26.69% (158/592) | 24.49% (145/592) |
| Shuffled B, original answers | 25.17% (149/592) | 27.03% (160/592) |
| Shuffled B, changed answers | 46.62% (276/592) | 75.17% (445/592) |

Regional supervision adds **27.70 percentage points** over the matched control. A second round adds **50.17 points** over the same model at K=1. Its K=2 answer NLL is 0.7278 nats, versus 1.0074 for the control. Regional validation accuracy rises from 46.21% at step 400 to 61.17% at step 500 and 75.00% at step 600.

Shuffling B replaces the first relation while preserving C and the query. Accuracy on the original answers falls to 27.03%, while accuracy on the answers implied by the replacement remains 75.17%. This supports query-dependent use of the relation, alongside the feedback and reset interventions. As in the previous phase, the shuffled cohort contains 524 distinct replacement combinations; it is a paired intervention on the 592 original cases, not another independent cohort.

K=4 remains worse than the trained K=2 computation. Useful recurrence is substantially stronger, but arbitrary extra rounds are not yet stable.

## Where the computation succeeds and fails

The existing auxiliary heads were scored without fitting on held-out cases:

| Internal measurement | Accuracy |
| --- | ---: |
| P after round one: first lookup | 92.23% (546/592) |
| Q after round two: composed answer | 74.83% (443/592) |
| B table, before and after query | 100.00% of entries and complete tables |
| C table, before and after query | 100.00% of entries and complete tables |

These local scores use the four-class training head; the primary result above uses the common 256-byte head. P now performs the first lookup reliably. The larger remaining gap is Q's second lookup and the stability of repeated regional processing.

The common answer path has not passed the lookup-mastery gate. B-side lookup is 70.00% on 20 test cases; C-side lookup is 71.43% on 28. On validation, those scores are 65.62% and 68.75%. Full composition with neither table identity reaches 75.37% on 544 test cases. The model therefore improves substantially without completing the separate 90% lookup target.

## Verification and execution

All **336 tests pass** across the five experiment suites. New checks cover separate mastery gates, split-safe balanced sampling, exact curriculum continuation, intermediate-output invariance, Q's first-round query boundary, gradients through regional targets and delayed edges, and exact continuation with the auxiliary head.

The artifact audit checks identical initial graph tensors, identical sampling continuation, source hashes, validation-only selection, raw prediction counts and execution totals. It also exports inference-only checkpoints after requiring exact logit agreement on 64 held-out cases with both auxiliary heads removed.

| Query rounds | Prefix expert calls/case | Query calls/case |
| --- | ---: | ---: |
| K=1 | 96 | 84 |
| K=2 | 96 | 168 |
| K=4 | 96 | 336 |

Each case provides eleven external bytes at every K. Both inference graphs contain 252,478 parameters and 384 physical experts. The control allocates 253,006 parameters during training; regional supervision allocates 253,138. Its extra head adds training arithmetic without changing expert execution counts.

Both runs resumed from step 500 after a runtime interruption. Execution totals describe the retained 600-update trajectories; discarded work before resumption is additional physical compute. The complete logs and checkpoint histories are retained. Training used PyTorch 2.15.0+cpu, Python 3.12.14 and one Torch thread per run. Concurrent runs and tests make these wall times unsuitable as speed measurements.

## Next development target

Keep this regional-supervision model as the composition candidate. Focus the next learning phase on bringing Q's lookup to P's reliability while preserving source retention and the existing first lookup. Retain the common-answer mastery checks and K/feedback/shuffle/reset controls. The wider one-pass comparison, independent initializations and combined memory/shared-private training remain required acceptance work. M7 is still open.

## Reproduction and files

The source package includes the implementation, tests and `docs/SHoEbox_Lookup_Protocol.md` with complete commands. The artifact package includes the parent checkpoint; initial, selected and final checkpoints for both completed runs; inference-only exports; the bounded mastery run; raw predictions; diagnostics; histories; tests; audit output; and the original mastery runtime for exact continuation.

Published measurements and compressed raw JSON are under `experiments/text_regions/results/lookup_mastery`. Python's standard `gzip` module can read each `.json.gz` file. The source archive contains the final regional runtime; the earlier mastery runtime snapshot is in the artifact archive. Earlier phases' unrelated checkpoint collections are not included.
