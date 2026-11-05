# SHoEbox: reliable second lookup

November 5, 2026

**Held-out composition reaches 95.78% (567/592), up from 75.00%.** Q's second lookup reaches 96.11%, close to P's 96.45% first lookup. Both complete source tables remain recoverable in every evaluated case, before and after the query. These gains come from continuing the existing regional objective with the same graph, inference head and loss weights.

## What this stage adds

The new continuation trainer preserves model weights, optimizer moments and counters, and both sampling and Torch RNG states. It continues at regional optimizer update 601 without restarting warmup. A capability-aware checkpoint selector requires the common answer, both lookup directions, nonidentity composition, P's first lookup and Q's second lookup to reach at least 90% validation accuracy. B and C must each retain 100% exact source tables before and after the query.

The previous run was still improving at its endpoint. This stage therefore continues the full original training-pair stream without adding a new objective or changing the architecture. Training remains K=2, batch 64, AdamW with beta=(0.9, 0.95), matrix weight decay 0.01, LR 0.001 and gradient clipping at 1. Source reconstruction retains weight 0.5 and regional supervision weight 1.0.

## Selection and stability

The declared budget is 600 additional updates, with all 528 unique validation cases checked every 100 updates. A checkpoint qualifies only if every capability condition passes. The lowest answer NLL among qualifying checkpoints selects the model. Two consecutive qualifying checks are a separate stability target and can stop training early.

Update 500 passes every condition: 96.21% overall validation accuracy, 96.88% B-side lookup, 100% C-side lookup, 96.04% nonidentity composition, 97.16% P first lookup, 96.40% Q second lookup and 100% source retention. Its answer NLL is 0.1581 nats. It is selected before test scoring, at regional optimizer update 1,100.

Training completes the 600-update budget. The final checkpoint still exceeds 90% on all six answer/lookup measures, but reconstructs only 527 of 528 complete B tables after the query. It does not replace the selected checkpoint. **One checkpoint passes every gate; the two-consecutive-pass stability target remains open.** The package retains both the selected and final states.

## Held-out results

The task remains `g(f(x))`: B independently receives the four-byte table for f, C receives the table for g, and A receives x afterward. All seven nodes execute each query round. P learns the first lookup; delayed regional exchange lets Q use that result in the second round. Targets enter training losses only. Inference chooses the largest of all 256 byte logits from the common answer head.

The whole-table-pair split is unchanged: 296 training, 132 validation and 148 test pairs. Every pair's four queries stay in one split. Individual tables are familiar; held-out combinations are not used for training. The original test cohort has 592 unique, answer-balanced cases.

| Evaluation | Previous selected model | This stage |
| --- | ---: | ---: |
| K=1 | 24.83% | 25.00% (148/592) |
| K=2 | 75.00% | 95.78% (567/592) |
| K=4 | 43.24% | 45.78% (271/592) |
| K=2, delayed exchange off | 24.83% | 25.00% (148/592) |
| K=2, memory reset | 24.49% | 21.96% (130/592) |
| Shuffled B, original answers | 27.03% | 27.03% (160/592) |
| Shuffled B, changed answers | 75.17% | 96.28% (570/592) |

The selected model improves K=2 accuracy by 20.78 percentage points over its parent. The second round adds 70.78 points over K=1 with the same weights and observations. Answer NLL falls from 0.7278 to 0.1657 nats. The comparison is a continuation of one training lineage, with additional training compute.

Changing B changes the relation the model must follow. It answers the replacement relation correctly in 96.28% of paired interventions while agreeing with the original answers in only 27.03%. The shuffled cohort contains 524 distinct replacement combinations across 592 interventions. Disabling delayed exchange and resetting memory also remove the gain.

K=4 remains much weaker than the trained K=2 computation. Extra processing rounds do not yet preserve a solved answer.

## Regional computation and retention

The existing local heads are scored without additional fitting. The first two rows below use the shared four-class regional head; the lookup and composition subsets use the common 256-byte answer head.

| Held-out measurement | Result |
| --- | ---: |
| P after round one: f(x) | 96.45% (571/592) |
| Q after round two: g(f(x)) | 96.11% (569/592) |
| B-side lookup, with C identity | 95.00% (19/20) |
| C-side lookup, with B identity | 100.00% (28/28) |
| Composition, neither table identity | 95.59% (520/544) |
| B table, before and after query | 100% of entries and complete tables |
| C table, before and after query | 100% of entries and complete tables |

Q has closed most of the previous gap to P. All selected-model capability thresholds also hold on the unchanged held-out test cohort. Each source-retention measurement covers 592 tables and 2,368 entries.

## Verification and execution

**All 342 tests pass** across five experiment suites. Six new checks cover individual gate failures, consecutive checks, phase/update offsets, full-cohort metric parity, exact interrupted/resumed training, and checkpoint selection that cannot replace a qualified model with an unqualified one.

The artifact audit verifies that incoming weights, optimizer slots and both RNG states are preserved exactly. It checks the complete training trajectory, execution counters, source hashes, validation-only selection, checkpoint hashes and all seven raw test prediction groups. The inference export removes both training-only heads and matches the selected model's logits exactly on 64 held-out cases.

The inference model still has 252,478 parameters and 384 physical experts; training allocates 253,138 parameters including the auxiliary heads. No inference capacity was added.

| Query rounds | Prefix expert calls/case | Query expert calls/case |
| --- | ---: | ---: |
| K=1 | 96 | 84 |
| K=2 | 96 | 168 |
| K=4 | 96 | 336 |

Each case supplies eleven external bytes at every K. The completed continuation accounts for 38,400 training episodes, 422,400 source-byte occurrences and 10,137,600 expert executions. The selected checkpoint occurs after 32,000 of those episodes. Earlier source pretraining and the parent regional phase precede this budget; evaluation is additional computation. The run used Python 3.12.14, PyTorch 2.15.0+cpu and one Torch thread. Concurrent tests make elapsed times unsuitable as speed measurements.

## Next development target

Keep the selected K=2 model as the composition candidate and carry its lookup and source-retention checks into integrated capability training. The next substantial step is one model that combines memory/correction, shared/private facts and composition. Longer-round stability, the comparable-budget one-pass control, independent initializations and the natural-text experiment remain acceptance work. M7 is still open.

## Reproduction and files

The source archive includes the trainer, tests, protocol and audited results under `experiments/text_regions/results/second_lookup`. The artifact archive contains the parent and its runtime snapshot, initial/selected/final checkpoints, the inference export, raw predictions, internal diagnostics, the full history, logs and this report. The protocol in `docs/SHoEbox_Second_Lookup_Protocol.md` and the artifact README give exact commands. Archives share a `SHoEbox/` root and include source-version metadata; artifact members are covered by a SHA-256 manifest.
