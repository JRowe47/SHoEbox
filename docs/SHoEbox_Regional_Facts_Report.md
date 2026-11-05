# SHoEbox: regional facts and corrections

Completed November 5, 2026. Two matched 100-update continuations; all remaining training, evaluation, export and audit work is complete.

## Result

Training-only local supervision did not solve the integrated memory and fact tasks in this run. The control scores 100.00% on held-out composition; regional supervision scores 99.83%. The validation rule selects `control/best.pt` at phase update 100. M7 remains open.

Regional supervision improves shared-fact answers from 32.42% to 37.70%, with smaller private-fact gains, while trained-delay recall and correction fall. Both arms continue the same integrated update-200 parent with identical initialization, optimizer moments, examples, carried stream states and random states. The sole experimental difference is local-loss weight 0 versus 1. The regional memory/fact heads add 528 training-only parameters; the exported inference graph remains unchanged.

## Matched endpoint scores

| Held-out task | Control | Regional supervision |
| --- | --- | --- |
| Composition, K=2 | 100.00% | 99.83% |
| Delayed recall, gap 16 | 43.75% | 35.35% |
| Delayed correction, gap 16 | 32.23% | 27.15% |
| Shared fact, all sources | 32.42% | 37.70% |
| Left-private fact, all sources | 27.15% | 28.91% |
| Right-private fact, all sources | 24.22% | 27.34% |
| A-only shared fact | 29.88% | 34.96% |
| A-only left-private fact | 24.22% | 29.30% |
| B-only shared fact | 27.73% | 25.78% |
| C-only shared fact | 26.76% | 24.80% |
| C-only right-private fact | 25.20% | 27.93% |

Composition uses all 592 held-out permutation cases. Each memory condition and fact-query type uses 512 fresh episodes. Missing-source controls use independently generated full-factorial cohorts. The four-value reference is 25%; every task answer is chosen from all 256 byte logits.

---

## Composition and source retention

| Composition condition | Control | Regional supervision |
| --- | --- | --- |
| One round | 25.00% | 24.66% |
| Two rounds | 100.00% | 99.83% |
| Four rounds | 41.55% | 41.55% |
| Two rounds, exchange disabled | 25.00% | 25.00% |
| Changed table, original answer | 27.03% | 27.03% |
| Changed table, changed answer | 100.00% | 99.66% |
| State reset before query | 22.80% | 22.80% |

The two-round answer remains dependent on exchange and retained source state. Replacing a table changes the expected answer: scoring against that changed answer recovers high accuracy. Four rounds do not improve this trained two-round computation.

| Held-out regional check | Control | Regional supervision |
| --- | --- | --- |
| P: first local lookup | 100.00% | 100.00% |
| Q: second local lookup | 99.83% | 99.83% |
| B: exact table after ingestion | 100.00% | 100.00% |
| C: exact table after ingestion | 100.00% | 100.00% |
| B: exact table after query | 100.00% | 100.00% |
| C: exact table after query | 100.00% | 100.00% |

Exact source-table retention is a 100% gate; the answer and local-lookup gates require at least 90%. The regional endpoint scored 98.48% on B's post-query exact-table validation check, missing that gate. Its held-out table checks above all pass; checkpoint selection still follows the validation results.

## Validation selection

| Run | Selected phase update | Weakest validation task |
| --- | --- | --- |
| control | 100 | 23.44% |
| regional | 0 | 23.44% |

Checkpoints are ranked using validation only: full qualification, composition preservation, weakest capability accuracy, macro accuracy, then lower macro NLL. The initial checkpoint is an allowed fallback. Held-out scores do not rank checkpoints. The selected control checkpoint is its update-100 endpoint; the regional arm retains its initial checkpoint as its validation choice.

---

## Delayed memory controls

| Memory condition | Control | Regional supervision |
| --- | --- | --- |
| recall, gap 16, carried | 43.75% | 35.35% |
| recall, gap 16, reset_query | 29.88% | 21.68% |
| correction, gap 16, carried | 32.23% | 27.15% |
| correction, gap 16, reset_query | 25.98% | 25.98% |
| correction, gap 16, withheld: old value | 27.73% | 27.73% |
| recall, gap 32, carried | 24.41% | 28.32% |
| recall, gap 32, reset_query | 29.88% | 21.68% |
| correction, gap 32, carried | 28.32% | 22.66% |
| correction, gap 32, reset_query | 25.98% | 25.98% |
| correction, gap 32, withheld: old value | 20.90% | 26.37% |

The gap-32 evaluation extends the delay beyond the gap-16 training setting. Reset controls remove carried state at the query. Withheld-correction rows score agreement with the old observed value; the correction was not supplied. Raw reports also include agreement with the withheld new value, full predictions, episode identities and execution counts.

## Reading the regional memory state

| Regional memory head | At write | At query |
| --- | --- | --- |
| recall: P | 59.38% | 28.12% |
| recall: Q | 51.56% | 21.88% |
| correction: P | 42.19% | 17.19% |
| correction: Q | 48.44% | 35.94% |

These frozen local-head diagnostics use 64 new held-out episodes per task, after checkpoint selection. They read P and Q at the observed write and again at the final query. The memory head can decode some newly written values, but its query-time accuracy is much weaker. Persistence and later readability remain the next concrete memory bottleneck.

---

## Fact acquisition diagnostics

| Regional fact head | At write | At query |
| --- | --- | --- |
| shared: P | 28.12% | 31.25% |
| shared: Q | 17.19% | 29.69% |
| left: P | 26.56% | 23.44% |
| right: Q | 25.00% | 23.44% |

The shared and private fact heads remain weak even at the observed-write positions in these downstream nodes. This run therefore does not establish successful fact acquisition followed only by forgetting. The saved diagnostic includes every eligible node/factor, not just the P/Q rows shown here. The control's local heads were not trained and are retained only as a matched reference.

## Completion and verification

Both runs resumed from update 90 with source hashes verified, then completed updates 91-100 and final validation. The reproduced control update-100 measurements exactly match the original pre-interruption log, excluding timing. Full resumable checkpoints, optimizer state, streams and all validation endpoints are retained.

Each arm executed 9,076,224 experts across 75,968 external events, with 6,400 composition episodes. Every update's examples, source counts and execution budget match between arms. All 355 regression tests across five suites passed before recovery; training-source hashes are unchanged.

The evaluation audit recomputes accuracy from 76 stored prediction groups, verifies generated targets and cohort identities, checks matched intervention budgets, and confirms validation-only selection. Inference exports strip all four auxiliary heads and reproduce the checked logits and recurrent states exactly. The additional completion audit verifies evaluation manifests, all completed stages and diagnostic arithmetic.

## Reproduce and resume

Extract the source and artifact archives into the same directory; both use the `SHoEbox/` root. Install `requirements.txt`, then run:

```bash
python scripts/audit_regional_facts.py
python scripts/audit_regional_facts.py --evaluations
python scripts/verify_regional_completion.py
```

For a fresh matched continuation, use the two commands in `docs/SHoEbox_Regional_Facts_Protocol.md`. To resume evaluation, rerun `scripts/evaluate_regional_facts.py` with the same checkpoint, output, count and batch arguments plus `--resume`. Completed stages are reused only when their manifest matches.

The final source archive includes code, protocol, this report and recorded JSON results. The artifact archive includes training and inference checkpoints, the integrated parent, recovery checkpoints, histories, predictions, diagnostics and logs. Its manifest records the source commit and publication status. No further training phase was started.
