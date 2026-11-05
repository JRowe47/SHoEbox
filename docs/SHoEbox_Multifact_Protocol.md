# SHoEbox: simultaneous shared and private retention

Development protocol, November 5, 2026. The confirmatory run writes its numeric settings and parent hashes to `protocol_lock.json` before training or test evaluation.

## Target and interface

Extend the published eight-node retained-composition graph to retain three logical facts simultaneously: one common fact observed by either of two independent sources, a left-private fact, and a right-private fact. Preserve facts across distractors, correct each independently, and use any requested fact as the continuous input to the frozen two-step composer.

The field query is an explicit address: shared, left-private, or right-private. This phase does not train natural-language query interpretation. The relation tables and final answer remain ordinary bytes; no fact value is decoded into an intermediate input byte.

## Ownership and initialization

Four producer nodes own separate contextual, expert and publication states: F/L observe the left stream; G/R observe the right stream. Both local producers see the complete raw stream. A source uses `identifier=value;` for its common field and `identifier:value;` for its private field. A learned producer gate decides whether to publish; the harness does not select observed value bytes or gate writes by the field label.

Each owner starts from the corresponding published joint-retention parent. For private-field owners, only the learned input-embedding rows for colon and equals are exchanged at initialization. Runtime bytes remain unchanged. This reuses the parent's byte-field convention without claiming that field parsing began from random weights.

The ordinary F/G/L/R-to-A edge carries the selected continuous publication. A shared read selects the valid common replica with the latest publication occurrence, with F winning an exact tie. This is explicit address and freshness handling, not a learned arbitration claim. Private reads select their dedicated owner. Query execution leaves all four owners inactive.

The seven-node composer and its common 256-byte decoder remain frozen. Only the four producers and their four edge value projections may learn. Edge gates and semantic projections stay at their inherited values. No model-level runtime defaults change.

## Data and local learning

Every episode has two independently ordered and padded source descriptions, fresh opaque identifiers, bare A-D distractors, invalid `=x` and `:x` strings, and independently timed observations. Both fields remain present in each source's raw stream. Values are four-way A-D.

Final fact triples use disjoint 32/16/16 train/validation/test combinations. All source-local value pairs occur in training. Missing-information controls use all 64 independent combinations, so the unobserved private value has a genuine conditional four-way prior. The audit checks actual identity uniqueness and split separation.

Recall, left-source common correction, right-source common correction, left-private correction, and right-private correction alternate. Gaps cycle through 0, 4, 8, 16. Field-specific causal labels are available only to the loss after an observed matching marker/value pair. No labels enter the runtime state or publications.

The local objective combines retained-payload alignment and write-candidate alignment to the model's own frozen A ingress representation, publication classification, the existing routing regularizers, and the mean of three final byte-query losses. Gate classification weights own-field writes by 0.5, opposite-field writes by 0.25, and ordinary holds by 0.25, with separate denominators. This prevents rare destructive writes from being diluted among easy filler bytes. Each 16-event chunk backpropagates before detaching its history; no replay store is introduced.

The confirmatory run uses 120 updates with batch 32. AdamW uses producer LR 0.0003, edge LR 0.001, beta=(0.9,0.95), weight decay 0.01, and global gradient clipping at 1. Validation uses 64 episodes per task at updates 0, 20, 40, 60, 80, 100, and 120. Selection includes update zero and maximizes the minimum of 15 validation task/field accuracies, then mean accuracy, then lower mean byte NLL. Exact ties retain the earlier checkpoint. Test results never select weights.

## Acceptance and controls

Each primary condition uses 512 fresh episodes per seed, with every episode queried for all three fields. All three published parent seeds are included.

- Every seed must reach at least 90% on each field for recall, each correction direction, and corrections withheld.
- Either source alone must support common and its own private answers at 90% or better. The unsupported private answer must remain at or below 35% on the independent cohort.
- After three live queries, an update-only suffix must support all three fields at 90% or better. The original facts are not presented again. Withholding that suffix must preserve the old answer.
- Blocked-memory and reset controls must remain at or below 35%; K=2 must exceed K=1 by at least five percentage points on the declared shared-correction task.
- Inactive source memories, export logits/state, frozen composer weights, declared execution counts, and all actual held-publication transitions must reproduce exactly.

Report 64-event delay, K=4, no delayed exchange, shuffled complete publications, unconditional publication, repeated requests, stale-replica selection, and equal-execution erased-source controls separately. Counterfactual shuffled and stale targets distinguish using substituted memory from merely producing a different answer.

All acquisition controls count duplicated encoder work: two raw source views are delivered to four independent field producers. Query controls preserve expert execution counts. Evaluation may reuse immutable acquisition states across query interventions; reported per-condition counts describe the complete standalone path. Training counters record actual executions.

The artifact bundle includes exact initial/selected/final optimizer checkpoints, inference state, protocol/source hashes, per-example predictions, independent reconstruction audits, and the separate development pilots. M7's natural-text experiment follows this integration stage.
