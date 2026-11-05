# Second lookup continuation protocol

This stage begins at the regional-supervision checkpoint from published commit
`3bc60c79e5fbdf918d85dc134e60678eeadd6313`: 75.00% held-out composition,
92.23% local first lookup, and 100% source-table retention.

## Learning and selection

Continue the existing regional objective with all model tensors, optimizer
moments, optimizer step counters and sampling/Torch RNG states intact. The
network architecture, inference interface, task distribution and loss weights
stay the same. The parent ended at optimizer update 600. The new phase's update
1 therefore executes optimizer update 601, without another LR warmup.

The configured maximum is 600 additional updates. Batch size is 64, LR 0.001,
AdamW and clipping remain as recorded by the parent. Source reconstruction has
weight 0.5; the shared regional objective has weight 1.0. Every batch comes from
the original training-pair sampler. The full table-pair split is unchanged.

Every 100 updates, evaluate all 528 unique validation cases. A checkpoint
qualifies only when it meets every condition below:

| Condition | Required validation accuracy |
| --- | ---: |
| Common 256-byte answer head, all cases | At least 90% |
| B-side lookup, C-side lookup, nonidentity composition, separately | At least 90% each |
| P's first lookup and Q's second lookup, existing local head | At least 90% each |
| B and C exact source tables, before and after query | 100% each |

Two consecutive qualifying validations establish this stage's mastery gate.
`--stop-on-mastery` ends training at the second check, or training ends at the
declared maximum. A failed check clears the consecutive-check streak.

Select the lowest validation answer NLL among qualifying checkpoints. If none
qualify, retain the lowest-NLL candidate and explicitly record that the gate is
open. The initial parent is included as a candidate. Test scoring occurs only
after `selection.json` records the selected checkpoint and its SHA-256.

## Checkpoints and exact continuation

`initial.pt` preserves the incoming model, optimizer and RNG without resetting
them. `best_candidate.pt` tracks lowest answer NLL. `best.pt` contains the best
capability-qualified checkpoint. `mastered.pt` records the first second
consecutive pass. `last.pt` permits exact resumption, including the gate streak.

Checkpoint `step` is local to this phase; `optimizer_step_offset + step` is the
optimizer update count inherited through continuation phases. Each history row
has both `phase_step` and global `step`. Source hashes and a parent-file hash
prevent resuming under a changed protocol.

```bash
python -m experiments.text_regions.composition_continue \
  --parent artifacts/second_lookup/parent.pt \
  --out artifacts/second_lookup/continuation \
  --steps 600 --validate-every 100 --stop-on-mastery
```

To resume an interrupted run, repeat the command with `--resume`. The optional
`--stop-after N` creates a bounded checkpoint without changing the maximum
budget or protocol. The parent file is the previous round's
`artifacts/mastery/regional/best.pt`.

## Evaluation and accounting

Read the finalized `selected_checkpoint` from `selection.json`. Evaluate that
file, using `best_candidate.pt` only if qualification failed:

```bash
python -m experiments.text_regions.composition_eval \
  artifacts/second_lookup/continuation/best.pt --controls --threads 1 \
  --out artifacts/second_lookup/continuation/test.json

python -m experiments.text_regions.results.lookup_mastery.diagnostics \
  artifacts/second_lookup/continuation/best.pt \
  --out artifacts/second_lookup/continuation/diagnostics.json

python -m experiments.text_regions.results.second_lookup.audit \
  --root artifacts/second_lookup --out artifacts/second_lookup/audit.json
```

The existing K=1/2/4, no-exchange, memory-reset and B-shuffle controls remain.
Both original and changed answers are scored for the shuffle. The original
test cohort contains 592 unique cases; shuffled cases are paired interventions.
No new training objective, decoder fit or architecture change is introduced.

Each retained update accounts for 64 episodes, 16,896 expert executions and
704 source-byte occurrences. Report the actual stopping update and distinguish
it from the maximum budget. Evaluation uses additional computation outside
these training counters. Wall times include contention and are not benchmarks.

The audit verifies initial optimizer/RNG preservation, the recorded training
trajectory, qualifying selection, predictions, source hashes and an inference
export with the training-only heads removed.
