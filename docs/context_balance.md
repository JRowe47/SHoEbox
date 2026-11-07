# Language and contextual memory in one update

The contextual-retention pass continues the task-trained regional graph from
`7f3ce944547d800e133e999046a77d4e029a1851`. It keeps the same seven nodes,
two rounds, seven recall slots and 312,440 trainable parameters. Publication,
state holds, receiver demand, payload adaptation and delayed credit all remain
active and independently ablatable.

## Shared learning

`ContextBalancedRun` processes one ordered language window and one procedural
memory window at identical weights. Each performs its own backward pass and
applies its own delayed publication credit before the gradients are captured.
Both source states and cursors advance independently. The two gradient sets
are then combined for one AdamW step; no extra model pass is needed.

The matched `sum` control directly adds both gradients. `group_clip` reuses the
existing gradient-balancing implementation, bounding each source's contribution
to each of four disjoint parameter groups at norm 1 before addition:

- Core: shared/private experts, recurrent regions, routing and edge projections.
- Readout: byte emission, readout mixing and normalization.
- Recall: selection roles, composer and regional recall controls.
- Publication: admission features, write policy, demand projection and payload adapter.

Both arms apply the same final global norm cap of 1. Moments and dot products
are measured in FP64. Logs retain each source/group norm, scale and cosine.
This bounds updates; it does not project away conflicting directions. Per-group
balancing is an optional training control, with the ordinary sum retained for
later ablation. The intended coordination between retention, demand and expert
learning remains a potentially threshold-dependent emergent feature.

## Context and supervision

The generator can fill half of each newly required acquisition/use gap with
ordinary text, leaving the rest for real corrections to competing facts. It
uses a separate deterministic RNG and surrounds corpus snippets with newlines.
The legacy generator is byte-for-byte unchanged when no corpus is supplied.
Natural text is an input distractor, with no new task or memory identifier.

Training snippets come from the training corpus partition. Validation snippets
come from validation; ordinary and long-delay tests use the test partition.
Fact-version provenance is computed after inserting context. Exact, semantic,
corrected and contextual-answer targets remain outside the model input.
The syntax objective excludes answer positions as before.

The study uses four entities, acquisition queries, six questions per cycle,
two cycles, minimum delays of 4/48/96 bytes and real interfering updates.
Long-delay tests triple those configured minima. Earlier language exposure is
continued by transferring the parent's language cursor. Runtime states and
optimizer start fresh; memory documents use a new seed namespace. All weights
transfer from the selected publication-learning parent and remain trainable.
Acquisition questions keep their short gaps in every split. Reporting therefore
also isolates exact answers more than 96 bytes after their source observation,
as well as answers beyond a gradient segment and corrected facts in that cohort.

## Measurements and recovery

The fixed selection rule is mean validation natural-byte and answer NLL at
updates 0, 32, 64, 96 and 128; ties retain the earlier checkpoint. Both sources
receive 128 windows of 4 by 32 bytes: 16,384 bytes each. Delayed credit detaches
at 16-byte segments. AdamW has constant learning rate 0.001, betas 0.9/0.95 and
weight decay 0.01. Source exposure, answer counts and expert executions must
match between arms. This is 256 source forward/backward passes and 128 AdamW
updates per arm, distinct from the preceding alternating-step schedule.

Validation uses 1,024 natural-byte targets and eight fresh documents. Final
evaluation doubles both. The preceding eight-document task test is retained
as a separate compatibility measurement. Selected checkpoints additionally
receive whole-rollout blocked-read, always-write, no-demand and always-commit
controls. These interventions include changes to later writes and state.
The common starting checkpoint is also measured on these exact held-out splits,
so continued learning can be separated from the gradient-control comparison.

At queries, the evaluator reports the age of selected valid publications, both
across all edges and restricted to A/B/C/U/V producers. It reports the fraction
older than the 16-byte gradient segment. A publication's age is distinct from
the age of the queried fact; current publications can carry older recurrent
information. Neither quantity alone proves useful retained content.

Every completed pair saves optimizer, both runtime states, demand, credit,
cursors, RNG, selection score and exposure totals. Resume validates corpus,
model, protocol and each stream fingerprint. Partial pairs are discarded after
an interruption. Evaluation caches completed measurements by weights,
configuration, protocol and corpus; selected/final identical weights reuse work.

## Reproduce

```bash
python scripts/run_context_balance.py --parent /path/to/graph_publication/credit/best.pt \
  --data data/tiny_shakespeare.txt --out artifacts/context_balance --parallel
python scripts/check_all.py
```

The runner resumes unfinished arms and skips fully evaluated arms. Checkpoints
and the exact corpus are included in the downloadable artifact archive.
