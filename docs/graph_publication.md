# Task-trained publication in the regional graph

The ordinary seven-node graph now supports the same learned admission, state
holding and delayed publication credit previously used only by the task-memory
model. Natural text and procedural memory documents train one set of regional
experts through the same byte input and output interfaces.

## Publication decisions

`GraphConfig.task_publication` selects participating owners. Each owner's policy
uses candidate content, a detached summary of its stored publication (including
validity, age and similarity), and downstream demand from the preceding round.
The memory and demand projections start at zero. Independent sigmoid admission
fits the graph's sequential stages; it does not impose a global write contest
between regions that execute at different times.

An empty active owner bootstraps a publication. Later hard writes increment its
version, while a hold preserves the complete publication and its dependencies.
Zero-valued straight-through tangents train admission from downstream task loss.
Optional low-rank payload adaptation exposes a compact delayed-credit target.
With `publication_state_commit='publication'`, contextualizer and recurrent
contents follow the same hard decision; execution clocks still advance.
Receiver feedback continues to track candidate computation independently.

## Causal demand

Each receiver reports mean and maximum soft slot-selection probability for each
valid incoming edge. An owner aggregates those observations across its active
readers. The completed-round demand becomes available only to the next round,
including the next byte when there is one round per byte. A writer cannot use a
future reader's decision in its current publication. Missing or blocked readers
contribute zero; document resets clear demand. Demand has live gradients inside
a training segment and detaches with the rest of runtime state at its boundary.

Forward edges still read completed current-round stages. Lateral and descending
edges still read the round-start snapshot. Recall remains a view of owned ports;
there is no episode buffer, replay store or task-ID input.

## Delayed credit and exact resume

`GraphEligibility` binds bounded sensitivities to owner order, publication
version and document epoch. It reuses the existing gate and factored payload
estimators. Activity records follow each owner's actual execution, flattened in
byte-then-round order. A document reset applies once before that byte's first
round. A detached boundary supplies one leaf for packed payloads and memberships;
its backward gradient is contracted with the saved sensitivities exactly once.

The estimator holds historical candidates, feature and demand networks, and
other regions' candidate histories fixed. It restores conditional admission and
payload-adapter credit, not the full historical recurrent Jacobian. Decay remains
0.995 per active owner execution, now a regional round rather than a byte.

The mixed runner shares its scheduling, objectives, optimizer and checkpoint
implementation with the preceding task-memory runner. Each data source owns its
runtime state, demand, eligibility and stream cursor. Restore validates the
corpus, configuration, protocol and stream fingerprints, then verifies that each
credit trace matches its live publications. Unscheduled pauses do not add model
selection opportunities.

Mixed training preserves ordinary FP32 gradient clipping exactly. If the norm
overflows while every gradient entry is finite, it computes that norm in FP64
and clips without losing the gradient direction. Nonfinite entries still abort
before an optimizer update. Recovery counts are stored separately by source.

## Controls and reproduction

All new graph behavior is opt-in. Existing configurations retain their weights,
initialization RNG, state loading and default forward outputs. The checked pilot
continues the preceding seven-slot, two-round graph checkpoint; A/B/C/U/V acquire
publication policies, while P/Q keep publishing their readout outputs.

```bash
python scripts/run_graph_publication.py --parent /path/to/parent.pt \
  --data data/tiny_shakespeare.txt --out artifacts/graph_publication --parallel
python scripts/check_all.py
```

The paired arms share initial weights, training bytes, expert executions and
selection criteria. Only application of delayed gate/payload gradients differs.
Whole-rollout controls independently block reads, remove demand, always commit
producer state, always publish, pool recall roles, or remove both feedback paths.
These controls include their downstream effects on later computation and writes.
The evaluator saves completed measurements and reuses them when selected/final
weights match. Cache identity includes weights, configuration, data protocol and
corpus; bump the evaluation schema when changing measurement semantics.

The learned demand and content controls are retained for larger-data experiments.
Their code marks the intended coordination as a potentially threshold-dependent
emergent feature, with explicit switches for later ablation.
