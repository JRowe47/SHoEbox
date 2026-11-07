# Credit through producer-state transitions

This pass continues `4d919d1`, using the selected direct-sum checkpoint from
the contextual-learning comparison. It adds a training-only option to carry
admission-policy credit through the graph's existing recurrent producer state.
The inference graph, checkpoint parameter names and parameter count are unchanged.

## What crosses a boundary

Publication credit already assigns later loss to a historical write decision
and its payload adapter. Producer-state credit adds the other route: a commit
changes a node's contextualizer and expert-block MinKron matrices, which influence
later computations even after its published payload has been replaced.

Each producer owns one sensitivity tensor per recurrent matrix. Its last axis
indexes that producer's existing admission-policy weights. Storage depends on
batch size, owner count, recurrent matrix sizes and policy width, not elapsed
stream length. No replay examples, old activations or extra inference memory
are introduced. Receipts at a gradient boundary are consumed once after backward.

For local state sensitivity E, actual commit a, policy probability p, fixed
historical policy features f and recurrent candidate C:

```text
E_next = (1-a) E + a J_state E + (C-S) outer (p(1-p) [f,1])
```

Forced decisions omit the final term. Always-commit mode transports E without
adding gate responsibility. Inactive rows preserve their trace; a document reset
clears it. Clocks and epochs bind traces to the state they describe.

For a fixed-input MinKron transition, J_state first applies the actual per-key
decay and then the actual key/value erase operation. The implementation applies
these small operations directly to E without allocating a state-by-state
Jacobian. There is no extra artificial trace decay.

This is a conditional derivative: historical module inputs, policy features,
inter-layer and inter-region dependencies remain fixed. It does not reconstruct
full BPTT, historical optimizer derivatives, or the state-to-publication path
that was already cut before the current segment. Ordinary live-window autograd
still trains every parameter. Tests compare against the exact corresponding
fixed-input untruncated derivative in FP64.

## Integration and cost

`StateCreditRun` extends the paired language/task trainer. Both sources see the
same weights before one optimizer update and own separate runtime states,
publication traces, producer traces and stream cursors. All of these resume
exactly. The `state_credit` protocol flag controls the new gradient contraction.
The original `ContextBalancedRun` remains available without the new trace cost.

Training hooks observe each current recurrent input, recover its local transition
coefficients and update the trace at the owner's commit. Hooks are removed on
normal completion and exceptions. The stage planner already respects hooks, so
tracked contextualizers use their per-node path while recording. Inference has
no hooks. Reusing kernel intermediates and retaining batched contextualizers are
future performance work; this pass prioritizes derivative correctness.

The new controls remain optional and independently ablatable. Their coordination
with publication and recall is marked in code as a potentially threshold-dependent
emergent feature.

## Matched continuation

The fixed budget is 32 paired updates from the same parent, with 4 streams,
32-byte windows, 16-byte gradient segments and constant learning rate 0.001.
Both arms collect identical kinds of traces; the publication-only control omits
only the producer-state gradient contraction. Each update directly sums language
and task gradients, followed by the existing global norm cap of 1.

The data grammar, natural-text distractors, correction mechanism and loss weights
are inherited from the parent study. The memory document seed is 483. The parent
language cursor continues; optimizer and runtime states start fresh. Validation
at updates 0, 16 and 32 selects the lowest mean language/answer NLL, with earlier
ties retained. Testing uses 1,024 natural targets and eight fresh task documents
per split; long-delay documents triple the configured retention gaps.

The causal probe selects the first exact query beyond 96 bytes in each long-delay
test document. Its paired input changes exactly one earlier fact-value byte.
Everything after that observation, including the query, stays identical. Positive
directional log-odds means the answer distribution moves toward the changed fact.
A second fork swaps only the paired producer-state contents immediately before
the final query byte. Ports, identities, ingress, feedback and demand stay in place.
It measures immediate state-content sensitivity, separate from whole-rollout
changes to earlier publication or routing.

## Reproduce

```bash
python scripts/run_state_credit.py --parent artifacts/context_balance/sum/best.pt
python scripts/evaluate_state_credit.py
python scripts/check_all.py
```

Training saves every completed pair. Repeating the runner resumes incomplete arms
and verifies their source, corpus, parent and protocol. Evaluation caches complete
measurements by weights, configuration, protocol, corpus and evaluator source.
Use a fresh output directory for a new comparison.
