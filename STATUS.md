# SHoEbox handoff

## Completed: credit through recurrent producer state

Continues public commit `4d919d1d4e23d6af3fb052987c2fe1cdd0bcad59` and the
context-balance direct-sum checkpoint at update 128.

`StateCreditRun` adds a training-only producer-state eligibility trace alongside
existing publication and payload credit. The local MinKron decay/erase Jacobian
transports sensitivity to each owner's existing admission weights. Forced
commits omit gate injection; inactive rows hold; document resets clear traces.
Epochs and clocks bind traces to their state, and boundary receipts are single-use.
All 312,440 parameters and the inference graph remain unchanged.

The derivative is conditional on historical module inputs and policy features.
Historical inter-layer/inter-region and already-detached state-to-publication
paths remain truncated. Keep the mechanism independently ablatable and marked
as a potentially threshold-dependent emergent feature. No replay or separate
higher-order-memory curriculum was added.

## Completed evidence

Both arms finished 32 paired updates: 4,096 natural bytes, 4,096 task-stream
bytes, 203 answer targets and 1,376,256 expert executions per arm. Every weight
trains. Both collect the new traces; only state-gradient contraction differs.
Validation at 0/16/32 selects update 32 in both arms. Final and selected weights
therefore coincide. The state arm's combined validation score is 2.436000 versus
2.437005 for publication-only credit.

| Held-out measure | Parent | State credit | Publication only |
|---|---:|---:|---:|
| Natural-byte NLL | 3.115499 | 3.121512 | 3.124802 |
| Answer NLL | 1.863883 | 1.773150 | 1.768821 |
| Exact NLL beyond 96 bytes | 1.609711 | 1.611467 | 1.602640 |
| Exact accuracy beyond 96 bytes | 32.14% | 32.14% | 32.14% |

Testing uses 1,024 natural targets, 144 ordinary answer targets and 56 exact
long-delay targets beyond 96 bytes. Eight paired observation-change probes
alter only an earlier relevant value; no checkpoint answers both members of
any pair correctly. Producer-state swaps have tiny query-time effects. The new
learning path is verified; this run establishes no recall advantage.

All 571 unique checks pass: the 568-check regression run and three separately
completed probe checks. The current check driver includes both. Fixed-input
FP64 derivative agreement, forced/inactive/reset behavior, disabled-credit
parity, backend agreement, cleanup, source isolation and exact resume pass.

Persistent producer sensitivities add 115,520 bytes per four-stream source.
The small CPU timing probe costs 1.40x the publication-only forward/backward
path. Recording currently recomputes coefficients and bypasses stage batching
for tracked contextualizers. The original ContextBalancedRun has no trace cost.

## Next development target

Prepare the integrated implementation for the planned laptop/GPU run. First
reuse MinKron transition intermediates and preserve batched contextualizers
while recording state credit; test GPU/device and precision behavior. Keep
language and task learning shared. Use greater ordinary stream exposure to test
useful retained content, without adding a special higher-order curriculum or
optimizing publication age itself. Retain all existing controls for later ablation.

For continuing this feature, the archive contains
`artifacts/state_credit/state/best.pt` and its publication-only comparator.
A new continuation should advance the task seed; the current runner's fixed
seed 483 reproduces this completed comparison. Both states/cursors/optimizers
are also retained for exact recovery of the existing protocol.

[Report](docs/SHoEbox_State_Credit_Report.md) |
[PDF](docs/SHoEbox_State_Credit_Report.pdf) |
[protocol](docs/state_credit.md) |
[metrics](experiments/text_regions/results/state_credit/summary.json).

## Reproduce

```bash
python scripts/run_state_credit.py --parent artifacts/state_credit/parent.pt --out artifacts/reproduction
python scripts/run_state_credit_evaluation.py --root artifacts/reproduction
python scripts/check_all.py
```

The archive includes runnable source, corpus, parent, selected/final checkpoints,
optimizer/runtime/credit states, complete logs, causal query traces and hashes.
