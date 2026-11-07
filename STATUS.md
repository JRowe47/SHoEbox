# SHoEbox handoff

## Latest: connected recall continuation and bounded activation memory

Both matched arms finish 16 paired-source updates from the selected update-32
state-credit parent: 8,192 natural bytes and 8,192 task bytes, with 374 answer
targets per arm. Four streams use 128-byte optimizer windows and 16-byte
execution chunks. Both select update 16 on validation.

Exact source coverage rises from 64/291 (22.0%) to 209/291 (71.8%) queries with
the connected path. Immediate acquisition coverage is 86/90 versus 51/90;
exact queries beyond 96 bytes receive 16/71 versus 0/71 connected derivatives.
The audit checks actual source occurrences and resets, not nominal delays.

Held-out natural/answer NLL is 3.1089/1.6659 for connected credit versus
3.1065/1.6524 for the old cutoff; the parent is 3.1215/1.7924. Balanced
four-value immediate recall remains 25%, with zero complete sets in both arms.
The run establishes no useful value-conditioned recall advantage.

Optional activation recomputation drops measured peak process RAM from
1,634.1 to 685.2 MiB on a matched 64-byte, four-stream paired update, with
38.59 to 55.37 seconds runtime. State, traces and cursors match exactly;
post-update weights differ by at most 2.98e-8 from accumulation order.
The option stays off by default; this paired study uses ordinary activations.
Training kernels remain selected during checkpoint forwards, including large
banks and batched stages. All 43 targeted checks pass; exact resume works.

Next: prepare device-aware batches and checkpoint restoration for the planned
laptop/GPU run, then broaden ordinary shared-stream exposure with larger
connected windows. Reliable immediate value-conditioned recall remains the
behavioral target. Keep the existing controls ablatable and add no special
higher-order curriculum. The repair trainer retains its 384-byte default.

[Report](docs/SHoEbox_Connected_Recall_Report.md) |
[Protocol](docs/connected_recall.md) |
[Audited metrics](experiments/text_regions/results/connected_recall/summary.json).

## Completed: repair the historical content path

`ContentCreditRun` separates 16-byte execution chunks from a default 384-byte
optimizer/gradient window. Internal chunks carry the complete live graph.
Historical eligibility is captured once at window entry: capture itself detaches
state, and applying internal receipts would double-count exact autograd credit.
Global norm cap 1.0, shared language/task updates and inference are unchanged.

The original 148-byte source-to-answer derivative is restored from zero to
0.013164902495110228, with zero difference from the full reference in both
gradient and logits. All 35 relevant regression checks pass, including exact
resume, row resets, older credit, all-parameter gradients and runtime fusion.

The default 384-byte smoke update completes: 384 language bytes plus 384 task
bytes, 19 answer targets, 757 changed parameter tensors and finite weights.
It takes 217.3 CPU seconds and peaks at 7.07 GiB process memory. Immediate
acquisition stays at 25% after this single update; its checkpoint is not
validation selected. [Measurements](experiments/text_regions/results/content_credit/summary.json).

Use the [connected continuation protocol](docs/content_credit.md) for subsequent
training. Exact content credit ends at optimizer boundaries; existing older
conditional traces retain their limited scope. The next target is ordinary
shared-stream training that establishes reliable value-conditioned short recall,
then longer-delay recall. Keep the broader architectural controls ablatable.

## Completed: signal and gradient headcheck

The November 7 audit of runtime `9f813f7` and the update-32 state-credit
checkpoint found live forward modulation and nonzero answer-only gradients in
all 757 parameter tensors. Global clipping is active: norm 3.807 becomes 1.000.
Publication gates are unsaturated, and recall/feedback interventions change outputs.

Native immediate acquisition still predicts `1` for every query (8/32 correct).
Full historical autograd detaches every 16 bytes. Explicit delayed additions
reach only 45 admission-policy and 20 payload-adapter parameters; older content
construction remains outside that credit path. A 148-byte observation derivative
is 0.0131649 with an intact graph and disconnected with 16-byte truncation, with
identical forward logits. Sixty-three relevant regression checks pass.

Prioritize broader content-level credit and reliable value-conditioned short
recall before interpreting longer-delay scores. Retain global cap 1.0, existing
activation normalization, shared language/task learning, and ablatable controls.
This diagnostic takes priority over the earlier GPU-preparation target below.
No model or trainer behavior was changed in the audit.

[Audit measurements and reproduction](experiments/text_regions/results/headcheck/README.md)

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
