# SHoEbox: credit through producer state

November 6, 2026 | Continuation of `4d919d1`

Delayed learning credit now follows the existing recurrent producer state across
gradient boundaries. A later task loss can train an earlier state-commit decision
after the associated publication has been replaced. Inference and all
312,440 trainable parameters are unchanged.

## Completed continuation

Two matched arms completed 32 paired updates from the
latest direct-sum checkpoint. Both maintain the new traces; the control omits only
the new producer-state gradient. Every update trains on language and memory at
identical weights, then takes one optimizer step. Each arm processes
4,096 natural bytes, 4,096 task-stream
bytes and 203 answer targets, with
1,376,256 expert executions.

| Held-out measurement | Parent | State credit | Publication only |
|---|---:|---:|---:|
| Selected continuation update | 0 | 32 | 32 |
| Natural-byte NLL | 3.1155 | 3.1215 | 3.1248 |
| All-answer NLL | 1.8639 | 1.7732 | 1.7688 |
| Exact-answer NLL | 1.6184 | 1.6275 | 1.6205 |
| Exact-answer accuracy | 24.6% | 26.3% | 24.6% |
| Exact NLL beyond 96 bytes | 1.6097 | 1.6115 | 1.6026 |
| Exact accuracy beyond 96 bytes | 32.1% | 32.1% | 32.1% |

Lower NLL is better. Relative to publication-only credit, the new control changes
natural-byte NLL by -0.0033, answer NLL by
+0.0043, and delayed exact NLL by +0.0088.
These use independently validation-selected checkpoints from the fixed
0/16/32 schedule. The completed final checkpoints are also retained and evaluated.

Testing covers 1,024 natural-byte targets and
144 answers in eight fresh contextual documents. The separate
long-delay set contains 56 exact targets
more than 96 bytes after their source fact. Four valid exact values give a 25%
uniform-choice baseline. This short continuation evaluates the added learning
path; durable content-sensitive recall remains the development target.

## Causal recall probe

Eight paired histories each change exactly one earlier fact-value byte while
keeping the later context and query identical. Positive directional log-odds
means the answer distribution moves toward the changed fact. A second fork swaps
only paired producer-state contents just before the final query byte; a positive
NLL delta means the swap worsens the answer.

| Query-local measurement | Parent | State credit | Publication only |
|---|---:|---:|---:|
| Fact-change directional log-odds | -0.000251 | +0.000244 | +0.001594 |
| Producer-state swap NLL delta | -0.000157 | -0.000040 | -0.000645 |

Both members of a fact-change pair are answered correctly in
0.0% of parent pairs,
0.0% with state credit, and
0.0% with publication-only credit.

## Mechanics and verification

Each producer's existing contextualizer and expert-block MinKron matrices carry
bounded-size sensitivities to its admission-policy weights. The actual decay and
erase operations transport those sensitivities. Holds preserve them; replacements
can carry them forward through the recurrent computation; document resets clear
them. At a gradient boundary, the new leaf's loss gradient is contracted with the
stored prefix sensitivity once. Live-window autograd still trains every weight.

The estimator fixes historical module inputs and policy features. Historical
cross-layer, cross-region and already-detached state-to-publication derivatives
remain truncated. The corresponding fixed-input conditional gradients match
untruncated autograd in FP64. This scope is explicit in the implementation.

All 571 unique checks pass, including the existing regression
suite, ten producer-credit checks and three causal-probe checks. Coverage includes
local Jacobians, full conditional gate derivatives, replacement survival, disabled
credit equivalence, inactive rows, reset identity, backend agreement, hook cleanup,
source isolation and exact optimizer/state/credit resume. Probe checks verify the
single changed byte, independent-document equivalence and isolated state swaps.

The median delayed state-gradient norm is 0.002258 for language
and 0.001078 for memory windows. Persistent producer sensitivities
add 115,520 bytes
(112.8 KiB) per four-stream source,
independent of elapsed stream length; live graphs and boundary receipts are
additional. No replay store or new inference state is introduced.

## Execution and next step

The optional recording path recomputes local coefficients and uses per-node
contextualizers while hooks are active. In a small 2-by-8-byte CPU forward/backward
probe, the median was 2.879 seconds with state credit
versus 2.051 seconds with publication credit
alone (1.40x, three measured repeats). This probe ran
alongside the study and checks. The original trainer retains its cheaper path.

Keep the feature independently ablatable for larger shared-stream runs. Next,
test whether the delayed state responsibility becomes useful content retention
with greater exposure, and reuse recurrent kernel intermediates before scaling
the optional trace path. No separate higher-order-memory curriculum was added.

## Reproduce

Run `python scripts/run_state_credit.py --parent artifacts/state_credit/parent.pt
--out artifacts/reproduction`, then evaluate that directory with
`python scripts/evaluate_state_credit.py --root artifacts/reproduction`.
Run `python scripts/check_all.py` for verification. The archive contains source,
corpus, parent and selected/final checkpoints, optimizer/runtime/credit state,
logs, query traces and file hashes. Detailed protocol: `docs/state_credit.md`.
