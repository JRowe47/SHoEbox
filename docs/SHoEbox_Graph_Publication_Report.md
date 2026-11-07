# SHoEbox: unified regional learning

2026-11-07 | Continuation of the regional-recall milestone

The ordinary seven-node regional graph now learns when to publish, retain state,
and adapt retained payloads. Natural language and memory tasks use the same
regional experts, byte interface, recall slots and learning path. Downstream
read demand becomes a trainable input to the next round's publication decisions.

## Completed learning run

| Selected model | Update | Language NLL | Answer NLL | Exact NLL | Long exact NLL |
|---|---:|---:|---:|---:|---:|
| Delayed credit | 160 | 3.4339 | 2.1084 | 1.8644 | 1.9450 |
| No delayed credit | 160 | 3.5689 | 2.5569 | 2.2291 | 2.2378 |

Lower NLL is better. Language uses 1,024 held-out byte
targets; task test uses 80 answer targets in
8 fresh documents. Long-delay documents triple the configured
delays. Exact answers are a subset of the combined answer score.
The credit arm's exact-answer accuracy is 40.7%
on the ordinary test and 10.3%
on the long-delay test.

The credit arm's validation language NLL moved from 4.2689
to 3.4429; answer NLL moved from
5.7476 to 2.1024.
At the selected checkpoints, adding delayed credit changed test language NLL by
-0.1350 and answer NLL by -0.4485 versus the matched arm.
This comparison jointly switches delayed gate and payload gradients; live-window
gradients remain enabled in both arms.

Continue tracking language and answer metrics separately as mixed-data exposure increases.

## What is integrated

- A/B/C/U/V learn independent write-or-hold policies from candidate content,
  stored content and preceding-round demand. P/Q continue publishing readout outputs.
- Holds preserve publication contents, identity and provenance. Optional state
  holds also preserve contextualizer and recurrent contents; execution clocks advance.
- Receiver demand aggregates mean and peak soft selection probability across
  valid readers. It cannot affect a writer earlier in the same round.
- Bounded owner/version/epoch-bound sensitivities carry conditional gate and
  low-rank payload credit across training cuts. No replay buffer is introduced.
- Language and memory sources share all trainable weights but retain separate
  runtime state, demand, credit and stream cursors.

## Matched exposure per arm

| Item | Value |
|---|---:|
| Parameters | 312,440 (729 added to the parent) |
| Regional nodes / rounds / recall slots per receiver | 7 / 2 / 7 |
| Updates / batch / byte window / gradient segment | 160 / 4 / 32 / 16 |
| Natural-text / procedural-memory bytes | 10,240 / 10,240 |
| Training answer targets | 624 |
| Expert executions, retained trajectory | 3,440,640 |
| Additional inference demand state per stream | 40 bytes |
| Training eligibility per stream | 8,128 bytes |

Both arms start from the same preceding graph weights, with fresh optimizer and
runtime state. All weights train. AdamW uses learning rate 0.002, constant
through the run. Selection minimizes the mean of validation language and answer
NLL at updates 0, 40, 80, 120 and 160. Pauses add no selection opportunities.

## Whole-rollout controls

These changes start from fresh test streams and include effects on later writes
and state. Positive changes mean the intervention worsened NLL.

| Control on selected credit model | Language delta | Answer delta |
|---|---:|---:|
| Block regional reads | +0.3813 | +1.0858 |
| Remove demand feedback | +0.0024 | -0.0001 |
| Always commit producer state | -0.0049 | +0.0004 |
| Always publish | +0.0032 | +0.0028 |
| Pool recall roles | +0.0045 | -0.0004 |
| Remove both feedback paths | +0.0043 | +0.0002 |

## Learned publication behavior

| Owner | Writes per execution | Mean demand | Peak demand |
|---|---:|---:|---:|
| A | 83.6% | 0.395 | 0.709 |
| B | 85.5% | 0.567 | 0.950 |
| C | 69.9% | 0.373 | 0.843 |
| U | 82.4% | 0.333 | 0.514 |
| V | 99.6% | 0.211 | 0.420 |

---

## Retention and next steps

At answer queries, selected valid publications averaged
0.18 bytes old;
0.0% were older than the
16-byte gradient segment. These measurements cover
selected edge reads across all receivers and rounds; they do not assign a
stored publication to a particular fact.

The learned demand projection norm is 0.0921;
the memory-feature norm is 0.4385;
the payload-adapter coefficient norm is 0.0768.
The training archive records delayed-gradient norms separately for both sources.

## Verification and interpretation

The full five-suite run passed 543 checks.
Targeted numerical and evaluation-cache checks bring the total to
547 unique checks. All 19 mixed/graph regressions passed
after the clipping update, followed by the evaluation-cache check.
Fourteen new integration checks cover causal demand, exact hard holds, bootstrap and resets,
multi-round recording, owner/version/epoch rejection, finite task gradients,
conditional gradient equivalence, single-use receipts, legacy behavior,
source isolation, and exact optimizer/state/credit/cursor resume.
Evaluation checkpoints each finished measurement and reuses results for identical weights.

The delayed estimator holds historical candidates, feature/demand networks and
cross-region candidate histories fixed. It restores the documented conditional
publication derivative; it does not reconstruct the full recurrent history.
Gate eligibility decays by 0.995 per active regional round.

Training also gained overflow-safe gradient clipping. It retains the preceding
FP32 arithmetic whenever the norm is finite, otherwise checks every gradient
entry and computes the norm in FP64 before scaling. Actual nonfinite entries
still stop the update. Recovery counts are included in the checkpoint totals.
The control resumed from update 120 after the original interruption; discarded
work is excluded from the retained-trajectory counts and adds no selection opportunity.

The architectural gap is closed: the general graph can now train useful
publication behavior with task feedback. The next pass should balance language
and answer learning, increase mixed-data exposure and inspect selective retention
under broader language contexts. Keep the switches for state commits, demand, role
composition and delayed credit. Code comments explicitly preserve the intended
coordination as a potentially threshold-dependent emergent feature for later ablation.

## Reproduce and inspect

The archive contains source, the exact corpus and parent, selected/final
checkpoints, optimizer/runtime/credit states, source cursors, query traces,
metrics, complete test output and a hash manifest.

```bash
cd SHoEbox
python scripts/run_graph_publication.py --parent ../graph_publication/parent.pt \
  --data data/tiny_shakespeare.txt --out artifacts/graph_publication --parallel
python scripts/check_all.py
python scripts/summarize_graph_publication.py
```

Architecture and controls: `docs/graph_publication.md`.
Machine-readable results: `experiments/text_regions/results/graph_publication/summary.json`.
