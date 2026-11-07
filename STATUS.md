# SHoEbox handoff

## Completed: unified regional publication and learning

Base: `6e23e7a83b6bfee06c248c53b94a2e2696a0ce40`.
The ordinary regional graph now supports task-trained publication admission,
producer-state holds, low-rank payload adaptation, and conditional delayed credit.
A/B/C/U/V use candidate content, stored content and preceding-round downstream
read demand. P/Q retain always-published readout outputs. Defaults remain compatible.

`GraphMixedRun` shares the prior mixed trainer's scheduling and checkpoint code.
Natural text and task documents use one graph and shared trainable weights, with
separate runtime state, demand, credit and cursors. Credit validates owner order,
publication version and document epoch. No replay buffer was introduced.

## Completed evidence

- Five-suite regression run: 543 passing checks. Three clipping checks and one
  evaluation-cache check bring the unique count to 547; all 19 affected mixed/graph
  checks passed after the clipping update. Exact resume includes optimizer, both
  cursors, demand and eligibility. Legacy behavior is compared exactly.
- Both arms finished and selected update 160. Seven nodes, seven recall slots,
  two rounds, 312,440 parameters: 729 more than the common graph parent.
- Each retained training trajectory: 10,240 natural bytes + 10,240 task bytes,
  624 answer targets, 3,440,640 expert executions. Batch 4, byte window 32,
  gradient segment 16, constant AdamW LR 0.002, seed 83, data seed 283.
- Both arms start from the preceding regional graph weights with fresh optimizer
  and runtime state. All weights train; only delayed gate/payload gradient
  application differs. Both delayed-gradient types are nonzero on both sources.

| Held-out metric | Delayed credit | No delayed credit |
|---|---:|---:|
| Natural-byte NLL, 1,024 targets | 3.433866 | 3.568910 |
| All-answer NLL, 80 targets | 2.108435 | 2.556905 |
| Exact-answer NLL, 54 targets | 1.864421 | 2.229105 |
| Exact-answer accuracy | 40.74% | 38.89% |
| Long-delay exact NLL, 58 targets | 1.945042 | 2.237804 |
| Long-delay exact accuracy | 10.34% | 10.34% |

Blocking regional reads increases credit-arm language/answer NLL by
+0.3813/+1.0858. Other whole-rollout controls have small effects in this pilot.
Selected valid publications average 0.18 bytes old; none at answer queries are
older than the 16-byte gradient segment. The result demonstrates the integrated
training path and useful reads, not long-held publication recall.

The no-credit arm encountered one finite-gradient FP32 norm overflow after
update 120. Clipping now falls back to an FP64 norm only when needed, preserving
ordinary arithmetic exactly and rejecting nonfinite entries. The control resumed
from 120 and completed normally. Counts exclude discarded work from that attempt.

Evaluation now checkpoints individual measurements and reuses results for
identical weights/configuration/data. The new cache behavior has a dedicated
resume/invalidation check. It was added while the final pilot evaluations ran.

[Report](docs/SHoEbox_Graph_Publication_Report.md) |
[PDF](docs/SHoEbox_Graph_Publication_Report.pdf) |
[architecture](docs/graph_publication.md) |
[metrics](experiments/text_regions/results/graph_publication/summary.json).
Artifacts are in `artifacts/graph_publication`; the export includes source, corpus,
common parent, selected/final checkpoints, optimizer/runtime/credit state,
query traces, complete check output, recovery record and hashes.

## Next development target

Increase mixed-language and long-delay task exposure through the unified graph.
Develop useful retention through distractors while monitoring language quality,
answer quality and task-gradient balance separately. Reuse the existing gradient
balance machinery if needed; avoid adding another separate memory model.

Keep demand feedback, state holding, payload credit and recall-role controls
independently ablatable. The pilot's small effects are not removal gates. Code
comments mark the intended coordination as potentially threshold-dependent.

## Reproduction

```bash
python scripts/run_graph_publication.py --parent /path/to/parent.pt \
  --data data/tiny_shakespeare.txt --out artifacts/graph_publication --parallel
python scripts/check_all.py
python scripts/summarize_graph_publication.py
python scripts/report_graph_publication.py
```

The paired runner resumes existing checkpoints and skips completed arms.
Unscheduled pauses preserve the selection schedule. Final evaluation also resumes
individual completed controls using its weight/configuration/data fingerprint.
