# SHoEbox handoff

## Completed: paired learning through broader context

Continues published checkpoint `7f3ce944547d800e133e999046a77d4e029a1851`.
`ContextBalancedRun` trains one language window and one memory window at identical
weights before one AdamW step. Each source retains separate runtime state,
demand, delayed credit and cursor. The optional `group_clip` control bounds each
source's core/readout/recall/publication gradients before addition; `sum` directly
adds them. Both receive the common final global clip. All controls remain ablatable.

The procedural stream now supports natural-text distractors alongside updates to
competing facts. Corpus partitions stay separate. Answers and provenance remain
outside model inputs. The existing generator is byte-identical without a corpus.
The inference architecture and 312,440 parameters are unchanged.

## Completed evidence

Both arms finish 128 paired updates: 16,384 language bytes, 16,384 task-stream
bytes (including embedded natural text), 727 answer targets and 5,505,024 expert
executions per arm. Every weight trains. The parent's language cursor continues;
optimizer/runtime states start fresh. The fixed validation rule selects clipping
update 96 and sum update 128.

| Held-out measure | Starting model | Group clipping | Direct sum |
|---|---:|---:|---:|
| Natural-byte NLL | 3.367160 | 3.092152 | 3.081378 |
| Answer NLL | 2.057892 | 1.928172 | 1.856024 |
| Exact NLL beyond 96 bytes | 1.949739 | 1.680759 | 1.632492 |
| Exact accuracy beyond 96 bytes | 23.47% | 22.45% | 23.47% |

The shared held-out protocol uses 2,048 language targets, 288 ordinary answer
targets and 98 exact targets beyond 96 bytes in the long-delay split. Direct sum
wins this matched comparison. Clipping stays optional for later scale/ablation.
Language and answer loss improve, but exact recall stays near the four-value
baseline. Selected producer reads older than a 16-byte segment remain absent.
Blocking regional reads worsens answer NLL by 2.2562 for clipping and 0.3714 for
sum. Other whole-rollout publication/demand/state controls have small effects.

## Verification and recovery

558 unique checks pass: all 547 preceding checks and the complete 11-test new
suite. The driver ended during new-suite reporting without its completion marker;
the whole new suite was rerun separately and passed. The live-output example also
passed. Initial gradients and validation agree exactly between arms.

Training resumed from completed pairs 119/121 after the execution session was
interrupted. Optimizer, both source states/cursors, demand, credit and RNG restore
exactly. The validation schedule stayed fixed. Exposure counts exclude discarded
partial work. Final evaluation caches each completed measurement and distinguishes
selected weights from final weights. The archive preserves the recovery record.

## Next development target

Develop useful long-delay retention through the existing regional graph. Prioritize
exact-answer accuracy and causal sensitivity to retained fact content. Publication
age is a diagnostic, not a learning target: recent publications can carry older
recurrent information. Investigate credit across producer-state transitions and
longer task exposure while maintaining language learning. Keep publication,
state-hold, demand, payload-credit, recall-role and gradient controls independently
ablatable; their intended coordination remains potentially threshold-dependent.

[Report](docs/SHoEbox_Context_Balance_Report.md) |
[PDF](docs/SHoEbox_Context_Balance_Report.pdf) |
[protocol](docs/context_balance.md) |
[metrics](experiments/text_regions/results/context_balance/summary.json).

## Reproduce

```bash
python scripts/run_context_balance.py --parent artifacts/context_balance/parent.pt --parallel
python scripts/check_all.py
python scripts/summarize_context_balance.py
python scripts/report_context_balance.py
```

The download includes source, corpus, parent, selected/final checkpoints, full
optimizer/runtime/credit state, histories, query traces, test logs and hashes.
To rerun training from the parent, select a new output directory with `--out`.
