# Balanced acquisition experiments

Memory is the current development focus. GPU preparation, broader integrated
acceptance and scaling work are deferred for this phase.

## Completed results

The graph remains at 312,440 trainable parameters. Both main lineages finish
96 optimizer updates with 1,536 answer targets and 12,288 natural-source bytes.
The retention branch adds 8,832 distractor bytes after the common update-48
parent. Both validation-selected checkpoints equal their fixed-budget finals.

| Held-out measurement | Starting model | Acquisition | Retention |
| --- | ---: | ---: | ---: |
| Immediate recall | 25.0% | 97.7% | 75.0% |
| +8 ordinary-text bytes | 25.0% | 24.2% | 75.0% |
| +24 ordinary-text bytes | 23.4% | 19.5% | 75.0% |
| +64 ordinary-text bytes | 21.1% | 16.4% | 75.0% |
| +128 ordinary-text bytes | 14.8% | 15.6% | 68.0% |
| Later exact queries, ordinary stream | 28.2% | 49.4% | 49.4% |
| Corrected exact queries, ordinary stream | 26.2% | 57.4% | 57.4% |
| Natural-text NLL, lower is better | 3.109 | 3.126 | 3.505 |

Each insertion condition has 32 balanced cases / 128 answers; the initial
source-to-query delay is 8 bytes. Acquisition passes 29/32 complete four-value
sets. Retention passes none: through +64 bytes, it identifies 0/2/3 correctly
and consistently emits 0 for stored 1. At +128 it answers 87/128 correctly.
The ordinary-stream later/corrected rows have 85/61 targets and overlap.

The harder long-delay stream with competing writes remains weak: retention
answers 29.6% of its 81 later exact queries correctly. Focused exact-value
training weakens semantic/context answers. Overall ordinary-task answer NLL
is 1.643 initially, 2.985 after acquisition and 2.707 after retention, despite
the higher exact-answer accuracy. The memory-only control finishes its planned
32-update screen; its final validation answer/language NLL is 1.993/5.331,
versus 1.138/3.353 for the joint run at the same update.

All four data-integrity checks pass. A restored gap-zero branch exactly matches
uninterrupted update 49 in weights, optimizer, language state/cursor, RNG and
document index. Training budgets, checkpoint selection and finite weights are
audited by `scripts/summarize_memory_acquisition.py`. Raw case predictions,
protocols and histories accompany the resumable checkpoints.

Next: start retention from the fully acquired update-96 checkpoint, then
develop selective recall among competing facts and corrections. The present
retention branch started at update 48, before all four values were separated.

[Audited summary](../experiments/text_regions/results/memory_acquisition/summary.json)

## Stage A: test value acquisition

Start both learning-rate pilots from the selected connected-recall checkpoint
at update 16. Transfer the unchanged 312,440-parameter graph and AdamW moments.
Use learning rates 0.001 and 0.003, global clipping at 1.0, four new base
documents per update and all four observed-value versions of each chosen
acquisition prefix (16 answer targets). Choose the native immediate question
round-robin across A/B/C/D. Only its earlier value byte differs across versions.
Answer bytes and provenance never enter the model. Train all parameters with
exact gradients through the complete prefix. Publication and recall policies
remain learned and unconstrained.

Each update also consumes 128 ordinary language bytes from an independent
carried stream. The objective is answer CE + 0.1 syntax CE + 0.25 language loss,
with existing routing auxiliary terms. Every base document has a new index;
finite grammar can naturally produce identical byte prefixes. No replay buffer
or deliberately higher-order curriculum is introduced.

First screen: 16 updates per learning rate. Validation is fixed at update 0
and every 16 updates, using 16 balanced cases / 64 answer targets. Selection
minimizes validation balanced-acquisition NLL, retaining earlier ties. Use this
screen to choose a continuation up to 96 updates; record the adaptive decision.
Language validation is monitored separately and does not select checkpoints.
All completed updates save resumable weights, optimizer, language state and
cursor, RNG, document counter and source totals.

The screen selected learning rate 0.003 (validation NLL 1.411768 versus
1.458243 at 0.001). A separate memory-only diagnostic starts from the same
original parent at 0.003 and sets the language weight to zero, initially for
32 updates. It otherwise uses the same bytes, masks, optimizer moments and
memory objective. This tests whether competing language gradients obstruct
acquisition; it is not a candidate replacement for joint learning by default.
The diagnostic was added using validation results only, before evaluating any
trained checkpoint on the held-out cohorts. Language forwards remain present
at zero weight so the computational path is matched.

## Stage B: learn retention through ordinary text

A validation-only probe at acquisition update 32 found that inserting 8 or
24 ordinary text bytes erased its emerging value discrimination. After the
update-48 acquisition validation, fork a retention continuation if its selected
checkpoint reaches at least 50% balanced acquisition accuracy. Train 48 further
updates with added natural-text gaps 0, 4, 8, 16 and 32 in a fixed cycle. Keep
learning rate 0.003 and language weight 0.25. Transfer the selected parent's
weights, AdamW moments, language state/cursor, RNG and next base-document index.
All prefixes still receive full content gradients; all existing gates remain
learned. No deliberately higher-order task is added.

Select this continuation by mean validation NLL across the five declared gaps,
using 16 balanced cases at each gap, at updates 0/16/32/48. The acquisition-only
run continues independently to update 96. If the fork is at update 48, the two
final lineages have equal optimizer-update counts and identical base-document
and language schedules after branching; the retention arm consumes additional
distractor bytes. The fixed held-out tests below are unchanged and are not used
to choose this training schedule.

## Held-out measurements

After selecting the training setting, evaluate independent test document
indices starting at 10,000, with data seed 683. Report 32 balanced acquisition
cases (128 answers), complete four-value sets, answer switching, and balanced
logit margin. Insert 0/8/24/64/128 held-out natural bytes before the same query
to measure delay transfer. Evaluate eight full ordinary documents and eight
long-delay documents, including corrections and semantic/context questions.
Report 1,024 held-out natural-text targets separately.

The existing general-stream task and the dense acquisition experiment have
different training distributions. A short-prefix gain is an acquisition result;
full-stream and delayed measurements establish its transfer. No test result
selects a checkpoint. A later training decision based on these results starts
a separately documented phase with fresh held-out data.

## Commands

```bash
python scripts/run_memory_acquisition.py --out artifacts/memory_acquisition/lr001 --lr .001 --steps 96 --stop-after 16
python scripts/run_memory_acquisition.py --out artifacts/memory_acquisition/lr003 --lr .003 --steps 96 --stop-after 16
# Continue to the declared fork, preserving its selected checkpoint:
python scripts/run_memory_acquisition.py --out artifacts/memory_acquisition/lr003 --lr .003 --steps 96 --stop-after 48 --resume
cp artifacts/memory_acquisition/lr003/best.pt artifacts/memory_acquisition/retention_parent.pt
# Continue acquisition and start its retention branch:
python scripts/run_memory_acquisition.py --out artifacts/memory_acquisition/lr003 --lr .003 --steps 96 --resume
python scripts/run_memory_acquisition.py --parent artifacts/memory_acquisition/retention_parent.pt --out artifacts/memory_acquisition/retention --lr .003 --steps 48 --gaps 0 4 8 16 32 --carry-cursor
# Memory-only diagnostic, ending at its declared 32-update screen:
python scripts/run_memory_acquisition.py --out artifacts/memory_acquisition/memory_only --lr .003 --language-weight 0 --steps 96 --stop-after 32
python scripts/evaluate_memory_acquisition.py --checkpoint artifacts/memory_acquisition/lr003/best.pt --out artifacts/memory_acquisition/lr003/evaluation.json
python scripts/evaluate_memory_acquisition.py --checkpoint artifacts/memory_acquisition/retention/best.pt --out artifacts/memory_acquisition/retention/evaluation.json
python scripts/summarize_memory_acquisition.py
```
