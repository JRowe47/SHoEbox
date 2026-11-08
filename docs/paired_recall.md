# Shared-history query selection

## Completed result: retained values, weak query binding

All 64 updates completed; validation selects update 64, with paired NLL falling
from 1.651034 to 0.852152. Capacity remains 312,440 unique trainable parameters.
The continuation uses older values more, but does not reliably select according
to the query. Keep `artifacts/memory_selectivity/selective/best.pt` as the working
joint-language/retention baseline. Preserve this continuation for diagnosis.

| Fresh test | Starting parent | Paired continuation |
|---|---:|---:|
| Single fact, +64 text bytes | 128/128 (100%) | 109/128 (85.2%) |
| Single fact, +128 | 111/128 (86.7%) | 87/128 (68.0%) |
| Older competing fact, +0 | 64/256 (25%) | 118/256 (46.1%) |
| Older competing fact, +64 | 64/256 (25%) | 113/256 (44.1%) |
| Newer competing fact, +64 | 256/256 (100%) | 188/256 (73.4%) |
| Both conflicting queries, +0/+24/+64 | 0/192 at each gap | 0/192, 1/192, 1/192 |
| Changed-value correction, +64 | 192/192 (100%) | 123/192 (64.1%) |
| Natural-text NLL | 3.132340 | 3.134060 |

Changing only the question's entity changes the selected model's answer in
2/192, 1/192 and 5/192 conflicting histories at the three gaps, versus zero for
the parent. Ordinary-stream later exact accuracy stays 30/84 (35.7%); corrected
exact falls from 27/60 to 26/60, and long-delay later exact from 26/77 to 25/77.
The higher older-fact score therefore comes largely from a changed preference
between retained values, with weaker retention and revision.

Separate positional ridge readouts recover both values in 256/256 fresh state
snapshots, before and after 32 intervening text bytes, in both checkpoints.
These are diagnostic readouts, never part of inference. An entity-conditioned
probe that receives the literal query key succeeds on both conflicting facts
in 177/192 immediate parent cases and 40/192 delayed cases. After training these
counts are 168/192 and 50/192. Content persists; entity-conditioned use remains
the bottleneck. Source-entity, source-value and query-entity gradients are nonzero.

The phase consumes 1,536 answer targets (1,024 paired, 256 single, 256 correction),
46,976 logical task bytes, 8,192 natural-language bytes and 112 new documents
(660 through 771). Sharing histories executes 40,384 rather than 52,352 padded
task positions, avoiding 11,968 duplicates (22.9%). This is an execution-count
saving, not a measured wall-time speedup. No core model architecture changes.

All 16 unique targeted checks pass. Production equivalence covers 757 parameter
tensors and 275 state tensors; gradient relative L2 error is 6.46e-7. Completed
resume leaves checkpoint/history bytes unchanged, and new phases inherit the
optimizer and source cursors. Parallel argument propagation and serial/parallel
pair scores agree. Budgets, selection, corpus identity and finite weights audit
successfully. [Audited summary](../experiments/text_regions/results/paired_recall/summary.json).

Next: add an ablatable learned interaction that makes query identity control
content selection through the existing expert/recall path. Compare against this
shared-history training control from the retained parent, and require paired
correctness alongside retention/correction preservation before promotion.

Parent: memory selectivity's selected update 48 (`deed39e` source).
The parent retains isolated facts at long delays but returns the latest value
for almost every query when two different entities have been observed.

## Declared experiment

Continue all 312,440 parameters, AdamW moments, natural-language state/cursor,
RNG and next training document (660). Preserve the architecture and all existing
learned controls. No replay buffer or answer labels enter inference.

Run 64 updates at learning rate 0.003. Cycle paired, paired, single, correction
updates, with 0/4/8/16/32 added natural-text bytes. A paired update crosses all
16 value assignments, executes each two-write history once, then forks its live
state for both native byte questions. The two branch losses flow into the same
history. Single/correction anchors have 16 targets each. Every update also learns
128 natural-language bytes with weight 0.25; syntax weight is 0.1, global gradient
clip 1.0. Each update averages its own answer targets, as in the parent runner.
Total: 1,536 answer targets, 8,192 language bytes, 112 new base documents.

Select at updates 0/16/32/48/64 by minimum mean older/newer NLL at gaps 0 and 32
on four validation templates, seed 683, start 2000; earlier ties win. Track
single-fact retention, correction and language loss without selecting on tests.
Both-conflicting-queries-correct is the principal behavioral check; query changes
alone do not imply correct selection. Cross-entropy needs both correct answers.
A query-blind predictor can answer 62.5% of all balanced targets by always
choosing one stored value. Both conflicting queries must succeed to demonstrate
selection; the aggregate score and loss alone are insufficient.

Only after selection, evaluate the parent and selected weights with fresh test
seed 1031, start 30000: 32 single templates, 16 two-fact templates (192 conflicting
pairs per gap), eight documents per ordinary stream, 1,024 natural-language bytes.
Use the existing evaluator and all its standard delay cohorts. This is a
continuation experiment, not a matched-budget comparison to the previous phase.

## Signal audit

Measure counterfactual old/new-value differences in floating state after the
first write, second write and intervening text. Record answer gradients at both
source keys/values and the query key. Fit diagnostic ridge readouts on frozen
state snapshots from separate fresh templates, then test on disjoint template
indices. These probes never participate in model inference or model training.
Check whether the older value remains accessible, and whether the learned answer
path uses it. Integer provenance and clocks are excluded from probe features.
The entity-conditioned diagnostic fits independent linear coefficient sets for
each literal query key. It gets no old/new position hint; it is never installed
in the model. Separate positional readouts test preservation of each value.

## Reproduction

```bash
python -m unittest experiments.text_regions.test_paired_recall -v
python scripts/diagnose_paired_recall.py --checkpoint artifacts/memory_selectivity/selective/best.pt --out artifacts/paired_recall/parent_diagnostic.json
python scripts/run_paired_recall.py --parent artifacts/memory_selectivity/selective/best.pt --out artifacts/paired_recall/train
python scripts/evaluate_memory_selectivity_parallel.py --checkpoint artifacts/memory_selectivity/selective/best.pt --out artifacts/paired_recall/parent_test.json --data-seed 1031 --start 30000
python scripts/evaluate_memory_selectivity_parallel.py --checkpoint artifacts/paired_recall/train/best.pt --out artifacts/paired_recall/selected_test.json --data-seed 1031 --start 30000
```

Use the same corpus digest as the parent. `--resume` continues the exact saved
update, data sources, optimizer and RNG; `--stop-after` changes only the execution
stopping point inside the fixed protocol. The shared-history implementation is
checked against complete duplicated prefixes for logits, state and all gradients.
