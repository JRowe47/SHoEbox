# Shared-history query selection

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
