# Acquired memory, retention and competing facts

## Decision fixed before this phase's held-out evaluation

Start from `da00d66` and the selected acquisition update-96 checkpoint (97.7%
immediate recall in the previous phase). Keep all 312,440 model parameters,
learned controls and inference paths unchanged. GPU preparation, scaling and
higher-order tasks remain deferred.

1. Continue retention for 48 updates, carrying optimizer moments, language
   state/cursor, RNG and the next document index. Use the existing joint loss,
   learning rate 0.003, global norm cap 1.0 and cyclic natural-text insertions
   0/4/8/16/32. Select minimum mean validation recall NLL at updates 0/16/32/48.
2. Starting from that selection, run a 48-update selective-recall continuation
   with balanced value assignments, queries of either of two facts, and
   corrections. Preserve single-fact examples and ordinary language training.
   Freeze the exact generator and selection rule before that run begins.
3. Evaluate on fresh test data seed 917 and document indices starting at
   20,000. Compare the acquisition parent, new retention selection and selective
   selection on identical cases. Report every value separately, complete
   balanced sets, old/new fact queries, corrections and natural-language NLL.
   Delay tests use added natural-text lengths 0/8/24/64/128. Use 32 base cases
   per single-fact gap and eight full ordinary documents per stream cohort.

Training labels and provenance remain outside model inputs. Keep full-prefix
gradients; do not force memory writes, disable fading, freeze a backbone, or
replay training examples. Use fresh base document indices for each update.

These are sequential training phases, not matched-compute architecture
ablations. Report the additional training budgets and selected checkpoint
steps explicitly. Held-out results do not select checkpoints.

Before the first retention validation checkpoint, early training showed large
updates disrupting acquired distinctions (preclip norms up to 72.8). Add a
matched retention arm at learning rate 0.001; retain the 0.003 arm unchanged.
Both run 48 updates with identical inputs and parent state. Select the lower
validation NLL across both selected checkpoints; on an exact tie retain the
0.003 arm. Only that checkpoint feeds the selective continuation. This choice
uses validation, not this phase's held-out results. The selective phase keeps
its declared learning rate 0.003. Preserve both retention arms and their tests.

## Retention command

```bash
python scripts/run_memory_acquisition.py \
  --parent artifacts/memory_selectivity/acquired_parent.pt \
  --out artifacts/memory_selectivity/retention \
  --lr .003 --steps 48 --gaps 0 4 8 16 32 --carry-cursor
python scripts/run_memory_acquisition.py \
  --parent artifacts/memory_selectivity/acquired_parent.pt \
  --out artifacts/memory_selectivity/retention_low \
  --lr .001 --steps 48 --gaps 0 4 8 16 32 --carry-cursor
```

## Selective continuation fixed before its run

Cycle through `single`, `older`, `newer`, `correction`, independently cycling
through gaps 0/4/8/16/32. Each update supervises 16 answers. Single-fact updates
use four fresh ordinary acquisition prefixes. Other updates use one fresh
two-write template with the complete 4 x 4 target/nuisance value product. Both
write orders, entity identities and value assignments are represented by
randomly chosen distinct entity pairs. Corrections repeat the same key; equal
old/new values are retained in the balanced product.

These compact two-write episodes use the existing `A x=0;B x=1;A x?` grammar.
They are controlled selection exercises; the complete ordinary task stream
remains a separate transfer test. No answer byte follows the question in the
input. Mode, nuisance labels, source positions and source versions are used
only by data generation and scoring. They never enter the model.

Keep the retention phase's learning rate, losses, clipping and language batch.
Carry its selected checkpoint's weights, optimizer, language state/cursor,
RNG and document cursor. Twelve updates per mode give 192 supervised targets
per mode, 768 total, alongside 6,144 ordinary language bytes. The validation
bank starts at index 1,000, seed 683, and has four fresh templates per mode at
gaps 0 and 32. Selection minimizes the unweighted mean NLL of these eight
cohorts at updates 0/16/32/48, with earlier ties winning. Single cohorts contain
16 answers each; two-write cohorts contain 64 each (416 validation answers).

The held-out controlled two-write tests use 16 templates per mode per gap
(256 answers each) at added lengths 0/24/64, seed 917, index 20,000. For older
and newer queries, additionally require *both* questions to be answered on the
same unchanged history, excluding equal-valued facts: 192 paired tests per gap.
This specifically detects a policy that always returns the latest value.

```bash
python scripts/select_retention_parent.py
python scripts/run_memory_selectivity.py \
  --parent artifacts/memory_selectivity/retention_parent.pt \
  --out artifacts/memory_selectivity/selective
python scripts/evaluate_memory_selectivity.py \
  --checkpoint artifacts/memory_selectivity/acquired_parent.pt \
  --out artifacts/memory_selectivity/parent_evaluation.json
python scripts/evaluate_memory_selectivity.py \
  --checkpoint artifacts/memory_selectivity/retention/best.pt \
  --out artifacts/memory_selectivity/retention/evaluation.json
python scripts/evaluate_memory_selectivity.py \
  --checkpoint artifacts/memory_selectivity/selective/best.pt \
  --out artifacts/memory_selectivity/selective/evaluation.json
```

`scripts/evaluate_memory_selectivity_parallel.py` accepts the same required
checkpoint/output arguments and runs the declared default cohorts in four
independent CPU workers. The complete parent evaluation, including query rows
and full-stream traces, agrees exactly with serial execution. Cohort identity
checks guard the join. This changes evaluation scheduling only.
