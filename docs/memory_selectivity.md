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

## Completed results

All three runs finish 48 updates and select update 48. The matched retention
comparison chooses 0.003: validation NLL 0.055799 / accuracy 99.4%, versus
1.056687 / 47.5% at 0.001. The selective continuation's mean validation NLL
falls from 1.154259 to 0.859912, while older-fact accuracy stays at 25%.

| Held-out measurement | Acquired parent | Retention 0.003 | Selective continuation |
| --- | ---: | ---: | ---: |
| Single fact, +0 bytes | 97.7% | 100.0% | 100.0% |
| Single fact, +8 bytes | 28.1% | 100.0% | 100.0% |
| Single fact, +24 bytes | 21.1% | 97.7% | 100.0% |
| Single fact, +64 bytes | 8.6% | 90.6% | 98.4% |
| Single fact, +128 bytes | 16.4% | 57.0% | 76.6% |
| Older competing fact, +0 bytes | 25.0% | 25.0% | 25.0% |
| Both conflicting queries correct, +0 bytes | 0/192 | 0/192 | 0/192 |
| Changed-value correction, +64 bytes | 11.5% | 94.8% | 100.0% |
| Natural-text NLL | 3.126 | 3.421 | 3.132 |

Each single-fact row contains 128 answers. The final model passes 30/32 complete
four-value sets at +64 and 14/32 at +128. At +64, values 0/2/3 each score 32/32;
value 1 scores 30/32, with two 1-to-0 mistakes. The previous systematic collapse
is resolved. These fresh test cohorts differ from the preceding phase's tests.

The final older-fact score is 25% at every tested gap, with no complete sets.
Matched query-pair correctness stays 0/192 at +0/+24/+64. Changing only the
queried entity changes the prediction 0/192, 0/192 and 1/192 times, respectively.
The next target is key-specific writing and reading. The NLL improvement does
not establish successful selection between conflicting facts.

| Full ordinary stream | Acquired parent | Retention 0.003 | Selective continuation |
| --- | ---: | ---: | ---: |
| Later exact queries (82 targets) | 29.3% | 35.4% | 36.6% |
| Corrected exact queries (64 targets) | 32.8% | 40.6% | 42.2% |
| Long-delay later exact queries (84 targets) | 23.8% | 29.8% | 28.6% |

The ordinary and long-delay cohorts each use eight full documents. Corrected
and later-exact cohorts overlap. Natural-text NLL uses 1,024 held-out bytes.
Controlled changed-value correction excludes equal old/new values and uses
192 targets. Neither controlled retention nor correction substitutes for
selective recall in the full competing-write stream.

The lower-rate control reaches 98.4% immediate recall but only 26.6% at +64 and
+128; its natural-text NLL is 3.341. Its complete results remain in the audit.
Each retention arm uses 768 answer targets, 34,944 task bytes and 6,144 language
bytes. Selective continuation uses 768 targets (192 per mode), 24,576 task bytes
and 6,144 language bytes. Architecture and inference remain unchanged.

All 15 targeted checks pass; exact resume, subsequent phase continuation and
the latest-value-shortcut control are verified. Completed production resume
preserves selected and final checkpoint bytes. Selection/budgets reconstruct,
all checkpoint weights are finite, and serial/parallel evaluation agrees exactly.

```bash
python scripts/summarize_memory_selectivity.py
python scripts/run_memory_selectivity.py \
  --parent artifacts/memory_selectivity/selective/best.pt \
  --out artifacts/memory_selectivity/next --steps 48
```

Starting another phase preserves weights, optimizer, language state and source
positions; its declared mode/gap cycle begins anew. The next experiment should
trace key influence at writes and reads, then supervise both query directions
from the same history within one update. Existing controls remain ablatable.
