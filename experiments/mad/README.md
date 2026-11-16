# MAD-style development protocol

A runnable capability harness for the current multiscale SHoEbox. It implements
deterministic tasks, independent label interpreters, difficulty sweeps, paired
training, development-only optimization selection, held-out scoring, frozen
mechanism interventions and exact checkpoint checks.

This is a **SHoEbox byte adaptation**, with additional executable tasks. Its
scores are not interchangeable with the original MAD benchmark. Sources:
[MAD-Lab](https://github.com/athms/mad-lab) and
[Poli et al., ICML 2024](https://proceedings.mlr.press/v235/poli24a.html).

## Run it

Python 3, NumPy and a C++17 compiler are required. No dataset download or GPU is
needed. Runs are serial, pinned to one available CPU where supported. BLAS is
restricted to one thread.

```bash
# Inspect the exact job count and total training-byte budget first.
python -m experiments.mad.run --preset smoke

# All nine tasks; corrected C versus two extra L2/L3 passes.
python -m experiments.mad.run --preset smoke --execute --audit-probes \
  --out /tmp/mad-smoke.json

# A focused development screen. Omitting --execute writes only a plan.
python -m experiments.mad.run --preset screen \
  --tasks recall compression pointer_chasing --settings base delay \
  --sizes 500000 1000000 --steps 0 2 --seeds 1 7 19 \
  --budget 100000 --eval-every 25000 --out /tmp/mad-plan.json

# Execute that same configuration, with frozen mechanism audits.
python -m experiments.mad.run --preset screen --execute --audit-probes \
  --tasks recall compression pointer_chasing --settings base delay \
  --sizes 500000 1000000 --steps 0 2 --seeds 1 7 19 \
  --budget 100000 --eval-every 25000 --out /tmp/mad-screen.json

python -m unittest experiments.mad.test_protocol -v
python -m experiments.mad.check_native
python -m experiments.mad.check_native --sanitize
```

The complete `screen` preset is a broad factorial research plan, not a quick
smoke test. Use task/setting/size subsets for individual questions. The runner
prints or saves a plan by default, requires `--execute` to train, and rejects an
existing output filename. A failed run keeps its completed cells with status
`running`; it is never silently reported as a completed experiment. Temporary
weights are removed when the process exits normally. Run JSON, source and
protocol hashes are the compact reusable evidence.

## Task definitions

| Task | Mechanism tested | Adjustable difficulty |
|---|---|---|
| `recall` | Multi-query retrieval of episode-specific key/value mappings | Keys, vocabulary, delay |
| `noisy_recall` | The same retrieval amid irrelevant tokens | Noise, keys, delay |
| `fuzzy_recall` | Retrieval with variable-length multi-byte keys and values | Maximum widths, vocabulary, keys |
| `selective_copy` | Copy marked values in their original order | Relevant length, noise, delay |
| `compression` | Recover a sequence through one 64-float L3 state | Sequence length, vocabulary, delay |
| `memorization` | Learn a fixed arbitrary key/value table in weights | Vocabulary/table size, examples |
| `variable_tracking` | Execute set/copy updates over four registers | Operations, vocabulary, delay |
| `composition` | Execute copy, modular add and xor compositions | Operations, vocabulary, delay |
| `pointer_chasing` | Follow an episode-specific directed cycle | Nodes, hop depth, noise, delay |

The base uses eight values and four items/operations. Noise defaults to 0.5 for
noisy recall and selective copy. `protocol.json` defines one-variable changes:
length 8, vocabulary 16, delay 128, noise 0.8, or only 64 training examples.
Fuzzy widths and pointer depth are explicit `TaskSpec` fields for further
registered settings. A noise setting is omitted for compression/memorization,
where that switch would not change the generator. The noise parameter controls
geometric insertion before relevant records; alignment padding is separate.

Each answer delimiter completes an actual 64-byte constituent, so even the
smallest problems exercise L3. Alignment padding is visible and fully charged
to the byte budget. The default sizes are **500K, 1M and 5M total fitted
parameters**. Sparse bank quantization gives slightly smaller actual totals;
the report records them. This four-level topology cannot fit every 100K budget.

## Causality, splits and scoring

Every episode has a byte stream and a separate answer mask. The encoder sees
only the byte stream. Native byte supervision is applied only at masked answer
positions; native local predictive/reconstruction objectives remain active at
their configured weights. Query positions, padding and distractors are excluded
from answer loss. The scoring code never performs a symbolic lookup for the
model. An independent parser/interpreter verifies generator labels.

Train/dev/test use SHA-256 folds (60/20/20) of the problem input, excluding scored
answers. Literal-free instruction/argument structures determine program folds.
Compression folds use the source sequence, excluding padding/delay. Duplicates
are rejected. Fixed-table memorization deliberately shares facts across splits;
its query contexts are disjoint, and the training set covers every key. The map
seed is distinct from model and dataset seeds. Every control/challenger receives
identical generated datasets, finite training-set order and input-byte budgets.

Primary task accuracy and task NLL restrict predictions to the declared value
alphabet. Native 256-byte accuracy/NLL are recorded separately. Native tasks
also receive greedy rollout evaluation: predictions replace earlier answer
tokens, with both restricted and full-byte decoding. Exact-answer accuracy
requires every scored token in the episode to be correct. Structural tokens
are supplied by the fixture; this is not a free-text parsing or EOS-generation
test. Development curves use teacher-forced metrics; final test reports include
rollouts. A random value guess scores `1 / vocab` per answer token.

Compression uses a 64→128→(positions × vocabulary) tanh MLP. It sees only the
single final L3 state; all reconstructed positions are predicted simultaneously,
with no answer feedback, skip path or retained-memory access by the decoder.
Mean task CE trains the head and supplies a native L3 state gradient before
retirement. All readout parameters are deducted from the requested core budget
and counted in the actual total. The ordinary runtime sources and checkpoint
format are unchanged; `.mad` envelopes include the native model and readout.

## Separate capacity, state and computation

Report total/core/readout parameters and the actual expert configuration when
adding a new adapter. The existing adapter records recurrent vectors, MinKron
matrices, allocated address matrices, retained records, child buffers, retained
slot count and loop training-tape allocation. These are explicit component
inventories, not a claim of total process RSS. L0 has an allocated address matrix
that its current computation does not use.

All current controls have four 64-wide recurrent states and 16 retained slots
per level. Increasing expert-bank parameters does **not** enlarge those states.
Loop controls share parameters and persistent state capacity; their temporary
training tapes and executed work are reported separately. Compression's exposed
decoder bottleneck is exactly 256 bytes. The runner rejects paired arms with
different recorded parameter or state capacities. A future wider-state or
additional-slot adapter must form a separate state-capacity experiment.

Loop depth means **extra** passes at L2/L3: 0, 1, 2 or 4. L1 and byte-scale loops
remain zero in this protocol. Fixed-budget comparisons disable convergence
stopping. `--audit-probes` tests 0/1/2/4 passes on each selected checkpoint,
removes addressed and retained memory reads together, and separately cycles
active L2 or L3 expert transformations while leaving router keys fixed. The
number of permuted active experts is recorded; fewer than two is not an effective
permutation. Compression additionally receives zero-bottleneck replacement.
An independent longer-delay test is generated without changing decoder shape.

## Selection and decision rules

1. Start with corrected C at several difficulty settings for each parameter
   budget. Use development chance-normalized accuracy
   `(accuracy - 1/vocab) / (1 - 1/vocab)` to identify a learnable, unsaturated
   setting in [0.1, 0.9], preferring the setting closest to 0.5. The selection is
   recorded separately for each task/size. A smoke run or fewer than three seeds
   or 128 development cases produces only provisional calibration. No suitable
   setting is reported as `uncalibrated`; reduce the problem or increase training.
2. Freeze the task setting, seeds, byte budget and state/parameter inventory for
   paired comparisons. Every architecture gets the same search budget. The
   default 3×2 grid varies a shared base-LR multiplier (0.5/1/2) and native
   auxiliary weight (0/0.1), which are supported by this local learner. This is
   an adaptation of MAD's optimizer sweep, not its AdamW LR/weight-decay grid.
3. Select each arm's hyperparameters by **mean final development task NLL across
   seeds**. Require identical seed coverage. Only selected configurations are
   evaluated on test; test metrics never select hyperparameters. Report total
   tuning CPU time as well as the selected fit's time. Macro scores average
   settings within tasks, then tasks equally.
4. Compare held-out answer accuracy/NLL at matched input bytes, bytes per CPU
   second, and development loss at the last **actually measured** checkpoint
   within the common training-time budget. Record the exact byte counts of these
   time comparisons; there is no interpolated or invented test checkpoint.
   Record time/bytes to two consecutive development checks at 90% accuracy, or
   null when the threshold is not reached. Plot curves from the stored points
   for longer studies; a last measurement is not an estimated asymptotic floor.
5. Use paired seed-level accuracy differences and bootstrap intervals (10,000
   resamples; no interval for a single seed). Require calibrated multi-seed
   held-out improvement, acceptable compute cost and meaningful frozen
   interventions before promoting an architectural mechanism. Also run the
   existing matched Tiny Shakespeare/FineWeb language evaluation as a separate
   guardrail. Synthetic results alone do not automatically change defaults.

Training CPU/wall timers include reset, encoder/readout learning and final
retirement. They exclude compilation, initialization, dataset generation,
validation, diagnostic inspection and checkpoint I/O. Inference timers include
clone/reset and scoring; for native tasks final scoring performs teacher-forced,
restricted-rollout and full-byte-rollout passes, so it is an evaluation cost,
not a single-generation throughput claim. All observed input bytes, native
supervised targets, native events and extra refinement passes are recorded.

## Extending the protocol

Keep `tasks.py` and split seeds fixed while changing the mechanism. `BenchModel`
in `backend.py` is the adapter boundary: train an episode under an exact input
budget, clone for frozen evaluation, score, inventory, and serialize/restore.
Add candidate policies there and register explicit paired cells in `run.py`.
Preserve the corrected-C control, equal search budget, split hashes and actual
state/parameter accounting. A change in decoder, label access or memory budget
needs its own control. The earlier final-only/head-only/state-auxiliary study
remains separately reproducible under `experiments/structured_state`.

The new native bridge adds no fields to production `Model`; it only performs
masked teaching, state extraction/credit, frozen scoring and memory inspection.
Unit/native tests cover ignored-label invariance, full-mask public-runtime
parity, label verification, split isolation, readout finite differences,
causality, readout bottleneck isolation, live checkpoint continuation and frozen
source preservation. ASan/UBSan runs disable LeakSanitizer on this managed host.
