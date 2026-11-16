# Stage A: supervised upper-scale state

Research adapter for corrected C at commit `600352c`. The production runtime,
checkpoint ABI, selected defaults and previous loop implementation are unchanged.
This isolates whether L2/L3 can learn useful state before adding more thought depth.

## Three matched arms

- `final_only`: byte language loss plus queried-register answer supervision.
- `head_only`: also supervise all four registers after each initialization/update,
  with the auxiliary gradient stopped at the task readout.
- `state_aux`: the same auxiliary labels and head updates, with gradients entering
  L2/L3 native states and their existing recurrent/content learning paths.

Every arm has the same 4,128-parameter linear softmax readout over the concatenated
64-wide L2 and L3 states. The four readout groups correspond to register indices;
the evaluator supplies the queried index. This is a supervised capability probe,
not a claim that the byte generator can parse an arbitrary question. The native
byte answer is measured separately. No answer value is supplied to inference.

The auxiliary loss is mean CE over the four register values. Its weight is 0.25.
State credit is accumulated in a bounded 8,224-byte external tape and averaged
separately over supervised events in each native retirement block. The final
answer channel has weight 1. Neither kind of label advances an input clock,
changes an already computed state, or calls native `teach()` again. All forward
features use only observed bytes. Exact resume includes pending auxiliary credit.

`adapter.py` compiles a private translation unit using two checked source anchors:
add task credit to the local native state VJP and clear that credit on retirement.
The public source files are never edited. A source change that invalidates an
anchor aborts compilation. The native regression checks compare the unlabelled
adapter with the public C runtime exactly.

## Verified curriculum

Four registers, eight values. Tracking uses set/copy/swap; composition uses
copy/modulo-8-add/xor. An independent parser/interpreter checks every evaluation
label. Training programs have 1–4 operations. Fixed 64-byte, left-padded ASCII
records align targets with real completed L3 constituents. A final query record
is followed by its answer byte. Training retains next-byte prediction over every
real target; intermediate state labels are separate from the byte stream.

A SHA-256 hash of the instruction/argument structure, including the query but
excluding literal values, assigns structural folds: buckets 2–9 train, 0 dev,
1 audit. No audit structure occurs in training. Training and evaluation have
independent fixed seeds. Further audits use 8/12 operations and unseen E–H names.

Training interleaves roughly 80% verified-program bytes and 20% the first 90% of
Tiny Shakespeare; the final 10% is held out. Padding counts as training bytes.
Models see the same episodes, labels, byte counts and optimizer schedule. The
linear shared learning-rate anchor moves from 1 to 0.1 over the byte budget, with
route oscillation disabled. Extra targets do not advance it.

The head-only arm controls for extra readout training. Frozen mean replacement
uses a mean estimated from separate **training-fold** examples. Active-expert
permutations cycle the transformations among experts selected during training,
leaving router keys fixed. Three permutation seeds are used. These are joint
L2/L3 interventions. They measure dependency; a loss change alone does not prove
that the dependency improves accuracy.

## Reproduce

```bash
python -m unittest experiments.structured_state.test_adapter -v
python -m experiments.structured_state.check_native
python -m experiments.structured_state.check_native --sanitize
python -m experiments.structured_state.run \
  --data /path/to/tinyshakespeare.txt \
  --out /tmp/state_screen.json --checkpoint-dir /tmp/state_checkpoints \
  --budget 2000000 --sizes 1000000 5000000 --seeds 7
```

NumPy, Python 3, and a C++17 compiler are required. Builds use O3, native CPU
instructions, and disabled floating-point contraction, matching the public
runtime. Sanitizers disable LeakSanitizer on this host and enable ASan/UBSan.
Checkpoints, corpora and binaries are excluded from git and the source attachment.
The `.task` checkpoint is explicitly a research envelope containing the native
model and task probe; use `Adapter.loads`, not the production checkpoint loader.

Training timers include Python curriculum generation, context resets, supervision
and final retirement; exclude compilation, validation and checkpoint I/O. Frozen
inference timings include clone/reset and prediction. Serial single-core runs are
used. Equal-time summaries interpolate the saved development curve at the fastest
arm's completion time; they are estimates, not extra trained checkpoints.

## Recorded screen

[Results and interpretation](RESULTS.md), [full curves and predictions](results.json),
[frozen probes](probes.json), [summary](summary.json), and
[verification](verification.txt). The report renderer includes the interpretation
of this recorded screen; review that text when running a different experiment.

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.structured_state.diagnose \
  --checkpoints /tmp/state_checkpoints --out /tmp/state_probes.json
python -m experiments.structured_state.report \
  --results experiments/structured_state/results.json \
  --probes experiments/structured_state/probes.json --out /tmp/state_report
```
