# Connected recall continuation

This phase continues the selected update-32 state-credit parent from the
content-credit repair at `614033e`. The first matched continuation targets
value-conditioned recall in the ordinary shared language/task stream.

## Fixed comparison

| Setting | Both arms |
| --- | --- |
| Updates | 16 |
| Batch / optimizer window | 4 streams / 128 bytes |
| Execution chunks | 16 bytes |
| Each source per update | 512 external bytes |
| Each source, complete run | 8,192 bytes |
| Initialization | Identical parent weights and AdamW moments |
| Learning rate / global clip | 0.001 / 1.0 |
| Training seed / data seed | 97 / 583 |
| Validation | Updates 0, 8, 16; mean language/answer NLL |
| Task grammar | Existing four entities, corrections, natural distractors and queries |

The connected arm preserves exact content gradients across execution chunks.
The control detaches at every 16-byte chunk and retains the previous conditional
credit behavior. Both maintain independent language/task runtime state, compute
their source gradients at identical weights, sum, clip and update together.
The 128-byte window is an explicit first continuation setting; it does not
change the repair trainer's 384-byte default. No new curriculum or replay is
introduced. Every completed optimizer update saves `last.pt`; validation alone
selects `best.pt`, including the unchanged parent as a candidate.

## Available credit and actual recall

Each task answer records whether its observed value falls inside the actual
optimizer window and inside the actual 16-byte segment. The audit uses document
IDs, source positions and row resets. A nominal delay shorter than the window
does not guarantee a connected source. The connected arm's exact coverage is
`within_window`; the control's is `within_segment`. Conditional older traces
remain in both arms and are not counted as exact content coverage.

Held-out evaluation uses 1,024 natural-text targets and eight documents in each
of the ordinary and longer-delay task conditions. A separate balanced causal
test makes four versions of each query prefix, changing only the earlier
observed value to 0, 1, 2 or 3. The model receives ordinary bytes; provenance and
expected answers stay in the evaluator. Native immediate queries and the first
exact query beyond 96 bytes in each long-delay document are scored separately.

Report full-vocabulary accuracy/NLL, the fraction of four-value sets answered
entirely correctly, the fraction that changes its answer, and a balanced logit
margin. A constant-answer predictor scores 25% accuracy, zero complete sets and
zero answer switches. Its balanced margin is zero apart from rounding. No
held-out result selects a checkpoint. Selected and fixed-budget final results
are retained separately, and identical weights share an evaluation cache.

## Optional activation recomputation

`--checkpoint-segments` flattens the complete graph-state tensor boundary and
recomputes each segment during `loss.backward()`. It does not shorten the
gradient window. Zero-size structural templates avoid retaining live internal
activations. Each execution creates its own producer recorder, preventing
duplicate trace advancement or stale hooks. Conditional traces are detached;
historical credit is still applied only at the optimizer-window entry.

The option is off by default. The recorded paired continuation uses ordinary
activation storage. Checkpointed training uses `loss.backward()`; derivative
probes using `autograd.grad` use the ordinary path. Inference is unchanged.
Training arithmetic stays selected during the no-grad checkpoint forward,
including large expert banks and batched contextualizers. Recompute and
ordinary training can accumulate parameter gradients in a different order;
tests compare their derivatives within floating-point tolerance and require
exact resume within each mode.

## Run and recover

```bash
python scripts/run_connected_recall.py --arm connected --out artifacts/connected_recall/connected
python scripts/run_connected_recall.py --arm cut_16 --out artifacts/connected_recall/cut_16
python scripts/evaluate_connected_recall.py
# Resume the same fixed protocol after an interruption:
python scripts/run_connected_recall.py --arm connected --out artifacts/connected_recall/connected --resume
# Measure one optional recomputed update in an isolated process:
python scripts/benchmark_content_checkpoint.py --horizon 64 --batch 4 \
  --checkpoint-segments --out artifacts/checkpoint_benchmark
```

Run the two training commands sequentially on machines with limited RAM.
`--stop-after` pauses without adding a validation opportunity. Changing a
protocol setting requires a new continuation; exact recovery enforces the
saved settings, corpus, optimizer, cursors, runtime state and conditional traces.
