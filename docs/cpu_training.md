# Longer CPU training runs

`shoebox.train_cpu` carries the sparse model across 32-observation windows, logs
every update, periodically validates, and writes atomic last/best checkpoints.
It counts external byte targets exactly; validation and sampling are separate.

```bash
python -m shoebox.train_cpu \
  --init artifacts/cpu_overhaul/final_learning/last.pt \
  --data data/cortical_fineweb/train.bin \
  --validation data/cortical_fineweb/validation.bin \
  --out artifacts/cpu_fineweb_100k \
  --targets 100000 --validation-targets 4000 \
  --batch 50 --context 32 --threads 2 --workers 1 \
  --checkpoint-every 8 --validation-every 16
```

The example continues the 2,048-target checkpoint's weights and AdamW moments.
It resets runtime state for the new 50-stream layout, trains 100,000 additional
targets, and records 102,048 lifetime training targets. The corpus includes the
earlier shakedown prefix, so this is a count of exposures, not distinct bytes.
Omit `--init` to initialize a fresh CPU model.

The learning rate warms from 0.0005 to 0.001 over the first 3,200 targets, then
remains constant. The example takes 63 optimizer updates: 62 full B50/T32 windows
and one B50/T16 window. Validation uses 4,000 held-out targets at the start,
25,600, 51,200, 76,800 and 100,000 targets. Last checkpoints occur every 12,800
targets and at completion; `best.pt` tracks the lowest validation NLL.

For exact continuation, use `--resume` with identical data, arguments, code and
runtime and choose a new output directory:

```bash
python -m shoebox.train_cpu \
  --resume artifacts/cpu_fineweb_100k/last.pt \
  --data data/cortical_fineweb/train.bin \
  --validation data/cortical_fineweb/validation.bin \
  --out artifacts/cpu_fineweb_100k_resumed \
  --targets 100000 --validation-targets 4000 \
  --batch 50 --context 32 --threads 2 --workers 1 \
  --checkpoint-every 8 --validation-every 16
```

`--max-updates N` stops an invocation after N additional updates and saves an
exact resumable checkpoint. `--init` carries weights/optimizer into new streams;
`--resume` also restores stream state, cursor, history and RNG. A focused check
verifies exact weights, optimizer, runtime state and RNG after split execution,
and rejects a changed learning-rate protocol.

`results.json` and `history.jsonl` expose progress throughout the run. A results
file can be ahead of `last.pt`; `checkpoint_trained_targets` records the latest
recoverable count. Final results add port entropy/diversity, expert and regional
usage, state size and three fixed-seed text samples.

## Completed 100k run

The 100,000-target continuation finishes at 3.269114 held-out NLL, from 4.434360,
at 61.54 training bytes/s. The training-fitted unigram scores 3.256191; all greedy
held-out predictions are spaces. Swapping input streams changes NLL by -0.000103.
This endpoint remains at byte-frequency level. See
[metrics, samples, diagnostics and reload audit](cpu_training_results.json).
