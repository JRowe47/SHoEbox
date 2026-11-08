# CPU execution and learned regional routing

`cortical_cpu()` adds 12,715 router parameters to the widened model, for
14,989,597 total parameters. Regional widths, expert banks, complete 4/8/16-byte
ports and the two-round protocol retain their previous capacities. The training
runner defaults to a 32-observation gradient window.

## Actual conditional execution

A1 and B1 observe each input byte. Their current latents and Y's preceding latent
form a routing query. A learned top-k selector chooses three optional activation
groups independently for each stream. A1, B1, S2 and Y form an always-active path.
The optional EC-C/EC-X pair is one group, keeping episodic content/context writes
aligned. Each stream therefore selects seven or eight of the 18 regions.

Each active destination learns a top-two selection among its permitted incoming
edges. An edge includes its complete port width and requested publication count;
individual bytes are never selected from within a port. At least one available
current edge is retained, and Y always includes the current S2 publication.
Previous Y output and EC hippocampal returns keep their existing special framing.

Unselected regional rows never enter their HAT, experts or output heads. Skipped
state remains live within the gradient window. Ports retain their last two real
publications independently per stream, with original events and revisions.
An active region can consume a previously published value from an inactive
source. Current-edge dependency waves preserve causality; missing startup inputs
are excluded, and a region with no available input is deferred.

Hard selections execute the graph. Straight-through strength gates carry task
gradients into region and edge scores. Training uses Gumbel exploration plus a
small balancing loss; evaluation disables exploration. Targets never enter the
router. This changes the learned model's execution policy; it is not an exact
output-preserving transformation of the dense regional schedule.

## CPU kernels and batching

- Hard byte embeddings use indexed reads and indexed weight-gradient sums while
  retaining the original soft-probability surrogate gradient.
- Different selected input recipes are grouped within a region. Variable-length
  native scans batch their real sequences; padding never decays or writes state
  and receives zero sequence gradient. The regional SHoE core runs once for the
  combined active rows. Metadata and byte projections still use padded tensor
  storage where necessary; their extra lanes are not recurrent observations.
- Batched outputs publish directly, without splitting and reassembling them.
  Complete-row replacement also bypasses redundant full-state copies.
- Native expert dot products use explicit AVX512 on supported hosts. The local
  build enables OpenMP when PyTorch uses it; without that compiler flag ATen's
  extension parallel loops would remain serial.
- `torch.set_num_threads(n)` controls intra-operation parallelism. Independent
  regional waves optionally use `set_cpu_workers(n)`. Each worker preserves grad
  mode and forward caches; publication occurs in deterministic order on the caller.

Native CPU FP32/FP64 supports first-order reverse-mode autograd. Builds are local
and keyed by source, compiler, PyTorch and CPU capability; binaries are not shipped.
Unsupported native environments retain the portable numerical path. The benchmark
runner requires a working native build so fallback cannot silently affect timings.

## Run

```python
import torch
from shoebox import CorticalModel, cortical_cpu

torch.set_num_threads(1)
model = CorticalModel(cortical_cpu()).set_execution_backend('cpu')
model.set_cpu_workers(1)
out = model(observed_bytes, targets=next_bytes)
out['loss'].backward()
state = out['state'].detach(compact=True)  # only at the TBPTT boundary
```

```bash
python scripts/check_all.py
python scripts/benchmark_cpu.py --data data/cortical_fineweb/train.bin \
  --out artifacts/cpu_sweep --repeats 3
python -m shoebox.cpu_run --mode train \
  --data data/cortical_fineweb/train.bin \
  --validation data/cortical_fineweb/validation.bin \
  --out artifacts/cpu_training --targets 2048 --batch 4 --context 32 \
  --threads 1 --workers 1
```

Tune `--region-topk` and `--edge-topk` explicitly when changing the compute budget.
`--preset wide --backend fused` supplies the dense comparison with identical
regional initialization. Existing dense presets/checkpoints keep dense scheduling.
Checkpoint loading preserves stored model precision and per-stream sparse state.

Timing is external byte positions summed across batch rows. Increasing the batch
improves aggregate throughput, not the latency of one sequential stream. Reported
forward-only timings observe a fixed byte sequence; they are not sampled decoding.
Benchmark timing excludes compilation, warmup, clipping and optimizer updates;
training timing includes forward, backward, clipping and AdamW. No profiler or
saved-tensor hooks contaminate either timing path. Wave-worker execution does not
propagate saved-tensor hooks, so use RSS rather than hook accounting for that mode.

The matched B4/T32 median improves 8.72 to 12.19 external bytes/s, with peak RSS
falling from 2,048 to 1,301 MiB. The 2,048-byte training comparison takes 8.56 versus
12.63 bytes/s including AdamW; final held-out NLL is 4.414 dense and 4.512 sparse.
See `cortical_cpu_results.json` for raw timings, batch/thread scaling, route counts,
validation and endpoint audit. The operator profile still contains thousands of
small PyTorch operations per observation; whole-region fusion is the next target.

For a larger aggregate-throughput run, `--batch 64 --targets 4096 --threads 2`
measured 69.1 bytes/s forward/backward and 239.8 bytes/s forward-only, with 5.64 GiB
peak RSS. This is one measured run. The B32 one-thread three-run median is 42.3
bytes/s; a repeated two-thread run is 41.7. Single-stream forward-only throughput
is 22.9 bytes/s. Keep these batch/latency distinctions when comparing results.
