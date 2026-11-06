# SHoEbox runtime profiling

November 5, 2026. Base source: `10f87459377d5d2062c51be2c14fa8feaf67ebd0`.

The optimized runtime reduces a complete joint text-and-memory update from **14.33 to 11.59 seconds**: **1.24x throughput**, or **19.1% less time**. The trained checkpoint's frozen text outputs and recurrent state match the untouched implementation exactly. The optimized PyTorch path is now the regional engine's default.

## Matched measurements

| Model / workload | Before (s) | After (s) | Throughput gain |
| --- | ---: | ---: | ---: |
| Trained 0.51M: Text forward + backward | 4.352 | 3.359 | 1.30x |
| Trained 0.51M: Text inference | 1.394 | 1.136 | 1.23x |
| Trained 0.51M: Full joint update | 14.329 | 11.592 | 1.24x |
| Trained 0.51M: 512-episode evaluation | 8.707 | 6.817 | 1.28x |
| Default 2.56M: Text forward + backward | 4.849 | 4.076 | 1.19x |
| Default 2.56M: Text inference | 1.659 | 1.388 | 1.20x |
| Scaled 24.91M: Text forward + backward | 5.233 | 4.510 | 1.16x |
| Scaled 24.91M: Text inference | 1.799 | 1.469 | 1.22x |

Each text call processes eight streams of sixteen bytes with two regional rounds: 128 targets for training, or 128 supplied bytes for inference. Inference is a supplied-byte forward pass; it is not a sampling benchmark. A joint update includes the text backward pass, sixteen new memory episodes with a sixteen-byte gap, three field queries, task/group gradient accounting and clipping, and AdamW. Evaluation covers 512 fresh shared-correction episodes with the same gap.

Medians use three measured pairs after one warmup per backend. Pair order alternates. The reference is loaded from the untouched Git checkout in the same process; model weights, optimizer moments, inputs and loss settings are matched. Model/checkpoint loading and optimizer restoration occur outside the timing window. No other experiment or test worker runs during these measurements. Individual times and the complete configurations are retained in `measurements.json`.

Host: AMD EPYC 9V74; Python 3.12.14; PyTorch 2.15.0+cpu; one CPU thread; PyTorch reports **AVX512** instruction support. The 2.56M and 24.91M runs use fixed-seed initial weights. Their timings measure capacity scaling, not trained language quality.

---

## Work removed or reused

| Change | Effect |
| --- | --- |
| Forward-scoped derived weights | Normalize symbol keys, group prototypes and lace templates once per forward. Reuse minKron decay exponentials. Gradient tracking stays intact. |
| Cached member addresses | Reuse normalized expert-routing addresses when the batch requests at least a full table's worth of rows. Small batches retain selected-address computation. Expert MLP arithmetic remains sparse. |
| One-event minKron | Remove sequence-axis transposes, repeated slices and one-piece concatenation. Compute each direction norm once for normalization and the zero guard. |
| Single-stream latent adapter | Call the known local recurrent stream directly instead of dispatching link IDs and gathering temporal indices. |
| Fully active batches | Operate on the original tensors and complete states instead of gathering every row and scattering all rows back. Mixed and absent rows preserve their existing semantics. |
| Reused projections | Reuse ingress and edge projections when the same immutable publication is consumed again within the forward. |
| Publication and observed laces | Replace uniform publication masks directly while preserving dependency tombstones. Observational lace routing stops after computing the IDs the node actually publishes. |

The trained text profile drops from **737,330 to 469,012 Python calls**, a **36.4% reduction**. Parameter names, tensor shapes, shared-bank identity, expert execution counts, regional rounds and checkpoint schemas are unchanged. The preserved SHoE experiment sources are unchanged.

Caches are local to a graph forward and are discarded on return or exception. They cannot outlive an optimizer update or connect unrelated backward passes. Observational reads detach cached derived values; they never detach the training route. NumPy conversion is unnecessary on this path: data and gradients stay in PyTorch, whose installed CPU kernels already support AVX-512.

## Remaining bottlenecks

Autograd and many small recurrent operations still consume much of each update. The next substantial kernel work is to fuse the one-event normalization/recurrence and its backward pass, then reduce allocation and dispatch in selected-expert execution. Expert-bank gradients and AdamW state are still dense even though expert arithmetic is sparse. Those costs matter increasingly as stored capacity grows.

The scaled configuration contains **24,912,484 parameters**, width 128, 864 experts per block, 432 shared experts per block, sixteen routing groups and four selected experts per block. It retains the eleven-node topology and two regional rounds. The default graph has seven nodes and selects eight experts per block; these are separate capacity profiles, not parameter-matched architecture comparisons.

---

## Verification

**All 422 regression checks pass.** New checks cover exact outputs, routing and state; gradients; mixed activity masks; selective resets; substituted edges; multiple live links; original initialization/RNG behavior; exception cleanup; and cache lifetime across independent backwards and weight loads. The existing suite exercises source isolation, bounded state, checkpoint and optimizer continuation, and serialized generation.

The untouched-source benchmark produces identical text-logit/state hashes for all six text cases. The 512-episode memory field results and all recorded execution counters also match exactly.

On the trained checkpoint, the full text gradient has relative L2 error **1.32e-07**. Reusing differentiable intermediates changes FP32 gradient summation order. Across three matched complete joint updates, the largest text-loss difference is **1.14e-05 nats/byte**, and the largest final weight difference is **4.77e-07**. Checkpoint verification retains every measured loss and norm.

The regression pass also exposed an older horizon trainer's FP32 clipping overflow after a changed rounding trajectory. The gradients were finite, with FP64 norm 7.30e21; their FP32 norm overflowed. That trainer now uses the existing overflow-safe clipping routine: ordinary clipping is unchanged, finite overflow falls back to FP64 norm reduction, and actual NaN/Inf gradients still raise. Its exact resume test passes.

## Reproduce

Extract the supplied source/benchmark ZIP. It contains the optimized tree, original Python reference sources, the selected checkpoint, raw measurements, profiles, verification evidence and this report. From `SHoEbox/`:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python scripts/benchmark_runtime.py \
  --checkpoint artifacts/joint_stability/study/20261401/group_clip/best.pt \
  --baseline-root ../SHoEbox_reference \
  --sizes checkpoint default 25m --repeats 3 --threads 1 \
  --memory-count 512 --profile --out artifacts/performance/recheck
python scripts/verify_runtime.py \
  --checkpoint artifacts/joint_stability/study/20261401/group_clip/best.pt \
  --out artifacts/performance/recheck_verification.json
```

For direct programmatic comparison, use `model.set_execution_backend('reference')` or `model.set_execution_backend('optimized')`. Existing regional training commands use the optimized path automatically. The artifact manifest identifies the exact source and checkpoint hashes used for this pass.
