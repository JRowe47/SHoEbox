# SHoEbox: faster regional execution

November 6, 2026. Compared with published commit `c80daed`.

This second pass raises full joint-update throughput by **20.0%**, reducing its median time from **12.58 to 10.49 seconds**. The comparison starts from the already optimized release. The new operator path is the default regional runtime.

## Matched measurements

| Model / workload | Published (s) | Updated (s) | Throughput gain |
| --- | ---: | ---: | ---: |
| 0.51M trained: Forward + backward | 3.942 | 3.552 | +11.0% |
| 0.51M trained: Text inference | 1.229 | 1.027 | +19.6% |
| 0.51M trained: Full joint update | 12.581 | 10.488 | +20.0% |
| 0.51M trained: 512-episode evaluation | 7.122 | 6.187 | +15.1% |
| 2.56M default: Forward + backward | 4.381 | 3.714 | +17.9% |
| 2.56M default: Text inference | 1.748 | 1.513 | +15.5% |
| 24.91M scaled: Forward + backward | 4.652 | 3.989 | +16.6% |
| 24.91M scaled: Text inference | 1.292 | 1.125 | +14.8% |

One CPU thread; PyTorch 2.15.0+cpu; Python 3.12.14; AVX512. Each text call processes eight streams of sixteen bytes, with two regional rounds. Inference supplies existing bytes; sampling is not included. The joint update also trains fresh memory episodes, combines and clips task gradients, and runs AdamW. The two larger models use fixed-seed initial weights for capacity profiling.

Values are paired medians after warmup, with alternating order. Seven pairs were used for the scaled inference recheck because the initial three-pair timings varied; all other rows use three pairs. Both sets of raw measurements are retained. Weights, optimizer state and inputs were matched; loading occurs outside timing. Benchmarks ran without another test or experiment worker.

---

## What changed

Recurrent updates, symbol readout and selected-expert arithmetic now execute as PyTorch scripted tensor programs, reducing Python dispatch. Algebraic fusion is disabled within the regional forward to retain the existing operator arithmetic. No external compiler is required.

When the query, key and value directions have equal widths, their normalization runs together. Expert dispatch directly indexes selected rows, removing repeated owner arrays and redundant singleton reductions. Routing tables and derived weights still expire after each forward.

The trained text profile falls from **469,191 to 285,044 Python calls**, a **39.2% reduction**. Parameter shapes, checkpoint keys, topology, shared-bank identity, expert selections and execution counts are retained.

## Verification

**426 regression checks pass**, including FP64 whole-graph derivatives, duplicate expert routes, unused expert banks, mixed activity, resets and checkpoint continuation. All six text benchmark cases retain identical output and state hashes; memory answers and execution counters also match.

The trained checkpoint's full-gradient relative L2 difference from the original reference is **1.32e-07**. Across three joint continuation updates, the largest text-loss difference is **1.14e-05 nats/byte**, and the largest final weight difference is **4.77e-07**. The per-parameter FP32 relative tolerance is 1e-4 for changed accumulation order; FP64 graph checks retain a 1e-10 bound, and checkpoint verification retains its 5e-6 aggregate gradient bound.

Autograd and repeated small tensor allocations remain the main training costs. Expert-bank gradients and AdamW state remain dense. A native fused recurrent backward pass is the next substantial kernel target.

## Reproduce

The downloadable package includes the current source, the published comparison source, the trained checkpoint, raw timings, profiles and verification logs. From `SHoEbox/`:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python scripts/benchmark_runtime.py \
  --checkpoint artifacts/performance2/best.pt \
  --baseline-root ../SHoEbox_reference --baseline-backend optimized \
  --sizes checkpoint default 25m --repeats 3 --threads 1 \
  --memory-count 512 --profile --out artifacts/performance2/recheck
python scripts/verify_runtime.py \
  --checkpoint artifacts/performance2/best.pt \
  --out artifacts/performance2/recheck_verification.json
```
