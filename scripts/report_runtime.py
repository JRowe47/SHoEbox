"""Generate the runtime report from measured timings and completed checks."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('artifacts/performance/benchmark/measurements.json'))
    p.add_argument('--verification', type=Path, default=Path('artifacts/performance/checkpoint_verification.json'))
    p.add_argument('--tests', type=Path, default=Path('artifacts/performance/regression_tests_final.log'))
    p.add_argument('--out', type=Path, default=Path('docs/SHoEbox_Runtime_Profiling_Report.md'))
    a = p.parse_args()
    results = json.loads(a.results.read_text())
    verification = json.loads(a.verification.read_text())
    tests = a.tests.read_text()
    assert verification['passed'] and 'All five experiment suites passed.' in tests
    count = sum(int(n) for n in re.findall(r'Ran (\d+) tests? in', tests))
    rows = results['measurements']
    assert len(rows) == 8
    assert all(r['checks']['reference'] == r['checks']['optimized'] for r in rows)
    lookup = {(r['size'], r['workload']): r for r in rows}
    joint = lookup['checkpoint', 'joint_update']
    text = lookup['checkpoint', 'text_train']
    scaled = lookup['25m', 'text_train']
    before, after = [joint['timings'][mode]['median_seconds'] for mode in ('reference', 'optimized')]
    names = {'checkpoint': 'Trained 0.51M', 'default': 'Default 2.56M', '25m': 'Scaled 24.91M'}
    tasks = {'text_train': 'Text forward + backward', 'text_inference': 'Text inference',
             'joint_update': 'Full joint update', 'memory_eval': '512-episode evaluation'}
    table = ['| Model / workload | Before (s) | After (s) | Throughput gain |',
             '| --- | ---: | ---: | ---: |']
    for r in rows:
        b, f = [r['timings'][mode]['median_seconds'] for mode in ('reference', 'optimized')]
        table.append(f"| {names[r['size']]}: {tasks[r['workload']]} | {b:.3f} | {f:.3f} | {r['speedup']:.2f}x |")
    calls = text['python_calls']
    call_reduction = 100*(1-calls['optimized']/calls['reference'])
    loss_gap = max(abs(x['text_nats']-y['text_nats']) for x, y in zip(
        verification['continuation']['reference'], verification['continuation']['optimized']))
    report = f'''# SHoEbox runtime profiling

November 5, 2026. Base source: `10f87459377d5d2062c51be2c14fa8feaf67ebd0`.

The optimized runtime reduces a complete joint text-and-memory update from **{before:.2f} to {after:.2f} seconds**: **{joint['speedup']:.2f}x throughput**, or **{100*(1-after/before):.1f}% less time**. The trained checkpoint's frozen text outputs and recurrent state match the untouched implementation exactly. The optimized PyTorch path is now the regional engine's default.

## Matched measurements

{chr(10).join(table)}

Each text call processes eight streams of sixteen bytes with two regional rounds: 128 targets for training, or 128 supplied bytes for inference. Inference is a supplied-byte forward pass; it is not a sampling benchmark. A joint update includes the text backward pass, sixteen new memory episodes with a sixteen-byte gap, three field queries, task/group gradient accounting and clipping, and AdamW. Evaluation covers 512 fresh shared-correction episodes with the same gap.

Medians use three measured pairs after one warmup per backend. Pair order alternates. The reference is loaded from the untouched Git checkout in the same process; model weights, optimizer moments, inputs and loss settings are matched. Model/checkpoint loading and optimizer restoration occur outside the timing window. No other experiment or test worker runs during these measurements. Individual times and the complete configurations are retained in `measurements.json`.

Host: AMD EPYC 9V74; Python {results['environment']['python']}; PyTorch {results['environment']['torch']}; one CPU thread; PyTorch reports **{results['environment']['cpu_capability']}** instruction support. The 2.56M and 24.91M runs use fixed-seed initial weights. Their timings measure capacity scaling, not trained language quality.

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

The trained text profile drops from **{calls['reference']:,} to {calls['optimized']:,} Python calls**, a **{call_reduction:.1f}% reduction**. Parameter names, tensor shapes, shared-bank identity, expert execution counts, regional rounds and checkpoint schemas are unchanged. The preserved SHoE experiment sources are unchanged.

Caches are local to a graph forward and are discarded on return or exception. They cannot outlive an optimizer update or connect unrelated backward passes. Observational reads detach cached derived values; they never detach the training route. NumPy conversion is unnecessary on this path: data and gradients stay in PyTorch, whose installed CPU kernels already support AVX-512.

## Remaining bottlenecks

Autograd and many small recurrent operations still consume much of each update. The next substantial kernel work is to fuse the one-event normalization/recurrence and its backward pass, then reduce allocation and dispatch in selected-expert execution. Expert-bank gradients and AdamW state are still dense even though expert arithmetic is sparse. Those costs matter increasingly as stored capacity grows.

The scaled configuration contains **{scaled['parameters']['total_unique_parameters']:,} parameters**, width 128, 864 experts per block, 432 shared experts per block, sixteen routing groups and four selected experts per block. It retains the eleven-node topology and two regional rounds. The default graph has seven nodes and selects eight experts per block; these are separate capacity profiles, not parameter-matched architecture comparisons.

---

## Verification

**All {count} regression checks pass.** New checks cover exact outputs, routing and state; gradients; mixed activity masks; selective resets; substituted edges; multiple live links; original initialization/RNG behavior; exception cleanup; and cache lifetime across independent backwards and weight loads. The existing suite exercises source isolation, bounded state, checkpoint and optimizer continuation, and serialized generation.

The untouched-source benchmark produces identical text-logit/state hashes for all six text cases. The 512-episode memory field results and all recorded execution counters also match exactly.

On the trained checkpoint, the full text gradient has relative L2 error **{verification['gradient_relative_l2_error']:.3g}**. Reusing differentiable intermediates changes FP32 gradient summation order. Across three matched complete joint updates, the largest text-loss difference is **{loss_gap:.3g} nats/byte**, and the largest final weight difference is **{verification['continuation_max_weight_difference']:.3g}**. Checkpoint verification retains every measured loss and norm.

The regression pass also exposed an older horizon trainer's FP32 clipping overflow after a changed rounding trajectory. The gradients were finite, with FP64 norm 7.30e21; their FP32 norm overflowed. That trainer now uses the existing overflow-safe clipping routine: ordinary clipping is unchanged, finite overflow falls back to FP64 norm reduction, and actual NaN/Inf gradients still raise. Its exact resume test passes.

## Reproduce

Extract the supplied source/benchmark ZIP. It contains the optimized tree, original Python reference sources, the selected checkpoint, raw measurements, profiles, verification evidence and this report. From `SHoEbox/`:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python scripts/benchmark_runtime.py \\
  --checkpoint artifacts/joint_stability/study/20261401/group_clip/best.pt \\
  --baseline-root ../SHoEbox_reference \\
  --sizes checkpoint default 25m --repeats 3 --threads 1 \\
  --memory-count 512 --profile --out artifacts/performance/recheck
python scripts/verify_runtime.py \\
  --checkpoint artifacts/joint_stability/study/20261401/group_clip/best.pt \\
  --out artifacts/performance/recheck_verification.json
```

For direct programmatic comparison, use `model.set_execution_backend('reference')` or `model.set_execution_backend('optimized')`. Existing regional training commands use the optimized path automatically. The artifact manifest identifies the exact source and checkpoint hashes used for this pass.
'''
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(report)
    print(a.out)


if __name__ == '__main__':
    main()
