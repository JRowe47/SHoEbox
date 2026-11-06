"""Summarize the second runtime pass from measured timings and verification."""
import argparse
import copy
import json
from pathlib import Path
import re

BASE = 'c80daedccfb85d44b9c576846f8222afa5ced42b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, default=Path('artifacts/performance2'))
    p.add_argument('--out', type=Path, default=Path('docs/SHoEbox_Operator_Performance_Report.md'))
    a = p.parse_args()
    results = json.loads((a.artifacts/'benchmark/measurements.json').read_text())
    recheck = json.loads((a.artifacts/'scaled_inference_recheck/measurements.json').read_text())
    verification = json.loads((a.artifacts/'checkpoint_verification.json').read_text())
    tests = (a.artifacts/'regression_tests.log').read_text()
    assert verification['passed'] and 'All five experiment suites passed.' in tests
    count = sum(int(n) for n in re.findall(r'Ran (\d+) tests? in', tests))
    rows = copy.deepcopy(results['measurements'])
    assert len(rows) == 8
    assert all(row['checks']['reference'] == row['checks']['optimized'] for row in rows)
    extended = recheck['measurements'][0]
    assert extended['checks']['reference'] == extended['checks']['optimized']
    assert (extended['size'], extended['workload']) == ('25m', 'text_inference')
    rows[-1] = extended
    sizes = dict(checkpoint='0.51M trained', default='2.56M default', **{'25m': '24.91M scaled'})
    workloads = dict(text_train='Forward + backward', text_inference='Text inference',
                     joint_update='Full joint update', memory_eval='512-episode evaluation')
    table = ['| Model / workload | Published (s) | Updated (s) | Throughput gain |',
             '| --- | ---: | ---: | ---: |']
    for row in rows:
        before, after = [row['timings'][mode]['median_seconds'] for mode in ('reference', 'optimized')]
        table.append(f"| {sizes[row['size']]}: {workloads[row['workload']]} | {before:.3f} | {after:.3f} | +{100*(row['speedup']-1):.1f}% |")
    joint = rows[2]
    before, after = [joint['timings'][mode]['median_seconds'] for mode in ('reference', 'optimized')]
    calls = rows[0]['python_calls']
    drop = 100*(1-calls['optimized']/calls['reference'])
    loss_gap = max(abs(x['text_nats']-y['text_nats']) for x,y in zip(
        verification['continuation']['reference'], verification['continuation']['optimized']))
    report = f'''# SHoEbox: faster regional execution

November 6, 2026. Compared with published commit `{BASE[:7]}`.

This second pass raises full joint-update throughput by **{100*(joint['speedup']-1):.1f}%**, reducing its median time from **{before:.2f} to {after:.2f} seconds**. The comparison starts from the already optimized release. The new operator path is the default regional runtime.

## Matched measurements

{chr(10).join(table)}

One CPU thread; PyTorch {results['environment']['torch']}; Python {results['environment']['python']}; {results['environment']['cpu_capability']}. Each text call processes eight streams of sixteen bytes, with two regional rounds. Inference supplies existing bytes; sampling is not included. The joint update also trains fresh memory episodes, combines and clips task gradients, and runs AdamW. The two larger models use fixed-seed initial weights for capacity profiling.

Values are paired medians after warmup, with alternating order. Seven pairs were used for the scaled inference recheck because the initial three-pair timings varied; all other rows use three pairs. Both sets of raw measurements are retained. Weights, optimizer state and inputs were matched; loading occurs outside timing. Benchmarks ran without another test or experiment worker.

---

## What changed

Recurrent updates, symbol readout and selected-expert arithmetic now execute as PyTorch scripted tensor programs, reducing Python dispatch. Algebraic fusion is disabled within the regional forward to retain the existing operator arithmetic. No external compiler is required.

When the query, key and value directions have equal widths, their normalization runs together. Expert dispatch directly indexes selected rows, removing repeated owner arrays and redundant singleton reductions. Routing tables and derived weights still expire after each forward.

The trained text profile falls from **{calls['reference']:,} to {calls['optimized']:,} Python calls**, a **{drop:.1f}% reduction**. Parameter shapes, checkpoint keys, topology, shared-bank identity, expert selections and execution counts are retained.

## Verification

**{count} regression checks pass**, including FP64 whole-graph derivatives, duplicate expert routes, unused expert banks, mixed activity, resets and checkpoint continuation. All six text benchmark cases retain identical output and state hashes; memory answers and execution counters also match.

The trained checkpoint's full-gradient relative L2 difference from the original reference is **{verification['gradient_relative_l2_error']:.2e}**. Across three joint continuation updates, the largest text-loss difference is **{loss_gap:.2e} nats/byte**, and the largest final weight difference is **{verification['continuation_max_weight_difference']:.2e}**. The per-parameter FP32 relative tolerance is 1e-4 for changed accumulation order; FP64 graph checks retain a 1e-10 bound, and checkpoint verification retains its 5e-6 aggregate gradient bound.

Autograd and repeated small tensor allocations remain the main training costs. Expert-bank gradients and AdamW state remain dense. A native fused recurrent backward pass is the next substantial kernel target.

## Reproduce

The downloadable package includes the current source, the published comparison source, the trained checkpoint, raw timings, profiles and verification logs. From `SHoEbox/`:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python scripts/benchmark_runtime.py \\
  --checkpoint artifacts/performance2/best.pt \\
  --baseline-root ../SHoEbox_reference --baseline-backend optimized \\
  --sizes checkpoint default 25m --repeats 3 --threads 1 \\
  --memory-count 512 --profile --out artifacts/performance2/recheck
python scripts/verify_runtime.py \\
  --checkpoint artifacts/performance2/best.pt \\
  --out artifacts/performance2/recheck_verification.json
```
'''
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(report)
    summary = dict(baseline_commit=BASE, environment=results['environment'],
                   measurements=[dict(size=r['size'], workload=r['workload'], parameters=r['parameters'],
                                      timings=r['timings'], speedup=r['speedup'],
                                      matching_checks=r['checks']['reference']==r['checks']['optimized']) for r in rows],
                   python_calls=calls, regression_checks=count,
                   verification=verification | {'checkpoint': 'artifacts/performance2/best.pt'})
    (a.artifacts/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(a.out)


if __name__ == '__main__':
    main()
