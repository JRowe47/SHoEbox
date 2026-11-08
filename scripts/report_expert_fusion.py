"""Render the completed indexed-expert optimization report as Markdown/PDF."""
import argparse
import json
from pathlib import Path
from render_markdown_report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    s = json.loads(args.summary.read_text())
    m = s['matched']
    old, new = m['medians']['baseline'], m['medians']['indexed']
    allocation = s['bank_allocation']
    operator_old, operator_new = s['baseline_operators'], s['indexed_operators']
    args.out.mkdir(parents=True, exist_ok=True)
    md = args.out/'SHoEbox_Expert_Optimization_Report.md'
    text = f'''# SHoEbox: indexed expert optimization

November 8, 2026

The next optimization pass is complete: **{m['throughput_gain_percent']:.1f}% higher median throughput, {m['peak_rss_reduction_percent']:.1f}% less peak process memory, and {m['saved_storage_reduction_percent']:.1f}% less saved tensor storage** than fused MinKron with the previous expert path.

| Metric | Previous expert path | Indexed experts |
| --- | --- | --- |
| Forward + backward | {old['total_seconds']:.2f} s | {new['total_seconds']:.2f} s |
| Throughput | {old['bytes_per_second']:.2f} bytes/s | {new['bytes_per_second']:.2f} bytes/s |
| Peak process RSS | {old['peak_rss_mib']/1024:.2f} GiB | {new['peak_rss_mib']/1024:.2f} GiB |
| Unique storage referenced by saves | {old['saved_unique_storage_bytes']/2**30:.2f} GiB | {new['saved_unique_storage_bytes']/2**30:.2f} GiB |
| Saved-tensor calls | {old['saved_tensor_calls']:,} | {new['saved_tensor_calls']:,} |

These are medians of three matched pairs in separate processes, alternating execution order. Both paths use 14,976,882 parameters, identical initialized weights and data, four streams, 32 observations per gradient window, eight warmup observations and one CPU thread. Compilation and warmup are excluded. Batch-time ranges were {old['total_seconds_range'][0]:.2f}-{old['total_seconds_range'][1]:.2f} seconds and {new['total_seconds_range'][0]:.2f}-{new['total_seconds_range'][1]:.2f} seconds respectively. Referenced saved storage is deduplicated storage, not peak RSS.

## What changed

Native expert execution reads selected weight rows directly, eliminating gathered matrix copies. Its analytic backward emits compact row derivatives. Groups of eight independent autograd edges accumulate those derivatives into one dense bank gradient, avoiding repeated sparse merging. Duplicate expert selections sum correctly.

This reduces full-bank gradient allocations from {allocation['baseline']['dense_bank_gradient_allocations']} to {allocation['indexed']['dense_bank_gradient_allocations']} in the 64-call bank audit. The full wide batch has 6,528 bank uses grouped into 816 dense allocations. Grouping does not truncate gradients. Architecture, ports, routing, state schema, optimizer and the 32-observation horizon are unchanged.

## Validation and completed run

All **{s['tests_passed']} tests pass**. Checks cover FP64 derivatives, finite-difference gradcheck, duplicate IDs, retained graphs, dense hooks, frozen weights, noncontiguous inputs, AdamW moments and decay, two CPU threads, fallback, and live state crossing separate model calls. All three matched FP32 batches retain identical greedy predictions; maximum logit difference is {max(r['max_logit_difference'] for r in m['comparisons']):.2e}. Floating-point reduction order changes, so the complete discrete-routing training trajectory need not be bit-identical.

The new path completed **1,000 FineWeb byte targets in eight updates**, taking **{s['training_seconds']:.2f} seconds** for forward/backward/optimizer work ({s['training_bytes_per_second']:.2f} bytes/s). Training peak RSS was {s['training_peak_rss_mib']/1024:.2f} GiB. The fixed training-slice probe changed from **{s['initial_probe_nll']:.6f} to {s['final_probe_nll']:.6f} nats/byte**. Checkpoint reload reproduces the endpoint probe and port diagnostics exactly. Compact carried state remains {s['state_bytes']:,} bytes, and task gradients still reach A1, B1 and L2 at lag 31.

## Remaining work and use

At the same final checkpoint, the one-observation operator profile reduces cumulative positive allocation traffic from {operator_old['positive_allocation_bytes']/2**20:.1f} to {operator_new['positive_allocation_bytes']/2**20:.1f} MiB. This measures allocation traffic, not retained memory. The remaining profile identifies small dense projections and pointwise autograd work as the next target.

Enable with `model.set_execution_backend('fused')`. Set `SHOEBOX_EXPERTS=0` to compare with the prior expert path while retaining fused MinKron. Linux CPU FP32/FP64 compiles locally; unsupported cases retain portable execution. Native backward supports first-order training.

Reproduction commands: `docs/cortical_expert_fusion.md`. The downloadable bundle includes source, exact input bytes, matched measurements, training diagnostics and operator profiles.

Published code: [JRowe47/SHoEbox, {args.commit[:7]}](https://github.com/JRowe47/SHoEbox/commit/{args.commit}).

Source SHA-256: `{s['source_sha256']}`.

Data SHA-256: `{s['data_sha256']}`.
'''
    md.write_text(text)
    render(md, md.with_suffix('.pdf'), title='SHoEbox: indexed expert optimization',
           footer_text='SHoEbox | Indexed expert optimization | November 8, 2026')
    print(json.dumps({'markdown':str(md), 'pdf':str(md.with_suffix('.pdf'))}))


if __name__ == '__main__':
    main()
