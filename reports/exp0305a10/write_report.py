"""Render already completed A10 evidence. Never loads or evaluates a model."""
import argparse,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);a=p.parse_args()
    report=Path(__file__).resolve().parent;root=report.parents[1];w=a.work
    read=lambda n:json.loads((w/n).read_text())
    bench=read('benchmark.json');counts=read('counts.json');pairs=read('checkpoints/result.json')
    for name in ('benchmark.json','timing_raw.jsonl','timing_started.json','timing_process.json','timing.txt',
                 'pre_timing_retry_process.json','pre_timing_retry.txt','pre_timing_remote_commit.json'):
        shutil.copyfile(w/name,report/name)
    processes={p.stem.removesuffix('_process'):json.loads(p.read_text()) for p in w.glob('*_process.json')
               if not p.name.startswith(('report','final_verify','package'))}
    py=[read('python_scalar.json'),read('python_candidate.json')]
    def sum_dict(rows):return {k:sum(x[k] for x in rows) for k in rows[0]}
    projected=sum_dict([x['counts'] for x in counts['rows']])
    original_projections={k:2*v for k,v in projected.items()}
    all_projections={k:3*v for k,v in projected.items()}
    python_events=sum_dict([v for x in py for v in x['updates'].values()])
    pair_events={k:3*v//2 for k,v in pairs['totals'].items()}
    research=read('research.json');history=read('history.json')
    ledger=dict(research_fitting_bytes=0,accuracy_rescoring=False,audit_episodes=0,
        pilots=0,sweeps=0,extra_inference_policy_experiments=0,
        checkpoint_correctness=dict(original_four_paths=pairs['totals'],counted_repeat_input_bytes=36000,
            all_six_paths_events=pair_events,original_execution_counts=original_projections,
            counted_repeat_execution_counts=projected,combined_execution_counts=all_projections,
            saved_byte_comparisons=sum(x['saved_byte_comparisons'] for x in pairs['rows']),
            exact_full_probability_float_comparisons=sum(x['full_probability_float_comparisons'] for x in pairs['rows']),
            resumed_step_hash_record_comparisons=18000,counted_probability_hash_comparisons=36000,
            counted_final_saved_hash_comparisons=36),
        python20=dict(tests=40,events=python_events,observed_bytes=sum(sum(x['observed_bytes'].values()) for x in py),
            rejected_calls=[v for x in py for v in x['rejected_calls']]),
        historical_python209=dict(successful_adapter_calls=research['successful_adapter_calls'],
            successful_adapter_bytes=sum(x['input_bytes'] for x in research['successful_adapter_calls'].values())),
        historical_checkpoint14=history,
        independent_affine=dict(builds=2,trials_per_build=320,projected_rows_per_build=35840,
            independent_scalar_rows_per_build=35840,original_scalar_rows_per_build=35840,
            affine_comparisons_per_build=71680,tanh_comparisons_per_build=35840,
            logged_combined_check_groups_per_build=71680,model_input_bytes=0),
        objective_microchecks=dict(python_backends=2,native_calls_per_python_backend=256,
            independent_calls_per_python_backend=256,code_comparisons_per_python_backend=131072,
            native_builds=4,native_calls_per_build=1029,independent_reference_calls_per_build=768,
            code_comparisons_per_build=460032,finite_difference_coordinates_per_build=256,
            actual_affine_gradient_rows_per_build=4176,expected_affine_gradient_rows_per_build=4096),
        presence_microchecks=dict(native_builds=4,logistic_events_per_build=812,
            finite_difference_coordinates_per_build=802,local_calls_per_build=640,
            actual_and_reference_positive_and_negative_record_rows_each_per_build=256,
            query_presence_rows_per_build=512,applicability_rows_per_build=512,
            hybrid_components_per_build=768,hybrid_code_comparisons_per_build=196608,
            hybrid_gradient_rows_per_build=12288,value_gradient_rows_per_build=4096,
            independent_expected_record_rows_per_build=1024),
        mask_microchecks=dict(native_builds=4,checks_per_build=57280,local_calls_per_build=1280,
            actual_hybrid_calls_per_build=512,reference_hybrid_calls_per_build=768,
            record_hybrid_executed_per_build=256,record_hybrid_omitted_per_build=256,
            actual_hybrid_gradient_rows_per_build=8192,reference_hybrid_gradient_rows_per_build=12288,
            actual_code_comparisons_per_build=131072,reference_code_comparisons_per_build=196608),
        scratch={k:v for k,v in counts.items() if k in ('coefficient_packing_buffer_bytes','separate_packing_passes',
            'new_value_projection_automatic_scratch_bytes','row_kernel_max_live_simd_source_values_bytes','packing_cpu')},
        timing={k:bench[k] for k in ('total_input_bytes','timed_bytes','warmup_bytes','padding_bytes','nuisance_bytes')},
        process_receipts=processes,total_registered_process_cpu_seconds=sum(x['cpu_seconds'] for x in processes.values()),
        limitations=['Inherited runtime/native microtrajectories retain full source, test counts and CPU, but do not expose every individual input/update or scalar-write counter. They are not included as zero or silently folded into a purported complete event total.',
            'Historical Python adapter counts cover successful run calls; separate legacy observe/teach calls and internal scalar writes are not separately tallied.',
            'Interactive file inspection/editing/publishing CPU is not captured by the child-process meter. Report, verification and archive process receipts are retained separately.'],
        failure=dict(receipt='controls/pre_timing_process.json',cpu_seconds=processes['pre_timing']['cpu_seconds'],
            reason='lscpu could not read restricted sysfs; metadata collector changed to retain probe failure and use /proc/cpuinfo',
            model_input_bytes=0,model_update_bytes=0,timing_started=False),
        method='All new checkpoint/control and timing events above are explicit; reused microcheck counts are fixed source-derived formulas. No model was run for report generation.')
    (report/'ACCOUNTING.json').write_text(json.dumps(ledger,indent=2)+'\n')
    names=bench['arms'];labels=['Native v19','Archived E8','Scalar A9','A10: four rows']
    r=bench['rows'];ratios=[100*r[n]['relative_native'] for n in names]
    colors=['#637387','#30816c','#647eaf','#c76538']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.6));bars=ax.barh(labels,ratios,color=colors,height=.58)
    ax.invert_yaxis();ax.set_xlim(0,113);ax.axvline(80,color='#a22635',ls='--',lw=1.7,label='Required: 80% of native')
    for bar,value in zip(bars,ratios):ax.text(value+1,bar.get_y()+bar.get_height()/2,f'{value:.2f}%',va='center',weight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=5)
    ax.set_xlabel('Throughput relative to native (ratio of median CPU times)');ax.set_title('A10 preserves exact results but fails the speed gate',loc='left',weight='bold',pad=18)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
    fig.text(.125,.025,'Fresh registered 16-world stream · 3 × 65,536 bytes per arm · CPU 0 · No accuracy rescoring',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(report/'SHoEbox_A10_Throughput.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.6))
    for n,label,color in zip(names,labels,colors):ax.plot(range(1,4),r[n]['cpu_seconds'],'o-',color=color,lw=2,label=f"{label}: median {r[n]['median_cpu_seconds']:.6f} s")
    ax.set_xticks([1,2,3]);ax.set_xlabel('Registered interleaved repeat');ax.set_ylabel('CPU seconds / 65,536 input bytes')
    ax.set_title('Every original A10 timing measurement',loc='left',weight='bold',pad=18);ax.grid(alpha=.15)
    ax.legend(frameon=False,fontsize=10,loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout(rect=(0,.06,1,1))
    fig.text(.09,.025,'Order rotates left each repeat · 1,024-byte reset warmup per arm/repeat · No measurements replaced',fontsize=9,color='#444444')
    fig.savefig(report/'SHoEbox_A10_Raw_Timings.png',dpi=180);plt.close(fig)
    table='\n'.join(f"| {label} | "+' | '.join(f'{x:.9f}' for x in r[n]['cpu_seconds'])+f" | {r[n]['median_cpu_seconds']:.9f} | {r[n]['bytes_per_cpu_second']:,.3f} | {100*r[n]['relative_native']:.6f}% |" for n,label in zip(names,labels))
    proc_table='\n'.join(f"| {name} | {x['exit_code']} | {x['cpu_seconds']:.6f} |" for name,x in sorted(processes.items()))
    speedup=100*(r['scalar_a9']['median_cpu_seconds']/r['row_parallel_a9']['median_cpu_seconds']-1)
    result=f'''# SHoEbox EXP03/05-A10 report

16 November 2026 (America/Denver). **A10 fails its execution gate.** Exactness
passes, but throughput is **76.020287% of native**, below the mandatory 80%.
The candidate is **{speedup:.6f}% faster than scalar A9** on the same fresh
stream. Public E8 remains unchanged. A9 still fails development; no accuracy
was rescored and no audit was accessed. Research fitting: **zero bytes**.

## Fixed experiment and result

Starting main: `8270e0d60bbefd75f7c7c9604d5961702615aac2`.
Passing source/control/workload lock was committed and published **before
timing** as [3bc9d54f84626c3c117f88419f8cdcec0564e325](https://github.com/JRowe47/SHoEbox/commit/3bc9d54f84626c3c117f88419f8cdcec0564e325).
Its tree is `68e09052a14c3c09bc0d1e780475953202f3b543`. The original local
commit `dd1435e` has that identical tree; the published commit was installed
locally before timing. All original computational sources and registrations
remain hash-identical. The sole execution addition is an explicitly selected
backend in `experiments/row_parallel_affine/`.

| Arm | Repeat 1 (CPU s) | Repeat 2 (CPU s) | Repeat 3 (CPU s) | Median (CPU s) | Bytes/CPU s | Native throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Candidate/scalar timed checksum: `-0x1.cc2217bf76a82p+16` in every repeat.
Their warmup checksum is `-0x1.12f8b02784500p+9` in every repeat. Exactness
and no-slowdown pass; native-speed and the combined execution gate fail.
All twelve first measurements are retained in `timing_raw.jsonl`.

![Median relative throughput](SHoEbox_A10_Throughput.png)

![All twelve raw timing measurements](SHoEbox_A10_Raw_Timings.png)

## Exact mechanism

Four independent rows occupy four SSE lanes. A 4×4 register transpose supplies
coefficients. Every row starts at positive zero and accumulates coordinates
0–63 in the original order, with separate float32 multiplication and addition.
Bias is added afterward, followed by original scalar tanh. No FMA instruction
appears in the generated candidate library. The six replaced sites are query
key, recognized record key, fallback record key, selected value, hybrid local
key preprojection, and local value preprojection. Native core, source vectors,
presence/applicability, selection, chronology, candidate enumeration and local
gradient/update arithmetic remain unchanged.

There is no persistent or heap coefficient packing buffer. Local value output
adds 32 automatic bytes per invocation; the kernel has a source-level maximum
of 144 bytes of simultaneously live SIMD values, not a claim about compiler
register allocation or spill size. Existing query/key output storage is reused.
Register transpose operations are included in the full measured interval;
there is no separate packing pass, packing timer or profiler experiment.
Model fields, magic, saved bytes, exact cursors and historical backends remain
unchanged. E8/A8 loads dispatch to their original execution libraries.

## Controls completed before timing

- Original 20 A9 Python tests pass on scalar and candidate: 40 tests. This
  includes enabled A8 parity, omitted record-key isolation, exact partial
  cursor, all-256 output, generic/local separation and unchanged policy source.
- Independent affine cases cover 320 trials, five key phases, four value
  phases, 8/16 rows, signed zero, bias, cancellation, subnormal/extreme and
  random finite inputs. Per optimized/sanitized build: 35,840 candidate rows,
  35,840 independent scalar rows and 35,840 original scalar rows, with 71,680
  affine comparisons plus 35,840 tanh comparisons. The log groups these as
  71,680 combined affine/tanh checks. Both builds pass bitwise.
- On each scalar/candidate, optimized and ASan/UBSan paths pass 581 E8,
  110 request, 83 admission/lifecycle, 13,089 exhaustive selection,
  75,295 hybrid objective, 7,826 presence and 57,280 masked-credit checks.
  Address/undefined-behavior checks are active; leak detection is disabled.
- Public/runtime: 60 Python tests, 824,111 native checks per build and
  723 public E8 checks per build pass. All 209 inherited research tests and
  46 protected-loader rejection checks pass.
- Fourteen historical constructor/checkpoint trajectories match the unchanged
  starting repository exactly. Historical v5/v6 source remains unchanged;
  those two are not independently instantiated by that existing suite.
- All 18 actual A9-F1 warmup/final checkpoint-cursor pairs pass: scalar and
  candidate, uninterrupted 1,000 versus 371/save/load/629. Every step compares
  all 256 outputs, loss/target count, inventory and complete saved bytes.
  There are 36,000 paired steps, 9,216,000 float-output comparisons,
  18,000 uninterrupted/resumed trace-record comparisons, and
  {ledger['checkpoint_correctness']['saved_byte_comparisons']:,} directly compared saved bytes.
  All 36 frozen model/cursor file hashes remain unchanged.
- A separately charged test-only counter replay adds 36,000 discarded bytes,
  reproduces every stored output hash and final saved hash, and confirms that
  relocated row counts exactly equal the scalar projections they replace.
  It is not linked into the measured backend.

The initial lock collector failed because `lscpu` could not read the container's
restricted sysfs. It ran no model inputs or updates and no timing. Its original
receipt/log and 1.312434 CPU seconds remain. The collector now records the probe
failure and reads `/proc/cpuinfo`; the retry passed in 1.408669 CPU seconds.
No numerical/control test failed and no timing measurement was repeated.

## Locked timing workload and environment

The exact registered enumeration/exclusions leave 2,020 eligible identities
after excluding 10,076 of 12,096 worlds. The one shuffle with seed 20261202
selects 16. The unchanged A9 exclusion union, all 896 original primary worlds
and all 128 F1 binary worlds are excluded. Only identity manifests were read
for excluded groups; excluded audit episodes were never instantiated/scored.

Conditions, world order, four variants, zero offset/spacers and alternating
updated key follow the registration. An independent literal constructor
matches all 384 episodes. Stream: 70,400 bytes, 41,792 PAD and 18,432 nuisance.

- Stream SHA-256: `2c26341b8f06acc74ebda3b1f118a6f6c32141cfa829df20de44e0965e3b6290`.
- Ordered sample SHA-256: `a75d7cd3635d4a7968b95f0a351e0a2093c6309caa2f260be4f50e1f44be05e9`.
- Exclusion SHA-256: `4541847efd3fb077d7dc61effe984feca5d147712ce1ed823bcebdc654002adb`.

The 16 identities are opened for future allocation exclusions in
`timing_worlds.lock.json`. No labels or expected answers reach timing.
The fresh native v19 uses nominal 500K, seed 7; E8 and both A9 arms use the
original frozen seed-7 final files. All compute jobs had completed before
timing. CPU 0 was pinned; AMD EPYC 9V74, GCC 14.2.0. Candidate flags:
`-std=c++17 -O3 -march=native -ffp-contract=off -shared -fPIC`.
Full environment, linked libraries, generated source and binary hashes are
retained in the lock. Compilation/loading/cloning are outside timed intervals.

There are exactly three rotating interleaved repeats, with reset, 1,024-byte
warmup, reset, 65,536-byte timing. Total inference inputs: 798,720;
786,432 measured and 12,288 warmup; 494,988 PAD and 182,952 nuisance.
Every input produces all 256 outputs. Research fitting, pilots, sweeps,
replay fits, new diagnosis, accuracy rescoring and audit episodes are zero.

## Cost and event ledger

`ACCOUNTING.json` retains per-policy events, projection/update counts,
reference comparisons, scratch bounds and every child-process receipt.
The 72,000 primary discarded checkpoint inputs plus 36,000 counted replay
inputs execute {all_projections['native_targets']:,} native targets,
{all_projections['affine_update_rows']:,} local affine row updates,
{all_projections['scalar_affine_rows']:,} scalar affine projections and
{all_projections['row_parallel_rows']:,} relocated vector-row projections
in {all_projections['row_parallel_calls']:,} vector calls. Counter replay
outputs are discarded and do not extend any research checkpoint or budget.
The two 20-test suites separately execute {python_events['input_bytes']:,}
run inputs and {ledger['python20']['observed_bytes']:,} observe inputs;
their event and rejected-call receipts remain separate.

Historical adapter run inputs total {ledger['historical_python209']['successful_adapter_bytes']:,};
the 14 historical comparison trajectories charge 114,716 inputs. Inherited
runtime/native microtrajectories retain full CPU, source and assertion counts,
but do not expose every individual byte/update or scalar-write count. Legacy
Python observe/teach calls outside the wrapped adapter run entry point are
also not separately tallied. Those missing counters are explicitly identified,
not recorded as zero or presented as a complete global event total.

| Process | Exit | Whole child-tree CPU seconds |
| --- | ---: | ---: |
{proc_table}

Research/control/timing processes total **{ledger['total_registered_process_cpu_seconds']:.6f} CPU
seconds** ({ledger['total_registered_process_cpu_seconds']/60:.6f} CPU minutes), including the failed metadata
probe and counted correctness repetition. Compilation belongs to these
process totals. Rendering, final file verification and packaging have separate
receipts. Interactive inspection, editing and publication CPU is not metered.
The original A9-F1 20.25M-byte campaign and every historical cost are preserved;
none is replaced or charged to a new fit.

## Interpretation and sole next registration

The exact kernel improves speed by {speedup:.6f}%, but the mandatory speed
gate fails. The original A9 68.202933% speed measurement used a different
registered stream and is retained separately. The paired scalar baseline
here is 68.374115%; these two historical measurements are not interchangeable.

A9's original competing 88.281250%, corrected overwrite 88.281250%, clean
94.531250% and diverse256 86.328125% remain its development results. Prior
teacher-fed subset diagnosis remains separate from full autoregression.
No result here repairs that accuracy failure or permits promotion. E8 remains
public and every development/qualification threshold stays unchanged.

The sole next registration is [A11 fixed eight-row exact affine execution](A11_REGISTRATION.md):
one fixed AVX2 8×8 register transpose with the same per-row arithmetic and
zero research fitting. It requires fresh timing identities excluding A10,
exact reference controls and complete inherited-event instrumentation before
timing, plus at least 80% native speed and no slowdown against scalar A9 or
A10. **A11 is not implemented, allocated, tested or measured.**
'''
    (report/'SHoEbox_EXP0305_A10_Report.md').write_text(result)
    print(json.dumps(dict(report=str(report),execution_pass=False,relative_native=ratios[-1],scalar_speedup_percent=speedup,
        process_cpu_seconds=ledger['total_registered_process_cpu_seconds'],model_calls=0),indent=2))

if __name__=='__main__':main()
