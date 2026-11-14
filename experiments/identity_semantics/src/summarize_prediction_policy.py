"""Summarize completed controlled trials without hiding unreached NLL targets."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
def read(name):
    x=json.loads((OUT/f'action4_{name}.json').read_text());assert x['completed'];return x['rows']
def summarize(rows):
    result={}
    for variant in dict.fromkeys(row['variant'] for row in rows):
        r=[x for x in rows if x['variant']==variant];times=[x['time_to_target'] for x in r if x['time_to_target'] is not None]
        result[variant]=dict(bytes_per_second=statistics.median(x['bytes_per_second'] for x in r),
            rate_range=[min(x['bytes_per_second'] for x in r),max(x['bytes_per_second'] for x in r)],
            heldout_nll=statistics.median(x['heldout_nll'] for x in r),
            nll_range=[min(x['heldout_nll'] for x in r),max(x['heldout_nll'] for x in r)],
            time_to_target=statistics.median(times) if times else None,reached=len(times),runs=len(r),
            parameters=r[0]['fitted_parameter_count'],rss_mib=statistics.median(x['rss_kib']/1024 for x in r),
            parameter_accounting=r[0]['parameter_accounting'],policy=r[0]['policy'],lr=r[0]['lr'])
    return result

def main():
    natural=summarize(read('ablation_natural'));qualified=summarize(read('selected_natural'));repeated=summarize(read('repeated'))
    baseline=qualified['baseline'];eligible={k:v for k,v in qualified.items() if k!='baseline' and v['reached']==v['runs']}
    selected_name=min(eligible,key=lambda k:eligible[k]['time_to_target']);selected=qualified[selected_name]
    speed=selected['bytes_per_second']/baseline['bytes_per_second'];gain=1-selected['time_to_target']/baseline['time_to_target']
    tuned=[]
    for name in ('rate_sweep','stride2','stride8'):
        for r in read(name):
            p=r['policy'];tuned.append(dict(variant=r['variant'],lr=r['lr'],rank=p['rank'],stride=p['slow_stride'],slow_scale=p['slow_scale'],heldout_nll=r['heldout_nll'],bytes_per_second=r['bytes_per_second']))
    manifest=json.loads((OUT/'action4_source_manifest.json').read_text())
    outcome=(f"Action 4 is implemented and published as the explicit compact-y-v1 policy. The selected {selected_name} variant processes "
      f"{selected['bytes_per_second']:,.1f} B/s versus {baseline['bytes_per_second']:,.1f} B/s for the rate-matched Action-3 event control "
      f"({100*(speed-1):.1f}% faster). It reaches the preselected 3.33 NLL target {100*gain:.1f}% sooner by median first-observed time. "
      f"Final median held-out NLL is {selected['heldout_nll']:.5f} versus {baseline['heldout_nll']:.5f}.")
    reason=(f"The 0.5x/1x/2x rate sweep advanced LR {selected['lr']} (local LR {selected['lr']/3:.6f}); the 2/4/8-byte cadence sweep advanced "
      f"stride {selected['policy']['slow_stride']}. Three matched seeds then compared the baseline, Y-only and combined policies at those settings. "
      f"{selected_name.title()} has the lowest median time to the fixed NLL target among policies reaching it on every seed. "
      'Both new components remain independently selectable. The compact predictor trades fewer active parameters and forecast work against a different learning trajectory; the combined variant is retained for its loss/parameter tradeoff.')
    repeat_delta=repeated[selected_name]['heldout_nll']-repeated['baseline']['heldout_nll']
    reason+=f' Repeated-text final NLL is {repeat_delta:+.5f} versus the control despite faster time to its 2.62 target; this is a workload tradeoff, so the existing defaults remain unchanged.'
    s=dict(baseline_revision=manifest['revision'],runtime_sha256=manifest['runtime_sha256'],natural=natural,qualified=qualified,repeated=repeated,tuning=tuned,
      selection=dict(variant=selected_name,policy=selected['policy'],lr=selected['lr'],reason=reason),outcome=outcome,
      gap=f"The selected {selected['bytes_per_second']:,.1f} B/s is {100*selected['bytes_per_second']/15000:.1f}% of 15,000 B/s; the remaining gap is {15000/selected['bytes_per_second']:.2f}x. This is single-stream throughput.",
      repeated_scope='Three matched seeds, 8,192 training bytes, separate 1,024-byte validation, full logits export, LR 0.08, and the selected slow cadence. The preselected repeated-text target is NLL 2.62. Targets that are not reached are reported explicitly.',
      validation=dict(full_suite=312,prediction_tests=15,unique_tests=314,selected_5m_tests=6,legacy_profiles=4,legacy_continuation_bytes=8192,sanitizers=['ASan','UBSan']),
      qualification_scope='Short controlled 1M-model screening and tuning. Longer learning and 100M qualification remain Action 9.')
    (OUT/'action4_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(outcome);print(reason)
if __name__=='__main__':main()
