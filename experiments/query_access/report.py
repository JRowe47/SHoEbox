"""Build compact results, gates, downloadable Markdown and primary figures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import ARMS
from experiments.query_readout.run import write

LABELS={'native':'Native','l3':'Shared L3','split':'Split frontier','pool':'Pooled views'}
COLORS={'native':'#636b76','l3':'#c97b28','split':'#7860b3','pool':'#16877d'}
def read(path):return json.loads(Path(path).read_text())
def mean(xs):return float(np.mean(xs))
def pct(x):return f'{100*x:.2f}%'

def training_cost(data):
    total=cpu=targets=0
    for r in data['rows']:
        pieces=[r['warmup'],*r['arms'].values(),*[a['separate_warmup'] for a in r['arms'].values() if a.get('separate_warmup')]]
        for p in pieces:total+=p['bytes'];cpu+=p['cpu_seconds'];targets+=p['targets']
    return dict(bytes=total,cpu_seconds=cpu,supervised_answer_targets=targets)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    train=read(a.runs/'train.json');audit=read(a.runs/'audit.json');pilot=read(a.pilot/'train.json');benchmark=read(a.runs/'benchmark.json');controls=read(a.controls/'positive_control.json')
    selection=read(a.runs/'selection.json');presence=read(a.runs/'source_presence_native.json')
    summary=dict(selected=selection['selected'],core='minimal-v19',production_promotion=False,arms={},costs={})
    for arm in ARMS:
        rows=[r['arms'][arm] for r in audit['rows']];fits=[r['arms'][arm] for r in train['rows']]
        exact=[r['test']['native_rollout']['native_exact'] for r in rows]
        offsets=np.array([[x['exact'] for x in r['offsets']] for r in rows]);delays=np.array([[x['exact'] for x in r['delays']] for r in rows])
        out=dict(native_exact=mean(exact),seed_exact=exact,first_answer=mean([r['test']['native_rollout']['first_answer'] for r in rows]),
            second_answer=mean([r['test']['native_rollout']['second_answer'] for r in rows]),
            native_nll_per_answer=mean([r['test']['native_rollout']['native_nll'] for r in rows]),
            both_query_orders=mean([r['counterfactual']['both_query_orders_exact'] for r in rows]),
            offset_mean=float(offsets.mean()),worst_offset_mean=float(offsets.mean(axis=0).min()),
            original_offset=float(offsets[:,0].mean()),delay_means=delays.mean(axis=0).tolist(),
            training_bytes_per_cpu_second=sum(x['bytes'] for x in fits)/sum(x['cpu_seconds'] for x in fits),
            inference_bytes_per_cpu_second=benchmark['rows'][arm]['bytes_per_cpu_second'],
            inference_relative=benchmark['rows'][arm]['bytes_per_cpu_second']/benchmark['rows']['native']['bytes_per_cpu_second'],
            inventory=fits[0]['inventory'],checkpoint_bytes=fits[0]['checkpoint_bytes'])
        if arm!='native':
            out['diagnostic']=np.mean([r['diagnostic']['by_answer'] for r in rows],axis=0).tolist()
            out['lesions']={k:mean([r['lesions'][k]['native_exact'] for r in rows]) for k in rows[0]['lesions']}
            out['lesion_unmodified']=mean([r['lesion_unmodified']['native_exact'] for r in rows])
            out['query_ablation_drop']=out['lesion_unmodified']-out['lesions']['no_query']
            out['gates']=dict(binding_mean=out['native_exact']>=.95,binding_every_seed=min(exact)>=.90,
                each_answer=all(mean([r['test']['native_rollout'][key] for r in rows])>=.95 and min(r['test']['native_rollout'][key] for r in rows)>=.90 for key in ('first_answer','second_answer')),
                counterfactual=out['both_query_orders']>=.90 and min(r['counterfactual']['both_query_orders_exact'] for r in rows)>=.85,
                boundaries=out['offset_mean']>=.95 and out['worst_offset_mean']>=.90,
                delays=bool(np.all(delays.mean(axis=0)>=.90) and np.all(delays>=.85)),
                query_mechanism=out['query_ablation_drop']>=.20 and out['lesions']['reverse']<out['lesion_unmodified'],
                inference_cost=out['inference_relative']>=.80,no_added_sequence_state=out['inventory']['added_persistent_sequence_state_bytes']==0,
                joint_language_copy_scaling_regressions='not run: binding gates failed')
        summary['arms'][arm]=out
    summary['costs']['formal']=training_cost(train);summary['costs']['pilot']=training_cost(pilot)
    summary['costs']['positive_controls']=dict(bytes=sum(r['bytes'] for r in controls['rows']),cpu_seconds=sum(r['cpu_seconds'] for r in controls['rows']))
    summary['costs']['total_training_bytes']=sum(summary['costs'][k]['bytes'] for k in ('formal','pilot','positive_controls'))
    summary['costs']['total_training_cpu_seconds']=sum(summary['costs'][k]['cpu_seconds'] for k in ('formal','pilot','positive_controls'))
    summary['source_presence']=presence
    write(a.out/'summary.json',summary)
    for source,name in [(a.runs/'plan.json','plan.json'),(a.runs/'train.json','train.json'),(a.runs/'audit.json','audit.json'),
        (a.runs/'selection.json','selection.json'),(a.runs/'benchmark.json','benchmark.json'),(a.pilot/'plan.json','pilot_plan.json'),
        (a.pilot/'train.json','pilot.json'),(a.controls/'control_plan.json','control_plan.json'),(a.controls/'positive_control.json','positive_control.json'),
        (a.runs/'source_presence_native.json','source_presence.json')]:shutil.copyfile(source,a.out/name)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfcfe','axes.facecolor':'#fbfcfe'})
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    fig.suptitle('SHoEbox EXP01-C — query cues improve; binding remains unqualified',fontsize=17,fontweight='bold')
    x=np.arange(4);width=.25
    for i,(key,label,color) in enumerate([('native_exact','Both answers exact','#254b72'),('first_answer','First answer','#16877d'),('second_answer','Second answer','#c97b28')]):
        bars=axes[0,0].bar(x+(i-1)*width,[100*summary['arms'][k][key] for k in ARMS],width,label=label,color=color)
        axes[0,0].bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    axes[0,0].set_xticks(x,[LABELS[k] for k in ARMS]);axes[0,0].set_ylim(0,106);axes[0,0].axhline(95,ls='--',color='#a33932',lw=1)
    axes[0,0].set_title('Held-out native answers · three seeds');axes[0,0].set_ylabel('Accuracy (%)');axes[0,0].legend(fontsize=8,loc='upper left')
    diag=np.array(summary['arms']['pool']['diagnostic'])
    labels=['Query','Key 0','Key 1','Value 0','Value 1','Selection'];indices=[1,2,3,4,5,0];x=np.arange(6)
    for i in range(2):
        bars=axes[0,1].bar(x+(i-.5)*.35,diag[i,indices]*100,.35,label=f'Answer {i+1}',color=['#16877d','#c97b28'][i]);axes[0,1].bar_label(bars,fmt='%.0f',fontsize=8,padding=2)
    axes[0,1].set_xticks(x,labels);axes[0,1].set_ylim(0,112);axes[0,1].set_title('Pooled-view diagnostic · teacher-fed prefixes');axes[0,1].set_ylabel('Correct diagnostic decoding (%)');axes[0,1].legend(fontsize=8)
    for arm in ARMS:
        rows=[r['arms'][arm] for r in audit['rows']]
        offsets=np.mean([[x['exact'] for x in r['offsets']] for r in rows],axis=0)
        axes[1,0].plot(range(64),100*offsets,label=LABELS[arm],color=COLORS[arm],lw=1.7)
        axes[1,1].plot([0,64,128,256],100*np.array(summary['arms'][arm]['delay_means']),marker='o',label=LABELS[arm],color=COLORS[arm])
    axes[1,0].set(title='Boundary audit · same worlds at every phase',xlabel='Leading padding offset (bytes)',ylabel='Both answers exact (%)',ylim=(0,101),xlim=(0,63));axes[1,0].axhline(95,ls='--',color='#a33932',lw=1);axes[1,0].legend(fontsize=8,ncol=2)
    axes[1,1].set(title='Independent delay audit · all 64 phases',xlabel='Extra delay after records (bytes)',ylabel='Both answers exact (%)',ylim=(0,101));axes[1,1].axhline(90,ls='--',color='#a33932',lw=1);axes[1,1].legend(fontsize=8,ncol=2)
    fig.savefig(a.out/'SHoEbox_EXP01C_Results.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    fig.suptitle('SHoEbox EXP01-C — learning per observed byte and measured cost',fontsize=17,fontweight='bold')
    for arm in ARMS:
        fits=[r['arms'][arm] for r in train['rows']]
        xs=np.mean([[c['bytes']+250000 for c in f['curve']] for f in fits],axis=0)/1e6
        loss=np.mean([[c['dev']['native_rollout']['native_nll'] for c in f['curve']] for f in fits],axis=0)
        exact=np.mean([[c['dev']['native_rollout']['native_exact'] for c in f['curve']] for f in fits],axis=0)*100
        secs=np.mean([[c['cpu_seconds'] for c in f['curve']] for f in fits],axis=0)
        axes[0,0].plot(xs,loss,marker='o',color=COLORS[arm],label=LABELS[arm]);axes[0,1].plot(secs,loss,marker='o',color=COLORS[arm],label=LABELS[arm])
        axes[1,0].plot(xs,exact,marker='o',color=COLORS[arm],label=LABELS[arm])
    axes[0,0].set(title='Development NLL vs charged input exposure',xlabel='Million observed bytes (including warmup)',ylabel='Nats per answer target')
    axes[0,1].set(title='Development NLL vs continuation training CPU',xlabel='Training CPU seconds (evaluation excluded)',ylabel='Nats per answer target')
    axes[1,0].set(title='Development exactness vs charged input exposure',xlabel='Million observed bytes (including warmup)',ylabel='Both answers exact (%)',ylim=(0,101));axes[1,0].axhline(95,ls='--',color='#a33932',lw=1)
    values=[summary['arms'][arm]['inference_bytes_per_cpu_second']/1000 for arm in ARMS]
    bars=axes[1,1].bar([LABELS[k] for k in ARMS],values,color=[COLORS[k] for k in ARMS]);axes[1,1].bar_label(bars,fmt='%.1f',padding=3)
    axes[1,1].axhline(values[0]*.8,ls='--',color='#a33932',lw=1,label='80% of native gate')
    axes[1,1].set(title='Isolated inference · full 256-byte output each byte',ylabel='Thousand bytes / CPU second',ylim=(0,max(values)*1.2))
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.savefig(a.out/'SHoEbox_EXP01C_Learning_Cost.png',dpi=170);plt.close(fig)
    S=summary['arms'];P=S['pool'];COST=summary['costs']
    table='\n'.join(f"| {LABELS[k]} | {pct(S[k]['native_exact'])} | {pct(min(S[k]['seed_exact']))}–{pct(max(S[k]['seed_exact']))} | {pct(S[k]['first_answer'])} | {pct(S[k]['second_answer'])} | {pct(S[k]['both_query_orders'])} |" for k in ARMS)
    perseed='\n'.join(f"| {r['seed']} | "+' | '.join(pct(r['arms'][k]['test']['native_rollout']['native_exact']) for k in ARMS)+' |' for r in audit['rows'])
    diags='\n'.join(f"| {i+1} | "+' | '.join(pct(v) for v in row)+' |' for i,row in enumerate(P['diagnostic']))
    phases='\n'.join(f"| {LABELS[k]} | {pct(S[k]['offset_mean'])} | {pct(S[k]['worst_offset_mean'])} | {pct(S[k]['original_offset'])} |" for k in ARMS)
    delays='\n'.join(f"| {LABELS[k]} | "+' | '.join(pct(v) for v in S[k]['delay_means'])+' |' for k in ARMS)
    lesions='\n'.join(f"| {k} | {pct(v)} |" for k,v in {'unmodified_same_cases':P['lesion_unmodified'],**P['lesions']}.items())
    timing='\n'.join(f"| {LABELS[k]} | {S[k]['training_bytes_per_cpu_second']:,.0f} | {S[k]['inference_bytes_per_cpu_second']:,.0f} | {100*S[k]['inference_relative']:.1f}% | {S[k]['native_nll_per_answer']:.4f} |" for k in ARMS)
    gates='\n'.join(f"| {k.replace('_',' ')} | {'Pass' if v is True else 'Fail' if v is False else v} |" for k,v in P['gates'].items())
    report=f'''# SHoEbox EXP01-C / EXP22 — stable query-time access

16 November 2026 (America/Denver). Starting main: `bda7bc8`.
Production baseline: minimal corrected C, v19. **No candidate is promoted.**

The new pooled view recovers the **current query byte at 100% in both answer
positions** in the held-out diagnostic, but native two-answer exactness is only
**{pct(P['native_exact'])}**, versus **{pct(S['native']['native_exact'])}** for native.
First-query selection reaches **{pct(P['diagnostic'][0][0])}**; second-query
selection falls to **{pct(P['diagnostic'][1][0])}**. Key/value association remains
the problem. The original query signature and record-containing constituents
remain present in every inspected phase/delay case, including native-generated
answer prefixes. This supports a record-preserving readout experiment next.

![Primary results](SHoEbox_EXP01C_Results.png)

## Implemented comparison

All candidates retain EXP01-B's five rank-16 projections, one query/key selection,
and residual into the full 256-byte native decoder. The 9,535 fitted head
parameters are deducted from the core budget: 489,919 core + 9,535 readout =
499,454 actual parameters, versus 498,367 for nominal-500K native. This 0.22%
bank-rounding difference is reported; all three research heads match exactly.

- **Shared L3:** the EXP01-B local-credit frozen-core algorithm, now trained
  across offsets. Numerical output and training parity are tested against its
  original implementation. Five duplicated normalizers use the new common
  layout; the original EXP01-B implementation/checkpoints are unchanged.
- **Split frontier:** compose valid partial children at query time, without
  committing a constituent; use completed L3 state for key/value evidence.
- **Pooled views:** learn separate query/key/value pools over existing retained
  stable pairs, completed constituents and valid partial children. Each lane's
  first projection row also scores its source pool, with both gradient paths
  validated. Input coordinates carry content, scale, bounded occurrence age,
  source type and child role. They contain no task-defined record index.

The pool tests fixed stable-content channels, not the retained learned-value
half. Its global pooling can mix evidence from different records. No writer,
admission policy, persistent sequence state, optimizer moments, extra inference
iteration, symbolic store or task parser is added. All candidate cores freeze
after their shared warmup. Prefix-byte reconstruction is task-specific training
supervision; labels and oracle locations are absent from native inference.

## Registered held-out results

Three seeds (1/7/19), fresh grouped world folds, 250K warmup and 2M continuation
bytes per branch. All 64 leading offsets appear during training. Development
selected pooled views ({pct(selection['development_means']['pool'])} versus
{pct(selection['development_means']['split'])} for split) before audit access.
All four registered arms were audited; no audit-based retuning occurred.

The primary audit has 1,024 presentations / 256 concrete worlds, balanced over
64 offsets and independent record/query order. All presentation siblings remain
in one split. Seeds share audit worlds; repeated presentations are not additional
independent worlds. Generated answer bytes feed later state.

| Arm | Both answers exact | Seed range | First answer | Second answer | Both query orders |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

| Seed | Native | Shared L3 | Split frontier | Pooled views |
| --- | ---: | ---: | ---: | ---: |
{perseed}

The record-encoded learned positive control reaches 100% on the new fold;
query removal reduces its exactness to 0%. Its parsed memory, task-alphabet
output and separate budget make it a learnability control, not a native cost
comparator. The strongest input-order shortcut is 50%.

EXP01-B's 83.37% used fixed-offset training/evaluation. These offset-trained
numbers answer a different question; comparing 44.56% directly with 83.37%
would confound the recipe. The matched shared-L3 control here reaches
{pct(S['l3']['native_exact'])}. The pooled candidate improves on that control,
but remains below native whole-episode exactness.

## Failure localization

Teacher-fed inspection on 256 presentations per seed; decoded labels are only
diagnostics. These are neither native output scores nor an inference oracle.

| Answer | Selection | Query byte | Key 0 | Key 1 | Value 0 | Value 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{diags}

The current query is now readable in both positions. Key decoding and selection
collapse at the second query, and value reconstruction remains weak in both.
The source-presence audit independently reconstructs exact stable signatures
for the current two-byte query pair and each eight-byte constituent pair that
contains a complete record. On seed 7, 64 presentations / 16 worlds × 64 offsets
× four delays × two prefix modes, all three signatures remain present at both
answers (100%). Teacher-fed and native-generated prefixes agree. This is exact
presence, not proof of a trained decoder's recoverability. Oracle span locations
are used only by the external inspector and never by native prediction.

## Boundaries, delays and counterfactuals

Boundary-only tests prepend padding before records while preserving every
record-to-query distance. Same 64 presentations / 16 worlds at every phase.

| Arm | Mean over 64 offsets | Worst offset, seed mean | Offset zero |
| --- | ---: | ---: | ---: |
{phases}

Independent delays insert whole blocks after records, holding answer phase
fixed. Each delay is averaged across all 64 phases, on 16 presentations / four
worlds per seed. Delay worlds are separate from the boundary panel's worlds.

| Arm | +0 bytes | +64 | +128 | +256 |
| --- | ---: | ---: | ---: | ---: |
{delays}

Counterfactuals fork exactly the same frozen record-prefix state and change
query order, preserving all padding. Pooled both-orders exactness is only
{pct(P['both_query_orders'])}; it does not qualify query-specific binding.

Frozen pooled-view lesions, all on the same 256 presentations:

| Intervention | Native both-answer exactness |
| --- | ---: |
{lesions}

Query removal costs {100*P['query_ablation_drop']:.2f} percentage points,
below the registered 20-point requirement. Removing completed sources nearly
eliminates success; removing partial sources has little effect on this fixture.
Uniform source pooling also removes success. These establish dependence on the
learned source selection, without establishing correct key/value pairing.

## Learning, compute and memory

![Learning and cost](SHoEbox_EXP01C_Learning_Cost.png)

| Arm | Training bytes / CPU-s | Inference bytes / CPU-s | Inference vs native | Audit NLL / answer target |
| --- | ---: | ---: | ---: | ---: |
{timing}

Training rates include observation, reset/retirement, prefix labels and online
updates, excluding evaluations and dataset construction. Every observed padding
byte is charged. Candidate cores are frozen, so their training rate is not an
equivalent-algorithm speedup over joint native fitting. NLL is measured per
scored answer byte; learning exposure is all input bytes, not only labels.

Inference runs alone, pinned to one CPU: three × 65,536-byte repetitions, seed
7, native 256-byte prediction at every byte in a C++ loop. Pooled throughput is
{100*(1-P['inference_relative']):.1f}% below native, failing the <=20% cost gate.
This is a short CPU timing, with raw repetitions retained in `benchmark.json`.

| Resource | Pooled adapter |
| --- | ---: |
| Added fitted weights | 9,535 / 38,140 bytes |
| Online normalization state | 5,160 bytes |
| Persistent sequence-state addition | 0 bytes |
| Optimizer moments | 0 bytes |
| Owned readout cache + error buffer | 27,656 bytes |
| Complete adapter object (including head/cache/statistics/pointer) | 71,008 bytes |
| Existing core Stream + retained arrays, four scales | 289,152 bytes |
| Saved adapter checkpoint | {P['checkpoint_bytes']:,} bytes |

The source bound is 80 allocated scratch entries (at most 77 populated).
The backward call also uses fixed stack gradient arrays: 38,140 bytes for the
head plus 1,280 bytes for five input gradients, and smaller local derivatives.
These are temporary compute buffers, not hidden sequence memory. The listed
289,152 bytes are the named core state arrays, not total process memory; process
RSS and complete checkpoint sizes are retained in `train.json`.

| Training category | Actual charged bytes | Training CPU seconds |
| --- | ---: | ---: |
| Development pilot | {COST['pilot']['bytes']:,} | {COST['pilot']['cpu_seconds']:.2f} |
| Formal native/head comparison | {COST['formal']['bytes']:,} | {COST['formal']['cpu_seconds']:.2f} |
| Positive controls | {COST['positive_controls']['bytes']:,} | {COST['positive_controls']['cpu_seconds']:.2f} |
| Total | **{COST['total_training_bytes']:,}** | **{COST['total_training_cpu_seconds']:.2f}** |

Formal accounting includes shared reduced warmups once and separate native
warmups. Logical per-branch exposure sums to 27M bytes; actual formal work is
25.5M. The pilot used 1.75M bytes and was not reused as formal training. No batch
probe or unreported fitting run was used. Source inspection is read-only, with
zero fitted parameters and zero training bytes. Compilation, evaluation and
correctness tests are separate from the training CPU totals.

## Qualification and next implementation

| Registered gate | Pooled result |
| --- | --- |
{gates}

105 Python tests pass: 48 runtime, 50 existing research and seven new access
tests. The new 36,497 native numerical/lifecycle checks pass optimized and
ASan/UBSan builds. The complete existing native runtime/sanitizer suite passes.
Checks cover gradients including tied pooling, causal labels, source-read
immutability, native/EXP01-B parity, exact checkpoint continuation across
partial boundaries and independent first/second-answer scoring. LeakSanitizer
was explicitly disabled for this host; ASan and UBSan remained enabled.

No production runtime file changes. Historical checkpoint policies and EXP01-B
remain available. Binding, boundary, delay, counterfactual and cost gates fail;
joint-credit, language/copy/reconstruction and 1M/5M promotion campaigns remain
gated, not passed or silently waived.

**Next: EXP01-D / EXP22 — preserve constituent-level key/value association.**
Decode keys and values together from individual existing retained constituents
before pooling, then apply shared query-conditioned weights to both. Handle
within-constituent phase explicitly and compare fixed stable versus learned
content channels at matched weights/state. Keep the now-readable current query
separate. First verify whether complete record pairs can be decoded across
phases and both queries; do not infer a need for new storage from a failed
global pool. Register fresh budgets/folds, preserve first/subsequent-query and
counterfactual gates, and return to joint credit only after native binding works.

## Primary literature and evidence

- [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/), section 3:
  separates address and returned-value encodings. Adapted here to neural state.
- [Dai et al., ACL 2019](https://aclanthology.org/P19-1285/), sections 3.2–3.3:
  completed-context reuse across segment boundaries. No new cache is added here.
- [Sun et al., 2023](https://arxiv.org/abs/2307.08621), section 2.1:
  within-chunk versus recurrent contributions motivate the split-frontier view;
  this experiment does not implement RetNet.
- [Yang et al., ICLR 2025](https://openreview.net/forum?id=r8H7xhYPwz):
  targeted memory correction and forgetting motivate separating access failure
  from a write-rule problem. No delta-rule mechanism is implemented here.

The SHoEbox measurements above are original to this experiment, not claims from
those papers. [Preregistered protocol](PROTOCOL.md) ·
[Implementation and reproduction](../../experiments/query_access/README.md).
Raw plans, source hashes, curves, audit results, lesions, delays, counters,
timings, presence checks and verification logs accompany this report.
'''
    (a.out/'SHoEbox_EXP01C_Query_Access.md').write_text(report)
    print(json.dumps({k:dict(exact=v['native_exact'],first=v['first_answer'],second=v['second_answer'],relative_speed=v['inference_relative']) for k,v in S.items()},indent=2))

if __name__=='__main__':main()
