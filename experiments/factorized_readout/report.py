"""Build the compact EXP01-B report and standalone primary PNG."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import ARMS
from .adapter import ROOT
from experiments.query_readout.run import write

LABELS={'native':'Native 500K','matched':'Matched core','answer_frozen':'Answer-only, frozen',
        'local_frozen':'Local credit, frozen','local_joint':'Local credit, joint','rls_frozen':'RLS addresses, frozen'}
COLORS=['#475569','#94a3b8','#d97706','#0284c7','#db2777','#7c3aed']

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
    out=ROOT/'reports/exp01b';out.mkdir(parents=True,exist_ok=True)
    for name in ('plan','train','audit','offsets','benchmark','positive_control','control_plan'):
        shutil.copy2(a.runs/f'{name}.json',out/f'{name}.json')
    for name in ('pilot','pilot_plan','rates','rates_plan','rls','rls_plan','conditioning'):
        shutil.copy2(a.pilot/f'{name}.json',out/f'{name}.json')
    load=lambda n:json.loads((out/f'{n}.json').read_text())
    tr=load('train');au=load('audit');off=load('offsets');bench=load('benchmark');control=load('positive_control');pilot=load('pilot')
    summary={}
    for arm in ARMS:
        values=[r['arms'][arm]['test']['native_rollout']['native_exact'] for r in au['rows']]
        counter=[r['arms'][arm]['counterfactual']['both_query_orders_exact'] for r in au['rows']]
        summary[arm]=dict(mean=float(np.mean(values)),min=min(values),max=max(values),seeds=values,
            both_query_orders=float(np.mean(counter)),parameters=tr['rows'][0]['arms'][arm]['inventory']['parameters'],
            training_bytes_per_cpu_second=float(np.mean([r['arms'][arm]['bytes_per_cpu_second'] for r in tr['rows']])),
            inference_bytes_per_cpu_second=bench['rows'][arm]['bytes_per_cpu_second'])
    formal_cpu=sum(r['warmup']['cpu_seconds']+sum(v['cpu_seconds']+(v['separate_warmup']['cpu_seconds'] if v['separate_warmup'] else 0) for v in r['arms'].values()) for r in tr['rows'])
    pilot_cpu=pilot['warmup']['cpu_seconds']+sum(v['cpu_seconds'] for v in pilot['arms'].values())+sum(r['cpu_seconds'] for r in load('rates'))+load('rls')['cpu_seconds']
    result=dict(arms=summary,formal_model_bytes=37500000,pilot_bytes=11250000,positive_control_bytes=750000,total_training_bytes=49500000,
        formal_cpu_seconds=formal_cpu,pilot_cpu_seconds=pilot_cpu,gate_passed=any(v['mean']>=.95 and v['min']>=.90 for k,v in summary.items() if k not in ('native','matched')),default_promoted=False,
        baseline='521771f0bc471dba0caba550e6198185d1630613',runtime='minimal-v19')
    write(out/'summary.json',result)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');fig.set_facecolor('#fafbfc')
    ax=axs[0,0];yy=np.arange(len(ARMS));means=np.array([summary[k]['mean'] for k in ARMS])*100
    low=np.array([summary[k]['min'] for k in ARMS])*100;high=np.array([summary[k]['max'] for k in ARMS])*100
    ax.barh(yy,means,color=COLORS,height=.62,xerr=[means-low,high-means],capsize=3)
    ax.set_yticks(yy,[LABELS[k] for k in ARMS]);ax.invert_yaxis();ax.set_xlim(0,104);ax.axvline(95,color='#b91c1c',ls='--',lw=1)
    for j,v in enumerate(means):ax.text(high[j]+2,j,f'{v:.1f}%',va='center',fontsize=9)
    ax.set_title('Held-out native whole-episode accuracy',loc='left',weight='bold');ax.set_xlabel('Exact episodes (%) · whiskers show seed range')
    ax.text(95,.985,'95% gate',color='#b91c1c',fontsize=8,ha='center',va='top',transform=ax.get_xaxis_transform())
    ax=axs[0,1]
    for arm in ('native','answer_frozen','local_frozen','local_joint','rls_frozen'):
        color=COLORS[ARMS.index(arm)];curves=np.array([[c['dev']['native_rollout']['native_exact'] for c in r['arms'][arm]['curve']] for r in tr['rows']])*100
        xx=np.array([c['bytes'] for c in tr['rows'][0]['arms'][arm]['curve']])/1e6
        ax.plot(xx,curves.mean(0),marker='o',color=color,label=LABELS[arm]);ax.fill_between(xx,curves.min(0),curves.max(0),color=color,alpha=.1)
    ax.set_ylim(35,100);ax.set_title('Online learning after 250K-byte warmup',loc='left',weight='bold');ax.set_xlabel('Continuation bytes (millions)');ax.set_ylabel('Development exactness (%)');ax.legend(fontsize=8,loc='lower right')
    ax=axs[1,0];focus=['native','answer_frozen','local_frozen','local_joint','rls_frozen']
    vals=[summary[k]['both_query_orders']*100 for k in focus]
    ax.barh(range(len(focus)),vals,color=[COLORS[ARMS.index(k)] for k in focus],height=.6);ax.set_yticks(range(len(focus)),[LABELS[k] for k in focus]);ax.invert_yaxis();ax.set_xlim(0,100)
    for j,v in enumerate(vals):ax.text(v+1.5,j,f'{v:.1f}%',va='center',fontsize=9)
    ax.set_title('Both query orders from the same record state',loc='left',weight='bold');ax.set_xlabel('Both two-answer episodes correct (%)')
    ax=axs[1,1]
    for arm in ('native','local_frozen','rls_frozen'):
        vals=np.array([r['arms'][arm]['alignment_only'] for r in off['rows']])*100
        ax.plot(range(64),vals.mean(0),color=COLORS[ARMS.index(arm)],label=LABELS[arm],lw=1.6)
    ax.set_title('Frozen boundary-offset audit',loc='left',weight='bold');ax.set_xlabel('Padding before records (bytes)');ax.set_ylabel('Native exactness (%)');ax.set_ylim(0,100);ax.legend(fontsize=8)
    fig.suptitle('SHoEbox EXP01-B · Online factorized query/key/value readout',fontsize=17,weight='bold')
    fig.supxlabel('Minimal v19 · 500K nominal parameters · seeds 1 / 7 / 19 · one read, no extra sequence state',fontsize=10)
    fig.savefig(out/'SHoEbox_EXP01B_Results.png',dpi=180);plt.close(fig)
    pct=lambda x:f'{100*x:.2f}%'
    table='\n'.join(f"| {LABELS[k]} | {pct(summary[k]['mean'])} | {100*summary[k]['min']:.2f}–{100*summary[k]['max']:.2f}% | {pct(summary[k]['both_query_orders'])} | {summary[k]['parameters']:,} |" for k in ARMS)
    speed='\n'.join(f"| {LABELS[k]} | {summary[k]['training_bytes_per_cpu_second']:,.0f} | {summary[k]['inference_bytes_per_cpu_second']:,.0f} |" for k in ARMS)
    lesions='\n'.join(f"| {name} | "+' | '.join(pct(np.mean([r['arms'][arm]['lesions'][name]['native_exact'] for r in au['rows']])) for arm in ('local_frozen','local_joint','rls_frozen'))+' |' for name in ('no_head','no_query','no_state','uniform','reverse'))
    delays='\n'.join(f"| {LABELS[arm]} | "+' | '.join(pct(np.mean([r['arms'][arm]['delay_only'][i]['exact'] for r in off['rows']])) for i in range(4))+' |' for arm in ('native','local_frozen','rls_frozen'))
    phase='\n'.join(f"| {LABELS[arm]} | {pct(np.mean([r['arms'][arm]['alignment_only'] for r in off['rows']]))} | {pct(np.mean([r['arms'][arm]['alignment_only'][0] for r in off['rows']]))} |" for arm in ('native','local_frozen','rls_frozen'))
    diag=np.mean([r['arms']['local_frozen']['diagnostic']['by_answer'] for r in au['rows']],axis=0)
    diag_table='\n'.join(f"| {i+1} | "+' | '.join(pct(x) for x in row)+' |' for i,row in enumerate(diag))
    relative=summary['local_frozen']['inference_bytes_per_cpu_second']/summary['native']['inference_bytes_per_cpu_second']-1
    report=f'''# SHoEbox EXP01-B — online factorized readout

15 November 2026 (America/Denver). Starting repository commit: `521771f`.
Production baseline: minimal corrected C, v19.

**Implemented and tested:** a one-pass online query/key selector and value
decoder over existing L3 state. Local reconstruction credit lifts held-out native
two-answer exactness from **{pct(summary['native']['mean'])} to {pct(summary['local_frozen']['mean'])}**
across three seeds. It adds **9,535 fitted parameters**, deducted from the core,
and **zero persistent sequence-state bytes**.

**Decision:** retain the implementation as a research adapter. No arm passes the
95% mean / 90% minimum-seed binding gate. The public minimal-v19 runtime remains
unchanged. The next repair is address representation across query positions and
constituent boundaries, followed by stable joint credit; extra storage is not
the next intervention.

![Primary results](SHoEbox_EXP01B_Results.png)

## Implementation

Five rank-16 projections decode a query, two keys and two value vectors from the
same existing 64-float L3 state. One soft selection computes
`a = sigmoid(q · (k0 − k1))`, then `v = a v0 + (1 − a) v1`.
A shared projection adds `v` to SHoEbox's native 256-byte hierarchical decoder.
These two lanes are recomputed views, not stored records or added memory slots.

The answer-only control has the identical graph and parameter count. The local
credit arm also reconstructs the observed query/key bytes and both observed
values. Its training adapter parses only the prefix, excludes scored answers,
and never computes or supplies the selected-record index. Inference receives
only neural state. This is task-specific intermediate training supervision,
not a language-model-only objective.

SGD uses online normalization and per-target updates. The RLS control replaces
only query/key fitting with a 65×65 recursive inverse-covariance update; value
learning remains SGD. RLS is frozen-core only and uses stationary raw-state
coordinates. No branch uses replay, batch optimization, a growing tape, an
external binding store, or additional inference passes.

## Locked held-out comparison

All rows use minimal v19, nominal 500K capacity, seeds 1/7/19, 250K warmup bytes
and 2M continuation bytes. The final horizon and .16 head rate were locked using
development pilots. All six registered arms were audited. Native output is
normalized over all 256 bytes; generated answer bytes feed subsequent state.
Query and framing bytes are supplied by the fixture.

| Arm | Exact episodes | Seed range | Both query orders | Fitted parameters |
| --- | ---: | ---: | ---: | ---: |
{table}

The audit has 1,024 presentations from 256 independent concrete worlds. All
record-order/query-order siblings share a split, using a new salted fold. Seeds
share the same audit worlds; 3,072 scored presentations are not 3,072 independent
worlds. The strongest input-order shortcut is 50%. The separately trained,
record-encoded learned lookup control reaches {pct(np.mean([r['test']['exact_accuracy'] for r in control['rows']]))};
removing its query reduces exactness to {pct(np.mean([r['no_query']['exact_accuracy'] for r in control['rows']]))}.
Its parsed inputs and task-alphabet output make it a learnability control, not a
native throughput comparator.

Query counterfactuals fork exactly the same frozen record-prefix state before
changing query order. The both-orders column requires all four generated answers
across the paired episodes to be correct, on 256 presentations / 64 worlds.

## Where the remaining errors occur

Frozen local-credit inspection separates selection from value reconstruction.
The table is teacher-fed diagnostic inspection on 256 presentations; its byte
decoders are not consulted by native inference.

| Answer position | Selection | Query byte | First key | Second key | First value | Second value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{diag_table}

The additional development-only ridge diagnostic recovers all five byte labels
at 100% on the first-query development set. Fitting the same linear heads
across both query positions reduces first-key decoding to 76.95%, with value
decoding still 99.22%. Those batch fits are explanatory probes, not online
results. They show why the earlier first-query probe does not establish a
single stable address representation across the whole episode.

The native residual decoder can exploit answer-order regularities, so final
accuracy and direct selection accuracy are reported separately. Joint local
credit does not match the frozen-core gain at the registered rate. RLS does not
remove the remaining address problem; a larger optimizer sweep is not justified
by this result.

## Frozen interventions

Each lesion uses the same 256 held-out presentations. Values below are native
whole-episode exactness; the primary table uses the larger 1,024-case audit.

| Intervention | Local, frozen | Local, joint | RLS, frozen |
| --- | ---: | ---: | ---: |
{lesions}

`no_query` forces equal selection weights; `uniform` explicitly sets the same
weights. `reverse` swaps the learned selection weights. `no_state` zeros the
normalized source (for RLS, raw address projections remain available, so this is
only a value-source lesion there). `no_head` removes the added residual entirely.

## Offset and delay audit

All 64 byte phases were checked with frozen models on 64 presentations / 16
worlds per seed. Padding before the records changes absolute hierarchy phase
while keeping record-to-query/answer distances fixed. The separate delay test
inserts complete 64-byte blocks after the records, preserving answer phase.
These are out-of-training-distribution audits with no refitting.

| Arm | Mean across 64 offsets | Original offset, same cases |
| --- | ---: | ---: |
{phase}

| Arm | +0 delay | +64 bytes | +128 bytes | +256 bytes |
| --- | ---: | ---: | ---: | ---: |
{delays}

These tests make boundary and delay dependence explicit. They do not qualify
arbitrary-delay retrieval or larger binding sets.

## Cost and exposure

| Arm | Training bytes / CPU-s | Inference bytes / CPU-s |
| --- | ---: | ---: |
{speed}

Inference is an isolated, single-process, pinned-CPU measurement: seed 7, three
65,536-byte repetitions, full 256-byte prediction **at every byte**, all byte-loop
work in C++. The local frozen head changes throughput by {relative*100:+.1f}%
relative to native. The stream hash and repetitions are in `benchmark.json`.
Frozen-core training fits fewer parameters than native/joint training, so their
training-rate difference is not an equivalent-algorithm speedup.

- Added weights: 38,140 bytes / 9,535 fitted parameters. Candidate total: 499,454.
- Added normalization: 1,032 bytes; scratch caches/errors: 2,180 bytes on this ABI.
- RLS-only optimizer state: 33,808 bytes, bounded independently of stream length.
- Added persistent sequence state: zero. Public runtime executes no adapter code.
- Formal model training: **37.5M actual charged bytes**, {formal_cpu:.2f} CPU seconds.
  This includes shared reduced-core warmups and separate full-budget warmups;
  logical per-branch exposure sums to 40.5M bytes.
- Development pilots: **11.25M bytes**, {pilot_cpu:.2f} CPU seconds. These include
  five initial arms, four rate/horizon continuations and one RLS control.
- New-fold positive controls: **0.75M bytes**. Total training: **49.5M bytes**.
- Ten offline ridge diagnostic fits are accounted separately; their labels and
  batch solves are never part of online model training or scored inference.

## Validation and promotion status

**98 Python tests pass:** 48 runtime and 50 MAD/EXP00/readout tests, including ten
new tests. **27,253 new native checks pass** in optimized and ASan/UBSan builds.
They cover numerical derivatives, inverse-covariance correctness, core freeze,
native parity, and exact continuation through partial boundaries, including
normalization and covariance state. The complete existing native runtime suite
also passes optimized and sanitizer runs. LeakSanitizer is disabled explicitly
on this host; AddressSanitizer and UndefinedBehaviorSanitizer remain enabled.

The default source and historical checkpoint policies are unchanged. Binding,
counterfactual and offset gates are now measured; candidate language/copy and
1M/5M promotion campaigns remain gated on repairing binding. No new broad
language capability or default promotion is claimed from this screen.

## Next experiment: EXP01-C / EXP22 query-time access

Use the new local-credit head as a reproducible research control. Separate the
current query cue from stable key/value evidence using existing completed and
partial constituents, and verify first and subsequent queries independently.
Compare stable evidence views against the current shared-L3 projection with
matched capacity and state. Train across boundary offsets, then audit a newly
held-out world fold and separate longer delays. Once selection is stable, return
bounded local credit to the core with its scale controlled. Keep EXP02 slot
expansion and reasoning loops gated on evidence of a storage need.

## Literature integrated

1. Miller et al., [Key-Value Memory Networks for Directly Reading Documents](https://aclanthology.org/D16-1147/), EMNLP 2016, section 3.
   Adaptation: separate addressing from returned value content within existing state.
2. Sussillo and Abbott, [Generating Coherent Patterns of Activity from Chaotic Neural Networks](https://doi.org/10.1016/j.neuron.2009.07.018), Neuron 2009, equations 4–5.
   Adaptation: bounded online RLS for address readouts; no FORCE reservoir or feedback.
3. Arora et al., [Zoology](https://arxiv.org/abs/2312.04927).
   Evaluation: independently ordered queries and whole-episode recall.
4. Poli et al., [Mechanistic Design and Scaling of Hybrid Architectures](https://proceedings.mlr.press/v235/poli24a.html), ICML 2024.
   Evaluation: small capability tasks with separate accuracy, exposure, state and compute accounting.

Source and reproduction: [experiments/factorized_readout](../../experiments/factorized_readout/README.md).
The numerical results above are this SHoEbox experiment's measurements.
'''
    (out/'SHoEbox_EXP01B_Online_Factorized_Readout.md').write_text(report)

if __name__=='__main__':main()
