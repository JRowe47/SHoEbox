import argparse
import json
import math
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from common import atomic,file_hash,Model,install

LABELS={'baseline':'Unchanged rates / routing','quarter_all':'Quarter learning rates','zero_bias':'Zero usage bias; updates off',
 'quarter_head':'Quarter readout rate only','quad_all':'4× all learning rates','quad_fast':'4× fast rate',
 'quad_slow':'4× slow rate','no_attraction':'No prototype attraction','freeze_bias':'Freeze existing usage bias',
 'no_aux':'No auxiliary objectives','upper_aux_tenth':'Upper contrastive weight ×0.1'}

def paired(a,b,seed=7):
    assert [d['row_idx'] for d in a]==[d['row_idx'] for d in b]
    w=np.array([d['targets'] for d in a]);v=np.array([y['nll']-x['nll'] for x,y in zip(a,b)])
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(w),(10000,len(w)))
    boot=(w[idx]*v[idx]).sum(axis=1)/w[idx].sum(axis=1)
    return dict(delta=float(np.average(v,weights=w)),ci95=np.quantile(boot,[.025,.975]).tolist(),
        improved_documents=int((v<0).sum()),documents=len(w),targets=int(w.sum()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 work=a.base/'diagnostic_40m';out=a.out;out.mkdir(parents=True,exist_ok=True);prefix='SHoEbox_100M_40M_Plateau_Diagnosis'
 read=lambda path:json.loads(Path(path).read_text())
 protocol=read(work/'protocol.json');selection=read(work/'selection.json')
 screen={f.stem:read(f) for f in sorted((work/'results/screen').glob('*.json'))}
 final={arm:read(work/'results'/arm/'results.json') for arm in selection['confirmed_arms']}
 parent=read(work/'results/probe_40_full.json');prior=read(work/'results/probe_20_full.json')
 snapshot=read(work/'results/snapshot_controls.json');gradient=read(work/'results/gradient_components.json')['40M']
 audit_start=read(work/'results/probe_40_audit_full.json')
 assert len(screen)==11
 assert all(r['targets']==3_000_000 and r.get('audit') for r in final.values())
 for r in final.values():
    assert r['metrics'][-1]['routing']['levels'][0]['events']==43_000_000
    for name in ['development','fresh','audit']:
        e=r[name];value=sum(d['nll']*d['targets'] for d in e['documents'])/sum(d['targets'] for d in e['documents'])
        assert abs(value-e['nll'])<1e-12
 hashes={str(path.relative_to(a.base)):file_hash(path) for path in [a.base/'run_100m/latest.shoe',a.base/'run_20m_40m/latest.shoe']}
 assert hashes['run_20m_40m/latest.shoe']==protocol['parent_sha256']
 assert hashes['run_100m/latest.shoe']=='234d274c887908badd5aad2afd71b514e8f78076c6ca31b08f8fb24a1c97e2db'
 assert file_hash(work/'data/train.bin')==protocol['training_data_sha256']
 comparison={arm:{split:paired(final['baseline'][split]['documents'],r[split]['documents']) for split in ['development','fresh','audit']} for arm,r in final.items() if arm!='baseline'}
 historical=paired(prior['documents'],parent['documents']);trace=np.array(parent['trace']);g=np.array(gradient['values'])
 ratios=(g[1:,2]/g[1:,1]).tolist()
 baseline=final['baseline'];best=min(final,key=lambda arm:final[arm]['audit']['nll'])
 install(work/'runtime_followup');usage={}
 for arm in final:
    with Model.load(work/'results'/arm/'latest.shoe') as model:
        levels=[]
        for l,n in enumerate(model.stats()['experts']):
            bank=np.zeros((n,5));model._check(model._lib.diag_bank(model._alive(),l,bank.ctypes.data))
            delta=bank[:,0]-np.array(parent['banks'][l])[:,0];assert np.all(delta>=0)
            freq=delta/delta.sum();nz=freq[freq>0]
            levels.append(dict(new_selections=int(delta.sum()),experts_used=int((delta>0).sum()),experts=n,
                effective=float(np.exp(-(nz*np.log(nz)).sum())),maximum_share=float(freq.max()),
                selection_quantiles=np.quantile(delta,[0,.1,.5,.9,1]).tolist()))
        usage[arm]=levels
 summary=dict(parent_checkpoint_sha256=protocol['parent_sha256'],historical_change=historical,
    frozen_development_nll=parent['nll'],zero_fast_bias_development_nll=snapshot['no_bias_fast']['nll'],
    fast_bias_override_fraction=float(trace[0,15]/trace[0,14]),
    fast_gate_derivative=float(trace[0,1]/trace[0,0]),
    fast_state_saturated_fraction=float(trace[0,6]/(64*trace[0,0])),
    upper_contrastive_to_language_gradient_energy_ratio=ratios,screen={k:{s:v[s]['nll'] for s in ['development','fresh']} for k,v in screen.items()},
    confirmation={k:{s:v[s]['nll'] for s in ['development','fresh','audit']} for k,v in final.items()},
    paired_confirmation=comparison,confirmed_arms=selection['confirmed_arms'],best_audit_arm=best,
    parent_audit_nll=audit_start['nll'],new_training_usage=usage,source_checkpoint_hashes_after=hashes,
    audit_targets=1_000_000,development_targets=262144,screen_targets=524288,
    verification=dict(runtime=read(work/'runtime_verification.json'),followup=read(work/'followup_verification.json'),checkpoints=read(work/'checkpoint_verification.json')))
 summary['diagnostic_source_hashes']={str(path.relative_to(Path(__file__).parent)):file_hash(path) for path in sorted(Path(__file__).parent.iterdir()) if path.suffix in {'.py','.cpp','.md'}}
 summary['runtime_source_hashes']={str(path.relative_to(work)):file_hash(path) for folder in ['runtime','runtime_followup'] for path in sorted((work/folder).iterdir()) if path.suffix in {'.cpp','.hpp'}}
 atomic(out/f'{prefix}_Results.json',summary)
 screen_rows='\n'.join(f"| {LABELS[arm]} | {r['development']['nll']:.6f} | {r['fresh']['nll']:.6f} | {r['development']['nll']-screen['baseline']['development']['nll']:+.6f} |" for arm,r in sorted(screen.items(),key=lambda item:item[1]['development']['nll']))
 final_rows='\n'.join(f"| {LABELS[arm]} | {r['development']['nll']:.6f} | {r['fresh']['nll']:.6f} | {r['audit']['nll']:.6f} |" for arm,r in final.items())
 ci_rows='\n'.join(f"| {LABELS[arm]} | {c['audit']['delta']:+.6f} | [{c['audit']['ci95'][0]:+.6f}, {c['audit']['ci95'][1]:+.6f}] | {c['audit']['improved_documents']}/{c['audit']['documents']} |" for arm,c in comparison.items())
 probes={f.stem.replace('probe_40_',''):read(f) for f in (work/'results').glob('probe_40_*.json')}
 upper_rows='\n'.join(f"| L{l} | {parent['routing'][l]['effective']:,.0f} / {len(parent['banks'][l]):,} | {probes['swap'+str(l)]['nll']-parent['nll']:+.6f} | {ratios[l-1]:.2f}× |" for l in [1,2,3])
 report=f'''# SHoEbox 100M: diagnosis of the 20–40M slowdown

The model is still learning, but its continuation rates and mature fast-bank usage correction impose a measurable loss penalty. The upper expert banks have a separate learning-efficiency problem; changing their auxiliary weights alone did not improve this short screen.

## What is actually happening

**1. The continuation schedule stayed too aggressive for this checkpoint.** The 100M run used a cosine schedule ending at 20M, then retained fast/query **0.0008**, slow **0.008**, and head **0.016** for 20–40M. These are four times the selected D reference's documented continuation rates: **0.0002 / 0.002 / 0.004**. That was the schedule used in this run, rather than an architectural default changing itself. It coincides with the point where the loss curve flattens.

Matched fresh-data interventions support an update-noise explanation. After 1M additional bytes, quarter-rate training reached **{screen['quarter_all']['development']['nll']:.6f}**, versus **{screen['baseline']['development']['nll']:.6f}** for unchanged continuation. Increasing all rates fourfold worsened loss to **{screen['quad_all']['development']['nll']:.6f}**; increasing the fast rate alone also worsened it. Reducing only the head rate reached **{screen['quarter_head']['development']['nll']:.6f}**, showing that the shared readout contributes to the noise. This is evidence from matched continuations, not a claim that the 1M reference schedule is universally optimal at 100M.

**2. Mature fast-bank balancing redirects useful routes.** On a frozen 40M checkpoint, removing only the fast-bank bias changes held-out loss from **{parent['nll']:.6f} to {snapshot['no_bias_fast']['nll']:.6f}**, without training or changing expert weights. Removing all biases gives **{probes['no_bias']['nll']:.6f}**. Scaling all biases to one quarter gives **{snapshot['quarter_bias_all']['nll']:.6f}**. The penalty is localized to the fast bank; removing L1, L2 or L3 bias separately changes loss by less than 0.0003.

In a sampled exact-search inspection, usage bias overrides the fast bank's best semantic match **{100*summary['fast_bias_override_fraction']:.2f}%** of the time. Its 11,816 experts are checked over only 256 selections: expected uniform usage is **{256/11816:.5f}** selections per expert, and the nominal 10%-of-uniform starvation threshold is **{.1*256/11816:.5f}**. With integer counts, the post-warmup test is therefore effectively “absent from this short window,” not sustained starvation. Most experts receive the same positive nudge; mean subtraction turns this into a relative penalty on recently used experts. All fast experts had already passed their 256-example warmup by the recorded 3.5M checkpoint. The weak correction remains afterward.

Simply freezing the existing biases was worse than the control, and disabling prototype attraction was essentially neutral. The useful change is to reduce the mature bias pressure, rather than freeze the router's current geometry wholesale.

**3. Upper expert capacity is not translating into language-specific computation.** The hierarchy itself helps: masking L1/L2/L3 increases loss by **{probes['mask1']['nll']-parent['nll']:.6f} / {probes['mask2']['nll']-parent['nll']:.6f} / {probes['mask3']['nll']-parent['nll']:.6f}**. However, cyclically replacing the selected L2 or L3 expert by its next neighbor has almost no effect. The shared state and readout paths contribute while those expert identities contribute little.

| Level | Effective / available experts on development text | NLL change from swapping its experts | Contrastive / language gradient energy |
|---|---:|---:|---:|
{upper_rows}

The gradient audit recomputed objective-specific content derivatives on 65,778 fresh training targets. L2 and L3 contrastive gradients dominate language-gradient energy, but their aggregate alignment with language gradients is near orthogonal, not strong direct opposition. The issue is weak task-specific learning relative to the local objective. Reducing only L2/L3 contrastive weight tenfold produced **{screen['upper_aux_tenth']['development']['nll']:.6f}** after 1M, versus **{screen['baseline']['development']['nll']:.6f}** for control; removing all auxiliary objectives also did not improve the screen. This is a structural follow-up, not the demonstrated immediate remedy.

## Controlled continuation results

Every arm started from the identical saved 40M model. All consumed the same document-aware, seed-7 shuffled fresh FineWeb stream. Eleven arms received 1M next-byte targets each. The two lowest sums of development and fresh-screen NLL were continued alongside the control to 3M. The separate 1M-byte audit set was first scored at confirmation.

### One-million-byte screen

| Change | Original development NLL | Fresh-screen NLL | Development change vs control |
|---|---:|---:|---:|
{screen_rows}

“Zero usage bias” clears all saved utilization biases and disables their subsequent updates, while means, prototypes, experts, and other learning remain active. “Freeze existing usage bias” retains the original bias values and disables updates. “No auxiliary objectives” disables contrastive, SDR-prediction and reconstruction learning. Quarter-rate changes preserve the prototype-attraction rate and objective weights.

### Three-million-byte confirmation

All rows now represent 43M cumulative training bytes on diagnostic copies.

| Setting | Original development NLL | Fresh-screen NLL | Previously unused audit NLL |
|---|---:|---:|---:|
{final_rows}

Original 40M audit loss: **{audit_start['nll']:.6f}**. The audit is a different text sample from the original development set; compare methods within each column.

| Setting vs unchanged control | Audit NLL difference | Paired document-bootstrap 95% interval | Audit documents improved |
|---|---:|---:|---:|
{ci_rows}

Intervals resample documents, weighted by their byte targets, with 10,000 bootstrap draws. They measure variation across these documents, not variation across independent training seeds.

### How much of the plateau remains?

The improvements mostly lower the loss quickly. From +1M to +3M, development NLL improves only **{screen['baseline']['development']['nll']-final['baseline']['development']['nll']:.6f}** for control, **{screen['quarter_all']['development']['nll']-final['quarter_all']['development']['nll']:.6f}** for quarter rates, and **{screen['zero_bias']['development']['nll']-final['zero_bias']['development']['nll']:.6f}** with usage bias removed. These interventions remove avoidable penalties; this experiment has not established a restored steep long-run learning slope. The weak task-specific contribution of upper experts remains a separate target.

## Ruled-out explanations

- **Complete learning lock:** 20–40M loss improved by {-historical['delta']:.6f} nats/byte; {historical['improved_documents']} of {historical['documents']} development documents improved. The paired interval for 40M minus 20M is [{historical['ci95'][0]:+.6f}, {historical['ci95'][1]:+.6f}]. Progress is small but real.
- **Route collapse:** the frozen 40M fast router used {parent['routing'][0]['distinct']:,} experts, with {parent['routing'][0]['effective']:,.0f} entropy-equivalent experts out of 11,816. Every expert received training selections. The evidence bundle records additional-use distributions for all confirmed arms.
- **Tanh or gate saturation:** fast-state coordinates above absolute 0.95 were {100*summary['fast_state_saturated_fraction']:.3f}%; the mean gate derivative was {summary['fast_gate_derivative']:.6f}, close to its 0.25 maximum. Slow input, content and state saturation was negligible.
- **Recent centering-mean drift as the main penalty:** replacing the 40M centering means with the 20M means changes loss only to {snapshot['mean_20M']['nll']:.6f}.
- **Dead memory or hierarchy:** removing retained recall raises loss to {probes['no_recall']['nll']:.6f}; removing all slow modulation and slow retained-read contributions raises it to {probes['fast_only']['nll']:.6f}. These components are materially used.

## Recommended next change

Restore D's continuation rates: **fast/query 0.0002, slow 0.002, head 0.004**. Replace the mature fast bank's 256-selection absence test with a bank-size-aware usage estimate that responds to persistent starvation. Retain initial coverage training and avoid forcing frequent contexts away from their best specialist.

The separate rate and bias changes have controlled evidence. Their combination was not tested here. After those changes, address upper-level language credit and verify that expert-specific computation becomes useful before adding more experts or routing depth. Increasing every slow learning rate was not effective in this screen.

## Reproducibility and verification

- Parent: 100,009,791 parameters, width 64, routing D, banks 11,816 / 5,908 / 3,544 / 2,365; 40,000,000 trained targets.
- Original development: 70 documents / 262,144 targets. Fresh screen: {len(protocol['fresh_screen_documents'])} documents / 524,288 targets. New audit: {len(protocol['fresh_audit_documents'])} documents / 1,000,000 targets. Training: {len(protocol['train_documents'])} documents / exactly 3,000,000 targets.
- All new documents are disjoint from prior training/development by document ID and full-text hash. Diagnostic training, screen and audit sets are disjoint by document. Partial final documents are excluded from the next split.
- The diagnostic runtime with neutral controls matched the original runtime in loss and serialized state after 8,192 training targets. The follow-up runtime passed the same check.
- The gradient decomposition agrees with the actual local flush to a maximum absolute error below 6e-8. Objective-component sums agree to below 1e-7 in the measured workload. Diagnostics run outside the production byte loop.
- All three 43M diagnostic checkpoints reload to their exact recorded development loss and pass a 1,024-target continuation comparison in loss and serialized state. These disposable verification targets are excluded from the reported models.
- Regression checks passed: 2,093 native, 25 Python, 19 semantic-lace and 44 routing checks. Native, lace and routing sanitizer runs passed ASan/UBSan; LeakSanitizer was explicitly disabled for this managed host.
- Both source checkpoints retain their original hashes. The 40M model remains at SHA-256 `{protocol['parent_sha256']}`.
- Experimental bias/auxiliary controls are restored by the diagnostic runner; they are not new serialized production policies. No global-default or GitHub change was made by this diagnosis.

The attached PNG shows the original late curve, matched continuations, fresh audit differences, and the frozen fast-bias intervention. The results archive includes exact per-document scores, routes/usage statistics, source hashes, protocols, verification, and the small diagnostic data splits. It excludes trained weight snapshots.
'''
 (out/f'{prefix}.md').write_text(report)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#0f172a','axes.labelcolor':'#334155','xtick.color':'#475569','ytick.color':'#475569'})
 colors={'baseline':'#64748b','quarter_all':'#d97706','zero_bias':'#087f8c'}
 fig,axes=plt.subplots(2,2,figsize=(13,9));ax=axes[0,0]
 history=read(a.base/'run_20m_40m/results.json')['metrics'];rows=[r for r in history if r['targets']>=10_000_000 and r['development_nll'] is not None]
 ax.plot([r['targets']/1e6 for r in rows],[r['development_nll'] for r in rows],color='#176d91',marker='o',ms=4,lw=2)
 ax.axvline(20,color='#94a3b8',ls='--');ax.set(title='Original run: slowdown after the rate floor',xlabel='Training bytes (millions)',ylabel='Development NLL (nats/byte)')
 ax=axes[0,1]
 for arm,r in final.items():
    initial=probes['no_bias']['nll'] if arm=='zero_bias' else parent['nll']
    ax.plot([40,41,43],[initial,screen[arm]['development']['nll'],r['development']['nll']],marker='o',lw=2,color=colors.get(arm),label=LABELS[arm])
 ax.set(title='Matched continuations from the 40M model',xlabel='Training bytes (millions)',ylabel='Development NLL (nats/byte)',xticks=[40,41,42,43]);ax.legend(frameon=False,fontsize=8,loc='best')
 ax=axes[1,0];names=[arm for arm in final if arm!='baseline'];vals=[comparison[arm]['audit']['delta'] for arm in names]
 low=[v-comparison[arm]['audit']['ci95'][0] for arm,v in zip(names,vals)];high=[comparison[arm]['audit']['ci95'][1]-v for arm,v in zip(names,vals)]
 ax.barh(range(len(names)),vals,color=[colors.get(arm,'#176d91') for arm in names],height=.48)
 ax.errorbar(vals,range(len(names)),xerr=[low,high],fmt='none',ecolor='#0f172a',capsize=4)
 short_labels={'zero_bias':'Usage bias off','quarter_all':'Quarter rates'}
 ax.axvline(0,color='#64748b',lw=1);ax.set(yticks=range(len(names)),yticklabels=[short_labels.get(arm,arm) for arm in names],xlabel='Audit NLL change vs unchanged continuation',title='Previously unused 1M-byte audit · after +3M')
 ax.text(.02,.50,'Negative = better · paired document-bootstrap 95% intervals',transform=ax.transAxes,fontsize=8,color='#475569',va='center')
 ax=axes[1,1];values=[parent['nll'],snapshot['no_bias_fast']['nll'],snapshot['quarter_bias_all']['nll']]
 ax.bar(range(3),values,color=['#64748b','#087f8c','#3b9caa'],width=.58)
 ax.set(xticks=range(3),xticklabels=['Saved 40M','Fast bias removed','All biases ×0.25'],ylim=(min(values)-.01,max(values)+.01),ylabel='Development NLL (nats/byte)',title='Frozen checkpoint: change routing bias only')
 for i,v in enumerate(values):ax.text(i,v+.0008,f'{v:.4f}',ha='center',fontsize=10,fontweight='bold')
 for ax in axes.flat:ax.grid(axis='y',color='#e2e8f0',lw=.7);ax.set_axisbelow(True)
 fig.suptitle('SHoEbox 100M · what is limiting progress?',x=.08,ha='left',fontsize=21,fontweight='bold')
 fig.text(.08,.92,'Schedule floor + mature fast-bank balancing · 11 matched screens, 3 confirmed continuations',fontsize=12,color='#475569')
 fig.text(.08,.025,'All losses are nats per UTF-8 byte. Zero-bias starting point is a frozen intervention, with no extra training. No trained weights changed in the lower-right test.',fontsize=9,color='#475569')
 fig.subplots_adjust(left=.11,right=.97,top=.86,bottom=.11,wspace=.38,hspace=.42)
 fig.savefig(out/f'{prefix}.png',dpi=180,facecolor='white');plt.close(fig)
 with zipfile.ZipFile(out/f'{prefix}_Results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(out.iterdir()):
        if path.suffix in {'.json','.md','.png'}:z.write(path,path.name)
    for path in sorted(work.rglob('*')):
        if path.is_file() and (path.suffix=='.json' or path.name.endswith('.log') or path.parent.name=='data' and path.suffix=='.bin'):
            z.write(path,'evidence/'+str(path.relative_to(work)))
    for path in [a.base/'run_100m/results.json',a.base/'run_20m_40m/results.json']:
        z.write(path,'historical/'+str(path.relative_to(a.base)))
    z.writestr('SOURCE.txt','Diagnostic source: SHoEbox/experiments/plateau_100m/\nLocal branch: diagnose-100m-40m-plateau\nParent training source: 5751121\nSource checkpoints are supplied separately in the preceding training deliveries.\n')
 print(json.dumps(dict(completed=True,confirmed=summary['confirmation'],paired_audit={arm:v['audit'] for arm,v in comparison.items()},best_audit_arm=best,outputs=[p.name for p in sorted(out.iterdir())])),flush=True)

if __name__=='__main__':main()
