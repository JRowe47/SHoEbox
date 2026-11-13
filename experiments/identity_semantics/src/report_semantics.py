"""Render experiment evidence as ordinary downloadable reports and curves."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
KERNELS=['binding','semantics_fixed','semantics']
labels={'binding':'Previous binding','semantics_fixed':'Corrected encoding, fixed','semantics':'Corrected encoding, learned'}
runs={(k,s):json.loads((ROOT/f'results/full_{k}_seed{s}/training.json').read_text()) for k in KERNELS for s in [1,2]}
assert all(r['status']=='complete' and r['tokens']==1000000 for r in runs.values())
perf=json.loads((ROOT/'results/throughput.json').read_text());probe=json.loads((ROOT/'results/encoding_probe.json').read_text());contrib=json.loads((ROOT/'results/semantic_contribution.json').read_text());selection=json.loads((ROOT/'results/rate_selection.json').read_text())
means={k:float(np.mean([runs[k,s]['test_nll'] for s in [1,2]])) for k in KERNELS}
speed=perf['median_bytes_per_second'];delta=means['semantics']-means['binding'];pct=100*delta/means['binding'];slow=100*(1-speed['semantics']/speed['binding'])
summary=dict(base_commit='6d06d60072c3032ac1ce13ceed04ee8f7d89a4cc',base_package='SHoEbox_Multisource_Binding',parameters={'binding':1023744,'semantics':1031936},added_semantic_coefficients=8192,mean_test_nll=means,change_nll=delta,relative_nll_change_percent=pct,throughput=perf,encoding_probe=probe,semantic_contribution=contrib,rate_selection=selection,test_count=62,checkpoint_version=30,composition_abi=4)
(OUT/'SHoEbox_Identity_Semantics_Results.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'SHoEbox_Identity_Semantics_Curves.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['kernel','seed','split','tokens','nll'])
 for (k,s),r in runs.items():
  for split in ['training','validation']:
   for row in r[split]:w.writerow([k,s,split,row['tokens'],row['nll']])
colors={'binding':'#637586','semantics_fixed':'#d89226','semantics':'#008c95'}
fig,axes=plt.subplots(1,2,figsize=(11.5,4.3));fig.patch.set_facecolor('white')
for k in KERNELS:
 a=np.array([[p['nll'] for p in runs[k,s]['training']] for s in [1,2]]);x=np.array([p['tokens'] for p in runs[k,1]['training']])/1e6
 smooth=np.array([np.convolve(z,np.ones(10)/10,mode='valid') for z in a]);axes[0].plot(x[9:],smooth.mean(0),color=colors[k],label=labels[k]);axes[0].fill_between(x[9:],smooth.min(0),smooth.max(0),color=colors[k],alpha=.10)
 v=np.array([[p['nll'] for p in runs[k,s]['validation'][1:]] for s in [1,2]]);vx=np.array([p['tokens'] for p in runs[k,1]['validation'][1:]])/1e6
 axes[1].plot(vx,v.mean(0),color=colors[k],marker='o',markersize=3);axes[1].fill_between(vx,v.min(0),v.max(0),color=colors[k],alpha=.10)
for ax in axes:
 ax.set_xlabel('Training byte targets (millions)');ax.set_ylabel('NLL (nats / byte)');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_title('Training loss - 50K-target moving mean');axes[1].set_title('Validation loss - 32,768 targets');axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('SHoEbox: identity and vector semantics | 1M model | two seeds',fontsize=13,fontweight='bold');fig.text(.5,.015,'Shaded bands show the two-seed range. Each arm uses its independently selected learning rate.',ha='center',fontsize=8,color='#555555');fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(OUT/'SHoEbox_Identity_Semantics_Curves.png',dpi=190);plt.close(fig)
rows='\n'.join(f"| {s} | {runs['binding',s]['test_nll']:.6f} | {runs['semantics_fixed',s]['test_nll']:.6f} | {runs['semantics',s]['test_nll']:.6f} |" for s in [1,2])
proberows=[]
change_names={'relative_start':'Relative start: 1 to 7','duration':'Second-child duration: 1 to 4','source_role':'Second-child source role','receiver_namespace':'Receiving-region namespace','absolute_offset':'Absolute offset: +7,000'}
for name,label in change_names.items():
 old=next(x for x in probe if x['kernel']=='binding' and x['change']==name);new=next(x for x in probe if x['kernel']=='semantics' and x['change']==name)
 proberows.append(f"| {label} | {'Yes' if old['id_changed'] else 'No'} / {old['code_l2']:.4f} | {'Yes' if new['id_changed'] else 'No'} / {new['code_l2']:.4f} |")
perfrows='\n'.join(f"| {labels[k]} | {speed[k]:,.0f} | {1e6/speed[k]:.1f} |" for k in KERNELS)
contribrows='\n'.join(f"| {x['seed']} | {x['baseline_nll']:.6f} | {x['zero_semantic_residual_nll']:.6f} | {x['delta_nll']:+.6f} |" for x in contrib)
regionrows=[]
for i,name in enumerate(['L2','S2','R2','F2','L3','S3','R3','F3','E4','F4','V4','M4','G5','EC-C','EC-X','Y']):
 if name in ['L2','L3','E4','V4','G5','EC-C','EC-X','Y']:
  regionrows.append(f"| {name} | {contrib[0]['regional_updates'][name]:,} | {contrib[0]['regional_parameter_l2'][i]:.6f} |")
assessment=(f"The repaired learned model lowers mean reserved-test NLL by {abs(delta):.6f} nats/byte ({abs(pct):.2f}%)." if delta<0 else f"The repaired learned model raises mean reserved-test NLL by {delta:.6f} nats/byte ({pct:.2f}%) in this comparison. The structural contract is repaired; this isolated change did not improve token-matched language loss.")
throughput_note=(f'{slow:.1f}% lower' if slow>=0 else f'{-slow:.1f}% higher')
text=f'''# SHoEbox: consistent identity and vector semantics

12 November 2026 | Second repair from the principled code audit | 1M-model study

## Result

The identity/vector mismatch is repaired on top of the multi-source binding implementation. Required structural distinctions now enter the content vector used by routing and local teaching, including L2's previously missing geometry and receiving-region distinctions. Exact structural IDs remain separate from learned semantic vectors and neural context.

**62 checks pass**, including the previous 35 checks, the binding suite against the new kernel, new semantic and gradient checks, and explicit checkpoint conversion. AddressSanitizer and UndefinedBehaviorSanitizer pass.

The learned version has **1,031,936 neural coefficients**, including 8,192 new semantic coefficients: **0.80% more** than the 1,023,744-parameter binding model. Each comparison model received one million training byte targets, using two seeds. {assessment}

Same-core median training throughput is **{speed['binding']:,.0f} bytes/s before and {speed['semantics']:,.0f} bytes/s after**, or **{throughput_note} throughput**. The corrected fixed-encoding control is measured separately below.

## Representation contract

| Object | Meaning and permitted changes |
| --- | --- |
| Exact content ID | 128-bit structural key. Includes receiving region, composition kind, ordered child IDs, source/recipe roles, present roles, relative starts and durations. Excludes absolute alignment, occurrence serial, expert choice and contextual activity. Byte atoms share the byte vocabulary; source distinctions enter compound roles. |
| Structural descriptor | Deterministic composition of children's immutable descriptors, plus an anchor derived from both halves of the compound ID. It carries role/geometry/namespace distinctions into the neural interface. |
| Semantic vector | Structural descriptor plus a learned bounded residual for that region. Learning may change this vector while keeping the exact ID fixed. |
| Neural context | The expert's recurrent, context-sensitive publication. It remains separate from content identity and descriptor. |

The local symbol is the exact content ID, not an expert number, a dictionary bucket, or an occurrence index. Count-table eviction therefore cannot reassign a symbol's vector. Parents build their own descriptors from immutable child descriptors, so a held child embedding from an earlier weight revision does not unpredictably change a parent's content code. Learned semantic adaptation belongs to each receiving region; learned neural publications continue to travel through the separate contextual path.

The learned code is a compact shared function of a local symbol's descriptor. It is not an unbounded per-symbol embedding table. Eight fixed signed projections yield eight bounded features. A regional 64-by-8 matrix maps these features into a residual bounded to +/-0.25 per coordinate. There are sixteen such matrices, for L2 through Y. They start at zero.

The exact ID remains the equality authority. A 64-dimensional similarity vector is not used as a collision-free identifier. The full-ID anchor makes required structural changes visible in the tested neural interface; learned residuals do not rewrite IDs or recognition counters.

## What was corrected

L2 now encodes constituent duration and source role in its ID, and carries the complete identity distinctions into its vector. A singleton still forwards its child's ID and vector. Mixed-source regions retain the declared slot positions and add the source namespace explicitly to the structural key. Their vectors now include an anchor for the completed compound ID, which covers the receiver namespace and present-role mask as well as the constituent fields.

This keeps identity, descriptor, semantic embedding and contextual response as separate concepts. Semantic learning does not use an expert's context-sensitive output as the symbol itself.

The following controlled probes use the same two child contents. Each cell shows **ID changed? / vector L2 distance**. The previous implementation's vector stayed exactly unchanged for every listed structural distinction.

| Controlled change | Previous binding | Repaired semantics |
| --- | ---: | ---: |
{chr(10).join(proberows)}

The inherited binding checks additionally exercise every immediate constituent slot in all fifteen mixed-source regions, absent inputs without role renumbering, and genuine older/newer occurrences.

## Semantic learning and saved provenance

Semantic matrices learn from local prediction and task/downstream credit entering a region's neural input, including retained-input credit. The update retains the publication's eight features, residual derivative, input scaling, tick and eligibility index. A later event cannot substitute its own embedding features for that saved producer.

The semantic residual is an input-side learned transform. It receives the saved expert-input Jacobian with the selected expert/gate held fixed; it does not differentiate hard routing decisions or turn this runtime into full backpropagation through time. The future-content target is read as a detached observed target. There is no update that moves the arriving target toward the prediction merely to reduce the error.

Finite differences verify the semantic-matrix derivative. Behavioral checks verify a loss-reducing semantic update with unchanged ID, preservation of saved features across later occurrences, unchanged parameters during evaluation, frozen-region behavior, and exact save/load/chunk continuation. Router cues and the local teacher's content input share the same content vector; the existing Y prefix-byte addition and normalization remain explicit.

Singleton cancellation of outstanding teaching and recognition status is still the next repair in the agreed sequence. This patch retains that behavior and does not claim to have repaired relay teaching.

## Matched language experiment

All three arms use the same Tiny Shakespeare byte stream, width 64, twelve experts per region, initial expert parameters, fixed transports and train/validation/test split. Each arm was screened on seed 3 at peak rates 0.02, 0.04 and 0.08 for 100,000 targets. Selection uses only 32,768 validation targets.

Selected peaks: previous binding **{selection['binding']['peak_lr']}**, corrected fixed encoding **{selection['semantics_fixed']['peak_lr']}**, and corrected learned encoding **{selection['semantics']['peak_lr']}**. Rates follow cosine decay to one tenth of the selected peak by one million targets. Local rate is one third of main; prototype rate is 0.002. Seeds 1 and 2 were then trained from fresh initialization for each arm.

The primary endpoint is the remaining **78,771 reserved-test targets**, evaluated from a fresh stream. As in the prior binding study, evaluation starts with cold episodic memory and permits causal memory admission from the observed test prefix. Neural weights, semantic coefficients and composition dictionaries stay frozen. The fixed-encoding arm holds the 8,192 semantic coefficients at zero throughout training.

| Seed | Previous binding NLL | Corrected fixed NLL | Corrected learned NLL |
| --- | ---: | ---: | ---: |
{rows}
| Mean | {means['binding']:.6f} | {means['semantics_fixed']:.6f} | {means['semantics']:.6f} |

The fixed and learned arms use independently selected rates, so their difference includes rate selection. The frozen intervention below isolates dependence on the learned semantic residual within each trained learned model.

![Training and validation loss](SHoEbox_Identity_Semantics_Curves.png)

## Learned residual contribution

Each learned checkpoint is evaluated with its semantic residual intact and then with all semantic matrices zeroed in a fresh evaluation copy. IDs, descriptors, dictionaries, expert weights and the evaluation protocol are retained. The intact pass reproduces the study's reserved-test loss to numerical precision. Positive changes mean that removing the residual worsens loss.

| Seed | Intact test NLL | Zero residual test NLL | Change |
| --- | ---: | ---: | ---: |
{contribrows}

This intervention measures dependence on the learned residual after training, while the fixed-encoding arm measures retraining without it. Regional neural-publication contribution checks remain a separate later step after the relay repair.

Selected seed-1 semantic learning diagnostics:

| Region | Semantic update calls | Learned matrix L2 norm |
| --- | ---: | ---: |
{chr(10).join(regionrows)}

## Throughput and state cost

Timing uses one pinned CPU core, six trials per arm in alternating order, 1,024 warmup targets and 16,384 timed targets per trial, after the training workers have exited. It includes forward execution, routing, dictionaries, episodic admission and neural learning. Compilation, snapshot creation, checkpoint writes and evaluation are excluded.

| Implementation | Median training bytes/s | Projected seconds per million |
| --- | ---: | ---: |
{perfrows}

Composition state grows from **3,103,240 to 3,317,000 bytes**, an increase of **213,760 bytes**. That includes immutable descriptors, saved semantic eligibility and 32,768 bytes of learned coefficients. The learned arm also computes input Jacobians for semantic credit; the timing measures that cost. These comparisons match training targets, not elapsed compute.

## Delivery and continuation

The package contains the full runtime, six trained checkpoints, exact data, tests, raw training records, rate selection, intervention evidence and report-generation scripts. The previous binding kernel remains available for reproduction.

```
model = fresh_model(kernel="semantics")
model = Model.load("results/full_semantics_seed1/last.npz")
```

Semantic checkpoints use **version 30 / composition ABI 4**. Save/load is exact for the new kernel. The loader rejects a version-29 binding composition blob rather than reinterpreting it. To reuse old neural weights, `src/convert_to_semantics.py` explicitly creates a new checkpoint with fresh semantic matrices, dictionaries, working context, eligibility, preprocessing means and episodic associations. That operation is a weight conversion, not exact continuation.

```
python -m unittest discover -s tests -v
python tests/run_semantics_sanitizer.py
python src/run_semantics_study.py pilots
python src/run_semantics_study.py full
python src/encoding_probe.py
python src/semantic_contribution.py
python src/semantics_performance.py
```

Source provenance: the recovered binding package's 49 recorded source/configuration hashes all match. Its ancestor is [SHoEbox commit 6d06d60](https://github.com/JRowe47/SHoEbox/commit/6d06d60072c3032ac1ce13ceed04ee8f7d89a4cc). This delivery is a tested source package; no GitHub publication is included in this step.
'''
(OUT/'SHoEbox_Identity_Semantics_Report.md').write_text(text)
print(json.dumps(dict(mean_test_nll=means,change=delta,percent=pct,median_bytes_per_second=speed),indent=2))
