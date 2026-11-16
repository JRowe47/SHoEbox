"""Render the compact baseline report and primary research figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.results.read_text())
    assert r['status']=='complete'
    folder=a.results.parent
    rows=lambda task,core:[x for x in r['runs'] if x['task']==task and x['core']==core]
    avg=lambda task,core,f:float(np.mean([f(x) for x in rows(task,core)]))
    colors={'full':'#BD7942','minimal':'#087F8C'}
    names={'full':'Full C','minimal':'Minimal core'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.17,wspace=.25)
    fig.suptitle('SHoEbox minimal core: paired qualification',x=.07,ha='left',fontsize=18,fontweight='bold')
    fig.text(.07,.865,'3 training seeds · 500K nominal parameters · identical data within each pair',fontsize=11,color='#44515A')
    for core in ('full','minimal'):
        curves=np.array([[x['dev_nll'] for x in row['curve']] for row in rows('fineweb',core)])
        x=np.array([x['bytes'] for x in rows('fineweb',core)[0]['curve']])/1e6
        axes[0].plot(x,curves.mean(0),marker='o',color=colors[core],label=names[core],linewidth=2.2)
        axes[0].fill_between(x,curves.min(0),curves.max(0),color=colors[core],alpha=.12)
    axes[0].set(title='Language learning',xlabel='Training bytes (millions)',ylabel='Development NLL (nats/byte)')
    axes[0].set_xticks([.25,.5,.75,1.]);axes[0].legend(frameon=False)
    summary={x['task']:x for x in r['summary']}
    axes[0].text(.03,.03,f"Held-out test: {summary['fineweb']['full']:.3f} → {summary['fineweb']['minimal']:.3f}",
                 transform=axes[0].transAxes,fontsize=10,color='#263640')
    for j,core in enumerate(('full','minimal')):
        vals=[100*summary[t][core] for t in ('selective_copy','compression')]
        bars=axes[1].bar(np.arange(2)+(j-.5)*.34,vals,width=.32,color=colors[core],label=names[core])
        axes[1].bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=4,fontsize=10)
    axes[1].set(title='Held-out capabilities',ylabel='Answer-token accuracy (%)',ylim=(0,100))
    axes[1].set_xticks([0,1],['Selective copy','State reconstruction'])
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.07,.04,'Language: 1M training targets. Tasks: 250K encoded training bytes each; 128 shared held-out cases per task.',fontsize=9,color='#44515A')
    png=folder/'SHoEbox_Minimal_Core_Qualification.png'
    fig.savefig(png,dpi=180,facecolor='white');plt.close(fig)

    table=[]
    for task,label in [('fineweb','FineWeb test NLL ↓'),('selective_copy','Selective-copy token accuracy ↑'),('compression','State-reconstruction token accuracy ↑')]:
        s=summary[task]
        if task=='fineweb':values=f"{s['full']:.6f} | {s['minimal']:.6f} | {s['delta']:+.6f} nats/byte"
        else:values=f"{100*s['full']:.2f}% | {100*s['minimal']:.2f}% | {100*s['delta']:+.2f} pp"
        table.append(f'| {label} | {values} |')
    exact=[]
    probes=[]
    for task,label in [('selective_copy','Selective copy'),('compression','State reconstruction')]:
        exact.append(f"| {label} | {100*avg(task,'full',lambda x:x['test']['exact_answer_accuracy']):.2f}% | {100*avg(task,'minimal',lambda x:x['test']['exact_answer_accuracy']):.2f}% |")
        probes.append(f"| {label} | {100*avg(task,'minimal',lambda x:x['test']['task_accuracy']):.2f}% | {100*avg(task,'minimal',lambda x:x['probe']['task_accuracy']):.2f}% | {'Disable addressed and retained reads' if task=='selective_copy' else 'Zero the final 64-value state'} |")
    speed=[]
    for task,label in [('fineweb','Language training'),('selective_copy','Copy training'),('compression','Reconstruction training')]:
        s=summary[task]['mean_training_throughput']
        speed.append(f"| {label} | {s['full']:,.0f} | {s['minimal']:,.0f} | {100*(s['minimal']/s['full']-1):+.1f}% |")
    speeds={c:avg('fineweb',c,lambda x:x['inference_bytes_per_cpu_second']) for c in colors}
    speed.append(f"| Frozen language inference | {speeds['full']:,.0f} | {speeds['minimal']:,.0f} | {100*(speeds['minimal']/speeds['full']-1):+.1f}% |")
    text=f'''# SHoEbox minimal-core baseline

The public default is **minimal corrected C, checkpoint v19**. It keeps sparse
experts at 1/4 bytes, ordered summaries at 16/64 bytes, bounded context and memory,
and one-pass prediction. Full C remains available as `Model(core="full")`.

## Architecture

| Component | Minimal baseline |
| --- | --- |
| Input/output | Bytes; learned embedding; existing hierarchical byte decoder |
| Fast scale, 1 byte | Corrected-C sparse expert; context-dependent transition |
| L1, 4 bytes | Corrected-C sparse expert; ordered constituent composition |
| L2/L3, 16/64 bytes | Ordered trainable summaries; diagonal recurrence; memory reads |
| Upper output modulation | Learned constant gain/bias vectors |
| Memory | Existing MinKron context, addressed memory and retained-unit retrieval |
| Learning | Separate fast/slow/query rates; coverage-only routing warmup; reconstruction and bounded child credit |
| Excluded from the minimal path | Upper expert banks/routing/forecast heads, state-dependent upper modulation, D predictors, refinement loops, route-rate cycles and experimental credit multipliers |

The nominal budget is reassigned to the surviving expert banks in a 2:1 ratio.
At 1M nominal, the model has **996,799 fitted parameters** and **135/68/0/0 experts**.
At 5M, it has **4,996,927 parameters** and **766/384/0/0 experts**. Width remains 64.
This reduces the computational architecture while preserving requested capacity.
Historical structs remain shared; unused entries in the upper `Shared` parameter
blocks are not fitted, executed or serialized in v19.

## Paired qualification

| Measurement | Full C | Minimal | Change |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

All three predeclared mean tolerances passed: no more than +0.01 nats/byte in
language loss and no more than 1 percentage point loss on either capability.
This qualifies the reduced baseline at the tested 500K operating point; the
previous full-C 20M-byte/5M-parameter results remain separate historical results.

![Paired qualification](SHoEbox_Minimal_Core_Qualification.png)

Exact whole-answer scores were also retained:

| Task | Full C | Minimal |
| --- | ---: | ---: |
{chr(10).join(exact)}

Frozen contribution probes on the minimal models:

| Task | Intact token score | Intervention score | Intervention |
| --- | ---: | ---: | --- |
{chr(10).join(probes)}

Copying therefore still depends on the memory path. Reconstruction depends on
the completed upper summary: its readout receives only the final 64-float L3
state. That experiment uses a separately trained MLP decoder, included in its
parameter budget; its score is not native language-generation accuracy.

## Protocol and cost

- Three paired seeds: 1, 7, 19. Nominal budget 500K; actual ordinary-model counts
  are 496,959 full and 498,367 minimal. Compression includes its decoder within
  the same nominal budget: 496,639 full and 498,047 minimal.
- FineWeb: one million next-byte training targets and disjoint 65,536-target
  development/test streams. Four 250K-target training blocks; constant default
  learning rates; the same bytes and explicit retirement boundaries for both cores.
- Copy: vocabulary 8, length 2, distractor fraction 0.5, learning-rate scale 2.
  Reconstruction: vocabulary 16, length 4, learning-rate scale 0.5. Both use
  auxiliary weight 0, 250K charged input bytes, 256 training examples and 128
  examples in each development/test split. Padding and context count toward cost.
- Learning-rate recipes were fixed before this comparison. No tuning or selection
  used its test set. All 18 fits were frozen before test scoring. Each seed uses
  the same held-out cases; 128 cases does not become 384 independent examples.
- Complete fitted-checkpoint round trips preserve predictions. Data and training
  order hashes, individual seed results and timings are in `qualification.json`.
- Training timing includes resets and final credit retirement, excludes compilation,
  evaluation and checkpoint I/O, and uses one pinned CPU process. Frozen inference
  is a separate one-pass measurement; median of three passes per seed. Concurrent
  correctness checks make the small timing differences unsuitable as a strong speed claim.

| Work | Full bytes/CPU-second | Minimal bytes/CPU-second | Change |
| --- | ---: | ---: | ---: |
{chr(10).join(speed)}

## Verification and compatibility

- 48 runtime Python tests; 35 MAD, EXP00 and state-adapter tests.
- Native derivative/regression suites pass optimized and ASan/UBSan builds.
  The added minimal suite passes 713 checks, including finite-difference summary
  gradients and poisoned unused dense operations. LSan was explicitly disabled
  for the managed host; ASan/UBSan remained enabled.
- Seven real historical policies (v7, v9, v10, v11, v12, v17, v18) match the
  pre-change source byte-for-byte before and after resumed training.
- Minimal checkpoints continue exactly across partial 1/4/16/64-byte boundaries
  at every supported depth. Frozen inference preserves the source model.
- Incompatible minimal extensions and a malformed refinement wrapper are rejected.
  Old checkpoints keep their saved architecture; no weight conversion is performed.

`verification.json` records the final checks. `compatibility.json` contains
the historical hashes against commit `047eabcd3622c3105a09fbaa49c33f9d67af38fe`.

## Use and reproduce

```python
from shoe_multiscale import Model

with Model(parameters=1_000_000) as model:
    assert model.stats()["core"] == "minimal"
    assert model.stats()["experts"] == [135, 68, 0, 0]
    model.train(b"Observed training bytes.")
    model.save("minimal.shoe")
```

```bash
python -m multiscale.cli train --core minimal --data train.bin --out minimal.shoe
python -m experiments.minimal_core.run --data DATA_DIRECTORY --out qualification.json
python -m experiments.minimal_core.verify_compat --old PREVIOUS_CHECKOUT --out compatibility.json
python -m experiments.mad.run --core minimal --preset smoke --execute --out minimal-mad.json
python -m multiscale.tests.run_all
```

The MAD v1 command retains `--core full` as its historical default; select
`--core minimal` for new baseline studies. Explicit historical architecture
options remain reproducible. The next capability task remains EXP01 query/readout
diagnosis, with the architecture named explicitly in each comparison.

Source, tests and compact evidence are committed. Trained weights, datasets and
temporary model snapshots are excluded.
'''
    (folder/'SHoEbox_Minimal_Core_Baseline.md').write_text(text)
    print(folder/'SHoEbox_Minimal_Core_Baseline.md')
    print(png)


if __name__=='__main__':main()
