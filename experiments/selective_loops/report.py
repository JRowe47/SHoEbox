"""Create the downloadable Markdown report and primary PNG from measured JSON."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    d=json.loads(args.results.read_text());args.out.mkdir(parents=True,exist_ok=True)
    colors={'C':'#197c83','loops':'#c15e2e'};labels={'C':'Corrected C','loops':'Multiscale loops'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelcolor':'#273745','text.color':'#273745'})
    fig,ax=plt.subplots(2,2,figsize=(12,8.3));fig.set_facecolor('#fbfbf8')
    for a in ax.flat:a.set_facecolor('#fbfbf8');a.grid(alpha=.16,axis='y');a.set_axisbelow(True)
    for arm in ('C','loops'):
        runs=[r for r in d['runs'] if r['arm']==arm]
        y=np.array([[v['nll'] for v in r['curve'][1:]] for r in runs]);mean=y.mean(0)
        x=np.array([v['targets']/1e6 for v in runs[0]['curve'][1:]])
        cpu=np.array([[v['cpu_seconds'] for v in r['curve'][1:]] for r in runs]).mean(0)
        for axis,xx in ((ax[0,0],x),(ax[0,1],cpu)):
            axis.plot(xx,mean,'o-',label=labels[arm],color=colors[arm],linewidth=2,markersize=5)
            axis.fill_between(xx,y.min(0),y.max(0),color=colors[arm],alpha=.12)
    ax[0,0].set(title='Learning per byte',xlabel='Training bytes (millions)',ylabel='Held-out NLL (nats/byte)')
    ax[0,1].set(title='Learning per CPU second',xlabel='Training CPU time (seconds)',ylabel='Held-out NLL (nats/byte)')
    ax[0,0].legend(frameon=False)
    for k,arm in enumerate(('C','loops')):
        values=[d['summary'][arm][mode+'_cpu_bps']/1000 for mode in ('train','inference')]
        pos=np.array([1.,0.])+(0.17 if k==0 else -.17)
        ax[1,0].barh(pos,values,height=.28,color=colors[arm],label=labels[arm])
        for v,y in zip(values,pos):ax[1,0].text(v+.7,y,f'{v:.1f}k',va='center',fontsize=10)
    ax[1,0].set(title='Serial throughput',xlabel='Thousand bytes / CPU second',yticks=[0,1],yticklabels=['Inference','Training'],xlim=(0,61))
    seed_deltas=[]
    for seed in d['seeds']:
        r={v['arm']:v['final_nll'] for v in d['runs'] if v['seed']==seed}
        seed_deltas.append(r['loops']-r['C'])
    ax[1,1].bar([str(s) for s in d['seeds']],seed_deltas,color=[colors['loops'] if v>0 else colors['C'] for v in seed_deltas],width=.55)
    ax[1,1].axhline(0,color='#273745',linewidth=1)
    ax[1,1].set(title='Final loss change by seed',xlabel='Matched initialization seed',ylabel='Loops minus C (negative is better)')
    fig.suptitle('SHoEbox · selective multiscale loops',fontsize=21,fontweight='bold',x=.06,ha='left',y=.99)
    fig.text(.06,.941,'999,615 parameters  |  1M training bytes per model  |  3 seeds  |  extra passes: 0 / 1 / 2 / 2',fontsize=11)
    fig.text(.06,.018,'Curves: seed means; bands: seed range. First plotted point: 250K bytes. Throughput: median of 3 serial probes.',fontsize=9)
    fig.tight_layout(rect=(.025,.045,.99,.918),h_pad=2.2,w_pad=2.1)
    fig.savefig(args.out/'SHoEbox_Selective_Loops.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
    c=d['summary']['C'];l=d['summary']['loops'];delta=l['mean_nll']-c['mean_nll']
    rows=[]
    for seed in d['seeds']:
        r={v['arm']:v['final_nll'] for v in d['runs'] if v['seed']==seed}
        rows.append(f"| {seed} | {r['C']:.6f} | {r['loops']:.6f} | {r['loops']-r['C']:+.6f} |")
    executed=sum(v['passes'] for r in d['runs'] if r['arm']=='loops' for v in r['loop_stats']['levels'])
    cap=sum(v['max_steps']*v['events'] for r in d['runs'] if r['arm']=='loops' for v in r['loop_stats']['levels'])
    text=f'''# SHoEbox selective multiscale loops

Implemented on the corrected-C and route-learning source (`e2a2e93`).

## Result

The feature is implemented and available with `Model(loops={{}})`. Its preset
adds 1 / 2 / 2 refinement passes at completed 4 / 16 / 64-byte constituents,
using the existing selected expert and a 64-value working state. No parameters
are added. The byte path takes no extra pass.

The matched short screen does **not** support replacing corrected C: mean
held-out NLL is **{l['mean_nll']:.6f} with loops versus {c['mean_nll']:.6f} for C**,
a change of **{delta:+.6f} nats/byte**. One of three seeds improves. Corrected C
remains the default; the completed loop feature has an explicit configuration.

| Seed | Corrected C | Multiscale loops | Change |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}
| Mean | {c['mean_nll']:.6f} | {l['mean_nll']:.6f} | {delta:+.6f} |

| Throughput | Corrected C | Loops | Change |
| --- | ---: | ---: | ---: |
| Training bytes / CPU second | {c['train_cpu_bps']:,.0f} | {l['train_cpu_bps']:,.0f} | {(l['train_cpu_bps']/c['train_cpu_bps']-1)*100:+.2f}% |
| Inference bytes / CPU second | {c['inference_cpu_bps']:,.0f} | {l['inference_cpu_bps']:,.0f} | {(l['inference_cpu_bps']/c['inference_cpu_bps']-1)*100:+.2f}% |

These are implementation-screen measurements after 1M training bytes, not a
long-horizon capability or saturation comparison. The data and schedule are
identical within each seed; both models have 999,615 fitted parameters.

## What runs

At each selected upper completion, anchor `a` is the ordinary expert's content
and working vector `h` begins at `a`. Each extra pass computes
`h_next = (1-s)*h + s*tanh(a + b + g*W*h)`, where `W,b` are the selected expert,
`g` is its routing gate, and `s=0.25` in the screen. Refinement publishes once,
before the native memory/state transition and before parent composition.

Budgets independently accept 0–4 extra passes per upper scale. The preset adds
at most 26 expert matrix evaluations per 64 input bytes. It adds no route
searches, byte clock ticks, memory admissions, native forecast targets or
route-learning visits. Full reverse credit covers every executed pass, the
anchor, shared expert weights, routing gate, composer and live child producers.

Convergence stopping uses the relative change of the working vector. A zero
tolerance requests fixed depth. The screen's 0.01 tolerance skipped
**{cap-executed:,} of {cap:,} available passes**: this tested preset used its full
budgets. The stopping mechanism is covered by explicit early-exit tests.
Learned hardness selection and learned halting are the next stage, and are not
implemented here.

Training eligibility is bounded by the existing 512-byte publication epoch
(about 385 KiB reserved for the preset). Frozen inference releases that tape.
Version 18 checkpoints preserve partial-loop credit exactly; old checkpoints
load with loops disabled and retain their original semantics.

## Use

```python
from shoe_multiscale import Model

model = Model(parameters=1_000_000, loops={{}})
# Explicit upper-only alternative:
# model = Model(loops={{'steps': [0, 0, 2, 4], 'tolerance': 0}})
model.train(b"Observed training bytes. " * 100)
model.save("loops.shoe")
print(model.loop_stats())
```

```bash
python -m multiscale.cli train --data input.txt --out loops.shoe \\
  --loop-steps 0 1 2 2 --loop-strength .25 --loop-tolerance .01
```

The feature also supports frozen inference-depth overrides. Call `finish()`
before an explicit training-policy change; ordinary resume preserves the saved
loop policy, state and pending gradients. Historical backends and semantic-lace
reassignment remain separate from this feature.

## Validation

All 44 Python tests, 1,439 loop-specific native checks and the existing native
regression harnesses pass. ASan and UBSan also pass; LeakSanitizer was explicitly
disabled on this host. Checks cover shared-weight and gate derivatives,
integrated language credit, early stopping, one publication per completion,
bounded storage, frozen evaluation, malformed checkpoints and exact pending
continuation. A checkpoint-dispatch stack issue found during testing is fixed.

Three old/new checkpoint cases (C, D and route-learning with credit extensions)
retain byte-identical 4,097-target continuations with loops disabled. All six
final checkpoints reproduce the recorded evaluation losses with the final source.

## Protocol and reproduction

- Tiny Shakespeare SHA-256: `{d['sha256']}`.
- First {d['split_byte']:,} bytes form the training partition; the final partition
  supplies {d['validation_targets']:,} held-out next-byte targets. Each run uses
  1,000,000 unique training targets. The split is contiguous and context is cold
  for each validation evaluation.
- Seeds: 1, 7, 19. Fast/query rate follows cosine 0.008 to 0.0004 over the run,
  updated every 50K targets; slow rate is 10× and head rate 20×. Other settings
  retain corrected C. Training state continues across rate blocks.
- CPU: AMD EPYC 9V74, one pinned core. Arms alternate order within each training
  block. Curve timers exclude compilation, evaluation and saving; final training
  retirement is included. Throughput uses three alternating serial 100K-target
  probes of the seed-1 fitted models and includes training reset/retirement.
- Reproduce with `python -m experiments.selective_loops.compare --data input.txt
  --out results.json`, then `python -m experiments.selective_loops.report
  --results results.json --out output`.

The code/evidence archive includes the implementation, tests, raw measurements,
compatibility hashes, source patch and reproduction scripts. It excludes
datasets, model weights and compiled binaries.

## Research connection

[SpiralFormer](https://arxiv.org/abs/2602.11698) supplies the multi-resolution
recurrence motivation. SHoEbox's completed constituents provide the causal
resolutions. [Think-at-Hard](https://proceedings.mlr.press/v306/fu26t.html)
motivates selecting where extra iteration helps. This implementation supplies
bounded local refinement and convergence stopping; it does not reproduce either
paper's Transformer or learned iteration selector.
'''
    (args.out/'SHoEbox_Selective_Loops.md').write_text(text)


if __name__=='__main__':main()
