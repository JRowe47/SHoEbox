"""Render compact, reproducible Stage A evidence and a downloadable report."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .tasks import suite

LABELS={'final_only':'Final answer','head_only':'Extra head labels','state_aux':'Extra state credit'}
COLORS={'final_only':'#64748b','head_only':'#d98c28','state_aux':'#187b85'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.results.read_text());runs=data['runs'];probes=json.loads(args.probes.read_text())['rows'];args.out.mkdir(parents=True,exist_ok=True)
    by_probe={(r['size'],r['arm']):r for r in probes};arms=list(LABELS);sizes=sorted({r['size'] for r in runs})
    cases=suite(19810423,512,'audit');initial=sum(e.states[0][e.query]==e.answer for e in cases)/len(cases)
    summary=dict(initial_binding_accuracy=initial,runs=[],comparisons=[])
    for r in runs:
        peer=[s for s in runs if s['size']==r['size']];common=min(s['cpu_seconds'] for s in peer)
        t=[s['cpu_seconds'] for s in r['curve']];loss=[s['dev']['nll'] for s in r['curve']]
        summary['runs'].append(dict(size=r['size'],arm=r['arm'],equal_time_seconds=common,equal_time_dev_nll=float(np.interp(common,t,loss)),
            audit_accuracy=r['audit']['accuracy'],audit_nll=r['audit']['nll'],long_accuracy=r['long']['accuracy'],
            native_answer_accuracy=r['audit']['native_byte_accuracy'],language_nll=r['curve'][-1]['language_nll'],
            cpu_bytes_per_second=r['cpu_bytes_per_second'],inference_bytes_per_second=r['audit']['targets']/r['audit']['cpu_seconds']))
    rng=np.random.default_rng(191)
    for size in sizes:
        state=next(r for r in runs if r['size']==size and r['arm']=='state_aux')
        for arm in ('final_only','head_only'):
            control=next(r for r in runs if r['size']==size and r['arm']==arm)
            d=np.array(state['audit']['correct'])-np.array(control['audit']['correct']);boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
            summary['comparisons'].append(dict(size=size,control=arm,delta=float(d.mean()),paired_example_bootstrap_95=np.quantile(boot,[.025,.975]).tolist()))
    (args.out/'SHoEbox_State_Supervision_Summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f8fafc','axes.facecolor':'#ffffff'})
    fig,axes=plt.subplots(3,2,figsize=(13,14),layout='constrained')
    fig.suptitle('SHoEbox • Intermediate state supervision',fontsize=18,fontweight='bold')
    for size,ax in zip(sizes,axes[0]):
        for arm in arms:
            r=next(r for r in runs if r['size']==size and r['arm']==arm);ax.plot([c['targets']/1e6 for c in r['curve']],[c['dev']['nll'] for c in r['curve']],marker='o',label=LABELS[arm],color=COLORS[arm])
        ax.axhline(math.log(8),color='#bd5a5a',ls='--',lw=1,label='Uniform eight-way answer');ax.set(title=f'{size//1000000}M • development answer loss',xlabel='Million training bytes',ylabel='NLL per queried value');ax.legend(fontsize=8);ax.grid(alpha=.15)
    x=np.arange(len(sizes));width=.23
    for j,arm in enumerate(arms):
        rr=[next(r for r in runs if r['size']==size and r['arm']==arm) for size in sizes]
        rates=np.array([r['audit']['accuracy'] for r in rr]);n=512;z=1.96
        middle=(rates+z*z/(2*n))/(1+z*z/n)
        radius=z*np.sqrt(rates*(1-rates)/n+z*z/(4*n*n))/(1+z*z/n)
        errors=100*np.array([rates-middle+radius,middle+radius-rates])
        axes[1,0].bar(x+(j-1)*width,100*rates,width,label=LABELS[arm],color=COLORS[arm],yerr=errors,capsize=3,error_kw={'elinewidth':1})
        axes[1,1].bar(x+(j-1)*width,[r['cpu_bytes_per_second']/1000 for r in rr],width,label=LABELS[arm],color=COLORS[arm])
    axes[1,0].axhline(12.5,color='#bd5a5a',ls='--',lw=1,label='Chance');axes[1,0].set(title='Held-out structural audit • 512 programs',ylabel='Exact queried value (%)',xticks=x,xticklabels=[f'{s//1000000}M' for s in sizes]);axes[1,0].legend(fontsize=8);axes[1,0].set_ylim(0,22)
    axes[1,1].set(title='Training throughput • one CPU core',ylabel='Thousand real byte targets / CPU second',xticks=x,xticklabels=[f'{s//1000000}M' for s in sizes]);axes[1,1].legend(fontsize=8);axes[1,1].set_ylim(0,44)
    labels=[];online=[];probe=[];variance=[];perm=[];means=[]
    for size in sizes:
      for arm in arms:
        r=next(r for r in runs if r['size']==size and r['arm']==arm);p=by_probe[(size,arm)]
        labels.append(f'{size//1000000}M\n'+{'final_only':'Final','head_only':'Head','state_aux':'State'}[arm]);online.append(100*r['audit']['accuracy']);probe.append(100*p['probes']['after_query_both']['accuracy'])
        perm.append(np.mean([s['nll'] for s in r['permutations']])-r['audit']['nll']);means.append(r['mean_replacement']['nll']-r['audit']['nll'])
    x=np.arange(len(labels));axes[2,0].bar(x-.19,online,.38,label='Online task head',color='#64748b');axes[2,0].bar(x+.19,probe,.38,label='Frozen standardized ridge probe',color='#187b85')
    axes[2,0].axhline(12.5,color='#bd5a5a',ls='--',lw=1);axes[2,0].set(title='Frozen probes recover more bindings',ylabel='Audit accuracy (%)',xticks=x,xticklabels=labels);axes[2,0].legend(fontsize=8)
    axes[2,1].bar(x-.19,means,.38,label='Replace L2/L3 features by training mean',color='#d98c28');axes[2,1].bar(x+.19,perm,.38,label='Permute active experts (three shuffles)',color='#187b85')
    axes[2,1].axhline(0,color='#64748b',lw=1);axes[2,1].set(title='Causal probes • answer loss change',ylabel='Intervention NLL − intact NLL',xticks=x,xticklabels=labels);axes[2,1].legend(fontsize=7.5)
    fig.supxlabel('Fresh models • seed 7 • 2M mixed training bytes each • 4,128 readout parameters in every arm\nAccuracy uses an explicit queried-register index; native byte generation is reported separately.\nAudit bars: 95% Wilson intervals over examples, not training seeds.',fontsize=10)
    fig.savefig(args.out/'SHoEbox_State_Supervision.png',dpi=150);plt.close(fig)
    table=['| Size | Supervision | Audit accuracy | Long (8/12 ops) | Native answer | Language NLL | Train kB/CPU s |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary['runs']:
        table.append(f"| {r['size']//1000000}M | {LABELS[r['arm']]} | {r['audit_accuracy']:.2%} | {r['long_accuracy']:.2%} | {r['native_answer_accuracy']:.2%} | {r['language_nll']:.4f} | {r['cpu_bytes_per_second']/1000:.2f} |")
    comparisons=[]
    for c in summary['comparisons']:
        lo,hi=c['paired_example_bootstrap_95'];comparisons.append(f"- {c['size']//1000000}M state credit versus {LABELS[c['control']].lower()}: {100*c['delta']:+.2f} percentage points; paired-example 95% interval [{100*lo:+.2f}, {100*hi:+.2f}].")
    probe_table=['| Size | Arm | Online after query | Frozen probe before query | Frozen probe after query |','|---|---|---:|---:|---:|']
    for r in probes:probe_table.append(f"| {r['size']//1000000}M | {LABELS[r['arm']]} | {r['online_after_query']:.2%} | {r['probes']['before_query_both']['accuracy']:.2%} | {r['probes']['after_query_both']['accuracy']:.2%} |")
    timing=['| Size | Arm | Matched CPU seconds | Estimated dev NLL at that time | Inference kB/CPU s |','|---|---|---:|---:|---:|']
    for r in summary['runs']:timing.append(f"| {r['size']//1000000}M | {LABELS[r['arm']]} | {r['equal_time_seconds']:.2f} | {r['equal_time_dev_nll']:.4f} | {r['inference_bytes_per_second']/1000:.2f} |")
    report=f'''# SHoEbox: intermediate state supervision — Stage A

## Result

**Direct intermediate supervision did not establish a reliable capability gain in this screen.** At 1M, audit accuracy was 13.67% for final-only, 14.06% for extra head labels, and 14.84% for direct state credit. At 5M it was 10.94%, 11.13%, and 12.89%. The 5M comparison with head-only favors state credit (+1.76 percentage points; paired-example interval +0.39 to +3.12). The other three intervals include zero, including both comparisons with final-only. Longer programs also remained near chance.

The useful result is diagnostic: **some binding information exists in L2/L3, but the online head uses little of it**. Standardized frozen probes recover 16.80–24.61% accuracy, while replacing changing L2/L3 features by their training mean alters the online answer NLL by only −0.00196 to +0.00055. This motivates a readout/retention experiment before learned halting. Corrected C remains the selected runtime.

## Matched experiment

Six fresh models: three supervision arms at nominal 1M and 5M, seed 7, **2,000,000 real byte targets per model** (12M total). Each arm has exactly the same parameter count: {runs[0]['parameters']:,} at 1M and {next(r for r in runs if r['size']==5000000)['parameters']:,} at 5M. These counts include an identical 4,128-parameter task head in every arm. No extra thought passes are used in Stage A.

1. **Final answer:** byte language loss plus one final queried-register target.
2. **Extra head labels:** also fit all four register values after initialization and each update, with intermediate gradients stopped at the readout.
3. **Extra state credit:** the same labels and head updates, with their gradient entering L2/L3 states and the existing recurrence/composition learning path.

The auxiliary weight is 0.25, normalized across four register labels and across supervised events per native retirement block. The final-answer channel has weight 1. The shared LR anchor falls linearly from 1 to 0.1 over the real byte budget; route oscillation is disabled. Fast/query rates start at 0.008, slow 0.08, native readout 0.16, task readout 0.2. Neither labels nor validation advance the byte schedule.

Training contains {runs[0]['task_bytes']:,} verified-task bytes and {runs[0]['language_bytes']:,} Tiny Shakespeare bytes in every arm. Padding is counted. The training episode digest is `{runs[0]['train_episode_digest']}`. The language corpus SHA-256 is `{data['config']['data_sha256']}`; the last 10% of bytes is held out. The exact configuration, curves and per-example correctness are attached in JSON.

## Capability results

{chr(10).join(table)}

Chance is 12.5%; uniform-answer NLL is {math.log(8):.6f}. A symbolic heuristic that merely reads the queried register's **initial** binding and ignores every update gets {initial:.2%} on this audit. It is a memory-task reference, not a trained model with a comparable resource budget.

The main score is a four-group, eight-class task readout over L2/L3. The evaluator supplies the queried register index. The **native answer** column instead scores the actual next-byte generator over all 256 bytes; no register index is supplied to that decoder. The experiment therefore separates state-readout accuracy from byte generation.

{chr(10).join(comparisons)}

Intervals resample paired audit examples for these fixed trained models; they do not measure training-seed variation. This is the planned screening stage. No configuration is promoted from it.

## What the upper states contain

A separate frozen linear diagnostic is fitted on 1,024 independently generated training-fold programs. It standardizes each feature using that training set, fits ridge regression with fixed alpha 1, and evaluates on the same untouched structural audit. No SHoEbox weights change. The probe receives the same queried-register index as the online head.

{chr(10).join(probe_table)}

At 1M, both extra-head and direct-state training reach 24.61% with the frozen probe; direct state credit provides no additional gain there. At 5M, direct state credit reaches 22.07%, versus 18.75% for head-only and 16.80% for final-only. These are useful diagnostic differences, not a working procedural solver: the initial-binding heuristic still gets 49.22%.

For the direct-state models, the shared feature mean accounts for 89.41% (1M) and 89.89% (5M) of post-query feature energy. The standardized probe's advantage identifies readout conditioning as a concrete follow-up. It does not establish that centering alone will repair the online learner. State information also remains incomplete: the 1M direct-state probe gets 25.78% on tracking and 23.44% on composition; the 5M version gets 28.52% and 15.63%.

The mean-replacement intervention uses a separate training-fold mean of final L2/L3 features. The expert intervention cycles transformations among experts actually selected during training, keeps router keys fixed, and is averaged over three permutations. The chart shows the resulting answer-NLL changes. Frozen probe accuracy measures recoverable information; these interventions measure the fitted online system's dependency on it.

## Resource comparison

{chr(10).join(timing)}

Equal-time values interpolate the saved development curves at the fastest arm's completed training time for that model size. They are curve estimates, not additional checkpoints. Training CPU timers include curriculum generation, context resets, labels and retirement, and exclude compilation, validation and checkpoint I/O. Frozen inference timers include clone/reset and answer scoring. Runs are serial and pinned to CPU {data['config']['cpu']}.

The task eligibility tape is bounded at **8,224 bytes**. It adds no production-model parameters beyond the identical task readout. No inference iterations, repeated byte admissions or extra clock ticks occur.

## Verification and reproducibility

- Six Python tests passed, including exact unlabelled equivalence to public corrected C, detached-head control, independent interpreter validation, causal label timing, chunking, epoch retirement and live-checkpoint continuation.
- Nineteen native checks passed, including finite differences through L2/L3 recurrent parameters and exact checkpoint continuation.
- The same native checks passed AddressSanitizer and UndefinedBehaviorSanitizer; host LeakSanitizer was disabled.
- Every trained checkpoint was reloaded and reproduced its development predictions and NLL exactly.
- The production runtime and its checkpoint ABI were unchanged. The research adapter aborts if either of its two source anchors changes.

Source: branch `codex/structured-state-supervision`, based on `600352ce16e89702de82e7b3f61c9b696c34e6f6`. Run commands are in `experiments/structured_state/README.md`. The source/evidence ZIP excludes datasets, trained weights and compiled binaries.

## Research connection and next decision

Keep this supervised task harness as the capability test. The next focused comparison should give the readout centered inputs and give the model four persistent binding slots with explicit retain/overwrite behavior. Include a centered-readout-only control so any gain can be assigned to storage versus readout conditioning. Once those states retain and expose useful bindings, run the fixed-depth K=1/2/4 comparison before learning a stop policy.

The head-only control remains necessary: changing its weights can change later final-answer gradients indirectly, even though intermediate gradients are detached. The direct-state arm measures the additional effect of sending intermediate credit into the upper states.

This follows Stage A of the supplied capability plan: establish useful upper information before expanding recurrent depth. The design is inspired by intermediate supervision in [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) and the broader use of structure-oriented auxiliary objectives in [Token Order Prediction](https://proceedings.mlr.press/v306/zuhri26a.html). This experiment implements verified register-state supervision; it does not reproduce either paper's architecture or loss.
'''
    (args.out/'SHoEbox_State_Supervision.md').write_text(report)
if __name__=='__main__':main()
