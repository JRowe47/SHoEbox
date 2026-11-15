"""Aggregate paired trials without treating documents as independent seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bootstrap_difference(base,changed):
    by_seed=[];counts=None
    for seed in (1,7,19):
        a=base[seed]['audit']['documents'];b=changed[seed]['audit']['documents']
        assert [(r['row_idx'],r['targets']) for r in a]==[(r['row_idx'],r['targets']) for r in b]
        by_seed.append(np.array([v['nll']-u['nll'] for u,v in zip(a,b)]))
        counts=np.array([r['targets'] for r in a])
    means=np.mean(by_seed,axis=0);rng=np.random.default_rng(139)
    ids=rng.integers(0,len(means),size=(10000,len(means)))
    resampled=(means[ids]*counts[ids]).sum(1)/counts[ids].sum(1)
    return dict(paired_seed_deltas=[float((v*counts).sum()/counts.sum()) for v in by_seed],
        document_bootstrap_95=np.quantile(resampled,[.025,.975]).tolist(),
        description='Resamples documents after averaging paired differences over fitted seeds. Seed variation is reported separately.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--timing',type=Path)
    p.add_argument('--means',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);arms=('C',a.arm)
    runs={arm:{} for arm in arms};probes={arm:{} for arm in arms};summary={'arms':{}}
    for arm,root in zip(arms,(a.baseline,a.candidate)):
        for seed in (1,7,19):
            folder=root/f'{arm}_s{seed}'
            runs[arm][seed]=json.loads((folder/'results.json').read_text());probes[arm][seed]=json.loads((folder/'probe.json').read_text())
            assert runs[arm][seed]['checkpoint_sha256']==probes[arm][seed]['checkpoint_sha256']
        dev=[runs[arm][s]['curve'][-1]['nll'] for s in (1,7,19)];audit=[probes[arm][s]['audit']['nll'] for s in (1,7,19)]
        swaps={f'L{l}':[np.mean([v['delta'] for k,v in probes[arm][s]['swaps'].items() if k.startswith(f'L{l}_')]) for s in (1,7,19)] for l in (1,2,3)}
        summary['arms'][arm]=dict(development=dev,audit=audit,development_mean=float(np.mean(dev)),development_sd=float(np.std(dev,ddof=1)),
            audit_mean=float(np.mean(audit)),audit_sd=float(np.std(audit,ddof=1)),swap_penalties=swaps)
    summary['audit_difference']=bootstrap_difference(probes['C'],probes[a.arm])
    if a.means:
        summary['fixed_mean_probes']={str(s):json.loads((a.means/f'means_s{s}.json').read_text()) for s in (1,7,19)}
    if a.timing:
        timing=json.loads(a.timing.read_text());summary['timing']=timing
        for arm in arms:summary['arms'][arm]['bytes_per_second']=float(np.median([v['bytes_per_second'] for v in timing['rows'] if v['arm']==arm]))
    (a.out/'SHoEbox_L2_L3_Summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors={'C':'#64748b',a.arm:'#087f8c'}
    labels={'C':'Corrected C',a.arm:'Extra recall credit to experts'}
    for arm in arms:
        rows=runs[arm];curves=[r['curve'] for r in rows.values()];points=sorted(set.intersection(*[set(v['targets'] for v in c) for c in curves]))
        vals=np.array([[next(v['nll'] for v in c if v['targets']==x) for x in points] for c in curves]);x=np.array(points)/1e6;mean=vals.mean(0);sd=vals.std(0,ddof=1)
        axs[0,0].plot(x,mean,color=colors[arm],lw=2,label=labels[arm]);axs[0,0].fill_between(x,mean-sd,mean+sd,color=colors[arm],alpha=.15)
        if a.timing and not a.means:
            seconds=np.array(points)/summary['arms'][arm]['bytes_per_second'];axs[0,1].plot(seconds,mean,color=colors[arm],lw=2,label=labels[arm])
    axs[0,0].set(xlabel='Training targets (million bytes)',ylabel='Development NLL (nats/byte)',title='Matched bytes · mean ± seed SD',xlim=(1,20),ylim=(1.68,2.05));axs[0,0].legend(frameon=False)
    if a.means:
        removed=[np.mean([probes['C'][s]['ablations'][f'L{l}_no_modulation']['delta'] for s in (1,7,19)]) for l in (2,3)]
        fixed=[np.mean([summary['fixed_mean_probes'][str(s)]['means'][f'L{l}_modulation']['delta'] for s in (1,7,19)]) for l in (2,3)]
        for shift,values,color,label in [(-.18,removed,'#d48a36','Remove modulation'),(.18,fixed,'#667fa6','Replace with training mean')]:
            bars=axs[0,1].bar(np.arange(2)+shift,values,width=.34,color=color,label=label)
            axs[0,1].bar_label(bars,labels=[f'{v:.5f}' for v in values],padding=4,fontsize=10)
        axs[0,1].set(xticks=[0,1],xticklabels=['L2','L3'],ylabel='Development NLL increase',title='Apparent contribution is mostly constant bias',ylim=(0,.073))
        axs[0,1].legend(frameon=False,fontsize=9,loc='upper left')
    else:axs[0,1].set(xlabel='Estimated serial training seconds',ylabel='Development NLL (nats/byte)',title='Matched time · separate serial benchmark',ylim=(1.68,1.85))
    for j,arm in enumerate(arms):
        samples=summary['arms'][arm]['swap_penalties'];means=[np.mean(samples[f'L{l}']) for l in (2,3)];sd=[np.std(samples[f'L{l}'],ddof=1) for l in (2,3)]
        axs[1,0].bar(np.arange(2)+(-.18 if j==0 else .18),means,width=.34,color=colors[arm],label=labels[arm],yerr=sd,capsize=4)
    axs[1,0].axhline(0,color='#334155',lw=.8);axs[1,0].set(xticks=[0,1],xticklabels=['L2','L3'],ylabel='NLL increase after expert permutation',title='Specialization · three rotations per seed');axs[1,0].legend(frameon=False)
    axs[1,1].axis('off');table=[]
    for seed_idx,seed in enumerate((1,7,19)):
        old=summary['arms']['C']['audit'][seed_idx];new=summary['arms'][a.arm]['audit'][seed_idx];table.append([str(seed),f'{old:.6f}',f'{new:.6f}',f'{new-old:+.6f}'])
    table.append(['Mean',f"{summary['arms']['C']['audit_mean']:.6f}",f"{summary['arms'][a.arm]['audit_mean']:.6f}",f"{summary['arms'][a.arm]['audit_mean']-summary['arms']['C']['audit_mean']:+.6f}"])
    t=axs[1,1].table(cellText=table,colLabels=['Seed','Corrected C','Extra credit','Difference'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.7)
    axs[1,1].set_title('Reserved audit · 310 documents / 1M targets',pad=12)
    fig.suptitle('SHoEbox · L2/L3 language-credit investigation\n5M parameters · 20M FineWeb training targets · seeds 1, 7, 19',fontsize=17,fontweight='bold')
    fig.savefig(a.out/'SHoEbox_L2_L3_Results.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
