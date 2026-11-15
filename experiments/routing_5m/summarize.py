"""Build compact, reproducible comparison tables and the primary PNG."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
arms=['D','C','D_warmup_only','C_warmup_only'];seeds=[1,7,19]
labels={'D':'D · continuing bias','C':'C · continuing bias','D_warmup_only':'D · warmup only','C_warmup_only':'C · warmup only'}
colors={'D':'#c64855','C':'#b48c26','D_warmup_only':'#087f8c','C_warmup_only':'#546bc3'}
data={arm:[json.loads((a.work/f'{arm}_s{s}/results.json').read_text()) for s in seeds] for arm in arms}
benchmark=json.loads((a.work/'serial_benchmark.json').read_text())['summary']
assert all(r['targets']==20_000_000 for rows in data.values() for r in rows)
summary={};csvrows=[]
for arm,rows in data.items():
    dev=np.array([r['curve'][-1]['nll'] for r in rows])
    audit=[json.loads((a.work/f'{arm}_s{s}/probe.json').read_text())['audit'] for s in seeds]
    av=np.array([q['nll'] for q in audit])
    summary[arm]=dict(development_mean=float(dev.mean()),development_sd=float(dev.std(ddof=1)),
        development_by_seed=dev.tolist(),audit_mean=float(av.mean()),audit_sd=float(av.std(ddof=1)),
        audit_by_seed=av.tolist(),parameters=rows[0]['parameters'],experts=rows[0]['experts'],
        serial_median_bps=benchmark[arm]['median_bps'])
    nll=np.array([[q['nll'] for q in r['curve']] for r in rows])
    for i,step in enumerate(rows[0]['curve']):
        for k,r in enumerate(rows):
            q=r['curve'][i];csvrows.append(dict(arm=arm,seed=seeds[k],targets=q['targets'],nll=q['nll'],train_seconds=q['train_seconds']))
    for target in (1.80,1.78,1.76,1.75,1.74):
        confirmed=[]
        for row in rows:
            curve=row['curve'];hit=next((curve[i]['targets'] for i in range(2,len(curve)) if all(q['nll']<=target for q in curve[i-2:i+1])),None)
            confirmed.append(hit)
        summary[arm][f'confirmed_{target:.2f}']=confirmed
with (a.out/'SHoEbox_5M_Routing_Curves.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(14,9.2),gridspec_kw={'height_ratios':[1.05,1]})
fig.patch.set_facecolor('#f7f8fa')
for ax in axs.flat:
    ax.set_facecolor('#ffffff');ax.grid(alpha=.17)
for arm,rows in data.items():
    x=np.array([q['targets'] for q in rows[0]['curve']])/1e6
    ys=np.array([[q['nll'] for q in r['curve']] for r in rows])
    for ax,minimum in ((axs[0,0],1),(axs[0,1],5)):
        mask=x>=minimum;y=ys[:,mask]
        horizontal=x[mask] if minimum==1 else x[mask]*1e6/benchmark[arm]['median_bps']
        ax.plot(horizontal,y.mean(0),color=colors[arm],label=labels[arm],lw=2.2)
        ax.fill_between(horizontal,y.min(0),y.max(0),color=colors[arm],alpha=.10)
axs[0,0].set(title='Learning on fresh FineWeb bytes',xlabel='Training targets (million bytes)',ylabel='Held-out loss (nats/byte)')
axs[0,1].set(title='Training time · serial-speed estimate',xlabel='Estimated total training seconds (points ≥5M)',ylabel='Held-out loss (nats/byte)')
axs[0,0].legend(frameon=False,fontsize=9)
baseline=np.array(summary['D']['audit_by_seed'])
for i,arm in enumerate(arms):
    delta=np.array(summary[arm]['audit_by_seed'])-baseline
    axs[1,0].barh(i,delta.mean(),color=colors[arm],alpha=.8,height=.55)
    axs[1,0].scatter(delta,[i-.10,i,i+.10],color='#263238',s=20,zorder=3)
    if delta.mean()<-.01:
        axs[1,0].text(delta.mean()/2,i,f'{delta.mean():+.5f}',ha='center',va='center',fontsize=10,color='white',fontweight='bold')
    else:
        axs[1,0].text(-.001,i,f'{delta.mean():+.5f}',ha='right',va='center',fontsize=10)
axs[1,0].set_yticks(range(4),[labels[a] for a in arms]);axs[1,0].invert_yaxis()
axs[1,0].axvline(0,color='#777777',lw=1);axs[1,0].set(title='Separate audit at 20M training bytes',xlabel='Loss difference versus D (lower is better)')
axs[1,0].margins(x=.25)
probe=json.loads((a.work/'D_s7/probe.json').read_text())
fixed=json.loads((a.work/'D_warmup_only_s7/probe.json').read_text())
for i,(name,q,color) in enumerate([('D',probe,colors['D']),('Warmup only',fixed,colors['D_warmup_only'])]):
    base=q['frozen']['full']['nll'];v=[q['frozen'][f'swap{l}']['nll']-base for l in (1,2,3)]
    axs[1,1].bar(np.arange(3)+(i-.5)*.32,v,width=.32,label=name,color=color)
    for j,y in enumerate(v):axs[1,1].annotate(f'{y:+.5f}',(j+(i-.5)*.32,y),xytext=(0,5 if y>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
axs[1,1].set_xticks(range(3),['L1','L2','L3']);axs[1,1].set(title='Upper expert swap penalty · seed 7',ylabel='Change in held-out loss (nats/byte)')
axs[1,1].axhline(0,color='#777777',lw=1);axs[1,1].legend(frameon=False,fontsize=9);axs[1,1].margins(y=.25)
fig.suptitle('SHoEbox · 5M-parameter routing diagnosis',fontsize=21,fontweight='bold',x=.05,ha='left',y=.99)
fig.text(.05,.941,'Four matched arms · seeds 1 / 7 / 19 · 20M fresh training targets each · lower loss is better',fontsize=12,color='#4a5560')
fig.text(.05,.015,'Bands show seed ranges. Audit: 1M targets / 310 disjoint documents. D and C share the same expert banks; D adds 12,288 predictor parameters.',fontsize=9,color='#4a5560')
fig.tight_layout(rect=[.025,.04,.99,.92],h_pad=2.5,w_pad=2.2)
fig.savefig(a.out/'SHoEbox_5M_Routing_Diagnosis.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
(a.out/'SHoEbox_5M_Routing_Summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
