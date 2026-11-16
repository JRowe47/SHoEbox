"""Primary E8 results and loss/cost figures from frozen measurement files."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();w=args.workspace;o=args.report;s=read(o/'summary.json');g=s['generic'];phase='audit' if 'audit' in g else 'development';colors=['#64748b','#7c3aed','#0f766e']
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 language=g[phase]['checks'][0];names=['Native v19','D3 joint','E8 local gate'];arms=['native','d3_joint','local_gate'];means=[language['scores'][k]['mean'] for k in arms]
 axes[0,0].scatter(range(3),means,color=colors,s=130,marker='_',linewidths=3,zorder=5);axes[0,0].set_xticks(range(3),names)
 for i,k in enumerate(arms):axes[0,0].scatter([i]*3,language['scores'][k]['values'],s=28,color='#111827',zorder=4)
 axes[0,0].axhline(means[0]+.01,color='#b45309',ls='--',label='Native +0.01 limit');axes[0,0].set(ylim=(min(means)-.035,max(means)+.035),ylabel='Nats per byte (lower is better)',title=f'Fresh generic language · {phase}');axes[0,0].legend(fontsize=9)
 selected='audit' if 'audit' in s['binding']['formal'] else 'development';r=s['binding']['formal'][selected]['local_gate'];ret=s['binding']['formal'].get('retained_'+selected,{}).get('local_gate')
 vals=[100*r['primary_mean'],100*ret['primary_mean'] if ret else 0,100*r['offset_mean'],100*r['worst_offset'],100*np.mean(r['both_orders'])]
 labels=['Binding','After 1M\nlanguage','Offset\nmean','Worst\noffset','Both\norders'];axes[0,1].bar(labels,vals,color='#0f766e',width=.65)
 for i,v in enumerate(vals):axes[0,1].text(i,v+.6,f'{v:.2f}',ha='center',fontsize=9)
 axes[0,1].set(ylim=(80,104),ylabel='Native exactness (%)',title=f'500K E8 · three seeds · {selected}')
 for size,color in [('formal','#0f766e'),('1m','#2563eb'),('5m','#7c3aed')]:
  data=s['binding'][size];p='audit' if 'audit' in data else 'development';axes[1,0].plot(range(64),np.array(data[p]['local_gate']['offsets'])*100,label='500K' if size=='formal' else size.upper(),color=color,alpha=.8)
 axes[1,0].axhline(95,color='#b45309',ls='--',label='Required offset mean 95%');axes[1,0].set(xlabel='Boundary offset (bytes)',ylabel='Mean native exactness (%)',ylim=(75,102),title='All 64 boundary offsets');axes[1,0].legend(fontsize=9)
 sizes=['formal','1m','5m'];speed=[s['binding'][k].get('benchmark',{}).get('relative',float('nan'))*100 for k in sizes]
 axes[1,1].bar(['500K','1M','5M'],speed,color=['#0f766e','#2563eb','#7c3aed'],width=.55)
 axes[1,1].axhline(80,color='#b45309',ls='--',label='80% required');axes[1,1].axhline(100,color='#64748b',ls=':',label='Matched native')
 for i,v in enumerate(speed):
  if np.isfinite(v):axes[1,1].text(i,v+1,f'{v:.2f}%',ha='center')
 axes[1,1].set(ylim=(0,110),ylabel='Native inference throughput retained (%)',title='Isolated full-256-byte inference');axes[1,1].legend(loc='lower right',fontsize=9)
 fig.suptitle('SHoEbox EXP01-E8 · fixed 1/256 local applicability prior',fontsize=16)
 fig.savefig(o/'SHoEbox_EXP01E8_Results.png',dpi=180);plt.close(fig)
 rows=read(w/'exp01e8-regressions/train.json')['rows'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for arm,color,label in zip(arms,colors,names):
  runs=[r for r in rows if r['arm']==arm and r['task']=='language'];y=np.mean([[p['dev_nll'] for p in r['curve']] for r in runs],axis=0);t=np.mean([[p['cpu_seconds'] for p in r['curve']] for r in runs],axis=0);b=np.array([p['bytes'] for p in runs[0]['curve']])/1e6
  axes[0].plot(b,y,'o-',color=color,label=label);axes[1].plot(t,y,'o-',color=color,label=label)
 for ax in axes:ax.set(ylabel='Development NLL (nats/byte)');ax.legend(fontsize=9)
 axes[0].set(xlabel='Generic training targets (millions)',title='Language learning by bytes · three seeds');axes[1].set(xlabel='Fitting CPU seconds (evaluation excluded)',title='Language learning by training CPU cost')
 fig.savefig(o/'SHoEbox_EXP01E8_Learning_Cost.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
