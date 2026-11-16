"""Rebuild compact development evidence and primary scientific PNGs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out;o.mkdir(exist_ok=True,parents=True)
 data=json.loads((w/'a1/results.json').read_text());diagnosis=json.loads((w/'diagnosis_corrected/results.json').read_text());conds=list(data['rows'][0]['arms']['e8']['development']);summary={}
 for arm in ('native','e8','a1'):
  rows=[r['arms'][arm] for r in data['rows']];summary[arm]={}
  for c in conds:
   x=[r['development'][c]['exact'] for r in rows];summary[arm][c]=dict(mean=float(np.mean(x)),minimum=min(x),seeds=x,positions=np.mean([r['development'][c]['positions'] for r in rows],axis=0).tolist())
 phases=[]
 for row in data['rows']:
  for arm,r in row['arms'].items():
   for phase,v in [('warmup',r['warmup']),('continuation',r)]:
    cur=v['cursor'];counts={k:v['inventory'][k]-v['initial_inventory'][k] for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows')}
    phases.append(dict(seed=row['seed'],arm=arm,phase=phase,input_bytes=cur['bytes'],padding_bytes=cur['padding_bytes'],targets=cur['targets'],answer_targets=cur['answer_targets'],cpu_seconds=v['cpu_seconds'],**counts))
 totals={k:sum(r[k] for r in phases) for k in ('input_bytes','padding_bytes','targets','answer_targets','cpu_seconds','local_projection_steps','local_presence_steps','gate_targets','local_training_windows')}
 processes={p.stem:json.loads(p.read_text()) for p in sorted(w.glob('*_process.json')) if not p.stem.startswith(('report','delivery'))}
 ledger=dict(fit_phases=phases,fit_totals=totals,processes=processes,total_metered_process_cpu_seconds=sum(r['cpu_seconds'] for r in processes.values()),
   scope='Whole-process totals are summed once; fitting/phase CPU is a subset, not added again. Unit/native microtrajectories are discarded engineering work, not research fitting. No local-head training on generic byte objectives.',new_research_pilot_bytes=0,audit_opened=False)
 ledger['artifact_processes']={p.stem:json.loads(p.read_text()) for p in sorted(w.glob('*_process.json')) if p.stem.startswith(('report','delivery'))}
 ledger['primary_development']={key:sum(v[key] for row in data['rows'] for arm in row['arms'].values() for v in arm['development'].values()) for key in ('inference_bytes','padding_bytes')}
 if (w/'extended/results.json').exists():
  extended=json.loads((w/'extended/results.json').read_text())
  ledger['extended_development']={key:sum(arm[key] for row in extended['rows'] for arm in row['arms'].values()) for key in ('inference_bytes','padding_bytes')}
 if (w/'history.json').exists():ledger['historical_discarded_input_bytes']=json.loads((w/'history.json').read_text())['correctness_only_discarded_input_bytes']
 for name,path in [('diagnostic_initial','diagnosis/partial_accounting.json'),('diagnostic_corrected','diagnosis_corrected/accounting.json'),('cursor_checks','cursors.json'),('throughput','benchmark.json'),('update_diagnostic','update_metrics.json')]:
  if (w/path).exists():ledger[name]=json.loads((w/path).read_text())
 write(o/'summary.json',dict(arms=summary,diagnosis=diagnosis,three_seeds=[1,7,19],development_worlds=128,audit_opened=False,public_default='qualified E8 unchanged'))
 write(o/'accounting.json',ledger)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 colors={'native':'#64748b','e8':'#cb6e36','a1':'#087f8c'}
 fig,ax=plt.subplots(1,2,figsize=(12.8,4.7),layout='constrained');x=np.arange(4);delays=[0,16,64,256]
 for cond,label,color,style in [('pad','Padding','#64748b','-'),('diverse','Diverse filler','#cb6e36','-'),('freeze','Diverse, cache frozen','#087f8c','--')]:
  keys=['pad0']+[(f'diverse{d}_freeze' if cond=='freeze' else f'{cond}{d}') for d in delays[1:]]
  ax[0].plot(x,[diagnosis[k]['exact']*100 for k in keys],marker='o',label=label,color=color,ls=style,lw=2)
  ax[1].plot(x,[diagnosis[k]['trace']['present']*100 for k in keys],marker='o',label=label,color=color,ls=style,lw=2)
 for aa in ax:aa.set_xticks(x,[str(d) for d in delays]);aa.set_xlabel('Inserted filler bytes');aa.set_ylim(-4,106);aa.grid(axis='y',alpha=.2)
 ax[0].set_ylabel('Native two-answer exactness (%)');ax[1].set_ylabel('Required L1 signature present (%)');ax[0].legend(loc='lower left',fontsize=9)
 ax[0].set_title('Qualified E8: filler content matters');ax[1].set_title('Eviction explains most long-filler failure')
 fig.suptitle('EXP03/05 diagnosis · 64 fresh development worlds · zero fitting',fontsize=15,fontweight='bold')
 fig.savefig(o/'SHoEbox_EXP0305_Diagnosis.png',dpi=190);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
 for arm,label in [('native','Native v19'),('e8','Matched E8'),('a1','A1 replacement')]:
  keys=['pad0','diverse16','diverse64','diverse256'];vals=[summary[arm][k]['mean']*100 for k in keys]
  ax[0].plot(x,vals,label=label,marker='o',lw=2,color=colors[arm])
 ax[0].set_xticks(x,['0','16','64','256']);ax[0].set_xlabel('Diverse filler bytes');ax[0].set_ylabel('Native two-answer exactness (%)');ax[0].legend(fontsize=9,loc='lower left');ax[0].set_title('Same slots, better retention')
 labels=['Clean','Repeated\n16 bytes','Diverse\n16 bytes','Two competing\nbindings','Overwrite*\n(binary)'];ks=['pad0','repeat16','diverse16','compete16','overwrite16'];xx=np.arange(5)
 for j,arm in enumerate(('native','e8','a1')):ax[1].bar(xx+(j-1)*.25,[summary[arm][k]['mean']*100 for k in ks],.24,color=colors[arm],label=arm)
 ax[1].set_xticks(xx,labels);ax[1].set_title('Competing records still fail retrieval');ax[1].text(0,-.2,'* Overwrite: perfect latest-value shortcut; diagnostic only.',transform=ax[1].transAxes,fontsize=9);ax[1].axhline(95,color='#a11',ls=':',lw=1);ax[1].text(4.45,96,'95% gate',ha='right',fontsize=8,color='#a11')
 for aa in ax:aa.set_ylim(0,105);aa.grid(axis='y',alpha=.2);aa.set_axisbelow(True)
 fig.suptitle('A1 is a partial retention repair; qualified E8 stays the default',fontsize=15,fontweight='bold')
 fig.savefig(o/'SHoEbox_EXP0305_Results.png',dpi=190);plt.close(fig)
 if (w/'benchmark.json').exists():
  b=json.loads((w/'benchmark.json').read_text());names=['native','e8','a1','public_e8'];fig,ax=plt.subplots(figsize=(8.5,4.5),layout='constrained')
  vals=[b['rows'][n]['relative_native']*100 for n in names];bars=ax.bar(['Native v19','Research E8','A1','Public E8'],vals,color=['#64748b','#cb6e36','#087f8c','#9c5a2f']);ax.axhline(80,color='#a11',ls='--',label='80% gate')
  for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+1,f'{val:.1f}%',ha='center')
  ax.set_ylim(0,112);ax.set_ylabel('Throughput relative to native (%)');ax.set_title('Isolated full-256-byte inference · interleaved median CPU');ax.legend();fig.savefig(o/'SHoEbox_EXP0305_Throughput.png',dpi=190);plt.close(fig)
 print(totals)
if __name__=='__main__':main()
