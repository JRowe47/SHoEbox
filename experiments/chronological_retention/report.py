"""Produce compact evidence and primary figures from locked A3 outcomes."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import write
ARMS=('e8','a2','a3')
NAMES={'pad0':'Clean binding','repeat16':'Repeat 16','diverse16':'Diverse 16','compete16':'Competing bindings','overwrite16':'Three-value overwrite','repeat64':'Repeat 64','repeat256':'Repeat 256','diverse64':'Diverse 64','diverse256':'Diverse 256'}
COLORS=['#718096','#2b6cb0','#dd6b20']

def compact_evidence(x):
 if isinstance(x,dict):return {k:compact_evidence(v) for k,v in x.items() if k!='autoreg_correct'}
 if isinstance(x,list):return [compact_evidence(v) for v in x]
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out;o.mkdir(exist_ok=True,parents=True);ev=o/'evidence';ev.mkdir(exist_ok=True)
 read=lambda p:json.loads((w/p).read_text())
 result=read('a3/results.json');ext=read('extended/results.json');speed=read('benchmark.json');updates=read('update_metrics.json');conflicts=read('conflicts.json');verification=read('verification.json')
 means={arm:{c:float(np.mean([r['arms'][arm]['development'][c]['exact'] for r in result['rows']])) for c in NAMES} for arm in ARMS}
 fit=[]
 for r in result['rows']:
  for arm,x in r['arms'].items():
   for phase,y in [('warmup',x['warmup']),('continuation',x)]:
    keys=('bytes','padding_bytes','targets','answer_targets','finish_events')
    row=dict(seed=r['seed'],arm=arm,phase=phase,cpu_seconds=y['cpu_seconds'],**{k:y['cursor'][k] for k in keys})
    for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):row[k]=y['inventory'][k]-y['initial_inventory'][k]
    fit.append(row)
 totals={k:sum(x[k] for x in fit) for k in ('bytes','padding_bytes','targets','answer_targets','local_projection_steps','local_presence_steps','gate_targets','local_training_windows','cpu_seconds')};assert totals['bytes']==20250000
 processes={p.stem:json.loads(p.read_text()) for p in w.glob('*_process.json') if '_artifact_' not in p.name}
 inference=dict(primary=sum(x['inference_bytes'] for r in result['rows'] for arm in ARMS for x in [*r['arms'][arm]['development'].values(),r['arms'][arm]['binary_regression']]),
  extended=sum(x['inference_bytes'] for r in ext['rows'] for x in r['arms'].values()),updates=updates['inference_bytes'],conflicts=conflicts['inference_bytes'],timing=speed['total_inference_bytes'])
 padding=dict(primary=sum(x['padding_bytes'] for r in result['rows'] for arm in ARMS for x in [*r['arms'][arm]['development'].values(),r['arms'][arm]['binary_regression']]),
  extended=sum(x['padding_bytes'] for r in ext['rows'] for x in r['arms'].values()),updates=updates['padding_bytes'],conflicts=conflicts['padding_bytes'],timing=speed['timing_padding_bytes'])
 accounting=dict(fit_phases=fit,fit_totals=totals,pilot_fitting_bytes=0,inference_bytes=inference,inference_padding_bytes=padding,
  exact_cursor_correctness=read('cursors.json')['discarded_correctness_updates'],prefit_python_correctness=read('prefit_correctness_bytes.json'),
  historical_correctness_bytes=read('history.json')['correctness_only_discarded_input_bytes'],processes=processes,
  research_and_verification_cpu_seconds=sum(r['cpu_seconds'] for r in processes.values()),
  scope='Process CPU includes descendants/compilation/repeated failed checks. Fit/evaluation subphase CPU is included, not added twice. Workspace setup, source editing, read-only inspection and lightweight syntax/report-text commands were not separately CPU-metered and performed no research fitting. Historical unit/native microtrajectory scalar updates are not instrumented; their complete CPU/logs are retained. Artifact generation and packaging are separately receipted.')
 write(o/'accounting.json',accounting);write(o/'verification.json',verification)
 compact=[]
 for r in result['rows']:
  compact.append(dict(seed=r['seed'],arms={arm:{k:v for k,v in x.items() if k in ('development','binary_regression','curve','inventory','initial_inventory','checkpoint_sha256','cursor_sha256','evaluation_cpu_seconds','cpu_seconds','wall_seconds')} for arm,x in r['arms'].items()}))
 write(ev/'development.json',compact_evidence(compact));write(ev/'conflicts_summary.json',[dict(seed=r['seed'],arm=r['arm'],summary=r['summary']) for r in conflicts['rows']])
 for source,dest in [('prefit.json','prefit.json'),('a3/plan.json','plan.json'),('a3/selection.json','selection.json'),('extended/results.json','extended.json'),('benchmark.json','benchmark.json'),('update_metrics.json','update_metrics.json'),('cursors.json','cursors.json'),('history.json','history.json'),('environment.json','environment.json'),('conflicts.lock.json','conflicts_lock.json')]:shutil.copy2(w/source,ev/dest)
 write(ev/'extended.json',compact_evidence(ext))
 for p in w.glob('*_process.json'):shutil.copy2(p,ev/p.name)
 for p in w.glob('*.txt'):shutil.copy2(p,ev/p.name)
 summary=dict(means=means,fit_totals=totals,public_default='E8 unchanged',a3_development_pass=verification['gates']['a3']['development_pass'],a1_audit_opened=False,a2_audit_opened=False,a3_audit_opened=False)
 write(o/'summary.json',summary)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 fig,axes=plt.subplots(1,2,figsize=(14,5.4),gridspec_kw={'width_ratios':[5,4]})
 for ax,conds,title,gate in [(axes[0],list(NAMES)[:5],'Primary development • three-value worlds',95),(axes[1],list(NAMES)[5:],'Long filler • trained at 16 bytes',90)]:
  xx=np.arange(len(conds))
  for i,arm in enumerate(ARMS):
   vals=[means[arm][c]*100 for c in conds];ax.bar(xx+(i-1)*.25,vals,.24,label=arm.upper(),color=COLORS[i])
   if arm=='a3':
    for x,y in zip(xx+.25,vals):ax.text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
  ax.axhline(gate,color='#6b7280',ls='--',lw=1,label=f'{gate}% mean gate');ax.set_ylim(0,108);ax.set_xticks(xx,[NAMES[c].replace(' ','\n',1) for c in conds]);ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[0].set_ylabel('Native whole-episode exactness (%)');axes[0].legend(ncol=2,loc='upper left',bbox_to_anchor=(0,-.23),frameon=False)
 fig.suptitle('EXP03/05-A3 • continuous chronological matching',fontsize=17,fontweight='bold');fig.tight_layout();fig.savefig(o/'SHoEbox_EXP0305_A3_Results.png',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5));names=list(speed['rows']);vals=[100*speed['rows'][n]['relative_native'] for n in names]
 ax.barh([n.replace('_',' ').upper() for n in names],vals,color=['#a0aec0',*COLORS,'#38a169']);ax.axvline(80,ls='--',color='#555');ax.set_xlim(0,max(110,max(vals)+12));ax.set_xlabel('Inference throughput relative to native (%)');ax.set_title('Isolated full-256-byte inference • median of three repeats')
 for i,y in enumerate(vals):ax.text(y+1,i,f'{y:.2f}%',va='center')
 fig.tight_layout();fig.savefig(o/'SHoEbox_EXP0305_A3_Throughput.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for ax,cond in zip(axes,('compete16','overwrite16')):
  keys=['target_present','target_recognized','target_shortlisted','native_teacher_fed'];xx=np.arange(4)
  for i,arm in enumerate(ARMS):
   rows=[r['summary'][cond] for r in conflicts['rows'] if r['arm']==arm]
   ax.bar(xx+(i-1)*.25,[100*np.mean([r[k] for r in rows]) for k in keys],.24,label=arm.upper(),color=COLORS[i])
  ax.set_xticks(xx,['Signature\npresent','Recognized','Shortlisted','Native\nanswer']);ax.set_ylim(0,105);ax.set_title(NAMES[cond]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[0].set_ylabel('Required answers (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.suptitle('Frozen state diagnosis • teacher-fed localization',fontsize=16);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(o/'SHoEbox_EXP0305_A3_Diagnosis.png');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
