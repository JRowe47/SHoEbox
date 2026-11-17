"""Produce compact evidence and primary figures from locked A8 outcomes."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import write
ARMS=('e8','a6','a8')
NAMES={'pad0':'Clean binding','repeat16':'Repeat 16','diverse16':'Diverse 16','compete16':'Competing bindings','overwrite16':'Three-value overwrite','repeat64':'Repeat 64','repeat256':'Repeat 256','diverse64':'Diverse 64','diverse256':'Diverse 256'}
COLORS=['#718096','#2b6cb0','#dd6b20']

def compact_evidence(x):
 if isinstance(x,dict):return {k:compact_evidence(v) for k,v in x.items() if k!='autoreg_correct'}
 if isinstance(x,list):return [compact_evidence(v) for v in x]
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out;o.mkdir(exist_ok=True,parents=True);ev=o/'evidence';ev.mkdir(exist_ok=True)
 read=lambda p:json.loads((w/p).read_text())
 result=read('a8/results.json');ext=read('extended/results.json');speed=read('benchmark.json');updates=read('update_metrics.json');conflicts=read('conflicts.json');retention=read('retention.json');geometry=read('geometry.json');verification=read('verification.json')
 signatures=read('signature_traces.json')
 for r,sr in zip(ext['rows'],signatures['rows']):
  for arm in ARMS:r['arms'][arm]['traces']=sr['arms'][arm]
 means={arm:{c:float(np.mean([r['arms'][arm]['development'][c]['exact'] for r in result['rows']])) for c in NAMES} for arm in ARMS}
 fit=[]
 for r in result['rows']:
  for arm,x in r['arms'].items():
   for phase,y in [('warmup',x['warmup']),('continuation',x)]:
    keys=('bytes','nuisance_bytes','replaced_padding_bytes','padding_bytes','targets','answer_targets','finish_events')
    row=dict(seed=r['seed'],arm=arm,phase=phase,cpu_seconds=y['cpu_seconds'],**{k:y['cursor'][k] for k in keys})
    for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):row[k]=y['inventory'][k]-y['initial_inventory'][k]
    fit.append(row)
 totals={k:sum(x[k] for x in fit) for k in ('bytes','nuisance_bytes','replaced_padding_bytes','padding_bytes','targets','answer_targets','local_projection_steps','local_presence_steps','gate_targets','local_training_windows','cpu_seconds')};assert totals['bytes']==20250000
 processes={p.stem:json.loads(p.read_text()) for p in w.glob('*_process.json') if '_artifact_' not in p.name}
 inference=dict(signatures=sum(v['inference_bytes'] for r in signatures['rows'] for ar in r['arms'].values() for v in ar.values()),primary=sum(x['inference_bytes'] for r in result['rows'] for arm in ARMS for x in [*r['arms'][arm]['development'].values(),r['arms'][arm]['binary_regression']]),
  extended=sum(x['inference_bytes'] for r in ext['rows'] for x in r['arms'].values()),updates=updates['inference_bytes'],conflicts=conflicts['inference_bytes'],retention=retention['inference_bytes'],geometry=geometry['inference_bytes'],timing=speed['total_inference_bytes'],survivors=read('survivors.json')['inference_bytes'])
 padding=dict(signatures=sum(v['padding_bytes'] for r in signatures['rows'] for ar in r['arms'].values() for v in ar.values()),primary=sum(x['padding_bytes'] for r in result['rows'] for arm in ARMS for x in [*r['arms'][arm]['development'].values(),r['arms'][arm]['binary_regression']]),
  extended=sum(x['padding_bytes'] for r in ext['rows'] for x in r['arms'].values()),updates=updates['padding_bytes'],conflicts=conflicts['padding_bytes'],retention=retention['padding_bytes'],geometry=geometry['padding_bytes'],timing=speed['timing_padding_bytes'],survivors=read('survivors.json')['padding_bytes'])
 accounting=dict(fit_phases=fit,fit_totals=totals,pilot_fitting_bytes=0,inference_bytes=inference,inference_padding_bytes=padding,
  local_hybrid_objective_accounting=read('prefit_correctness_bytes.json').get('research_totals'),
  exact_cursor_correctness=read('cursors.json')['discarded_correctness_updates'],prefit_python_correctness=read('prefit_correctness_bytes.json'),
  historical_correctness_bytes=read('history.json')['correctness_only_discarded_input_bytes'],processes=processes,
  research_and_verification_cpu_seconds=sum(r['cpu_seconds'] for r in processes.values()),
  scope='Process CPU includes descendants/compilation/repeated failed checks. Fit/evaluation subphase CPU is included, not added twice. Workspace setup, source editing, read-only inspection and lightweight syntax/report-text commands were not separately CPU-metered and performed no research fitting. Historical unit/native microtrajectory scalar updates are not instrumented; their complete CPU/logs are retained. Artifact generation and packaging are separately receipted.')
 write(o/'accounting.json',accounting);write(o/'verification.json',verification)
 compact=[]
 for r in result['rows']:
  compact.append(dict(seed=r['seed'],arms={arm:{k:v for k,v in x.items() if k in ('development','binary_regression','curve','inventory','initial_inventory','checkpoint_sha256','cursor_sha256','evaluation_cpu_seconds','cpu_seconds','wall_seconds')} for arm,x in r['arms'].items()}))
 write(ev/'development.json',compact_evidence(compact));write(ev/'conflicts_summary.json',[dict(seed=r['seed'],arm=r['arm'],summary=r['summary']) for r in conflicts['rows']])
 for source,dest in [('prefit.json','prefit.json'),('codebook.json','codebook.json'),('identity_attribution.json','identity_attribution.json'),('survivors.lock.json','survivors_lock.json'),('prefit_correctness_bytes.json','local_objective_accounting.json'),('a8/plan.json','plan.json'),('a8/selection.json','selection.json'),('extended/results.json','extended.json'),('benchmark.json','benchmark.json'),('update_metrics.json','update_metrics.json'),('cursors.json','cursors.json'),('history.json','history.json'),('environment.json','environment.json'),('matching_geometry.json','matching_geometry.json'),('conflicts.lock.json','conflicts_lock.json')]:shutil.copy2(w/source,ev/dest)
 write(ev/'extended.json',compact_evidence(ext))
 for name in ('coverage.json','frozen_fit.lock.json','evaluation.lock.json','frozen_diagnosis.lock.json','delivery_verification.json','python_counts.json'):
  shutil.copy2(w/name,ev/name)
 write(ev/'survivors_summary.json',[dict(seed=r['seed'],arm=r['arm'],summary=r['summary']) for r in read('survivors.json')['rows']])
 write(ev/'retention_summary.json',[dict(seed=r['seed'],arm=r['arm'],summary=r['summary']) for r in retention['rows']])
 shutil.copy2(w/'retention.lock.json',ev/'retention_lock.json')
 write(ev/'geometry_summary.json',[dict(seed=r['seed'],summary=r['summary']) for r in geometry['rows']])
 shutil.copy2(w/'geometry.lock.json',ev/'geometry_lock.json')
 for p in w.glob('*_process.json'):shutil.copy2(p,ev/p.name)
 for p in w.glob('*.txt'):shutil.copy2(p,ev/p.name)
 summary=dict(means=means,fit_totals=totals,public_default='E8 unchanged',a8_development_pass=verification['gates']['a8']['development_pass'],a1_audit_opened=False,a2_audit_opened=False,a3_audit_opened=False,a4_audit_opened=False,a5_audit_opened=False,a6_audit_opened=False,a7_audit_opened=False,a8_audit_opened=False)
 write(o/'summary.json',summary)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 fig,axes=plt.subplots(1,2,figsize=(14,5.4),gridspec_kw={'width_ratios':[5,4]})
 for ax,conds,title,gate in [(axes[0],list(NAMES)[:5],'Primary development • three-value worlds',95),(axes[1],list(NAMES)[5:],'Long filler • trained at 16 bytes',90)]:
  xx=np.arange(len(conds))
  for i,arm in enumerate(ARMS):
   vals=[means[arm][c]*100 for c in conds];ax.bar(xx+(i-1)*.25,vals,.24,label=arm.upper(),color=COLORS[i])
   if arm=='a8':
    for x,y in zip(xx+.25,vals):ax.text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
  ax.axhline(gate,color='#6b7280',ls='--',lw=1,label=f'{gate}% mean gate');ax.set_ylim(0,108);ax.set_xticks(xx,[NAMES[c].replace(' ','\n',1) for c in conds]);ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[1].text(.99,91.5,'90% mean gate',transform=axes[1].get_yaxis_transform(),ha='right',fontsize=9,color='#6b7280')
 axes[0].set_ylabel('Native whole-episode exactness (%)');axes[0].legend(ncol=2,loc='upper left',bbox_to_anchor=(0,-.23),frameon=False)
 fig.suptitle('EXP03/05-A8 • phase-balanced PAD/diverse curriculum',fontsize=17,fontweight='bold');fig.tight_layout();fig.savefig(o/'SHoEbox_EXP0305_A8_Results.png',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5));names=list(speed['rows']);vals=[100*speed['rows'][n]['relative_native'] for n in names]
 ax.barh([n.replace('_',' ').upper() for n in names],vals,color=['#a0aec0',*COLORS,'#38a169']);ax.axvline(80,ls='--',color='#555');ax.set_xlim(0,max(110,max(vals)+12));ax.set_xlabel('Inference throughput relative to native (%)');ax.set_title('Isolated full-256-byte inference • median of three repeats')
 for i,y in enumerate(vals):ax.text(y+1,i,f'{y:.2f}%',va='center')
 fig.tight_layout();fig.savefig(o/'SHoEbox_EXP0305_A8_Throughput.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for ax,cond in zip(axes,('compete16','overwrite16')):
  keys=['target_present','target_recognized','target_shortlisted','native_teacher_fed'];xx=np.arange(4)
  for i,arm in enumerate(ARMS):
   rows=[r['summary'][cond] for r in conflicts['rows'] if r['arm']==arm]
   ax.bar(xx+(i-1)*.25,[100*np.mean([r[k] for r in rows]) for k in keys],.24,label=arm.upper(),color=COLORS[i])
  ax.set_xticks(xx,['Signature\npresent','Recognized','Shortlisted','Native\nanswer']);ax.set_ylim(0,105);ax.set_title(NAMES[cond]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[0].set_ylabel('Required answers (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.suptitle('Frozen state diagnosis • teacher-fed localization',fontsize=16);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(o/'SHoEbox_EXP0305_A8_Diagnosis.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 x=np.arange(3);labels=['16 bytes','64 bytes','256 bytes']
 for i,arm in enumerate(ARMS):
  y=[100*np.mean([r['arms'][arm]['traces']['diverse'+str(d)]['trace']['present'] for r in ext['rows']]) for d in (16,64,256)]
  axes[0].plot(x,y,'o-',label=arm.upper(),color=COLORS[i])
 axes[0].set_xticks(x,labels);axes[0].set_ylim(0,105);axes[0].set_ylabel('Required signature present (%)');axes[0].set_title('Memory survival through diverse filler');axes[0].legend(frameon=False)
 loss=[100*np.mean([r['summary']['target_lost_fraction'] for r in retention['rows'] if r['arm']==arm]) for arm in ARMS]
 axes[1].bar([a.upper() for a in ARMS],loss,color=COLORS);axes[1].set_ylabel('Original targets removed at least once (%)');axes[1].set_ylim(0,105);axes[1].set_title('Separate 256-byte admission trace')
 for i,y in enumerate(loss):axes[1].text(i,y+2,f'{y:.1f}%',ha='center')
 fig.tight_layout();fig.savefig(o/'SHoEbox_EXP0305_A8_Retention.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.6))
 rstats={arm:[r['summary'] for r in retention['rows'] if r['arm']==arm] for arm in ('a6','a8')}
 sstats={arm:[r['summary'] for r in read('survivors.json')['rows'] if r['arm']==arm] for arm in ('a6','a8')}
 for i,arm in enumerate(('a6','a8')):
  rs=rstats[arm];ss=sstats[arm];xx=np.arange(2)+(i-.5)*.3
  vals=[100*sum(r['full_diverse_recognized'] for r in rs)/sum(r['full_diverse_pairs'] for r in rs),100*sum(r['true_record_phase_false_negatives'] for r in rs)/sum(r['true_record_phase_events'] for r in rs)]
  axes[0].bar(xx,vals,.29,color=COLORS[i+1],label=arm.upper())
  for x,y in zip(xx,vals):axes[0].text(x,y+.3,f'{y:.2f}%',ha='center',fontsize=9)
  xx=np.arange(3)+(i-.5)*.3;vals=[100*np.mean([r[k] for r in ss]) for k in ('survivor_accuracy','survivor_shortlisted','incorrect_survivor_unannotated_mass')]
  axes[1].bar(xx,vals,.29,color=COLORS[i+1],label=arm.upper())
  for x,y in zip(xx,vals):axes[1].text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
 axes[0].set_xticks([0,1],['Diverse-pair false\nrecognition','True record-phase\nfalse negatives']);axes[0].set_ylabel('Events (%)');axes[0].set_ylim(0,max(20,axes[0].get_ylim()[1]*1.12));axes[0].set_title('Presence errors at window completion')
 axes[1].set_xticks([0,1,2],['Correct given\ntarget survives','Surviving target\nshortlisted','Filler mass on\nwrong survivor answers']);axes[1].set_ylim(0,108);axes[1].set_ylabel('Answers / selected mass (%)');axes[1].set_title('Readout competition with a surviving target')
 axes[0].legend(frameon=False);fig.suptitle('A8 balanced record context • diverse256 frozen diagnosis',fontsize=15);fig.tight_layout();fig.savefig(o/'SHoEbox_EXP0305_A8_Presence.png');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
