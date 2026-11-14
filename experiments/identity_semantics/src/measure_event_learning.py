"""Stage-5 placement/learning-rate sweep and isolated paired throughput.

Every policy starts from identical seed-7 content and bottom parameters. Each
gets its own 1,024-byte warmup because context and credit contracts differ.
Timing excludes compilation, checkpoint IO, evaluation and generation. Only
compact JSON is retained. No old checkpoint is silently migrated.
"""
import argparse,hashlib,json,os,resource,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from compare_optimized_runtime import TRAIN,source_hash

def worker(a):
 import numpy as np
 from runtime import fresh_model
 from reference_acceptance import collect,check_integrity,check_hierarchy
 from run_reference_acceptance import generate
 os.sched_setaffinity(0,{a.cpu})
 config=json.loads(a.policy);policy=config['event'];scale=config.get('scale',1.)
 n=a.warmup+a.targets+1
 raw=a.corpus.read_bytes()[:n] if a.corpus else (TRAIN*(n//len(TRAIN)+1))[:n]
 valraw=a.validation.read_bytes()[:1025] if a.validation else (b'The bird crossed the valley. The queen walked through the gate. '*20)[:1025]
 if len(raw)!=n:raise ValueError('Insufficient training data')
 x=np.frombuffer(raw,np.uint8);val=np.frombuffer(valraw,np.uint8)
 m=fresh_model(seed=7,event=policy)
 m.run(x[:a.warmup],x[1:a.warmup+1],mode='periodic',lr=.04*scale,lp=.04*scale/3)
 for name in ('reset_event_metrics','reset_reverse_metrics','reset_content_factor_metrics'):
  if hasattr(m,name):getattr(m,name)()
 points=[];seconds=0.;weighted=0.
 for offset in range(0,a.targets,512):
  lo=a.warmup+offset;k=min(512,a.targets-offset);start=time.perf_counter()
  result=m.run(x[lo:lo+k],x[lo+1:lo+k+1],mode='periodic',lr=.04*scale,lp=.04*scale/3,logits=a.logits)
  elapsed=time.perf_counter()-start;nll=result[0] if a.logits else result
  seconds+=elapsed;weighted+=float(nll)*k;points.append(dict(targets=offset+k,seconds=seconds,train_nll=float(nll)))
 integrity=collect(m);check_integrity(integrity)
 check=m.evaluation_copy(memory='cold_frozen');heldout=float(check.run(val[:-1],val[1:],mode='eval'));check.close()
 # Bounds describe the actual measured work, including every terminal replay.
 metrics=m.event_metrics();reverse=m.reverse_metrics()
 if m.event['learning']=='local':
  if sum(v['replay_applied'] for v in metrics.values())>a.targets*m.event['replay_budget']:raise AssertionError('Replay budget exceeded')
 row=dict(config=config,effective=m.event,parameters=m.parameter_count,targets=a.targets,warmup=a.warmup,
  seconds=seconds,bytes_per_second=a.targets/seconds,train_nll=weighted/a.targets,heldout_nll=heldout,curve=points,
  event_metrics=metrics,reverse_metrics=reverse,credit_integrity=True,
  generation=generate(m,b'The river ',count=64),fingerprint=m.fingerprint(),
  archive=m.neural_metrics(),composition_bytes=len(m.composition_state()),event_bytes=len(m.event_state()),
  content_bytes=len(m.content_expert_state()),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  train_sha256=hashlib.sha256(raw).hexdigest(),validation_sha256=hashlib.sha256(valraw).hexdigest(),
  primary=integrity['acceptance'],paired_memory=integrity['paired_memory'])
 m.close();a.output.write_text(json.dumps(row,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--corpus',type=Path);p.add_argument('--validation',type=Path);p.add_argument('--warmup',type=int,default=1024)
 p.add_argument('--targets',type=int,default=2048);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
 p.add_argument('--policy');p.add_argument('--worker',action='store_true');p.add_argument('--logits',action='store_true')
 p.add_argument('--suite',choices=['sweep','placement','paired','long'],default='sweep');p.add_argument('--repeats',type=int,default=3)
 a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.worker:return worker(a)
 base=dict(context_heads=0,learning='ancestry');local=dict(context_heads=0,learning='local');combined=dict(context_heads=1,learning='local')
 if a.suite=='sweep':
  configs=[dict(name='stage4',event=base,scale=1),dict(name='context1',event={**base,'context_heads':1},scale=1),
           dict(name='context2',event={**base,'context_heads':2},scale=1)]
  configs += [dict(name=f'{name}_lr{scale:g}',event=event,scale=scale)
              for scale in (.5,1.,2.) for name,event in [('local',local),('combined1',combined),('combined2',{**combined,'context_heads':2})]]
  configs += [dict(name=f'stage4_lr{scale:g}',event=base,scale=scale) for scale in (.5,2.)]
 elif a.suite=='placement':
  configs=[dict(name=name,event=event,scale=1.) for name,event in [('stage4',base),('context1',{**base,'context_heads':1}),('context2',{**base,'context_heads':2}),('local',local),('combined1',combined),('combined2',{**combined,'context_heads':2})]]
 else:
  configs=[]
  for repeat in range(a.repeats):
   rows=[dict(name='stage4',event=base,scale=1,repeat=repeat),dict(name='stage5',event=combined,scale=1,repeat=repeat)]
   configs.extend(rows if repeat%2==0 else reversed(rows))
 result=dict(protocol=__doc__,cpu=a.cpu,experts=int(os.environ.get('SHOEBOX_EXPERTS','12')),logits=a.logits,
             source_sha256=source_hash(ROOT/'runtime'),runs=[])
 import tempfile
 with tempfile.TemporaryDirectory(prefix='shoebox-event-measure-') as tmp:
  for i,config in enumerate(configs):
   output=Path(tmp)/'row.json';cmd=[sys.executable,__file__,'--worker','--output',str(output),'--policy',json.dumps(config),
    '--warmup',str(a.warmup),'--targets',str(a.targets),'--cpu',str(a.cpu)]
   if a.corpus:cmd+=['--corpus',str(a.corpus)]
   if a.validation:cmd+=['--validation',str(a.validation)]
   if a.logits:cmd+=['--logits']
   subprocess.run(cmd,check=True);row=json.loads(output.read_text());result['runs'].append(row)
   a.output.write_text(json.dumps(result,indent=2)+'\n')
   print(i+1,len(configs),config['name'],round(row['bytes_per_second'],1),'B/s',round(row['heldout_nll'],6),'heldout',flush=True)
 if a.suite in ('paired','long'):
  result['medians']={name:statistics.median(r['bytes_per_second'] for r in result['runs'] if r['config']['name']==name) for name in ('stage4','stage5')}
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
