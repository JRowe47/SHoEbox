"""Checkpointed version-12 continuation; no training occurs until invoked."""
import argparse,hashlib,json,os,time
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--experts',type=int,choices=[12,64],required=True)
ap.add_argument('--checkpoint',type=Path,required=True)
ap.add_argument('--data',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--tokens',type=int,default=1000000)
ap.add_argument('--eval-every',type=int,default=100000)
ap.add_argument('--cpu',type=int,default=0)
ap.add_argument('--new-stream',action='store_true')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--lr',type=float,default=.012)
ap.add_argument('--local-lr',type=float,default=.004)
ap.add_argument('--route-lr',type=float,default=.002)
args=ap.parse_args()
if args.tokens<1 or args.eval_every<1:ap.error('Positive tokens and eval-every required')
os.environ['SHOEBOX_EXPERTS']=str(args.experts);os.environ['OPENBLAS_NUM_THREADS']='1'
os.sched_setaffinity(0,{args.cpu})
import numpy as np
from model import Model,ROOT,evaluate

blob=args.data.read_bytes()
if len(blob)<args.tokens+1:raise ValueError(f'Need {args.tokens+1:,} contiguous input bytes')
data=np.frombuffer(blob,np.uint8);digest=hashlib.sha256(blob).hexdigest()
validation=np.frombuffer((ROOT.parent.parent/'data/validation.bin').read_bytes(),np.uint8)
args.out.mkdir(parents=True,exist_ok=True)
manifest=args.out/'run.json';checkpoint=args.out/'last.npz'
rates=dict(lr=args.lr,lp=args.local_lr,route=args.route_lr)
def write_record(r):
    temporary=manifest.with_suffix('.pending');temporary.write_text(json.dumps(r,indent=2));temporary.replace(manifest)
if args.resume:
    r=json.loads(manifest.read_text())
    if (r['data_sha256'],r['target_tokens'],r['experts'],r['rates'])!=(digest,args.tokens,args.experts,rates):raise ValueError('Resume data, budget, size or learning-rate mismatch')
    m=Model.load(checkpoint);position=r['completed_tokens']
    if int(m.stats()['observations'])!=r['origin_observations']+position:raise ValueError('Checkpoint and progress manifest do not match')
else:
    if manifest.exists() or checkpoint.exists():raise FileExistsError('Use a new output directory or --resume')
    m=Model.load(args.checkpoint)
    if args.new_stream:m.reset_context()
    position=0;r=dict(experts=args.experts,parameters=m.parameter_count,source_checkpoint=str(args.checkpoint),source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),data_sha256=digest,target_tokens=args.tokens,completed_tokens=0,origin_observations=int(m.stats()['observations']),new_stream=args.new_stream,rates=rates,state_config=m.state_config,history=[],training_seconds=0.,completed=False)
    r['initial_validation']=evaluate(m,validation);m.save(checkpoint);write_record(r)
while position<args.tokens:
    boundary=min(args.tokens,((position//args.eval_every)+1)*args.eval_every)
    loss_sum=0;count=0
    while position<boundary:
        n=min(10000,boundary-position);tic=time.perf_counter()
        loss=m.run(data[position:position+n],data[position+1:position+n+1],'periodic',**rates)
        r['training_seconds']+=time.perf_counter()-tic
        if not np.isfinite(loss) or not np.isfinite(m.flat).all() or not np.isfinite(m.extra_flat).all():raise FloatingPointError('Nonfinite training result; prior checkpoint remains intact')
        loss_sum+=loss*n;count+=n;position+=n
    before=m.fingerprint();score=evaluate(m,validation,blocks=True);assert m.fingerprint()==before
    row=dict(targets=position,training_nll=loss_sum/count,validation=score,metrics=m.learned_metrics())
    r['history'].append(row);r['completed_tokens']=position;r['completed']=position==args.tokens
    if score['nll']<r.get('best_validation_nll',r['initial_validation']['nll']):
        r['best_validation_nll']=score['nll'];r['best_validation_tokens']=position;m.save(args.out/'best_validation.npz')
    m.save(checkpoint);write_record(r)
    print(json.dumps(dict(targets=position,training_nll=row['training_nll'],validation_nll=score['nll'],bytes_per_second=position/r['training_seconds'])),flush=True)
m.close()
