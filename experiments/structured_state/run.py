"""Stage A: matched-byte/state-supervision screen on verified programs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from .adapter import Adapter,ARMS
from .tasks import episode,suite,verify

def evaluate(model,cases,mean=None,permutation=None):
    rows=[];started=time.process_time()
    with model.clone(frozen=True) as m:
        active={}
        if permutation is not None:
            for l in (2,3):active[str(l)]=m.permute(l,permutation+l)
        for ep in cases:
            m.reset();text=ep.text;m.run(text[:len(ep.records)*64],text[1:len(ep.records)*64+1]);x,p=m.read()
            if mean is not None:p=m.read(mean)[1]
            native=m.native_probs();q=ep.query;y=ep.answer
            rows.append(dict(family=ep.family,steps=ep.steps,correct=int(p[q].argmax()==y),
                nll=-math.log(max(float(p[q,y]),1e-30)),all_correct=int(np.all(p.argmax(1)==ep.states[-1])),
                native_correct=int(native.argmax()==ord('a')+y),native_restricted=int(native[97:105].argmax()==y)))
    def aggregate(rs):
        return dict(n=len(rs),accuracy=float(np.mean([r['correct'] for r in rs])),
                    nll=float(np.mean([r['nll'] for r in rs])),all_register_accuracy=float(np.mean([r['all_correct'] for r in rs])),
                    native_byte_accuracy=float(np.mean([r['native_correct'] for r in rs])),
                    native_restricted_accuracy=float(np.mean([r['native_restricted'] for r in rs])))
    out=aggregate(rows);out['by_family']={k:aggregate([r for r in rows if r['family']==k]) for k in ('tracking','composition')}
    out['by_steps']={str(k):aggregate([r for r in rows if r['steps']==k]) for k in sorted({r['steps'] for r in rows})}
    out['cpu_seconds']=time.process_time()-started;out['targets']=sum(len(ep.records)*64 for ep in cases);out['active_permuted']=active
    out['correct']=[r['correct'] for r in rows];return out

def language_nll(model,data):
    with model.clone(frozen=True) as m:return m.run(data[:-1],data[1:])

def training_mean(model,cases):
    xs=[]
    with model.clone(frozen=True) as m:
        for ep in cases:
            m.reset();text=ep.text;m.run(text[:len(ep.records)*64],text[1:len(ep.records)*64+1]);xs.append(m.read()[0])
    return np.mean(xs,axis=0)

def train_episode(m,ep,remaining):
    m.reset();text=ep.text;used=0;loss=0
    for j in range(len(ep.records)):
        n=min(64,remaining-used)
        if n<=0:break
        start=j*64;loss+=n*m.run(text[start:start+n],text[start+1:start+n+1]);used+=n
        if n<64:break
        if j<len(ep.records)-1:m.supervise(ep.states[j],True)
        else:
            target=[-1]*4;target[ep.query]=ep.answer;m.supervise(target)
    if used==len(ep.records)*64 and used<remaining:
        loss+=m.run(text[-2:-1],text[-1:]);used+=1
    return used,loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budget',type=int,default=2000000);p.add_argument('--sizes',type=int,nargs='+',default=[1000000,5000000]);p.add_argument('--seeds',type=int,nargs='+',default=[7])
    p.add_argument('--arms',nargs='+',default=list(ARMS));p.add_argument('--eval-every',type=int,default=500000);p.add_argument('--cases',type=int,default=256)
    p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--skip-audit',action='store_true');args=p.parse_args()
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});data=args.data.read_bytes();split=int(len(data)*.9);train=data[:split];valid=data[split:]
    dev=suite(20261115,args.cases,'dev');audit=suite(19810423,512,'audit');long=suite(19810424,512,'audit',lengths=(8,12));rename=suite(19810425,256,'audit',renamed=True)
    for ep in dev+audit+long+rename:verify(ep)
    calib_rng=random.Random(1067);calib=[episode(calib_rng,('tracking','composition')[i%2],2+2*(i//2%2),'train') for i in range(256)]
    config=dict(stage='A',base_commit='600352ce16e89702de82e7b3f61c9b696c34e6f6',budget=args.budget,seeds=args.seeds,sizes=args.sizes,arms=args.arms,cpu=cpu,
        data_sha256=hashlib.sha256(data).hexdigest(),data_split=split,dev_cases=args.cases,
        schedule='linear shared anchor 1 to 0.1 over real byte targets; amplitude 0; fast/query .008, slow .08, native head .16, task head .2',
        auxiliary_weight=.25,added_parameters=4128,auxiliary='mean CE over four registers; state gradient averaged per native retirement block',
        train_lengths=[1,2,3,4],task_fraction_target=.8,task_seed=20261116,dev_seed=20261115,audit_seeds=[19810423,19810424,19810425],
        dataset_hashes={name:hashlib.sha256(b''.join(ep.text for ep in cases)).hexdigest() for name,cases in [('dev',dev),('audit',audit),('long',long),('rename',rename)]})
    result=dict(config=config,runs=[]);args.out.parent.mkdir(parents=True,exist_ok=True);args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    for size in args.sizes:
      for seed in args.seeds:
        for arm in args.arms:
            rng=random.Random(config['task_seed']);m=Adapter(size,seed,arm,args.budget);row=dict(size=size,seed=seed,arm=arm,parameters=int(m.info()[0]),curve=[])
            cpu_seconds=wall_seconds=0.;total=tasks_bytes=language_bytes=episodes=language_offset=0;next_eval=0
            pending_language=0;training_loss=0.;label_hash=hashlib.sha256()
            while total<args.budget:
                if total>=next_eval:
                    point=dict(targets=total,cpu_seconds=cpu_seconds,wall_seconds=wall_seconds,dev=evaluate(m,dev),language_nll=language_nll(m,valid))
                    row['curve'].append(point);print(size,seed,arm,total,point['dev']['accuracy'],point['dev']['nll'],point['language_nll'],flush=True);next_eval+=args.eval_every
                begin=time.process_time();wall=time.perf_counter()
                if pending_language>=1024:
                    m.reset();n=min(pending_language,args.budget-total,len(train)-1-language_offset)
                    if n:
                        chunk=train[language_offset:language_offset+n+1];training_loss+=n*m.run(chunk[:-1],chunk[1:]);total+=n;language_bytes+=n;language_offset+=n;pending_language-=n
                    if language_offset==len(train)-1:language_offset=0
                else:
                    ep=episode(rng,('tracking','composition')[rng.randrange(2)],rng.randrange(1,5),'train')
                    label_hash.update(ep.text);used,loss=train_episode(m,ep,args.budget-total);total+=used;tasks_bytes+=used;pending_language+=used//4;training_loss+=loss;episodes+=1
                cpu_seconds+=time.process_time()-begin;wall_seconds+=time.perf_counter()-wall
            begin=time.process_time();wall=time.perf_counter();m.finish();cpu_seconds+=time.process_time()-begin;wall_seconds+=time.perf_counter()-wall
            row['curve'].append(dict(targets=total,cpu_seconds=cpu_seconds,wall_seconds=wall_seconds,dev=evaluate(m,dev),language_nll=language_nll(m,valid)))
            row.update(cpu_seconds=cpu_seconds,wall_seconds=wall_seconds,cpu_bytes_per_second=total/cpu_seconds,
                task_bytes=tasks_bytes,language_bytes=language_bytes,train_episode_digest=label_hash.hexdigest(),mean_train_nll=training_loss/total,info=m.info())
            checkpoint=args.checkpoint_dir/f'{size}_{seed}_{arm}.task';checkpoint.write_bytes(m.dumps())
            # Evaluation and resumption use the serialized adapter, including live task eligibility.
            with Adapter.loads(checkpoint.read_bytes()) as restored:
                check=evaluate(restored,dev);assert check['correct']==row['curve'][-1]['dev']['correct'] and check['nll']==row['curve'][-1]['dev']['nll']
            if not args.skip_audit:
                mean=training_mean(m,calib);row['audit']=evaluate(m,audit);row['long']=evaluate(m,long);row['rename']=evaluate(m,rename)
                row['mean_replacement']=evaluate(m,audit,mean=mean)
                row['permutations']=[evaluate(m,audit,permutation=s) for s in (23,47,71)]
            result['runs'].append(row);save();print('DONE',size,seed,arm,row['curve'][-1]['dev']['accuracy'],row['cpu_bytes_per_second'],flush=True);m.close()
    save()

if __name__=='__main__':main()
