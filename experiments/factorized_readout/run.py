"""EXP01-B locked train/audit phases; native autoregressive endpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .adapter import Adapter,PARAMETERS,HERE,ROOT,library,prefix_labels
from .tasks import dataset,Spec,SALT
from experiments.mad_next.tasks import fingerprint,shortcuts,reencode,QUERY,PAD
from experiments.query_readout.run import train,evaluate,write

ARMS=['native','matched','answer_frozen','local_frozen','local_joint','rls_frozen']

def job(args):
    seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
    stem=Path(folder)/str(seed);stem.mkdir(exist_ok=True)
    spec=Spec('binding',vocab=2);tr=dataset(spec,2048,'train');dv=dataset(spec,512,'dev')
    row=dict(seed=seed,core='minimal',datasets=dict(train=fingerprint(tr),dev=fingerprint(dv)),arms={})
    with Adapter(parameters=plan['parameters']-PARAMETERS,seed=seed,kind='native') as base:
        row['warmup']=train(base,tr,dv,plan['warmup_bytes'],plan['warmup_bytes'],601)
        for arm in ARMS:
            if arm=='matched':m=base.clone(frozen=False)
            else:
                kind=arm.split('_')[0];m=Adapter(seed=seed,kind=kind,joint=not arm.endswith('frozen'),rate=plan['head_rate'])
                if kind!='native':m.adopt(base)
            with m:
                warm=None
                if arm=='native':warm=train(m,tr,dv,plan['warmup_bytes'],plan['warmup_bytes'],601)
                result=train(m,tr,dv,plan['continuation_bytes'],plan['eval_interval'],602)
                result['inventory']=m.inventory();result['separate_warmup']=warm
                raw=m.dumps();(stem/f'{arm}.qr').write_bytes(raw)
                with Adapter.loads(raw) as restored:
                    assert evaluate(m,dv[:8])==evaluate(restored,dv[:8])
                result['continuation_reload_exact']=True;row['arms'][arm]=result;write(stem/'train.json',row)
                print(seed,arm,result['curve'][-1]['dev']['native_rollout']['native_exact'],flush=True)
    return row

def frozen_query_pairs(model,cases,spec):
    """Clone the *same* frozen record-prefix state before each changed query."""
    pair=changed=first_correct=0
    with model.clone() as f:
        for ep in cases:
            other=reencode(ep,spec,reverse_queries=True);at=ep.data.index(QUERY,2)
            assert ep.data[:at]==other.data[:at]
            f.reset()
            for byte in ep.data[:at]:f.observe(byte)
            answers=[];success=[]
            for variant in (ep,other):
                with f.clone() as branch:
                    predicted=[];correct=True
                    for t in range(at,len(variant.data)):
                        byte=variant.data[t]
                        if variant.mask[t]:
                            byte=int(branch.log_probs().argmax());predicted.append(byte);correct &= byte==variant.data[t]
                        if t+1<len(variant.data):branch.observe(byte)
                    success.append(correct);answers.append(predicted)
            pair+=all(success);changed+=answers[0][0]!=answers[1][0]
            first_correct+=answers[0][0]==ep.data[ep.mask.index(1)] and answers[1][0]==other.data[other.mask.index(1)]
    return dict(cases=len(cases),both_query_orders_exact=pair/len(cases),first_answers_both_correct=first_correct/len(cases),first_answer_changed=changed/len(cases))

def diagnostic(model,cases):
    totals=np.zeros((2,6));count=np.zeros(2)
    with model.clone() as f:
        for ep in cases:
            f.reset();labels=prefix_labels(ep);j=0
            for t,byte in enumerate(ep.data[:-1]):
                f.observe(byte)
                if ep.mask[t+1]:
                    d=f.diagnostic();lab=labels[t];totals[j,0]+=(d[0]>=.5)==(lab[0]==lab[1]);totals[j,1:]+=d[1:]==lab
                    count[j]+=1;j+=1
    return dict(scope='teacher-forced inspection only; labels do not enter inference',columns=['selection','query_byte','key0_byte','key1_byte','value0_byte','value1_byte'],by_answer=(totals/count[:,None]).tolist())

def audit_job(args):
    seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
    spec=Spec('binding',vocab=2);cases=dataset(spec,1024,'test');rows={};stem=Path(folder)/str(seed)
    for arm in ARMS:
        with Adapter.loads((stem/f'{arm}.qr').read_bytes()) as m:
            row=dict(test=evaluate(m,cases),counterfactual=frozen_query_pairs(m,cases[:256],spec))
            if arm not in ('native','matched'):
                row['lesions']={name:evaluate(m,cases[:256],lesion=name)['native_rollout'] for name in ('no_head','no_query','no_state','uniform','reverse')}
                row['diagnostic']=diagnostic(m,cases[:256])
            rows[arm]=row;print('audit',seed,arm,row['test']['native_rollout']['native_exact'],flush=True)
    result=dict(seed=seed,arms=rows,test_sha256=fingerprint(cases),test_cases=len(cases),independent_worlds=len({e.split_key for e in cases}),shortcuts=shortcuts(cases,spec))
    write(stem/'audit.json',result);return result

def offset_job(args):
    seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
    spec=Spec('binding',vocab=2);cases=dataset(spec,64,'test');rows={}
    for arm in ('native','local_frozen','rls_frozen'):
        with Adapter.loads((Path(folder)/str(seed)/f'{arm}.qr').read_bytes()) as m,m.clone() as f:
            offsets=[];delays=[]
            for offset in range(64):
                # Prefix padding changes the absolute constituent phase while
                # keeping all record-to-query/answer distances exactly fixed.
                variants=[replace(e,data=e.data[:2]+bytes([PAD])*offset+e.data[2:],mask=e.mask[:2]+bytes(offset)+e.mask[2:]) for e in cases]
                offsets.append(sum(f.score(e,2,2)[5] for e in variants)/len(variants))
            for delay in (0,64,128,256):
                # Whole-block insertion increases delay while preserving the
                # first answer's absolute offset modulo 64.
                variants=[]
                for e in cases:
                    at=e.data.index(QUERY,2);variants.append(replace(e,data=e.data[:at]+bytes([PAD])*delay+e.data[at:],mask=e.mask[:at]+bytes(delay)+e.mask[at:]))
                delays.append(dict(extra_delay=delay,exact=sum(f.score(e,2,2)[5] for e in variants)/len(variants)))
            rows[arm]=dict(alignment_only=offsets,delay_only=delays)
    result=dict(seed=seed,cases=len(cases),worlds=len({e.split_key for e in cases}),arms=rows);write(Path(folder)/str(seed)/'offsets.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit','offsets']);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
    if a.phase=='train':
        files=[*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.cpp'),ROOT/'experiments/query_readout/run.py',ROOT/'experiments/mad_next/tasks.py',ROOT/'multiscale/native.cpp',*sorted((ROOT/'multiscale').glob('*.hpp'))]
        plan=dict(experiment='EXP01-B',core='minimal-v19',seeds=[1,7,19],parameters=500000,arms=ARMS,
            warmup_bytes=250000,continuation_bytes=2000000,eval_interval=500000,head_rate=.16,local_weight=1,
            training_cases=2048,development_cases=512,test_cases=1024,key_pool=48,vocab=2,
            fold_salt=SALT.decode(),selection='all fixed-horizon arms audited; no test selection',
            gate=dict(mean_native_exact=.95,min_seed_native_exact=.90),offset_audit_cases=64,counterfactual_cases=256,
            cpus=sorted(os.sched_getaffinity(0))[:a.workers],source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
        write(a.out/'plan.json',plan);fn=job
    else:plan=json.loads((a.out/'plan.json').read_text());fn=audit_job if a.phase=='audit' else offset_job
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(seed,plan,a.out) for seed in plan['seeds']]))
    write(a.out/(a.phase+'.json'),dict(plan=plan,rows=rows))

if __name__=='__main__':main()
