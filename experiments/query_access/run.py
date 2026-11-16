"""Registered online fits, development selection, then a separate locked audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
from .adapter import Adapter, PARAMETERS, HERE, ROOT, library, prefix_labels
from .tasks import dataset, shifted, balanced_offsets, SALT
from experiments.mad_next.tasks import fingerprint, QUERY, Spec, shortcuts
from experiments.query_readout.run import write

ARMS=['native','l3','split','pool']

def evaluate(model,cases,lesion=None,all_modes=True):
    rows={}
    with model.clone(intervention=lesion) as frozen:
        for mode in ((0,1,2) if all_modes else (2,)):
            total=np.zeros(9)
            for ep in cases:total+=frozen.score(ep,2,mode)
            rows[str(mode)]=dict(token_accuracy=float(total[3]/total[0]),exact=float(total[4]/len(cases)),
                native_token_accuracy=float(total[6]/total[0]),native_exact=float(total[5]/len(cases)),
                first_answer=float(total[7]/len(cases)),second_answer=float(total[8]/len(cases)),
                task_nll=float(total[2]/total[0]),native_nll=float(total[1]/total[0]))
    return dict(cases=len(cases),native_rollout=rows['2'],**({'teacher_forced':rows['0'],'restricted_rollout':rows['1']} if all_modes else {}))

def train(model,cases,dev,budget,interval,order_seed):
    total=targets=cpu=wall=0;order=[];rng=random.Random(order_seed);digest=hashlib.sha256();curve=[]
    next_eval=interval;episode=0;offset_counts=[0]*64
    while total<budget:
        if not order:order=list(range(len(cases)));rng.shuffle(order)
        idx=order.pop();offset=episode%64;episode+=1;ep=shifted(cases[idx],offset);offset_counts[offset]+=1
        c=time.process_time();w=time.perf_counter();used,n,_,_=model.train_episode(ep,budget-total)
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w;total+=used;targets+=n
        digest.update(f'{idx}:{offset}:{used};'.encode())
        if total>=next_eval or total==budget:
            c=time.process_time();w=time.perf_counter();model.finish()
            cpu+=time.process_time()-c;wall+=time.perf_counter()-w
            curve.append(dict(bytes=total,cpu_seconds=cpu,dev=evaluate(model,dev,all_modes=False)))
            while next_eval<=total:next_eval+=interval
    return dict(bytes=total,targets=targets,cpu_seconds=cpu,wall_seconds=wall,bytes_per_cpu_second=total/cpu,
                order_sha256=digest.hexdigest(),offset_counts=offset_counts,curve=curve)

def registration(folder,pilot,workers):
    files=[ROOT/'reports/exp01c/PROTOCOL.md',*sorted(HERE.glob('*.py')),*sorted(HERE.glob('*.hpp')),*sorted(HERE.glob('*.cpp')),
           ROOT/'experiments/factorized_readout/adapter.py',ROOT/'experiments/mad_next/tasks.py',ROOT/'multiscale/native.cpp',*sorted((ROOT/'multiscale').glob('*.hpp'))]
    plan=dict(experiment='EXP01-C/EXP22',registered_utc=datetime.now(timezone.utc).isoformat(),starting_commit='bda7bc8',
        phase='pilot' if pilot else 'formal',core='minimal-v19',seeds=[7] if pilot else [1,7,19],
        arms=ARMS[1:] if pilot else ARMS,parameters=500000,warmup_bytes=250000,
        continuation_bytes=500000 if pilot else 2000000,eval_interval=500000,head_rate=.16,local_weight=1,
        training_cases=2048,development_cases=512,test_cases=1024,data_seed=20261117,fold_salt=SALT.decode(),
        selection='final development mean selects view; all registered arms audited; no test-based changes',
        gate=dict(mean_native_exact=.95,min_seed_native_exact=.90,mean_each_answer=.95,min_seed_each_answer=.90,
            mean_both_orders=.90,min_seed_both_orders=.85,min_offset_mean=.90,mean_each_delay=.90,min_seed_each_delay=.85,
            query_ablation_drop=.20,min_relative_inference_throughput=.80),
        cpus=sorted(os.sched_getaffinity(0))[:workers],
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path=folder/'plan.json'
    if path.exists():raise FileExistsError('refusing to overwrite a preregistered run')
    write(path,plan);return plan

def job(args):
    seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
    stem=Path(folder)/str(seed);stem.mkdir(exist_ok=True)
    tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'))
    row=dict(seed=seed,core='minimal',datasets=dict(train=fingerprint(tr),dev=fingerprint(dv)),arms={})
    with Adapter(parameters=500000-PARAMETERS,seed=seed,kind='native',joint=True) as base:
        row['warmup']=train(base,tr,dv,250000,250000,701)
        for arm in plan['arms']:
            with Adapter(seed=seed,kind=arm,joint=arm=='native',rate=plan['head_rate']) as m:
                warm=None
                if arm=='native':warm=train(m,tr,dv,250000,250000,701)
                else:m.adopt(base)
                result=train(m,tr,dv,plan['continuation_bytes'],plan['eval_interval'],702)
                result['inventory']=m.inventory();result['separate_warmup']=warm
                raw=m.dumps();(stem/f'{arm}.qr').write_bytes(raw)
                result['checkpoint_bytes']=len(raw);result['checkpoint_sha256']=hashlib.sha256(raw).hexdigest()
                with Adapter.loads(raw) as restored:assert evaluate(m,dv[:8])==evaluate(restored,dv[:8])
                result['reload_exact']=True;row['arms'][arm]=result;write(stem/'train.json',row)
                print(seed,arm,result['curve'][-1]['dev']['native_rollout'],flush=True)
    row['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(stem/'train.json',row);return row

def swapped_queries(ep):
    data=bytearray(ep.data);queries=[i for i,b in enumerate(data) if b==QUERY];answers=[i for i,b in enumerate(ep.mask) if b]
    assert len(queries)==len(answers)==2
    data[queries[0]+1],data[queries[1]+1]=data[queries[1]+1],data[queries[0]+1]
    data[answers[0]],data[answers[1]]=data[answers[1]],data[answers[0]]
    return replace(ep,data=bytes(data),answer=tuple(reversed(ep.answer)))

def frozen_query_pairs(model,cases):
    pair=changed=first=0
    with model.clone() as f:
        for ep in cases:
            other=swapped_queries(ep);at=ep.data.index(QUERY,2);assert ep.data[:at]==other.data[:at]
            f.reset()
            for byte in ep.data[:at]:f.observe(byte)
            predictions=[];correct=[]
            for variant in (ep,other):
                with f.clone() as b:
                    answer=[];success=True
                    for t in range(at,len(variant.data)):
                        byte=variant.data[t]
                        if variant.mask[t]:
                            byte=int(b.log_probs().argmax());answer.append(byte);success&=byte==variant.data[t]
                        if t+1<len(variant.data):b.observe(byte)
                    predictions.append(answer);correct.append(success)
            pair+=all(correct);changed+=predictions[0][0]!=predictions[1][0]
            first+=predictions[0][0]==ep.data[ep.mask.index(1)] and predictions[1][0]==other.data[other.mask.index(1)]
    return dict(cases=len(cases),both_query_orders_exact=pair/len(cases),first_answers_both_correct=first/len(cases),first_answer_changed=changed/len(cases))

def diagnostic(model,cases):
    sums=np.zeros((2,6));counts=np.zeros(2)
    with model.clone() as f:
        for ep in cases:
            f.reset();labs=prefix_labels(ep);j=0
            for t,byte in enumerate(ep.data[:-1]):
                f.observe(byte)
                if ep.mask[t+1]:
                    d=f.diagnostic();lab=labs[t];sums[j,0]+=(d[0]>=.5)==(lab[0]==lab[1]);sums[j,1:]+=d[1:]==lab
                    counts[j]+=1;j+=1
    return dict(scope='teacher-fed diagnostic only; labels not inference inputs',columns=['selection','query','key0','key1','value0','value1'],by_answer=(sums/counts[:,None]).tolist())

def audit_job(args):
    seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
    base=dataset(1024,'test');cases=balanced_offsets(base);rows={};stem=Path(folder)/str(seed)
    for arm in plan['arms']:
        with Adapter.loads((stem/f'{arm}.qr').read_bytes()) as m:
            row=dict(test=evaluate(m,cases),counterfactual=frozen_query_pairs(m,cases[:256]))
            if arm!='native':
                names=['no_head','no_query','no_state','uniform','reverse']
                if arm=='pool':names+=['no_completed','no_partial','uniform_pool']
                row['lesions']={name:evaluate(m,cases[:256],name,False)['native_rollout'] for name in names}
                row['lesion_unmodified']=evaluate(m,cases[:256],all_modes=False)['native_rollout']
                row['diagnostic']=diagnostic(m,cases[:256])
            with m.clone() as f:
                row['offsets']=[]
                for offset in range(64):
                    totals=np.zeros(9)
                    for ep in base[:64]:totals+=f.score(shifted(ep,offset),2,2)
                    row['offsets'].append(dict(offset=offset,exact=totals[5]/64,first=totals[7]/64,second=totals[8]/64))
                row['delays']=[]
                for delay in (0,64,128,256):
                    totals=np.zeros(9)
                    for offset in range(64):
                        for ep in base[64:80]:totals+=f.score(shifted(ep,offset,delay),2,2)
                    row['delays'].append(dict(delay=delay,exact=totals[5]/1024,first=totals[7]/1024,second=totals[8]/1024))
            rows[arm]=row;write(stem/'audit_partial.json',rows)
            print('audit',seed,arm,row['test']['native_rollout'],flush=True)
    result=dict(seed=seed,arms=rows,test_sha256=fingerprint(cases),test_cases=len(cases),
        independent_worlds=len({e.split_key for e in cases}),offset_worlds=16,delay_worlds=4,
        shortcuts=shortcuts(cases,Spec('binding',vocab=2)))
    write(stem/'audit.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','train','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
    if a.phase in ('pilot','train'):plan=registration(a.out,a.phase=='pilot',a.workers);fn=job
    else:
        plan=json.loads((a.out/'plan.json').read_text());fn=audit_job
        if plan['phase']!='formal':raise ValueError('pilot audit forbidden')
        if not (a.out/'selection.json').exists():raise ValueError('development selection must be locked first')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(s,plan,a.out) for s in plan['seeds']]))
    write(a.out/('audit.json' if a.phase=='audit' else 'train.json'),dict(plan=plan,rows=rows))
    if a.phase=='train':
        scores={arm:float(np.mean([r['arms'][arm]['curve'][-1]['dev']['native_rollout']['native_exact'] for r in rows])) for arm in ('split','pool')}
        chosen=max(scores,key=scores.get)
        write(a.out/'selection.json',dict(selected=chosen,development_means=scores,reason='highest final mean native development exactness',locked_before_audit=True))

if __name__=='__main__':main()
