"""Fixed-recipe EXP01 screen. Explicit train and locked audit phases."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

from .adapter import Adapter, PARAMETERS, HERE, ROOT, library
from experiments.mad_next.tasks import Spec, dataset, fingerprint, shortcuts, reencode

ARMS=('native','native_matched_core','additive_frozen','bilinear_frozen','additive_joint','bilinear_joint')


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def evaluate(model,cases,vocab=2,lesion=None):
    rows={}
    with model.clone(intervention=lesion) as frozen:
        for mode in (0,1,2):
            total=np.zeros(7)
            for ep in cases:total+=frozen.score(ep,vocab,mode)
            rows[str(mode)]={'token_accuracy':float(total[3]/total[0]),'exact':float(total[4]/len(cases)),
                'native_token_accuracy':float(total[6]/total[0]),'native_exact':float(total[5]/len(cases)),
                'task_nll':float(total[2]/total[0]),'native_nll':float(total[1]/total[0])}
    return {'cases':len(cases),'teacher_forced':rows['0'],'restricted_rollout':rows['1'],'native_rollout':rows['2']}


def train(model,cases,dev,budget,interval,order_seed,vocab=2):
    total=targets=cpu=wall=0;order=[];rng=random.Random(order_seed);digest=hashlib.sha256();curve=[]
    next_eval=interval
    while total<budget:
        if not order:order=list(range(len(cases)));rng.shuffle(order)
        idx=order.pop();ep=cases[idx];c=time.process_time();w=time.perf_counter()
        used,n,_,_=model.train_episode(ep,budget-total)
        cpu+=time.process_time()-c;wall+=time.perf_counter()-w;total+=used;targets+=n
        digest.update(f'{idx}:{used};'.encode())
        if total>=next_eval or total==budget:
            # A consistent retirement boundary is part of the fixed recipe.
            c=time.process_time();w=time.perf_counter();model.finish()
            cpu+=time.process_time()-c;wall+=time.perf_counter()-w
            curve.append({'bytes':total,'cpu_seconds':cpu,'dev':evaluate(model,dev,vocab)})
            while next_eval<=total:next_eval+=interval
    return dict(bytes=total,targets=targets,cpu_seconds=cpu,wall_seconds=wall,
                bytes_per_cpu_second=total/cpu,order_sha256=digest.hexdigest(),curve=curve)


def job(core,seed,plan,folder):
    if hasattr(os,'sched_setaffinity'):
        cpus=plan['cpus'];os.sched_setaffinity(0,{cpus[(plan['seeds'].index(seed)*2+plan['cores'].index(core))%len(cpus)]})
    spec=Spec('binding',vocab=plan['vocab']);tr=dataset(spec,2048,'train',plan['data_seed']);dv=dataset(spec,512,'dev',plan['data_seed'])
    row={'core':core,'seed':seed,'datasets':{'train_sha256':fingerprint(tr),'dev_sha256':fingerprint(dv)},'arms':{}}
    stem=Path(folder)/f'{core}_{seed}';stem.mkdir(parents=True,exist_ok=True)
    reduced=Adapter(parameters=plan['parameters']-PARAMETERS,seed=seed,core=core,kind='native')
    try:
        row['warmup']=train(reduced,tr,dv,plan['warmup_bytes'],plan['warmup_bytes'],1067,plan['vocab'])
        (stem/'warmup.qr').write_bytes(reduced.dumps())
        for arm in ARMS:
            kind=arm.split('_')[0];kind=kind if kind in ('additive','bilinear') else 'native'
            if arm=='native_matched_core':m=reduced.clone(frozen=False)
            else:
                m=Adapter(parameters=plan['parameters'],seed=seed,core=core,kind=kind,joint=not arm.endswith('frozen'),rate=plan['head_rate'])
                if kind!='native':m.adopt(reduced)
            with m:
                warmup=None
                if arm=='native':warmup=train(m,tr,dv,plan['warmup_bytes'],plan['warmup_bytes'],1067,plan['vocab'])
                inv=m.inventory()
                result=train(m,tr,dv,plan['continuation_bytes'],plan['eval_interval'],1068,plan['vocab'])
                result['inventory']=inv
                result['warmup_cpu_seconds']=warmup['cpu_seconds'] if warmup else row['warmup']['cpu_seconds']
                result['warmup_shared']=arm!='native'
                raw=m.dumps();(stem/f'{arm}.qr').write_bytes(raw)
                with Adapter.loads(raw) as restored:
                    assert evaluate(restored,dv[:8],plan['vocab'])==evaluate(m,dv[:8],plan['vocab'])
                result['checkpoint_reload_exact']=True
                row['arms'][arm]=result
                write(stem/'train.json',row)
                print(core,seed,arm,round(result['curve'][-1]['dev']['native_rollout']['native_exact'],4),flush=True)
        row['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row['shortcuts_dev']=shortcuts(dv,spec)
        write(stem/'train.json',row);return row
    finally:reduced.close()


def audit(core,seed,plan,folder):
    spec=Spec('binding',vocab=plan['vocab']);cases=dataset(spec,1024,'test',plan['data_seed'])
    stem=Path(folder)/f'{core}_{seed}';rows={}
    for arm in ARMS:
        with Adapter.loads((stem/f'{arm}.qr').read_bytes()) as m:
            rows[arm]={'test':evaluate(m,cases,plan['vocab'])}
            # Counterfactual pair gate: both original and reversed queries must
            # work. Each variant starts from an independently reset observation.
            with m.clone() as f:
                pair=0
                for ep in cases[:256]:
                    pair+=bool(f.score(ep,plan['vocab'],2)[5] and f.score(reencode(ep,spec,reverse_queries=True),plan['vocab'],2)[5])
                rows[arm]['query_pair_exact']=pair/256
            if arm.startswith(('bilinear','additive')):
                rows[arm]['lesions']={k:evaluate(m,cases[:256],plan['vocab'],k) for k in ('no_head','no_query','no_reads')}
            # Single-pass score, including full 256-byte normalization at each
            # requested answer; separate from the three-pass diagnostic above.
            times=[]
            with m.clone() as f:
                for _ in range(3):
                    c=time.process_time()
                    for ep in cases[:128]:f.score(ep,plan['vocab'],2)
                    times.append(time.process_time()-c)
            rows[arm]['single_pass_cpu_seconds_per_episode']=float(np.median(times)/128)
        print('audit',core,seed,arm,rows[arm]['test']['native_rollout']['native_exact'],flush=True)
    result={'core':core,'seed':seed,'arms':rows,'test_sha256':fingerprint(cases),'shortcuts':shortcuts(cases,spec)}
    write(stem/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,7,19]);p.add_argument('--cores',nargs='+',default=['minimal','full'])
    p.add_argument('--warmup',type=int,default=250000);p.add_argument('--budget',type=int,default=500000)
    p.add_argument('--head-rate',type=float,default=.16);p.add_argument('--workers',type=int,default=2);p.add_argument('--vocab',type=int,default=2)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
    if a.phase=='train':
        paths=[*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.cpp'),ROOT/'multiscale/native.cpp',*sorted((ROOT/'multiscale').glob('*.hpp'))]
        plan=dict(experiment='EXP01',version=1,parameters=500000,seeds=a.seeds,cores=a.cores,vocab=a.vocab,
            warmup_bytes=a.warmup,continuation_bytes=a.budget,eval_interval=max(1,a.budget//2),head_rate=a.head_rate,
            data_seed=20261115,arms=list(ARMS),selection='fixed final horizon; all registered arms audited; no test selection',
            cpus=sorted(os.sched_getaffinity(0))[:a.workers],
            source_hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
            scope='diagnostic fixed-recipe screen, not full EXP01 promotion grid')
        write(a.out/'plan.json',plan);fn=job
    else:plan=json.loads((a.out/'plan.json').read_text());fn=audit
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures=[ex.submit(fn,c,s,plan,a.out) for c in plan['cores'] for s in plan['seeds']]
        rows=[f.result() for f in as_completed(futures)]
    write(a.out/(a.phase+'.json'),{'plan':plan,'rows':sorted(rows,key=lambda r:(r['core'],r['seed']))})


if __name__=='__main__':main()
