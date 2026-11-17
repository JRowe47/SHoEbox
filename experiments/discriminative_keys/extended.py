"""Frozen development boundaries, padding, lesions and causal state localization."""
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .backend import Model
from .tasks import worlds,make,cases
from .diagnose import evaluate
from .run import write
from experiments.query_access.run import frozen_query_pairs

CONDITIONS=('pad0','repeat16','diverse16','compete16','overwrite16')

def job(args):
    seed,runs,out,cpu=args;os.sched_setaffinity(0,{cpu});start=time.process_time();row={'seed':seed,'arms':{}}
    for arm in ('e8','a4','a5'):
        policy=arm
        with Model.loads((Path(runs)/str(seed)/(arm+'.qr')).read_bytes(),policy) as m:
            result={};total=pad=0
            result['offsets']={}
            # All registered primary conditions cross the same 64 presentations
            # with every offset. This also checks phase-specific interference.
            for condition in CONDITIONS:
                scores=[]
                for offset in range(64):
                    cs=[make(w,condition,v,offset,updated=i%2) for i,w in enumerate(worlds('dev',16)) for v in range(4)]
                    r=evaluate(m,cs);scores.append(r['exact']);total+=r['inference_bytes'];pad+=r['padding_bytes']
                result['offsets'][condition]=scores
            result['padding']={}
            for delay in (0,64,128,256):
                cs=[make(w,'pad'+str(delay),v,offset) for offset in range(64) for w in worlds('dev',20)[16:20] for v in range(4)]
                r=evaluate(m,cs);result['padding'][str(delay)]=r;total+=r['inference_bytes'];pad+=r['padding_bytes']
            result['both_orders']={}
            for condition in CONDITIONS:
                cs=cases('dev',64,condition=condition)
                result['both_orders'][condition]=frozen_query_pairs(m,[c.ep for c in cs])
                for c in cs:
                    at=c.distractor_end
                    # Counterfactual scorer observes prefix once and each query suffix.
                    total+=2*(len(c.ep.data)-1)-at
                    pad+=c.ep.data[:at].count(9)+2*c.ep.data[at:-1].count(9)
            result['lesions']={}
            for lesion in (None,'no_head','no_query','no_state','uniform','reverse','unpair','gate_off','gate_on'):
                with m.clone(intervention=lesion) as f:
                    # evaluate() would clone again and clear intervention; score directly.
                    cs=cases('dev',64);records=[f.score(c.ep,3,2) for c in cs]
                    result['lesions'][str(lesion)]=float(np.mean([r[5] for r in records]));total+=sum(len(c.ep.data)-1 for c in cs);pad+=sum(c.ep.data[:-1].count(9) for c in cs)
            result['traces']={}
            for condition in ('pad0','diverse16','diverse64','diverse256','compete16','overwrite16'):
                r=evaluate(m,cases('dev',32,condition=condition),trace_state=True);result['traces'][condition]=r
                total+=r['inference_bytes'];pad+=r['padding_bytes']
            result['binary_regression']={}
            binary=result['binary_regression'];binary['offsets']=[]
            for offset in range(64):
                cs=[make(w,'pad0',v,offset) for w in worlds('dev',16,vocab=2) for v in range(4)]
                r=evaluate(m,cs);binary['offsets'].append(r['exact']);total+=r['inference_bytes'];pad+=r['padding_bytes']
            binary['padding']={}
            for delay in (0,64,128,256):
                cs=[make(w,'pad'+str(delay),v,offset) for offset in range(64) for w in worlds('dev',20,vocab=2)[16:20] for v in range(4)]
                r=evaluate(m,cs);binary['padding'][str(delay)]=r;total+=r['inference_bytes'];pad+=r['padding_bytes']
            cs=cases('dev',64,vocab=2);binary['both_orders']=frozen_query_pairs(m,[c.ep for c in cs])
            total+=sum(2*(len(c.ep.data)-1)-c.distractor_end for c in cs)
            pad+=sum(c.ep.data[:c.distractor_end].count(9)+2*c.ep.data[c.distractor_end:-1].count(9) for c in cs)
            binary['lesions']={}
            for lesion in (None,'no_head','no_query','no_state','uniform','reverse','unpair','gate_off','gate_on'):
                with m.clone(intervention=lesion) as f:
                    binary['lesions'][str(lesion)]=float(np.mean([f.score(c.ep,3,2)[5] for c in cs]))
                    total+=sum(len(c.ep.data)-1 for c in cs);pad+=sum(c.ep.data[:-1].count(9) for c in cs)
            result['inference_bytes']=total;result['padding_bytes']=pad
            row['arms'][arm]=result;write(Path(out)/(str(seed)+'.json'),row)
            print('extended',seed,arm,{c:round(v['trace']['present'],4) for c,v in result['traces'].items()},flush=True)
    row['cpu_seconds']=time.process_time()-start;write(Path(out)/(str(seed)+'.json'),row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0))[:3]
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(job,[(s,str(a.runs),str(a.out),c) for s,c in zip((1,7,19),cpus)]))
    write(a.out/'results.json',dict(rows=rows,audit_opened=False))
if __name__=='__main__':main()
