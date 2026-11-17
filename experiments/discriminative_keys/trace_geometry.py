"""Read-only identity/version geometry; no new candidate policy is evaluated."""
import argparse,time
from pathlib import Path
import numpy as np
from .backend import Model
from .tasks import cases
from .run import write,sha
from experiments.joint_cold_gate.adapter import prefix_labels

def projection(weights,x):
    w=weights.astype(np.float32);v=x.astype(np.float32)*np.float32(8)
    return np.tanh(np.cumsum(w[:,:64]*v,axis=1,dtype=np.float32)[:,-1]+w[:,64])
def distance(x,y):
    d=np.asarray(x,dtype=np.float32)-np.asarray(y,dtype=np.float32)
    return float(np.cumsum(d*d,dtype=np.float32)[-1])
def summary(xs):
    return dict(n=len(xs),mean=float(np.mean(xs)),minimum=float(min(xs)),maximum=float(max(xs)),quantiles=np.quantile(xs,[.1,.5,.9]).tolist()) if xs else {}
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    hashes={str(p.relative_to(a.runs)):sha(p) for p in a.runs.glob('*/a5.qr')}
    write(a.out.with_suffix('.lock.json'),dict(source_sha256=sha(Path(__file__)),checkpoints=hashes,fitting_bytes=0,audit_opened=False))
    total=pad=0;rows=[];start=time.process_time()
    for seed in (1,7,19):
        records=[]
        with Model.loads((a.runs/str(seed)/'a5.qr').read_bytes(),'a5') as m,m.clone() as f:
            weights=f.probe()[325:5525].reshape(5,16,65)
            for condition in ('compete16','overwrite16'):
                for index,c in enumerate(cases('dev',32,condition=condition)):
                    f.reset();labels=prefix_labels(c.ep);known={};answer=0
                    for t,byte in enumerate(c.ep.data[:-1]):
                        f.observe(byte);total+=1;pad+=int(byte==9)
                        for ph,packed in enumerate(labels[t,1:5]):
                            if packed>=0:
                                sig=f.state()[2];known[(int(packed)//256,int(packed)%256,ph,sig.tobytes())]=sig.copy()
                        if not c.ep.mask[t+1]:continue
                        count,_,_,units=f.state();_,decoded=f.decoded();q=f.probe()[9690:9706];candidates=[]
                        for k in range(count):
                            for ph in range(4):
                                if decoded[k,ph,0]<=0:continue
                                owners=[(key,value) for (key,value,phase,_),sig in known.items() if phase==ph and np.array_equal(sig,units[k,1:])]
                                key=projection(weights[ph],units[k,1:]);native_distance=float(decoded[k,ph,4])
                                assert abs(distance(key,q)-native_distance)<2e-5
                                candidates.append(dict(slot=k,phase=ph,born=int(units[k,0]),owners=owners,key=key.tolist(),query_distance=native_distance))
                        nearest=min(candidates,key=lambda x:(x['query_distance'],-x['born'],x['slot'],x['phase']))
                        pairs=[]
                        for i,x in enumerate(candidates):
                            for y in candidates[i+1:]:
                                if not x['owners'] or not y['owners']:continue
                                same=any(k==j for k,v in x['owners'] for j,u in y['owners'])
                                pairs.append(dict(same_key=same,distance=distance(x['key'],y['key'])))
                        records.append(dict(condition=condition,presentation=index,answer=answer,query=c.query_keys[answer],target=c.expected[answer],
                            nearest_identity_correct=any(k==c.query_keys[answer] for k,v in nearest['owners']),
                            nearest_version_correct=(c.query_keys[answer],c.expected[answer]) in nearest['owners'],pairs=pairs,candidates=candidates,query_vector=q.tolist()))
                        answer+=1
        stats={}
        for cond in ('compete16','overwrite16'):
            rs=[r for r in records if r['condition']==cond]
            stats[cond]=dict(nearest_identity_correct=float(np.mean([r['nearest_identity_correct'] for r in rs])),
                nearest_version_correct=float(np.mean([r['nearest_version_correct'] for r in rs])),
                same_key_distance=summary([p['distance'] for r in rs for p in r['pairs'] if p['same_key']]),
                different_key_distance=summary([p['distance'] for r in rs for p in r['pairs'] if not p['same_key']]))
        rows.append(dict(seed=seed,summary=stats,records=records));print(seed,stats,flush=True)
        write(a.out,dict(rows=rows,inference_bytes=total,padding_bytes=pad,cpu_seconds=time.process_time()-start,fitting_bytes=0,audit_opened=False))
    assert hashes=={str(p.relative_to(a.runs)):sha(p) for p in a.runs.glob('*/a5.qr')}
if __name__=='__main__':main()
