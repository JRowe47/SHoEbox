"""Read-only first-answer probes: no preceding answer is ever observed.

Fit normalizers and identical ridge readouts on train only. Raw/centered/
standardized arms differ only in conditioning. These are representation
diagnostics, not online model accuracy or proof that an undecoded fact is absent.
"""
import argparse
import json
import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from .adapter import Adapter
from .run import write
from experiments.mad_next.tasks import Spec,dataset,parse,KEY,VALUE,QUERY,fingerprint

SLICES={'fast':slice(0,64),'l1':slice(64,128),'l2':slice(128,192),'l3':slice(192,256),
        'retained_reads':slice(256,384),'addressed_requery':slice(384,432),
        'states':slice(0,256),'all':slice(0,432)}


def collect(model,cases):
    x={'query_byte':[],'answer':[]};labels=[]
    with model.clone() as f:
        for ep in cases:
            records,queries=parse(ep)
            stop=ep.mask.index(1);qi=ep.data.index(QUERY,2)+1
            f.reset()
            for t,byte in enumerate(ep.data[:stop]):
                f.observe(byte)
                if t==qi:x['query_byte'].append(f.features())
            x['answer'].append(f.features())
            labels.append([queries[0],records[0][2]-VALUE,records[1][2]-VALUE,ep.answer[0],records[0][1],records[1][1]])
    return {k:np.asarray(v,dtype=np.float64) for k,v in x.items()},np.asarray(labels)


def ridge(x,y,z,classes,condition):
    mean=x.mean(0) if condition!='raw' else np.zeros(x.shape[1])
    scale=np.maximum(x.std(0),1e-6) if condition=='standardized' else np.ones(x.shape[1])
    a=(x-mean)/scale;b=(z-mean)/scale
    # Unpenalized intercept makes raw/centered linear models mathematically
    # equivalent; standardization changes the feature metric, not the solver.
    offset=a.mean(0);target=np.eye(classes)[y];ymean=target.mean(0)
    a=a-offset;b=b-offset
    w=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@(target-ymean))
    return (b@w+ymean).argmax(1),dict(parameters=w.size+classes)


def run(path,out,audit=False):
    spec=Spec('binding',vocab=2)
    tr=dataset(spec,2048,'train');dv=dataset(spec,512,'dev')
    write(Path(out).with_suffix('.plan.json'),dict(checkpoint=str(path),audit=audit,
        primary='L3 standardized, ridge lambda 1, separately decoded keys/values/query; strict unique predicted-key match',
        normalization='train only',sources=list(SLICES),conditions=['raw','centered','standardized']))
    te=dataset(spec,1024,'test') if audit else []
    with Adapter.loads(Path(path).read_bytes()) as m:
        x,y=collect(m,tr);z,v=collect(m,dv+te)
    nd=len(dv)
    rows=[];factorized=[]
    for time in x:
        for name,sl in SLICES.items():
            for condition in ('raw','centered','standardized'):
                predictions={}
                for task,column,classes in [('query_identity',0,48),('first_value',1,2),('second_value',2,2),('queried_value',3,2),('first_key',4,48),('second_key',5,48)]:
                    pred,info=ridge(x[time][:,sl],y[:,column],z[time][:,sl],classes,condition)
                    predictions[task]=pred
                    rows.append(dict(at=time,source=name,condition=condition,label=task,
                                     accuracy=float(np.mean(pred[:nd]==v[:nd,column])),
                                     test_accuracy=float(np.mean(pred[nd:]==v[nd:,column])) if audit else None,**info))
                # Diagnostic factorization: independently decode each key and
                # value and the query from the SAME neural vector. Equality is
                # applied to predicted symbols, not parsed validation inputs.
                # The oracle below separately grants the true record index.
                # Both use intermediate training labels; neither is native LM
                # capability. This follows the address/value separation in
                # Miller et al., https://aclanthology.org/D16-1147/ .
                p=predictions;match0=p['query_identity']==p['first_key'];match1=p['query_identity']==p['second_key']
                inferred=np.where(match0,p['first_value'],p['second_value'])
                oracle=np.where(v[:,0]==v[:,4],p['first_value'],p['second_value'])
                entry=dict(at=time,source=name,condition=condition)
                for prefix,slc in [('',slice(0,nd)),*([('test_',slice(nd,None))] if audit else [])]:
                    entry[prefix+'inferred_match_accuracy']=float(np.mean(inferred[slc]==v[slc,3]))
                    entry[prefix+'unique_match_fraction']=float(np.mean((match0!=match1)[slc]))
                    entry[prefix+'strict_binding_accuracy']=float(np.mean(((match0!=match1)&(inferred==v[:,3]))[slc]))
                    entry[prefix+'oracle_record_index_accuracy']=float(np.mean(oracle[slc]==v[slc,3]))
                factorized.append(entry)
    result=dict(checkpoint_sha256=__import__('hashlib').sha256(Path(path).read_bytes()).hexdigest(),
                train_sha256=fingerprint(tr),dev_sha256=fingerprint(dv),train_examples=len(tr),dev_examples=len(dv),
                test_sha256=fingerprint(te) if audit else None,test_examples=len(te),
                scope='first answer only; train-fit ridge lambda=1, unpenalized intercept; development diagnostic',rows=rows,factorized=factorized)
    write(out,result)
    for row in rows:
        if row['condition']=='standardized' and row['label'] in ('query_identity','queried_value'):
            print(row['at'],row['source'],row['label'],round(row['accuracy'],4))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--audit',action='store_true');a=p.parse_args()
    run(a.checkpoint,a.out,a.audit)
