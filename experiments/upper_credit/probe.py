"""Frozen active-expert permutations and level-specific path ablations."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.upper_credit.screen import install
from experiments.plateau_100m.common import score,take_documents
from experiments.scale_100m.train import atomic,file_hash,frozen_score
from shoe_multiscale import Model

def probe(model,dev,docs,access,full=True):
    result=score(model,dev,docs,details=True)
    for row in result['routing']:row.pop('routes',None)
    output={'full':result,'swaps':{},'ablations':{}}
    with model.clone() as retired:
        retired.finish()
        for level in (1,2,3):
            ids=np.flatnonzero(result['routing'][level]['counts']).astype(np.int32)
            shifts=sorted(set([1,len(ids)//3+1,len(ids)//2+1])) if len(ids)>1 else []
            for shift in shifts:
                with retired.inference_copy(cold=True,clear_memory=True) as m:
                    assert access.upper_rotate(m._alive(),level,ids.ctypes.data,len(ids),shift%len(ids))==0
                    total=0;count=0;rows=[]
                    for d in docs:
                        m.reset_context(clear_memory=True);a,b=d['start'],d['stop'];n=b-a-1
                        loss=m.run(dev[a:b-1],dev[a+1:b]);total+=loss*n;count+=n
                        rows.append(dict(row_idx=d['row_idx'],targets=n,nll=loss))
                    output['swaps'][f'L{level}_shift{shift}']=dict(nll=total/count,delta=total/count-result['nll'],documents=rows)
            if full:
                for kind,label in enumerate(('no_modulation','no_recall','no_learned_recall','no_fixed_recall')):
                    with retired.inference_copy(cold=True,clear_memory=True) as m:
                        assert access.upper_ablate(m._alive(),level,kind)==0
                        total=0;count=0
                        for d in docs:
                            m.reset_context(clear_memory=True);a,b=d['start'],d['stop'];n=b-a-1
                            total+=m.run(dev[a:b-1],dev[a+1:b])*n;count+=n
                        output['ablations'][f'L{level}_{label}']=dict(nll=total/count,delta=total/count-result['nll'])
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--cpu',type=int,default=5)
    p.add_argument('--full',action='store_true');p.add_argument('--audit',type=Path);a=p.parse_args()
    os.sched_setaffinity(0,{a.cpu})
    if a.arm in ('cmean','cnorm'):
        from experiments.upper_credit.centered import install as center_install
        _,lib=center_install(a.work/'runtime',a.arm)
    elif a.arm.startswith('s'):
        from experiments.upper_credit.state_expert import install as state_install
        _,lib=state_install(a.work/'runtime',a.arm)
    elif a.arm.startswith('lme'):
        from experiments.upper_credit.local_expert import install as local_install
        _,lib=local_install(a.work/'runtime',a.arm)
    elif a.arm.startswith('lm'):
        from experiments.upper_credit.local_head import install as local_install
        _,lib=local_install(a.work/'runtime',a.arm)
    elif a.arm.startswith('e'):
        from experiments.upper_credit.specialize import install as specialized_install
        _,lib=specialized_install(a.work/'runtime',a.arm)
    else:_,lib=install(a.work/'runtime',a.arm)
    access=C.CDLL(str((a.work/'accessors.so').resolve()))
    access.upper_rotate.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int];access.upper_rotate.restype=C.c_int
    access.upper_ablate.argtypes=[C.c_void_p,C.c_int,C.c_int];access.upper_ablate.restype=C.c_int
    out=a.work/f'{a.arm}_s{a.seed}';meta=json.loads((a.data/'manifest.json').read_text())
    dev=np.fromfile(a.data/'development.bin',np.uint8);docs=meta['files']['development']['documents']
    with Model.load(out/'latest.shoe') as m:
        r=probe(m,dev,docs,access,a.full);r.update(arm=a.arm,seed=a.seed,checkpoint_sha256=file_hash(out/'latest.shoe'))
        if a.audit:
            audit=np.fromfile(a.audit/'train.bin',np.uint8);protocol=json.loads((a.audit/'protocol.json').read_text())
            for field in ('id','text_sha256'):
                assert not {d[field] for d in meta['files']['train']['documents']}&{d[field] for d in protocol['fresh_audit_documents']}
            nll,details=frozen_score(m,audit,protocol['fresh_audit_documents']);r['audit']=dict(nll=nll,documents=details)
        atomic(out/'probe.json',r)
        print(json.dumps({k:v for k,v in r.items() if k in ('arm','seed')}|{'nll':r['full']['nll'],'swaps':{k:v['delta'] for k,v in r['swaps'].items()}}),flush=True)
if __name__=='__main__':main()
