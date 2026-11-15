"""Offline frozen routing counterfactuals and independent document scores."""
import argparse
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.routing_5m.run import install
from experiments.plateau_100m.common import Model,score,atomic
from experiments.scale_100m.train import frozen_score

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--audit-data',type=Path,required=True);p.add_argument('--audit-protocol',type=Path,required=True)
p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--cpu',type=int,default=8)
a=p.parse_args();os.sched_setaffinity(0,{a.cpu});install(a.work/'runtime',a.arm)
out=a.work/f'{a.arm}_s{a.seed}';r=json.loads((out/'results.json').read_text())
lock=(out/'probe.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'probe.json').exists():
    cached=json.loads((out/'probe.json').read_text())
    if cached.get('checkpoint_sha256')==r['checkpoint_sha256'] and cached['targets']==r['targets']:
        print(json.dumps(dict(cached=True,arm=a.arm,seed=a.seed)),flush=True)
        raise SystemExit(0)
meta=json.loads((a.data/'manifest.json').read_text());dev=np.fromfile(a.data/'development.bin',np.uint8)
docs=meta['files']['development']['documents'];audit=np.fromfile(a.audit_data,np.uint8)
audit_protocol=json.loads(a.audit_protocol.read_text())
audit_docs=audit_protocol['fresh_audit_documents']
for key in ('id','text_sha256'):
    assert not {d[key] for d in meta['files']['train']['documents']}&{d[key] for d in audit_docs}
control=C.CDLL(str(a.work/'accessors.so'));control.scale_bias.argtypes=[C.c_void_p,C.c_int,C.c_float];control.scale_bias.restype=C.c_int
started=time.perf_counter()
with Model.load(out/'latest.shoe') as model:
    frozen={}
    for intervention in ['full','no_bias','swap0','swap1','swap2','swap3','mask1','mask2','mask3']:
        s=score(model,dev,docs,intervention,details=intervention=='full' and a.arm!='A')
        if 'routing' in s:
            for row in s['routing']:row.pop('routes',None)
        frozen[intervention]=s
    for level in range(4):
        with model.clone() as retired:
            retired.finish()
            with retired.inference_copy(cold=True,clear_memory=True) as m:
                control.scale_bias(m._alive(),level,0.)
                total=0;count=0
                for d in docs:
                    m.reset_context(clear_memory=True);x,y=d['start'],d['stop'];n=y-x-1
                    total+=m.run(dev[x:y-1],dev[x+1:y])*n;count+=n
                frozen[f'no_bias_{level}']=dict(nll=total/count)
    audit_loss,perdoc=frozen_score(model,audit,audit_docs)
    assert abs(frozen['full']['nll']-r['curve'][-1]['nll'])<1e-12
    result=dict(arm=a.arm,seed=a.seed,targets=r['targets'],frozen=frozen,
        audit=dict(nll=audit_loss,documents=perdoc),seconds=time.perf_counter()-started,
        checkpoint_verified=True,checkpoint_sha256=r['checkpoint_sha256'])
    if a.seed==7:
        lib=model._lib
        lib.diag_gradient.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p]
        lib.diag_gradient.restype=C.c_int
        aggregate=np.zeros((4,20));total=0
        with model.clone() as gradient_model:
            for d in audit_protocol['train_documents'][:24]:
                gradient_model.reset_context(clear_memory=True);x=d['start'];n=min(4096,d['targets'])
                values=np.zeros((4,20))
                gradient_model._check(lib.diag_gradient(gradient_model._alive(),audit[x:x+n].ctypes.data,
                    audit[x+1:x+n+1].ctypes.data,n,values.ctypes.data))
                aggregate[:,:12]+=values[:,:12]
                aggregate[:,12:14]=np.maximum(aggregate[:,12:14],values[:,12:14]);total+=n
        assert np.all(aggregate[:,12]<2e-6)
        result['gradient']=dict(targets=total,channels=['language','contrastive','SDR_prediction','reconstruction'],
            columns=['blocks','language_energy','contrastive_energy','SDR_energy','reconstruction_energy',
                'language_dot_contrastive','language_dot_SDR','language_dot_reconstruction','contrastive_dot_SDR',
                'contrastive_dot_reconstruction','SDR_dot_reconstruction','combined_energy','maximum_linearity_error','maximum_gradient'],
            values=aggregate[:,:14].tolist())
atomic(out/'probe.json',result)
print(json.dumps(dict(arm=a.arm,seed=a.seed,targets=r['targets'],audit=audit_loss,
    bias_delta=frozen['no_bias']['nll']-frozen['full']['nll'])),flush=True)
