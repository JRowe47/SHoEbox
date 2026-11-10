import argparse, fcntl, hashlib, json, os, time
from pathlib import Path
import numpy as np
from model import ROOT,Model,core,evaluate
DATA=ROOT.parent/'scale_probe/data'
SIZE='1m' if core.E==12 else '5m'
PROFILES={
 'control':{},
 'center_y':dict(center_y=1),
 'split_norm':dict(split_norm=1),
 'center_split':dict(center_y=1,split_norm=1),
 'residual':dict(residual=1,split_norm=1),
 'center_residual':dict(center_y=1,residual=1,split_norm=1),
 'no_upper_local':dict(upper_local_gain=0),
 'innovation_local':dict(innovation=1),
 'residual_no_local':dict(center_y=1,residual=1,split_norm=1,upper_local_gain=0),
 'a1_only':dict(center_y=1,split_norm=1,reserved1=1),
}
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def make(name,seed=1):
    with np.load(ROOT.parent/'scale_probe/sources'/SIZE/'last.npz') as a:
        p,_=core.initialize(seed=seed)
        return Model(config=json.loads(str(a['config_json'])),params=p,fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=json.loads(str(a['lace_json'])),state_config=json.loads(str(a['state_config_json'])),probe=dict(route_mode=1,center_rate=.001),utility=PROFILES[name])
def run(name,total,seed=1):
    data=np.frombuffer((DATA/'fresh.bin').read_bytes(),np.uint8);val=np.frombuffer((DATA/'validation.bin').read_bytes(),np.uint8)
    assert total<len(data)
    out=ROOT/'results'/SIZE/f'seed{seed}'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists():
        result=json.loads((out/'training.json').read_text());pos=result['targets'];m=Model.load(out/'last.npz')
    else:
        m=make(name,seed);pos=0
        result=dict(size=SIZE,seed=seed,profile=name,parameters=m.parameter_count,targets=0,seconds=0.,history=[],validation=[],utility=m.utility,training_sha256=hashlib.sha256(data).hexdigest())
    while pos<total:
        n=min(5000,total-pos);lr=float(.008+.036*(1+np.cos(np.pi*min(1,(pos+n/2)/1e6))))
        t=time.perf_counter();loss=m.run(data[pos:pos+n],data[pos+1:pos+n+1],mode='periodic',lr=lr,lp=lr/3,route=.002);result['seconds']+=time.perf_counter()-t;pos+=n
        result['history'].append(dict(targets=pos,nll=loss,lr=lr))
        if pos%100000==0 or pos==total:
            score=evaluate(m,val[:32769] if total<=300000 else val)
            result['validation'].append(dict(training_targets=pos,evaluation_targets=score['targets'],nll=score['nll'],effective_experts=score['effective_experts']))
            print(json.dumps(dict(size=SIZE,seed=seed,profile=name,targets=pos,val=score['nll'])),flush=True)
            # Save each 100K so interrupted sessions restart from a concrete point.
            result['targets']=pos
            assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all()
            m.save(out/'last.npz');atomic(out/'training.json',result)
    m.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',choices=PROFILES);p.add_argument('--targets',type=int,default=300000);p.add_argument('--seed',type=int,default=1);a=p.parse_args()
    os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))})
    lockdir=ROOT/'results'/SIZE/f'seed{a.seed}'/a.profile;lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'.training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(a.profile,a.targets,a.seed)
