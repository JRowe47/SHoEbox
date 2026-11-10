import json, os
import numpy as np
from model import ROOT, parent, evaluate
from experiment import make,PROFILES,DATA,SIZE,atomic
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))})
data=np.frombuffer((DATA/'fresh.bin').read_bytes(),np.uint8)
rates=dict(mode='periodic',lr=.03,lp=.01,route=.002)
checks={}
a=make('control');b=parent.Model(config=a.config,params=a.params,fixed=a.fixed,lace=a.lace,state_config=a.state_config,extra=a.extra_flat,probe=a.probe)
x=a.run(data[:4096],data[1:4097],**rates);y=b.run(data[:4096],data[1:4097],**rates)
assert x==y and np.array_equal(a.flat,b.flat) and np.array_equal(a.extra_flat,b.extra_flat)
for method in ['probe_state','lace_state','learned_state']:assert np.array_equal(getattr(a,method)(),getattr(b,method)())
checks['control_exactly_matches_centered_v15']=True;a.close();b.close()
for name in PROFILES:
    a=make(name);b=make(name);a.run(data[:1024],data[1:1025],**rates);b.run(data[:379],data[1:380],**rates)
    path=ROOT/f'check_{SIZE}_{name}.npz';b.save(path);b.close();b=type(a).load(path)
    b.run(data[379:1024],data[380:1025],**rates);assert a.fingerprint()==b.fingerprint(),name
    before=a.fingerprint();f=a.frozen_copy();g=a.frozen_copy()
    _,p=f.run(data[:64],data[1:65],logits=True);_,q=g.run(data[:64],np.zeros(64,np.uint8),logits=True)
    assert np.array_equal(p,q),name;assert np.array_equal(f.publications(),g.publications()),name
    assert np.array_equal(f.ymean(),a.ymean());f.close();g.close()
    evaluate(a,data[:257],mean_y=True);assert before==a.fingerprint(),name
    a.reset();f=a.frozen_copy();_,p=a.run(data[:128],data[1:129],logits=True);_,q=f.run(data[:128],data[1:129],logits=True)
    assert np.array_equal(p,q),name
    a.close();b.close();f.close();path.unlink()
    checks[name]=dict(reload_chunk_exact=True,evaluation_preserves_trainer=True,target_independent=True,reset_matches_frozen_copy=True)

# Verify the residual objective's readout update against an independent equation.
a=make('center_residual');a.set_ymean(np.linspace(-.1,.1,64,dtype=np.float32));f=a.frozen_copy()
before_w=a.params['readout'].copy();before_b=a.params['rbias'].copy()
xx=data[:1];yy=data[1:2];f.run(xx,yy);h=f.publications();features=np.r_[h[0],h[17]-a.ymean()]
expected_w=before_w.astype(np.float64);expected_b=before_b.astype(np.float64);lr=.001;node=0
for d in range(8):
    bit=(int(yy[0])>>(7-d))&1
    base=float(before_b[node])+np.dot(before_w[node,:64].astype(float),features[:64])
    z=base+np.dot(before_w[node,64:].astype(float),features[64:])
    e0=1/(1+np.exp(-base))-bit;e1=1/(1+np.exp(-z))-bit
    expected_b[node]-=lr*.05*e0
    expected_w[node,:64]-=lr/(1+np.dot(features[:64].astype(float),features[:64]))*e0*features[:64]
    expected_w[node,64:]-=lr/(1+np.dot(features[64:].astype(float),features[64:]))*e1*features[64:]
    node=2*node+1+bit
a.run(xx,yy,mode='periodic',lr=lr,lp=0,route=0)
assert np.allclose(a.params['readout'],expected_w,rtol=1e-6,atol=1e-8)
assert np.allclose(a.params['rbias'],expected_b,rtol=1e-6,atol=1e-8)
checks['residual_head_matches_independent_equations']=True;a.close();f.close()
a=make('a1_only');before=a.params['readout'][:,64:].copy();a.run(data[:512],data[1:513],**rates)
assert np.array_equal(before,a.params['readout'][:,64:])
f=a.frozen_copy();g=a.frozen_copy();g.params['readout'][:,64:]=100
_,p=f.run(data[:32],data[1:33],logits=True);_,q=g.run(data[:32],data[1:33],logits=True)
assert np.array_equal(p,q)
checks['a1_only_decoder_ignores_and_does_not_train_y_weights']=True;a.close();f.close();g.close()
atomic(ROOT/'results'/f'checks_{SIZE}.json',checks);print(SIZE,'all checks passed',flush=True)
