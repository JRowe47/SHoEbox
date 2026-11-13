"""Balanced four-way relation probe of the actual E4 content representation.

The diagnostic readout is separate from the language decoder. Each source alone
determines one of two bits, hence at most 50% of balanced four-class labels.
"""
import ctypes as ct,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model
libpath=ROOT/'results/joint_fixture.so'
subprocess.run(['g++','-O2','-march=native','-std=c++17','-shared','-fPIC',str(ROOT/'tests/binding_fixture.cpp'),'-o',str(libpath)],check=True)
fx=ct.CDLL(str(libpath));fx.fixture_stage.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int64,ct.c_int64,ct.c_int64,ct.c_float]
fx.fixture_inputs.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p];fx.fixture_driver_code.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
m=fresh_model(seed=1,kernel='binding')
def features(offset,remove_r3=False):
 joint=[];driver=[];labels=[]
 for a in [0,1]:
  for b in [0,1]:
   for src,age,value in [(7,1,12),(7,0,20+31*a),(8,1,71),(8,0,80+31*(0 if remove_r3 else b)),(6,0,169)]:
    fx.fixture_stage(m.handle,src,age,value,2-age,offset+8-2*age,offset+10-2*age,0)
   q=np.zeros(64,np.float32);c=q.copy();d=q.copy();fx.fixture_inputs(m.handle,10,q.ctypes.data,c.ctypes.data);fx.fixture_driver_code(m.handle,10,d.ctypes.data)
   joint.append(q);driver.append(d);labels.append(2*a+b)
 return np.array(joint),np.array(driver),np.array(labels)
def train(x,y):
 w=np.zeros((64,4));bias=np.zeros(4)
 for _ in range(500):
  z=x@w+bias;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);p[np.arange(len(y)),y]-=1;p/=len(y)
  w-=.3*x.T@p;bias-=.3*p.sum(0)
 return w,bias
def assess(x,y,w,b):
 z=x@w+b;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
 return dict(accuracy=float(np.mean(z.argmax(1)==y)),nll=float(-np.log(p[np.arange(len(y)),y]).mean()))
j,d,y=features(0);jt,dt,yt=features(12345);ja,_,_=features(12345,True)
np.testing.assert_array_equal(j,jt);wj,bj=train(j,y);wd,bd=train(d,y)
r=dict(parameters=m.parameter_count,classes=4,region='E4',task='Predict ordered pair (S3 bit, R3 bit); all other roles fixed',labels_leak_into_input=False,source_only_accuracy_ceiling=.5,train_patterns=4,evaluation='Same relations at held-out absolute alignment; diagnostic linear readout, not a language benchmark',bound=assess(jt,yt,wj,bj),first_source_only=assess(dt,yt,wd,bd),bound_with_secondary_source_removed=assess(ja,yt,wj,bj),alignment_code_max_diff=float(np.max(np.abs(j-jt))))
assert r['bound']['accuracy']==1 and r['first_source_only']['accuracy']<=.5 and r['bound_with_secondary_source_removed']['accuracy']<=.5
(ROOT/'results/joint_probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));m.close()
