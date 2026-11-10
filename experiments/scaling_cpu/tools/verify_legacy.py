"""Compare the generalized kernel against an extracted 175M runtime/checkpoint."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--targets',type=int,default=50_000)
a=p.parse_args()
with np.load(a.checkpoint,allow_pickle=False) as z:os.environ['SHOEBOX_EXPERTS']=str(int(z['experts'][0]))
sys.path.insert(0,str(a.runtime.resolve()))
from runtime import Model as Legacy
from shoebox_runtime import Model
from shoebox_runtime.model import atomic_json

data=np.fromfile(a.data,np.uint8)[:a.targets+1]
if len(data)!=a.targets+1:raise ValueError('Not enough verification bytes')
start=time.perf_counter();old=Legacy.load(a.checkpoint);new=Model.load(a.checkpoint)
np.testing.assert_array_equal(old.flat,new.flat);np.testing.assert_array_equal(old.extra_flat,new.extra_flat)
# Verify numerical predictions before learning and continuation after learning.
oldprobe=old.frozen_copy();newprobe=new.frozen_copy()
ov,ol=oldprobe.run(data[:512],data[1:513],logits=True)
nv,nl=newprobe.run(data[:512],data[1:513],logits=True)
assert ov==nv;np.testing.assert_array_equal(ol,nl)
oldprobe.close();newprobe.close()
rates=dict(mode='periodic',lr=.0004,lp=.0004/3,route=.002)
old_loss=old.run(data[:-1],data[1:],**rates);new_loss=new.run(data[:-1],data[1:],**rates)
np.testing.assert_array_equal(old.flat,new.flat);np.testing.assert_array_equal(old.extra_flat,new.extra_flat)
assert old_loss==new_loss
import ctypes as ct
arrays=new.state_arrays();checks={}
for key,(size,export,_) in __import__('shoebox_runtime.model',fromlist=['STATE_ARRAYS']).STATE_ARRAYS.items():
 arr=np.zeros(getattr(old.lib,size)(),np.uint8);getattr(old.lib,export)(old.handle,arr.ctypes.data)
 np.testing.assert_array_equal(arr,arrays[key]);checks[key]=True
out=dict(checkpoint=str(a.checkpoint),targets=a.targets,parameters=new.parameter_count,
         loss=old_loss,exact_logits=True,exact_parameters=True,exact_extra_parameters=True,
         exact_state=checks,seconds=time.perf_counter()-start)
atomic_json(a.output,out);print(json.dumps(out,indent=2));old.close();new.close()
