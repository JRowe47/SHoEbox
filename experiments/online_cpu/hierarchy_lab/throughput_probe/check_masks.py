import ctypes as ct,json,time
from pathlib import Path
import numpy as np
import model
ROOT=Path(__file__).resolve().parent;model.KERNEL='mask_radix'
lib=ct.CDLL(str(model.build()));lib.check_masks.argtypes=[ct.POINTER(ct.c_float),ct.POINTER(ct.c_uint64),ct.c_int,ct.c_int,ct.c_int]
rng=np.random.default_rng(173);values=np.concatenate([rng.standard_normal((10000,64)).astype(np.float32),rng.integers(-2,3,size=(2000,64)).astype(np.float32),np.zeros((1,64),np.float32),np.full((1,64),np.inf,np.float32),np.full((1,64),-np.inf,np.float32),np.ones((1,64),np.float32)])
out=np.zeros(len(values),np.uint64);expected=np.zeros_like(out);records=[]
for k in [0,1,2,8,16,32,63,64]:
 lib.check_masks(values.ctypes.data_as(ct.POINTER(ct.c_float)),expected.ctypes.data_as(ct.POINTER(ct.c_uint64)),len(values),k,0)
 for method in [1,2]:
  t=time.perf_counter();lib.check_masks(values.ctypes.data_as(ct.POINTER(ct.c_float)),out.ctypes.data_as(ct.POINTER(ct.c_uint64)),len(values),k,method)
  records.append(dict(k=k,method=method,seconds=time.perf_counter()-t,exact=bool(np.array_equal(out,expected))))
  assert records[-1]['exact'],(method,k,np.flatnonzero(out!=expected)[:5])
suffix='_avx2' if '-mno-avx512f' in model.FLAGS else ''
(ROOT/'results'/f'mask_checks{suffix}.json').write_text(json.dumps(dict(vectors=len(values),compiler_flags=model.FLAGS,checks=records),indent=2)+'\n');print('Exact masks:',len(values),'vectors x 8 feature counts x 2 implementations')
for method in [0,1,2]:
 times=[]
 for rep in range(5):
  t=time.perf_counter();lib.check_masks(values.ctypes.data_as(ct.POINTER(ct.c_float)),out.ctypes.data_as(ct.POINTER(ct.c_uint64)),len(values),16,method);times.append(time.perf_counter()-t)
 print(method,'median nanoseconds per mask',float(np.median(times))*1e9/len(values))
