import json,os
import numpy as np
from model import ROOT,Model,core,evaluate
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))});size='1m' if core.E==12 else '5m'
src=ROOT/'sources'/size/'last.npz';data=np.frombuffer((ROOT/'data/validation.bin').read_bytes(),np.uint8)
calib=Model.load(src,reset=True);calib.lib.diagnostic_enable(calib.handle,1)
past=np.frombuffer((ROOT/'data/fresh.bin').read_bytes(),np.uint8)[-65537:];calib.run(past[:-1],past[1:])
import ctypes as ct
a=np.zeros(18+18*4*64+18*4*64*64+18*8,np.float64);calib.lib.diagnostic_export(calib.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
means=a[18:18+18*4*64].reshape(18,4,64)/np.maximum(1,a[:18,None,None]);calib.close()
scores={}
for name in ['baseline','no_A1_decoder','no_Y_decoder','mean_A1_decoder','mean_Y_decoder']:
 m=Model.load(src)
 if name=='mean_A1_decoder':
  m.params['rbias']+=m.params['readout'][:,:64]@means[0,3].astype(np.float32);m.params['readout'][:,:64]=0
 if name=='mean_Y_decoder':
  m.params['rbias']+=m.params['readout'][:,64:]@means[17,3].astype(np.float32);m.params['readout'][:,64:]=0
 if name=='no_A1_decoder':m.params['readout'][:,:64]=0
 if name=='no_Y_decoder':m.params['readout'][:,64:]=0
 scores[name]=evaluate(m,data);m.close()
(ROOT/'results'/f'decoder_lesions_{size}.json').write_text(json.dumps(scores,indent=2))
print(size,{k:round(v['nll']-scores['baseline']['nll'],6) for k,v in scores.items()},flush=True)
