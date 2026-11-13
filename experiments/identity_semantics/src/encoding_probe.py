import ctypes as ct,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model
rows=[]
for kernel in ['binding','semantics']:
 path=ROOT/f'results/probe_{kernel}.so'
 subprocess.run(['g++','-O2','-march=native','-std=c++17','-shared','-fPIC',*(['-DSEMANTIC_PROBE=1'] if kernel=='semantics' else []),str(ROOT/'tests/encoding_probe.cpp'),'-o',str(path)],check=True)
 lib=ct.CDLL(str(path));lib.probe_encode.argtypes=[ct.c_void_p]+[ct.c_int]*5+[ct.c_void_p]*2
 m=fresh_model(kernel=kernel)
 def encode(args):
  ids=np.zeros(2,np.uint64);code=np.zeros(64,np.float32);lib.probe_encode(m.handle,*args,ids.ctypes.data,code.ctypes.data);return ids,code
 base=encode((2,1,1,0,0))
 for name,args in [('relative_start',(2,7,1,0,0)),('duration',(2,1,4,0,0)),('source_role',(2,1,1,1,0)),('receiver_namespace',(16,1,1,0,0)),('absolute_offset',(2,1,1,0,7000))]:
  ids,code=encode(args);rows.append(dict(kernel=kernel,change=name,id_changed=bool(np.any(ids!=base[0])),code_l2=float(np.linalg.norm(code-base[1])),max_abs=float(np.max(np.abs(code-base[1])))))
 m.close()
(ROOT/'results/encoding_probe.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
