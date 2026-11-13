"""Mute neural publications while preserving bound content and event timing."""
import ctypes as ct,json,os,subprocess,sys
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model,core
source=ROOT/'runtime/hierarchy_lab/throughput_probe';text=(source/'composition.cpp').read_text();needle='s.h[r][i]=std::tanh(z);';assert text.count(needle)==1
text='#define SHOEBOX_MULTI_BINDING 1\nstatic int selected_region=-1;\n#define run contribution_internal_run\n'+text.replace(needle,needle+'if(r==selected_region)s.h[r][i]=0;')
text+='\n#undef run\nextern "C" double contribution_run(void*p,const unsigned char*x,const unsigned char*y,int n,int region){selected_region=region;float unused;return contribution_internal_run(p,x,y,n,0,0,0,0,0,&unused);}\n'
(ROOT/'results/contribution.cpp').write_text(text)
subprocess.run(['g++','-O3','-march=native','-std=c++17','-I'+str(source),'-shared','-fPIC',str(ROOT/'results/contribution.cpp'),'-o',str(ROOT/'results/contribution.so')],check=True)
lib=ct.CDLL(str(ROOT/'results/contribution.so'));lib.contribution_run.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int];lib.contribution_run.restype=ct.c_double
held=np.fromfile(ROOT/'data/heldout.bin',np.uint8)[32768:];x=held[:-1].copy();y=held[1:].copy();rows=[]
for seed in [1,2]:
 model=Model.load(ROOT/f'results/full_binding_seed{seed}/last.npz');m=model.frozen_copy();baseline=m.run(x,y);m.close()
 m=model.frozen_copy();parity=lib.contribution_run(m.handle,x.ctypes.data,y.ctypes.data,len(x),-1);m.close();assert parity==baseline,(parity,baseline)
 for r in [6,10,12,14]:
  m=model.frozen_copy();nll=lib.contribution_run(m.handle,x.ctypes.data,y.ctypes.data,len(x),r);m.close();row=dict(seed=seed,region=core.NAMES[r],baseline=baseline,intervened_nll=nll,delta=nll-baseline,targets=len(x));rows.append(row);print(row,flush=True)
 model.close();(ROOT/'results/neural_contribution.json').write_text(json.dumps(rows,indent=2)+'\n')
