"""Frozen learned-residual intervention, with the complete symbol contract retained."""
from pathlib import Path
import ctypes as ct,json,os,subprocess,sys
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model
path=ROOT/'results/semantic_intervention.so'
subprocess.run(['g++','-O2','-march=native','-std=c++17','-shared','-fPIC',str(ROOT/'tests/semantics_fixture.cpp'),'-o',str(path)],check=True)
lib=ct.CDLL(str(path));lib.semantic_fixture_ablate.argtypes=[ct.c_void_p]
data=np.fromfile(ROOT/'data/heldout.bin',np.uint8)[32768:];rows=[]
for seed in [1,2]:
 original=Model.load(ROOT/f'results/full_semantics_seed{seed}/last.npz')
 learned=original.frozen_copy();disabled=original.frozen_copy();lib.semantic_fixture_ablate(disabled.handle)
 base=learned.run(data[:-1],data[1:]);muted=disabled.run(data[:-1],data[1:])
 expected=json.loads((ROOT/f'results/full_semantics_seed{seed}/training.json').read_text())['test_nll']
 assert abs(base-expected)<1e-12,(base,expected)
 row=dict(seed=seed,baseline_nll=base,zero_semantic_residual_nll=muted,delta_nll=muted-base,parameters_l2=float(np.linalg.norm(original.semantic_parameters())),regional_updates=original.semantic_updates(),regional_parameter_l2=np.linalg.norm(original.semantic_parameters().reshape(16,-1),axis=1).tolist())
 rows.append(row);print(json.dumps(row),flush=True)
 original.close();learned.close();disabled.close()
(ROOT/'results/semantic_contribution.json').write_text(json.dumps(rows,indent=2)+'\n')
