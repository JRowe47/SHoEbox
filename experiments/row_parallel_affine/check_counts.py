"""Test-only integer execution counters; never used by the timing backend.

One separately charged discarded 1,000-byte trajectory per archived pair and
backend. Every probability vector hash and final saved byte hash must reproduce
the already recorded uninstrumented exactness trace. No accuracy is computed.
"""
import argparse,ctypes as C,hashlib,json,subprocess
from pathlib import Path
from . import backend
from .verify_checkpoints import sha
from experiments.selective_record_key_credit.stream import advance

NAMES=('input_bytes','native_targets','scalar_affine_rows','affine_update_rows',
       'row_parallel_calls','row_parallel_rows','hybrid_objectives','full_probabilities','source_vectors')

def counted_library(policy,execution,folder):
    original=backend.scalar.library(policy)
    text=backend.source() if policy=='a9' else backend.scalar.source(policy)
    head=backend.header() if policy=='a9' and execution=='row_parallel' else backend.scalar.header(policy)
    head='#include <cstdint>\nstatic uint64_t a10_test_counts[9]{};\n'+head
    changes={
      'inline float affine(const Affine& p,const Vec& x){':'inline float affine(const Affine& p,const Vec& x){++a10_test_counts[2];',
      'inline void affine_step(Affine& p,const Vec& x,float error,float rate){':'inline void affine_step(Affine& p,const Vec& x,float error,float rate){++a10_test_counts[3];',
      'inline Vec source_vector(const Vec& stable){':'inline Vec source_vector(const Vec& stable){++a10_test_counts[8];',
    }
    if policy in ('a8','a9'):changes['inline A5Objective a5_objective(const LowRank& z,int observed){']='inline A5Objective a5_objective(const LowRank& z,int observed){++a10_test_counts[6];'
    if policy=='a9' and execution=='row_parallel':changes['inline void a10_affine(const Affine* rows,const Vec& x,float* out){']='inline void a10_affine(const Affine* rows,const Vec& x,float* out){++a10_test_counts[4];a10_test_counts[5]+=Rows;'
    for old,new in changes.items():assert head.count(old)==1;head=head.replace(old,new)
    for old,new in {
      'void Model::observe(uint8_t byte){':'void Model::observe(uint8_t byte){++a10_test_counts[0];',
      'float Model::teach(uint8_t target){':'float Model::teach(uint8_t target){++a10_test_counts[1];',
      'void Model::log_probs(float* out)const{':'void Model::log_probs(float* out)const{++a10_test_counts[7];',
    }.items():assert text.count(old)==1;text=text.replace(old,new)
    text+='\nextern "C" void a10_counts(uint64_t* out,int reset){for(int i=0;i<9;++i){out[i]=a10_test_counts[i];if(reset)a10_test_counts[i]=0;}}\n'
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'readout.hpp').write_text(head);(folder/'model.cpp').write_text(text)
    subprocess.run(['g++',*backend.FLAGS,'-I'+str(folder),'-I'+str(backend.ROOT/'multiscale'),str(folder/'model.cpp'),'-o',str(folder/'model.so')],check=True)
    lib=C.CDLL(str(folder/'model.so'))
    for name in dir(original):
        if isinstance(getattr(original,name),C._CFuncPtr):
            source=getattr(original,name);target=getattr(lib,name);target.argtypes=source.argtypes;target.restype=source.restype
    lib.a10_counts.argtypes=[C.POINTER(C.c_uint64),C.c_int];lib.a10_counts.restype=None
    return lib

class Observed:
    def __init__(self,model,trace):self.model=model;self.policy=model.policy;self.trace=trace;self.i=0
    def inventory(self):return self.model.inventory()
    def reset(self):self.model.reset()
    def finish(self):self.model.finish()
    def run(self,x,y,mask=None,labels=None,**kw):
        loss=0.;targets=0
        for i in range(len(x)):
            l,t=self.model.run(x[i:i+1],y[i:i+1],None if mask is None else mask[i:i+1],None if labels is None else labels[i:i+1],**kw)
            loss+=l;targets+=t
            assert sha(self.model.log_probs().tobytes())==self.trace[self.i]['probability_sha256'];self.i+=1
        return loss,targets

def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
    rows=[];original=backend.scalar.library
    for execution in ('scalar','row_parallel'):
        libs={p:counted_library(p,execution,a.work/'counts_build'/execution/p) for p in ('e8','a8','a9')}
        backend.scalar.library=lambda policy='e8':libs[policy]
        for path in sorted((a.prior/'a9').glob('*/*.qr')):
            name=str(path.relative_to(a.prior/'a9'));policy=path.name.split('.')[0]
            trace_path=a.work/'checkpoints'/(name.replace('/','_')+'.trace.json')
            assert trace_path.exists(),'Uninstrumented exactness must finish first'
            trace=json.loads(trace_path.read_text());cursor=json.loads(path.with_suffix('.cursor.json').read_text())
            with backend.scalar.Model.loads(path.read_bytes(),policy) as model:
                counts=(C.c_uint64*9)();libs[policy].a10_counts(counts,1)
                proxy=Observed(model,trace);s,_=advance(proxy,cursor,1000)
                libs[policy].a10_counts(counts,1)
                assert sha(model.dumps())==trace[-1]['saved_sha256']
                row=dict(execution=execution,file=name,counts=dict(zip(NAMES,counts)),exact=True)
                rows.append(row);print(execution,name,'exact counted trajectory',flush=True)
        backend.scalar.library=original
    # Source branch counts must agree after accounting for relocated rows.
    for l,r in zip(rows[:18],rows[18:]):
        assert l['file']==r['file']
        lc=l['counts'];rc=r['counts']
        for name in ('input_bytes','native_targets','affine_update_rows','hybrid_objectives','full_probabilities','source_vectors'):assert lc[name]==rc[name]
        assert lc['scalar_affine_rows']==rc['scalar_affine_rows']+rc['row_parallel_rows']
    result=dict(passed=True,rows=rows,discarded_input_bytes=36000,research_fitting_bytes=0,
        probability_vector_comparisons=36000,final_saved_hash_comparisons=36,
        original_four_path_multiplier_per_execution=2,
        scope='Integer test counters only. One discarded trajectory per pair/execution; exact probabilities/final saved bytes against uninstrumented trace. Multiply each backend by two for original uninterrupted/resumed paths.',
        coefficient_packing_buffer_bytes=0,separate_packing_passes=0,
        new_value_projection_automatic_scratch_bytes=32,
        row_kernel_max_live_simd_source_values_bytes=144,
        packing_cpu='No separate coefficient packing pass or buffer. Register shuffles are included in whole-process and later registered inference CPU; no profiler or microbenchmark.')
    (a.work/'counts.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
