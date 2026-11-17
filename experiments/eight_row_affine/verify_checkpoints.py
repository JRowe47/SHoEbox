"""All actual cursors, six parallel trajectories, discarded; no scores."""
import argparse,ctypes as C,hashlib,json
from contextlib import ExitStack
from pathlib import Path
from . import backend
from experiments.row_parallel_affine import backend as a10
from experiments.selective_record_key_credit import backend as scalar
from experiments.selective_record_key_credit.stream import advance
from experiments.selective_record_key_credit.credit import Counter,augmented

def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):assert not p.exists(),p;p.write_text(json.dumps(x,indent=2)+'\n')
def attach(lib,raw,policy):
    assert sha(raw[32:])==raw[:32].hex()
    m=scalar.Model.__new__(scalar.Model);m.policy=policy;m.lib=lib;m.ptr=lib.qr_load(raw[32:],len(raw)-32)
    if not m.ptr:raise RuntimeError(lib.ms_error().decode())
    return m
class Group:
    def __init__(self,models,policy):
        self.models=models;self.policy=policy;self.trace=[];self.saved_byte_comparisons=0
        self.counter=Counter();self.counter.n=models[0].inventory()['bytes']
    def inventory(self):return self.models[0].inventory()
    def reset(self):
        for m in self.models:m.reset()
        self.counter.reset()
    def finish(self):
        for m in self.models:m.finish()
    def compare(self):
        raw=[m.dumps() for m in self.models];probs=[m.log_probs().tobytes() for m in self.models]
        # Both output computation and serialization must preserve all state.
        assert all(x==raw[0] for x in raw[1:])
        assert all(x==probs[0] for x in probs[1:])
        inv=[m.inventory() for m in self.models];assert all(x==inv[0] for x in inv[1:])
        self.saved_byte_comparisons+=5*len(raw[0])
        self.trace.append(dict(saved_sha256=sha(raw[0]),probability_sha256=sha(probs[0]),saved_size=len(raw[0])))
    def run(self,x,y,mask=None,labels=None,**kw):
        loss=0.;targets=0
        for i in range(len(x)):
            args=(x[i:i+1],y[i:i+1],None if mask is None else mask[i:i+1],None if labels is None else labels[i:i+1])
            results=[m.run(*args,**kw) for m in self.models];assert all(x==results[0] for x in results[1:])
            loss+=results[0][0];targets+=results[0][1];self.counter.run(*args,**kw);self.compare()
        return loss,targets
def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('production',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    counted={'e8':scalar.library('e8'),'a8':scalar.library('a8'),'scalar':scalar.library('a9'),'a10':a10.library(),'a11':backend.library()}
    manifest=json.loads((a.production/'result.json').read_text());plain={}
    for name,entry in manifest['libraries'].items():
        path=Path(entry['path']);assert sha(path.read_bytes())==entry['sha256'];lib=C.CDLL(str(path))
        for fname in dir(counted[name]):
            f=getattr(counted[name],fname)
            if isinstance(f,C._CFuncPtr):g=getattr(lib,fname);g.argtypes=f.argtypes;g.restype=f.restype
        plain[name]=lib
    files=sorted([*a.runs.glob('*/*.qr'),*a.runs.glob('*/*.cursor.json')]);hashes={str(f.relative_to(a.runs)):sha(f.read_bytes()) for f in files};assert len(hashes)==36
    write(a.out/'frozen_before.json',hashes);rows=[]
    for path in sorted(a.runs.glob('*/*.qr')):
        raw=path.read_bytes();cursor=json.loads(path.with_suffix('.cursor.json').read_text());policy=path.name.split('.')[0]
        names=['scalar','a10','a11'] if policy=='a9' else [policy]*3
        libs=[pool[name] for pool in (counted,plain) for name in names]
        with ExitStack() as stack:
            group=Group([stack.enter_context(attach(lib,raw,policy)) for lib in libs],policy);s,_=advance(group,cursor,1000)
            full=group.models[0].dumps();trace=group.trace;counts=group.counter.total;compared=group.saved_byte_comparisons
        with ExitStack() as stack:
            group=Group([stack.enter_context(attach(lib,raw,policy)) for lib in libs],policy);t,_=advance(group,cursor,371)
            saved=[m.dumps() for m in group.models]
            group.models=[stack.enter_context(attach(lib,state,policy)) for lib,state in zip(libs,saved)]
            u,_=advance(group,t,629)
            assert s==u and all(full==m.dumps() for m in group.models)
            assert trace==group.trace and counts==group.counter.total
        name=str(path.relative_to(a.runs));write(a.out/(name.replace('/','_')+'.trace.json'),trace)
        row=dict(file=name,policy=policy,exact=True,discarded_input_bytes=12000,counted_input_bytes=6000,uncounted_input_bytes=6000,
            full_probability_float_comparisons=2000*5*256,uninterrupted_vs_resumed_trace_comparisons=1000,
            saved_byte_comparisons=compared+group.saved_byte_comparisons,
            counts={k:12*v for k,v in augmented(counts,policy).items()},
            final_saved_sha256=sha(full),final_cursor_sha256=sha(json.dumps(s,sort_keys=True).encode()))
        rows.append(row);write(a.out/(name.replace('/','_')+'.receipt.json'),row);print(name,'six paths exact',flush=True)
    assert hashes=={str(f.relative_to(a.runs)):sha(f.read_bytes()) for f in files}
    write(a.out/'result.json',dict(passed=True,count=len(rows),rows=rows,research_fitting_bytes=0,
        discarded_correctness_input_bytes=sum(r['discarded_input_bytes'] for r in rows),
        counted_and_uncounted_exact=True,frozen_hashes_unchanged=True,accuracy_scored=False,audit_episodes=0,
        totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}))
if __name__=='__main__':main()
