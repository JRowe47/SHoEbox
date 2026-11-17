"""Discarded per-byte exactness at all 18 frozen A9-F1 actual cursors."""
import argparse,hashlib,json
from pathlib import Path
from .backend import Model as Candidate
from experiments.selective_record_key_credit.backend import Model as Scalar
from experiments.selective_record_key_credit.stream import advance
from experiments.selective_record_key_credit.credit import Counter,augmented

def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,indent=2)+'\n')

class Pair:
    def __init__(self,left,right,policy):
        self.models=[left,right];self.policy=policy;self.trace=[];self.saved_bytes_compared=0
        self.counter=Counter();self.counter.n=left.inventory()['bytes']
    def inventory(self):return self.models[0].inventory()
    def compare(self):
        a,b=self.models
        pa=a.log_probs().tobytes();pb=b.log_probs().tobytes();assert pa==pb
        sa=a.dumps();sb=b.dumps();assert sa==sb
        assert a.inventory()==b.inventory()
        self.saved_bytes_compared+=len(sa)
        record=dict(saved_sha256=sha(sa),probability_sha256=sha(pa),saved_size=len(sa))
        self.trace.append(record)
    def reset(self):
        for m in self.models:m.reset()
        self.counter.reset()
    def finish(self):
        for m in self.models:m.finish()
    def run(self,x,y,mask=None,labels=None,**kw):
        loss=0.;targets=0
        for i in range(len(x)):
            args=(x[i:i+1],y[i:i+1],None if mask is None else mask[i:i+1],None if labels is None else labels[i:i+1])
            ra=self.models[0].run(*args,**kw);rb=self.models[1].run(*args,**kw)
            assert ra==rb
            loss+=ra[0];targets+=ra[1];self.counter.run(*args,**kw);self.compare()
        return loss,targets

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    files=sorted([*a.runs.glob('*/*.qr'),*a.runs.glob('*/*.cursor.json')])
    hashes={str(f.relative_to(a.runs)):sha(f.read_bytes()) for f in files}
    assert len(hashes)==36
    write(a.out/'frozen_before.json',hashes)
    rows=[]
    for path in sorted(a.runs.glob('*/*.qr')):
        raw=path.read_bytes();cursor=json.loads(path.with_suffix('.cursor.json').read_text());policy=path.name.split('.')[0]
        with Scalar.loads(raw,policy) as left,Candidate.loads(raw,policy) as right:
            pair=Pair(left,right,policy);s,_=advance(pair,cursor,1000)
            full=left.dumps();trace=pair.trace;counts=pair.counter.total;compared=pair.saved_bytes_compared
        with Scalar.loads(raw,policy) as left,Candidate.loads(raw,policy) as right:
            pair=Pair(left,right,policy);t,_=advance(pair,cursor,371)
            with Scalar.loads(left.dumps(),policy) as l2,Candidate.loads(right.dumps(),policy) as r2:
                pair.models=[l2,r2];u,_=advance(pair,t,629)
                assert s==u and full==l2.dumps()==r2.dumps()
                assert trace==pair.trace
                assert counts==pair.counter.total
        name=str(path.relative_to(a.runs));write(a.out/(name.replace('/','_')+'.trace.json'),trace)
        row=dict(file=name,policy=policy,exact=True,discarded_input_bytes=4000,
            paired_steps=2000,full_probability_float_comparisons=2000*256,
            uninterrupted_vs_resumed_trace_comparisons=1000,
            saved_byte_comparisons=compared+pair.saved_bytes_compared,
            counts={k:4*v for k,v in augmented(counts,policy).items()},
            final_saved_sha256=sha(full),final_cursor_sha256=sha(json.dumps(s,sort_keys=True).encode()))
        rows.append(row);write(a.out/(name.replace('/','_')+'.receipt.json'),row)
        print(name,'exact',flush=True)
    assert hashes=={str(f.relative_to(a.runs)):sha(f.read_bytes()) for f in files}
    write(a.out/'result.json',dict(passed=True,count=len(rows),rows=rows,
        research_fitting_bytes=0,discarded_correctness_input_bytes=sum(r['discarded_input_bytes'] for r in rows),
        frozen_hashes_unchanged=True,accuracy_scored=False,audit_episodes=0,
        totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}))
if __name__=='__main__':main()
