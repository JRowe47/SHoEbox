"""Read-only existing-state localization, never an inference oracle."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from .backend import Model,ROOT
from .tasks import CONDITIONS,cases,manifest
from experiments.joint_cold_gate.adapter import prefix_labels

def trace(m,c,freeze=False):
    ep=c.ep;m.reset();labels=prefix_labels(ep);known={};rows=[];i=0;inference=0
    while i<len(ep.data)-1:
        if freeze and i==((c.distractor_start+3)//4)*4:
            data=ep.data[i:c.distractor_end];m.freeze_span(data);inference+=len(data);i=c.distractor_end
            continue
        m.observe(ep.data[i]);inference+=1
        lab=labels[i]
        if any(lab[1:5]>=0):
            count,clock,last,units=m.state()
            for phase,packed in enumerate(lab[1:5]):
                if packed>=0:known[int(packed)//256]=(int(packed)%256,last.copy(),phase)
        if ep.mask[i+1]:
            count,clock,last,units=m.state();selected,decoded=m.decoded();diag=m.diagnostic();q=c.query_keys[len(rows)];target=c.expected[len(rows)]
            value,sig,phase=known[q]
            slots=[k for k in range(count) if np.array_equal(units[k,1:],sig)]
            readable=any(decoded[k,p,0]>0 and decoded[k,p,1]==q and decoded[k,p,2]==target for k in range(count) for p in range(4))
            included=any(k in slots and p==phase for k,p in selected)
            native=int(m.log_probs().argmax())
            rows.append(dict(present=bool(slots),decodable=bool(readable),shortlisted=bool(included),query=bool(diag[1]==q),
                selected_value=bool(diag[4+int(diag[0]<.5)]==target),native=bool(native==target),occupancy=count,clock=clock))
        i+=1
    return rows,inference

def evaluate(m,cs,trace_state=False,freeze=False):
    native=[];positions=[];traces=[];n=pad=0
    with m.clone() as f:
        for c in cs:
            if freeze:
                # Autoregressive full alphabet, with only a diagnostic cache freeze.
                f.reset();j=0;correct=[];i=0
                while i<len(c.ep.data)-1:
                    if i==((c.distractor_start+3)//4)*4:
                        data=c.ep.data[i:c.distractor_end];f.freeze_span(data);n+=len(data);pad+=data.count(9);i=c.distractor_end;continue
                    byte=c.ep.data[i]
                    if c.ep.mask[i]:byte=int(f.log_probs().argmax())
                    f.observe(byte);n+=1;pad+=int(c.ep.data[i]==9)
                    if c.ep.mask[i+1]:correct.append(int(f.log_probs().argmax())==c.ep.data[i+1])
                    i+=1
                native.append(all(correct));positions.append(correct)
            else:
                r=f.score(c.ep,3,2);native.append(r[5]);positions.append(r[7:9]);n+=len(c.ep.data)-1;pad+=c.ep.data[:-1].count(9)
            if trace_state:
                rows,used=trace(f,c,freeze);traces.extend(rows);n+=used;pad+=c.ep.data[:-1].count(9)
    result=dict(exact=float(np.mean(native)),positions=np.mean(positions,axis=0).tolist(),presentations=len(cs),worlds=len({c.ep.split_key for c in cs}),
        inference_bytes=n,padding_bytes=pad,autoreg_correct=list(map(bool,native)))
    if traces:result['trace']={k:float(np.mean([r[k] for r in traces])) for k in traces[0]}
    return result
