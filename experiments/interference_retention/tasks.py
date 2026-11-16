"""Fresh grouped fixtures. Labels/interpreter are training/diagnostic only.

MQAR separates distractor count from delay (Arora et al., ICLR 2024).
https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib, json, random
from experiments.mad_next.tasks import World, Spec, encode
from experiments.mad.tasks import MAP, KEY, SEP, VALUE, END, QUERY, PAD
SALT=b'SHoEbox-EXP0305-bounded-state-v1\0'
CONDITIONS=['pad0','pad16','pad64','pad256','repeat16','repeat64','repeat256',
            'diverse16','diverse64','diverse256','compete16','overwrite16']

@lru_cache(None)
def excluded(key):
    # Historical binding fixtures used keys 0..47; higher keys cannot be opened.
    if any(k>=48 for k,v in json.loads(key)[1]):return False
    # Exclude WHOLE declared test folds, a strict superset of opened samples.
    from experiments.mad_next.tasks import fold as a
    from experiments.factorized_readout.tasks import fold as b
    from experiments.query_access.tasks import fold as c
    from experiments.constituent_readout.tasks import fold as d2
    from experiments.constituent_readout.tasks_v3 import fold as d3
    from experiments.joint_readout.tasks import fold as e1
    from experiments.joint_switch.tasks import fold as e2
    return any(fn(key)=='test' for fn in (a,b,c,d2,d3,e1,e2))

def fold(key):
    n=int.from_bytes(hashlib.sha256(SALT+key).digest()[:8],'big')%10
    return 'train' if n<6 else 'dev' if n<8 else 'audit'

@lru_cache(None)
def worlds(split,count,pool=64):
    rng=random.Random(20261123);all_worlds=[]
    for a in range(pool):
        for b in range(a+1,pool):
            for v in range(2):
                w=World('binding',((a,v),(b,1-v)))
                if not excluded(w.group_key) and fold(w.group_key)==split:all_worlds.append(w)
    rng.shuffle(all_worlds)
    if len(all_worlds)<count:raise ValueError(f'only {len(all_worlds)} {split} worlds at pool {pool}')
    return tuple(all_worlds[:count])

@dataclass(frozen=True)
class Case:
    ep: object
    condition: str
    query_keys: tuple
    expected: tuple
    distractor_start: int
    distractor_end: int

def make(w,condition='pad0',variant=0,offset=0,spacers=(0,0)):
    order=[0,1] if variant//2==0 else [1,0]
    qs=[k for k,v in w.records]
    if variant%2:qs.reverse()
    ep=encode(w,Spec('binding',vocab=2,key_pool=64),order,qs)
    raw=bytearray(ep.data[:2]);mask=bytearray(ep.mask[:2])
    raw.extend([PAD]*offset);mask.extend(bytes(offset))
    for i,ix in enumerate(order):
        k,v=w.records[ix];raw.extend([MAP,KEY+k,SEP,VALUE+v,END]);mask.extend(bytes(5))
        raw.extend([PAD]*spacers[i]);mask.extend(bytes(spacers[i]))
    start=len(raw);state=dict(w.records)
    length=int(''.join(c for c in condition if c.isdigit()))
    kind=condition.rstrip('0123456789')
    rng=random.Random(int.from_bytes(hashlib.sha256(w.group_key+b'noise').digest()[:8],'big'))
    if kind=='pad':noise=bytes([PAD])*length
    elif kind=='repeat':noise=bytes([200])*length
    elif kind=='diverse':noise=bytes(rng.randrange(160,256) for _ in range(length))
    elif kind=='compete':
        keys=rng.sample([k for k in range(48) if k not in state],2)
        noise=bytes([MAP,KEY+keys[0],SEP,VALUE,END,MAP,KEY+keys[1],SEP,VALUE+1,END])+bytes([PAD])*6
    elif kind=='overwrite':
        k,v=w.records[0];state[k]=1-v
        noise=bytes([MAP,KEY+k,SEP,VALUE+1-v,END])+bytes([PAD])*11
    else:raise ValueError(condition)
    raw.extend(noise);mask.extend(bytes(len(noise)));end=len(raw)
    at=ep.data.index(QUERY,2);tail=bytearray(ep.data[at:]);tailmask=ep.mask[at:]
    for i,b in enumerate(tail):
        if tailmask[i]:
            j=sum(tailmask[:i]);tail[i]=VALUE+state[qs[j]]
    raw.extend(tail);mask.extend(tailmask)
    out=replace(ep,data=bytes(raw),mask=bytes(mask),answer=tuple(state[k] for k in qs))
    case=Case(out,condition,tuple(KEY+k for k in qs),tuple(VALUE+state[k] for k in qs),start,end)
    assert interpret(out)==case.expected
    return case

def interpret(ep):
    """Independent byte interpreter, ignores masked payloads and never reads answer."""
    state={};answers=[];d=ep.data;i=2
    while i<len(d):
        if d[i]==MAP and i+4<len(d) and d[i+2]==SEP and d[i+4]==END:
            state[d[i+1]]=d[i+3];i+=5
        elif d[i]==QUERY and i+2<len(d) and d[i+2]==SEP:
            answers.append(state[d[i+1]]);i+=3
        else:i+=1
    return tuple(answers)

def cases(split,n=64,pool=64,condition='pad0'):
    return [make(w,condition,v,i%64) for i,w in enumerate(worlds(split,n,pool)) for v in range(4)]

def manifest():
    result={}
    for part,n in [('train',512),('dev',128),('audit',256)]:
        ws=worlds(part,n)
        result[part]=dict(worlds=n,sha256=hashlib.sha256(b'\n'.join(w.group_key for w in ws)).hexdigest())
    result['excluded_policy']='entire EXP00/01, B, C, D2, D3, E1 and E2-family test folds; no generic document/task data used'
    return result
