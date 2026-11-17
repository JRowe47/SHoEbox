"""Fresh grouped fixtures. Labels/interpreter are training/diagnostic only.

MQAR separates distractor count from delay (Arora et al., ICLR 2024).
https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib, json, random
from experiments.mad_next.tasks import World, Spec, encode
from experiments.mad.tasks import MAP, KEY, SEP, VALUE, END, QUERY, PAD
SALT=b'SHoEbox-EXP0305-selective-record-key-credit-v9\0'
CONDITIONS=['pad0','pad16','pad64','pad256','repeat16','repeat64','repeat256',
            'diverse16','diverse64','diverse256','compete16','overwrite16']

@lru_cache(None)
def reserved_a1():
    from experiments.interference_retention.tasks import worlds as a1_worlds
    return frozenset(w.group_key for w in a1_worlds('audit',256))

@lru_cache(None)
def reserved_a2():
    from experiments.query_first_retention.tasks import worlds as a2_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a2_worlds(part,n,vocab=v))

@lru_cache(None)
def reserved_a3():
    from experiments.chronological_retention.tasks import worlds as a3_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a3_worlds(part,n,vocab=v))

@lru_cache(None)
def reserved_a4():
    from experiments.relative_chronological_retention.tasks import worlds as a4_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a4_worlds(part,n,vocab=v))

@lru_cache(None)
def reserved_a5():
    from experiments.discriminative_keys.tasks import worlds as a5_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a5_worlds(part,n,vocab=v))

@lru_cache(None)
def reserved_a6():
    from experiments.record_presence_cost.tasks import worlds as a6_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a6_worlds(part,n,vocab=v))

@lru_cache(None)
def reserved_a7():
    from experiments.diverse_record_context.tasks import worlds as a7_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a7_worlds(part,n,vocab=v))

@lru_cache(None)
def reserved_a8():
    from experiments.balanced_record_context.tasks import worlds as a8_worlds
    return frozenset(w.group_key for part,n,v in [('audit',256,3),('dev',128,3),('dev',128,2)]
                     for w in a8_worlds(part,n,vocab=v))

@lru_cache(None)
def excluded(key):
    from experiments.interference_retention.tasks import excluded as historical
    return historical(key) or key in reserved_a1() or key in reserved_a2() or key in reserved_a3() or key in reserved_a4() or key in reserved_a5() or key in reserved_a6() or key in reserved_a7() or key in reserved_a8()

def fold(key):
    n=int.from_bytes(hashlib.sha256(SALT+key).digest()[:8],'big')%10
    return 'train' if n<6 else 'dev' if n<8 else 'audit'

@lru_cache(None)
def salted_worlds(split,count,pool=64,vocab=3):
    """Original A9 allocator, retained to reproduce the 96-world failure."""
    rng=random.Random(20261201);all_worlds=[]
    for a in range(pool):
        for b in range(a+1,pool):
            for v in range(vocab):
                for u in range(vocab):
                    if v==u:continue
                    w=World('binding',((a,v),(b,u)))
                    if not excluded(w.group_key) and fold(w.group_key)==split:all_worlds.append(w)
    rng.shuffle(all_worlds)
    if len(all_worlds)<count:raise ValueError(f'only {len(all_worlds)} {split} worlds at pool {pool}')
    return tuple(all_worlds[:count])

@lru_cache(None)
def binary_development():
    """Exact A9-F1 binary-only allocation; never used by the fitting cursor."""
    primary={w.group_key for part,n in [('train',512),('dev',128),('audit',256)]
             for w in salted_worlds(part,n)}
    eligible=[]
    for key0 in range(64):
        for key1 in range(key0+1,64):
            for value0 in range(2):
                for value1 in range(2):
                    if value0==value1:continue
                    w=World('binding',((key0,value0),(key1,value1)))
                    if excluded(w.group_key) or w.group_key in primary:continue
                    eligible.append(w)
    assert len(primary)==896 and len(eligible)==336
    random.Random(20261201).shuffle(eligible)
    return tuple(eligible[:128])

@lru_cache(None)
def worlds(split,count,pool=64,vocab=3):
    if vocab==2:
        if split!='dev' or pool!=64 or not 0<=count<=128:
            raise ValueError('A9-F1 binary worlds are a fixed 128-world development-only sample')
        return binary_development()[:count]
    return salted_worlds(split,count,pool,vocab)

@dataclass(frozen=True)
class Case:
    ep: object
    condition: str
    query_keys: tuple
    expected: tuple
    distractor_start: int
    distractor_end: int
    updated_key: int = -1

def make(w,condition='pad0',variant=0,offset=0,spacers=(0,0),updated=0):
    order=[0,1] if variant//2==0 else [1,0]
    qs=[k for k,v in w.records]
    if variant%2:qs.reverse()
    ep=encode(w,Spec('binding',vocab=3,key_pool=64),order,qs)
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
        k,v=w.records[updated];third=3-sum(v for _,v in w.records);state[k]=third
        noise=bytes([MAP,KEY+k,SEP,VALUE+third,END])+bytes([PAD])*11
    else:raise ValueError(condition)
    raw.extend(noise);mask.extend(bytes(len(noise)));end=len(raw)
    at=ep.data.index(QUERY,2);tail=bytearray(ep.data[at:]);tailmask=ep.mask[at:]
    for i,b in enumerate(tail):
        if tailmask[i]:
            j=sum(tailmask[:i]);tail[i]=VALUE+state[qs[j]]
    raw.extend(tail);mask.extend(tailmask)
    out=replace(ep,data=bytes(raw),mask=bytes(mask),answer=tuple(state[k] for k in qs))
    case=Case(out,condition,tuple(KEY+k for k in qs),tuple(VALUE+state[k] for k in qs),start,end,KEY+w.records[updated][0] if kind=='overwrite' else -1)
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

def cases(split,n=64,pool=64,condition='pad0',vocab=3):
    return [make(w,condition,v,i%64,updated=i%2) for i,w in enumerate(worlds(split,n,pool,vocab)) for v in range(4)]

def manifest():
    result={}
    for part,n in [('train',512),('dev',128),('audit',256)]:
        ws=worlds(part,n)
        result[part]=dict(worlds=n,sha256=hashlib.sha256(b'\n'.join(w.group_key for w in ws)).hexdigest())
    result['binary_dev']=dict(worlds=128,sha256=hashlib.sha256(b'\n'.join(w.group_key for w in worlds('dev',128,vocab=2))).hexdigest())
    result['binary_allocation']='A9-F1: all 336 eligible binary worlds, excluding all 896 primary groups, RNG 20261201, first 128; development only'
    result['reserved_a1_audit_worlds']=len(reserved_a1())
    result['reserved_a2_audit_and_development_worlds']=len(reserved_a2())
    result['reserved_a3_audit_and_development_worlds']=len(reserved_a3())
    result['reserved_a4_audit_and_development_worlds']=len(reserved_a4())
    result['reserved_a5_audit_and_development_worlds']=len(reserved_a5())
    result['reserved_a6_audit_and_development_worlds']=len(reserved_a6())
    result['reserved_a7_audit_and_development_worlds']=len(reserved_a7())
    result['reserved_a8_audit_and_development_worlds']=len(reserved_a8())
    result['excluded_policy']='reserved A1/A2/A3/A4/A5/A6/A7/A8 audits, A2/A3/A4/A5/A6/A7/A8 development and binary regression, entire EXP00/01, B, C, D2, D3, E1 and E2-family test folds; no generic document/task data used'
    return result
