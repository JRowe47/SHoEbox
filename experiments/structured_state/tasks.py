"""Deterministic executable register programs with disjoint structural folds."""
from dataclasses import dataclass
import hashlib
import random

VALUES='abcdefgh'
@dataclass(frozen=True)
class Episode:
    family:str
    steps:int
    records:tuple
    states:tuple
    query:int
    answer:int
    template:str
    renamed:bool=False
    @property
    def text(self):return b''.join(self.records)+VALUES[self.answer].encode()+b'\n'

def record(text):
    raw=text.encode('ascii')
    if len(raw)>64:raise ValueError('record exceeds constituent span')
    return b' '*(64-len(raw))+raw

def execute(state,op):
    """Reference interpreter, independent of model features and labels."""
    out=list(state);kind,a,b=op
    if kind=='set':out[a]=b
    elif kind=='copy':out[a]=state[b]
    elif kind=='swap':out[a],out[b]=state[b],state[a]
    elif kind=='add':out[a]=(state[a]+state[b])%8
    elif kind=='xor':out[a]=state[a]^state[b]
    else:raise ValueError(kind)
    return tuple(out)

def episode(rng,family,steps,fold,renamed=False):
    """Fold is defined by instruction/argument structure, excluding literals.

    Train uses hash buckets 2..9, development 0, audit 1. Initial values and
    query answers are never used to assign the split. Longer tests use 8/12
    operations absent from training (1..4). Renaming is a separate test.
    """
    kinds=('set','copy','swap') if family=='tracking' else ('copy','add','xor')
    while True:
        ops=[]
        for _ in range(steps):
            kind=rng.choice(kinds);a=rng.randrange(4)
            b=rng.randrange(8) if kind=='set' else rng.choice([i for i in range(4) if i!=a])
            ops.append((kind,a,b))
        q=rng.randrange(4)
        template=';'.join(f'{k}:{a}:{"#" if k=="set" else b}' for k,a,b in ops)+f'?{q}'
        bucket=int(hashlib.sha256(template.encode()).hexdigest()[:8],16)%10
        if (fold=='train' and bucket>=2) or (fold=='dev' and bucket==0) or (fold=='audit' and bucket==1):break
    names='EFGH' if renamed else 'ABCD';state=tuple(rng.randrange(8) for _ in range(4));order=list(range(4));rng.shuffle(order)
    records=[record('init '+', '.join(f'{names[r]}={VALUES[state[r]]}' for r in order)+'\n')];states=[state]
    for op in ops:
        k,a,b=op;records.append(record(f'{k} {names[a]} {VALUES[b] if k=="set" else names[b]}\n'))
        state=execute(state,op);states.append(state)
    records.append(record(f'value {names[q]} = '));states.append(state)
    return Episode(family,steps,tuple(records),tuple(states),q,state[q],template,renamed)

def suite(seed,n=256,fold='dev',lengths=(2,4),renamed=False):
    rng=random.Random(seed)
    return [episode(rng,('tracking','composition')[i%2],lengths[(i//2)%len(lengths)],fold,renamed) for i in range(n)]

def verify(ep):
    """Parse emitted bytes afresh; compare every stored target to execution."""
    names='EFGH' if ep.renamed else 'ABCD';initial=ep.records[0].decode().strip()[5:]
    state=[None]*4
    for assignment in initial.split(', '):
        name,value=assignment.split('=');state[names.index(name)]=VALUES.index(value)
    expected=[tuple(state)]
    for raw in ep.records[1:-1]:
        k,a,b=raw.decode().split();i=names.index(a)
        if k=='set':state[i]=VALUES.index(b)
        elif k=='copy':state[i]=state[names.index(b)]
        elif k=='swap':j=names.index(b);state[i],state[j]=state[j],state[i]
        elif k=='add':state[i]=(state[i]+state[names.index(b)])&7
        elif k=='xor':state[i]^=state[names.index(b)]
        else:raise ValueError(k)
        expected.append(tuple(state))
    expected.append(tuple(state))
    q=names.index(ep.records[-1].decode().split()[1])
    assert tuple(expected)==ep.states and q==ep.query and state[q]==ep.answer
    assert all(len(r)==64 for r in ep.records)
    return True
