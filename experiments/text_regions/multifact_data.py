"""Two independent byte descriptions with shared and source-private fields.

The '=' field is shared; ':' is private to its source. The runtime sees raw
bytes and explicit source availability. Causal field labels are loss-only.
"""
from dataclasses import dataclass
import hashlib
import itertools
import random
import torch

OWNERS = ('F', 'L', 'G', 'R')
SOURCE = dict(F=0, L=0, G=1, R=1)
FIELD = dict(F=0, L=1, G=0, R=2)
MARKER = dict(F=61, L=58, G=61, R=58)
KINDS = ('recall', 'shared_left', 'shared_right', 'left_private', 'right_private')
CONTROLS = ('intact', 'withheld', 'left_only', 'right_only', 'erase_left', 'erase_right')


def identity(split,index,seed):
    """A keyed 24-bit Feistel permutation: opaque and collision-free per seed."""
    namespaces={'train':0,'validation':1,'test':2,'independent':3}
    if split not in namespaces or not 0<=index<2**22:raise ValueError('Invalid identity namespace/index.')
    ordinal=(namespaces[split]<<22)|index;left,right=ordinal>>12,ordinal&4095
    for round_index in range(4):
        digest=hashlib.sha256(f'{seed}/{round_index}/{right}'.encode()).digest()
        left,right=right,left^(int.from_bytes(digest[:2],'big')&4095)
    return f'{(left<<12)|right:06x}'


def factor_split(values):
    return ('train', 'train', 'validation', 'test')[(values[0]+values[1]+values[2]) % 4]


def factors(split):
    if split not in ('train', 'validation', 'test', 'independent'):
        raise ValueError('Unknown factor split.')
    return tuple(v for v in itertools.product(range(65,69), repeat=3)
                 if split == 'independent' or factor_split(v) == split)


@dataclass(frozen=True)
class MultiFactBatch:
    inputs: torch.Tensor  # [B,T,2], independently rendered source streams
    reset: torch.Tensor
    available: torch.Tensor
    writes: dict[str, torch.Tensor]
    retained: dict[str, torch.Tensor]
    answers: torch.Tensor  # [B,3], shared / left-private / right-private
    old_answers: torch.Tensor
    identities: tuple[str, ...]
    boundary: int


def labels(inputs, reset, available):
    writes, retained = {}, {}
    for owner in OWNERS:
        current = torch.full_like(inputs[:,0,0], -100)
        previous = current.clone(); ww, rr = [], []
        source = SOURCE[owner]
        for t in range(inputs.shape[1]):
            current = current.masked_fill(reset[:,t], -100)
            previous = previous.masked_fill(reset[:,t], -100)
            value = inputs[:,t,source]; observed = available[:,t,source]
            write = observed & previous.eq(MARKER[owner]) & value.ge(65) & value.le(68)
            current = torch.where(write, value, current)
            ww.append(write); rr.append(current)
            previous = torch.where(observed, value, -100)
        writes[owner], retained[owner] = torch.stack(ww,1), torch.stack(rr,1)
    return writes, retained


def make_batch(split, start, count, seed, gap=16, kind='recall', control='intact'):
    choices = factors(split)
    if kind not in KINDS or control not in CONTROLS or min(start,gap)<0 or count<1 or start+count>2**22:
        raise ValueError('Invalid task, control, or dimensions.')
    boundary = 28 + gap//2
    length = boundary + 14 + gap
    rows, old_values, values, ids = [], [], [], []
    # Permuting the full factor cohort makes every complete cohort balanced,
    # including conditional missing-information controls.
    order = list(choices); random.Random(seed+7919).shuffle(order)
    for index in range(start,start+count):
        digest = hashlib.sha256(f'multifact-v1/{seed}/{split}/{index}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        identifier = identity(split,index,seed)
        final = list(order[index % len(order)]); old = final.copy()
        changed = {'shared_left':0,'shared_right':0,'left_private':1,'right_private':2}.get(kind)
        if changed is not None: old[changed] = rng.choice([v for v in range(65,69) if v != final[changed]])
        sources = []
        for source in range(2):
            def noise(n):
                data = bytearray()
                while len(data)<n:
                    token = rng.choice((b'A',b'B',b'C',b'D',b'x',b'?',b'.',b'=x',b':x'))
                    data.extend(token if len(data)+len(token)<=n else b'.')
                return data
            data = noise(rng.randrange(4)); spans = []
            fields = [0,source+1]; rng.shuffle(fields)
            for field in fields:
                begin = len(data)
                data.extend(identifier.encode()+bytes([61 if field==0 else 58,old[field]])+b';')
                spans.append((begin,len(data)))
                data.extend(noise(rng.randrange(4)))
            data.extend(noise(boundary-len(data)))
            data.extend(noise(rng.randrange(4)))
            begin = len(data)
            applies = kind in (('shared_left','left_private') if source==0 else ('shared_right','right_private'))
            if applies:
                data.extend(identifier.encode()+bytes([61 if changed==0 else 58,final[changed]])+b';')
                spans.append((begin,len(data)))
                if control=='withheld': data[begin:] = b'.'*(len(data)-begin)
            data.extend(noise(length-len(data)))
            if control == ('erase_left' if source==0 else 'erase_right'):
                for a,b in spans:data[a:b]=b'.'*(b-a)
            sources.append(list(data))
        rows.append(list(zip(*sources))); ids.append(identifier)
        old_values.append(old); values.append(old if control=='withheld' else final)
    inputs=torch.tensor(rows,dtype=torch.long)
    reset=torch.zeros(inputs.shape[:2],dtype=torch.bool);reset[:,0]=True
    available=torch.ones_like(inputs,dtype=torch.bool)
    if control=='left_only':available[:,:,1]=False
    if control=='right_only':available[:,:,0]=False
    writes,retained=labels(inputs,reset,available)
    return MultiFactBatch(inputs,reset,available,writes,retained,torch.tensor(values),
                          torch.tensor(old_values),tuple(ids),boundary)


def batch_digest(batch):
    digest=hashlib.sha256()
    for name in ('inputs','reset','available','answers','old_answers'):
        digest.update(name.encode());digest.update(getattr(batch,name).numpy().tobytes())
    for collection in ('writes','retained'):
        for owner in OWNERS:
            digest.update((collection+owner).encode());digest.update(getattr(batch,collection)[owner].numpy().tobytes())
    return digest.hexdigest()
