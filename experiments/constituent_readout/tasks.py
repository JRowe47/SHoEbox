"""EXP01-D grouped world split and independent phase/delay interventions."""
from dataclasses import replace
import hashlib
import random
from experiments.mad_next.tasks import Spec, sample_world, variants, solve, PAD, QUERY

SALT=b'SHoEbox-EXP01C-constituent-pairing-v1\0'

def fold(key):
    bucket=int.from_bytes(hashlib.sha256(SALT+key).digest()[:8],'big')%10
    return 'train' if bucket<6 else 'dev' if bucket<8 else 'test'

def dataset(count,split,seed=20261118):
    if count<1 or split not in ('train','dev','test'):raise ValueError('invalid split')
    spec=Spec('binding',vocab=2);rng=random.Random(seed);result=[];seen=set()
    for _ in range(max(10000,count*1000)):
        w=sample_world(spec,rng)
        if w.content_key in seen or fold(w.group_key)!=split:continue
        seen.add(w.content_key);family=variants(w,spec,rng)
        assert all(solve(ep,spec)==ep.answer for ep in family)
        result.extend(family)
        if len(result)>=count:return result
    raise ValueError('insufficient worlds')

def shifted(ep,offset=0,delay=0):
    if not 0<=offset<64 or delay<0 or delay%64:raise ValueError('invalid phase/delay')
    at=ep.data.index(QUERY,2)
    return replace(ep,data=ep.data[:2]+bytes([PAD])*offset+ep.data[2:at]+bytes([PAD])*delay+ep.data[at:],
                   mask=ep.mask[:2]+bytes(offset)+ep.mask[2:at]+bytes(delay)+ep.mask[at:])

def balanced_offsets(cases):
    # A complete four-presentation world family uses the same offset. All 64
    # offsets receive equal weight when len(cases) is a multiple of 256.
    return [shifted(ep,(i//4)%64) for i,ep in enumerate(cases)]
