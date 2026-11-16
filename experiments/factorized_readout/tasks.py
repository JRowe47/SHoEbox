"""New, preregistered world fold for EXP01-B; original EXP00 folds stay intact."""
import hashlib
import random
from experiments.mad_next.tasks import Spec,sample_world,variants,solve

SALT=b'SHoEbox-EXP01B-online-factorized-v1\0'

def fold(key):
    bucket=int.from_bytes(hashlib.sha256(SALT+key).digest()[:8],'big')%10
    return 'train' if bucket<6 else 'dev' if bucket<8 else 'test'

def dataset(spec,count,split,seed=20261116):
    if count<1 or split not in ('train','dev','test'):raise ValueError('invalid split/count')
    rng=random.Random(seed);result=[];seen=set()
    for _ in range(max(10000,count*1000)):
        w=sample_world(spec,rng)
        if w.content_key in seen or fold(w.group_key)!=split:continue
        seen.add(w.content_key);family=variants(w,spec,rng)
        if any(solve(ep,spec)!=ep.answer for ep in family):raise AssertionError('interpreter mismatch')
        result.extend(family)
        if len(result)>=count:return result
    raise ValueError('insufficient worlds')
