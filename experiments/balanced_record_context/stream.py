"""Exact byte cursor: schedule events, partial examples and RNGs are checkpointed."""
import copy,hashlib,random
from .tasks import worlds,make
from .spacers import augment
from experiments.joint_cold_gate.adapter import prefix_labels
from experiments.joint_cold_gate.stream import tuples
TRAIN_CONDITIONS=('pad0','repeat16','diverse16','compete16','overwrite16')

def new_cursor(warmup=False,local=True,policy="a8"):
    if policy not in ("e8","a6","a8"):raise ValueError(policy)
    seed=10701 if warmup else 10702;interval=250000 if warmup else 500000
    return dict(policy=policy,curriculum='balanced_record_context_v8' if policy=='a8' and not warmup else 'pad_spacers',nuisance_bytes=0,replaced_padding_bytes=0,schema='exp0305-a8-cursor-1',bytes=0,targets=0,padding_bytes=0,answer_targets=0,episodes=0,
        epoch=-1,order=[],current=None,model_bytes=0,order_rng=random.Random(seed).getstate(),augmentation_rng=random.Random(seed+1000).getstate(),
        warmup=warmup,local=local,order_sha256='0'*64,finish_interval=interval,next_finish=interval,finish_events=0)

def advance(m,cursor,n):
    if type(n) is not int or n<0 or cursor['schema']!='exp0305-a8-cursor-1':raise ValueError('invalid cursor/budget')
    if getattr(m,'policy',cursor['policy'])!=cursor['policy']:raise ValueError('model/cursor policy mismatch')
    if m.inventory()['bytes']!=cursor['model_bytes']:raise ValueError('model/cursor mismatch')
    s=copy.deepcopy(cursor);ws=worlds('train',512);conditions=('pad0',) if s['warmup'] else TRAIN_CONDITIONS
    rng=random.Random();rng.setstate(tuples(s['order_rng']));aug=random.Random();aug.setstate(tuples(s['augmentation_rng']));remaining=n;loss=0
    while remaining:
        if s['current'] is None:
            if not s['order']:
                s['order']=list(range(len(ws)*4*len(conditions)));rng.shuffle(s['order']);s['epoch']+=1
            ix=s['order'].pop();wi=ix//(4*len(conditions));variant=(ix//len(conditions))%4;condition=conditions[ix%len(conditions)]
            spacers=[aug.randrange(8),aug.randrange(8)];length=int(''.join(c for c in condition if c.isdigit()))
            offset=(s['episodes']%64-sum(spacers)-length)%64;s['episodes']+=1
            s['current']=dict(index=wi,variant=variant,condition=condition,spacers=spacers,offset=offset,updated=(wi+s['epoch'])%2,used=0)
            s['order_sha256']=hashlib.sha256(bytes.fromhex(s['order_sha256'])+repr(s['current']).encode()).hexdigest();m.reset()
        cur=s['current'];ep=make(ws[cur['index']],cur['condition'],cur['variant'],cur['offset'],cur['spacers'],cur['updated']).ep
        spacer_positions=[]
        if s['curriculum']=='balanced_record_context_v8':
            ep,spacer_positions=augment(ep,ws[cur['index']],s['episodes']-1,cur['offset'],cur['spacers'])
        start=cur['used'];stop=min(len(ep.data)-1,start+remaining,start+s['next_finish']-s['bytes'])
        mask=ep.mask[start+1:stop+1] if s['warmup'] else None
        labels=prefix_labels(ep)[start:stop] if s['local'] else None
        l,t=m.run(ep.data[start:stop],ep.data[start+1:stop+1],mask,labels);used=stop-start
        s['nuisance_bytes']+=sum(160<=b<=255 for b in ep.data[start:stop]);s['replaced_padding_bytes']+=sum(start<=i<stop for i in spacer_positions)
        loss+=l;remaining-=used;s['bytes']+=used;s['targets']+=t;s['padding_bytes']+=ep.data[start:stop].count(9);s['answer_targets']+=sum(ep.mask[start+1:stop+1]);s['model_bytes']=stop;cur['used']=stop
        if stop==len(ep.data)-1:s['current']=None
        if s['bytes']==s['next_finish']:
            m.finish();s['finish_events']+=1;s['next_finish']+=s['finish_interval']
    s['order_rng']=rng.getstate();s['augmentation_rng']=aug.getstate()
    return s,loss
