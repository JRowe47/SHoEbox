"""Pre-fit complete per-arm stream digests and independent byte substitution."""
import argparse, hashlib, json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
from . import stream
from .tasks import worlds, interpret
from .test_spacers import reference
from experiments.record_presence_cost.tasks import make as historical_make
from experiments.record_presence_cost.account_correctness import Counter
from experiments.joint_cold_gate.adapter import prefix_labels

class DigestCounter(Counter):
    def __init__(self,policy):
        super().__init__();self.policy=policy
        self.digests={k:hashlib.sha256() for k in ('input','target','mask','labels')}
        self.episode=-1;self.coverage=defaultdict(lambda:dict(episodes=0,input_bytes=0,padding_bytes=0,nuisance_bytes=0))
    def reset(self):
        super().reset();self.episode+=1;self.new_episode=True
    def run(self,x,y,mask=None,labels=None):
        regime='diverse' if self.policy=='a8' and labels is not None and (self.episode//64)%2 else 'pad'
        row=self.coverage[(regime,self.condition,self.episode%64)]
        row['episodes']+=int(self.new_episode);self.new_episode=False
        row['input_bytes']+=len(x);row['padding_bytes']+=x.count(9);row['nuisance_bytes']+=sum(b>=160 for b in x)
        self.digests['input'].update(x);self.digests['target'].update(y)
        self.digests['mask'].update(bytes([1])*len(x) if mask is None else bytes(mask))
        if labels is not None:self.digests['labels'].update(np.asarray(labels,dtype='<i4').tobytes())
        return super().run(x,y,mask,labels)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    original=stream.augment;counts={'episodes_checked':0,'spacer_bytes_checked':0}
    def checked(ep,w,episode,offset,lengths):
        candidate,positions=original(ep,w,episode,offset,lengths)
        raw=bytearray(ep.data);at=2+offset
        diverse=(episode % 128)>=64
        for physical,length in enumerate(lengths):
            at+=5
            if diverse:raw[at:at+length]=reference(w.group_key,episode,physical,length)
            at+=length
        assert candidate.data==bytes(raw)
        assert candidate.mask==ep.mask and candidate.answer==ep.answer
        assert interpret(candidate)==interpret(ep)
        np.testing.assert_array_equal(prefix_labels(candidate),prefix_labels(ep))
        assert positions==[i for i,(x,y) in enumerate(zip(ep.data,candidate.data)) if x!=y]
        counts['episodes_checked']+=1;counts['spacer_bytes_checked']+=len(positions)
        transitions[episode]=dict(regime='diverse' if diverse else 'pad',phase=episode%64,spacers=lengths)
        return candidate,positions
    original_make=stream.make
    def checked_make(*args,**kwargs):
        c=original_make(*args,**kwargs);h=historical_make(*args,**kwargs)
        m.condition=c.condition
        assert c.ep==h.ep and c.expected==h.expected and c.distractor_start==h.distractor_start and c.distractor_end==h.distractor_end
        return c
    stream.augment=checked;stream.make=checked_make;rows=[];transitions={}
    try:
        for warmup in (True,False):
            for policy in ('e8','a6','a8'):
                m=DigestCounter(policy);s,_=stream.advance(m,stream.new_cursor(warmup,not warmup,policy),250000 if warmup else 2000000)
                coverage=[dict(regime=k[0],condition=k[1],phase=k[2],**v) for k,v in sorted(m.coverage.items())]
                rows.append(dict(arm=policy,phase='warmup' if warmup else 'continuation',cursor=s,counts=m.total,coverage=coverage,digests={k:v.hexdigest() for k,v in m.digests.items()}))
    finally:
        stream.augment=original;stream.make=original_make
    for phase in ('warmup','continuation'):
        rs=[r for r in rows if r['phase']==phase]
        for key in ('order_rng','augmentation_rng','order','current','episodes','order_sha256','bytes','targets','answer_targets','finish_events'):
            assert all(r['cursor'][key]==rs[0]['cursor'][key] for r in rs)
        assert rs[0]['digests']==rs[1]['digests']
        for key in ('mask','labels'):assert len({r['digests'][key] for r in rs})==1
        if phase=='warmup':assert rs[0]['digests']==rs[2]['digests']
        else:
            assert rs[0]['digests']['input']!=rs[2]['digests']['input']
            assert rs[0]['cursor']['padding_bytes']-rs[2]['cursor']['padding_bytes']==rs[2]['cursor']['replaced_padding_bytes']
    for episode,regime in ((63,'pad'),(64,'diverse'),(127,'diverse'),(128,'pad')):
        assert transitions[episode]['regime']==regime
    for regime in ('pad','diverse'):
        assert {v['phase'] for v in transitions.values() if v['regime']==regime}==set(range(64))
    assert set(transitions)==set(range(rows[-1]['cursor']['episodes']))
    a.out.write_text(json.dumps(dict(passed=True,rows=rows,independent_checks=counts,block_boundaries={k:transitions[k] for k in (63,64,127,128)},terminal_episode=transitions[max(transitions)],research_fit_bytes=0,scope='Data-only exact streams, shared across seeds; model fitting is independently charged for every arm and seed. Coverage counts consumed bytes and encountered episodes, including the incomplete terminal episode; no balancing.'),indent=2)+'\n')
    print('All six per-arm phase streams locked; every augmented episode agrees with independent literal rule.',counts)

if __name__=='__main__':main()
