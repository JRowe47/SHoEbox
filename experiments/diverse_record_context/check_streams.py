"""Pre-fit complete per-arm stream digests and independent byte substitution."""
import argparse, hashlib, json
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
    def run(self,x,y,mask=None,labels=None):
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
        for physical,length in enumerate(lengths):
            at+=5;raw[at:at+length]=reference(w.group_key,episode,physical,length);at+=length
        assert candidate.data==bytes(raw)
        assert candidate.mask==ep.mask and candidate.answer==ep.answer
        assert interpret(candidate)==interpret(ep)
        np.testing.assert_array_equal(prefix_labels(candidate),prefix_labels(ep))
        assert positions==[i for i,(x,y) in enumerate(zip(ep.data,candidate.data)) if x!=y]
        counts['episodes_checked']+=1;counts['spacer_bytes_checked']+=len(positions)
        return candidate,positions
    original_make=stream.make
    def checked_make(*args,**kwargs):
        c=original_make(*args,**kwargs);h=historical_make(*args,**kwargs)
        assert c.ep==h.ep and c.expected==h.expected and c.distractor_start==h.distractor_start and c.distractor_end==h.distractor_end
        return c
    stream.augment=checked;stream.make=checked_make;rows=[]
    try:
        for warmup in (True,False):
            for policy in ('e8','a6','a7'):
                m=DigestCounter(policy);s,_=stream.advance(m,stream.new_cursor(warmup,not warmup,policy),250000 if warmup else 2000000)
                rows.append(dict(arm=policy,phase='warmup' if warmup else 'continuation',cursor=s,counts=m.total,digests={k:v.hexdigest() for k,v in m.digests.items()}))
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
    a.out.write_text(json.dumps(dict(passed=True,rows=rows,independent_checks=counts,research_fit_bytes=0,scope='Data-only exact streams, shared across seeds; model fitting is independently charged for every arm and seed.'),indent=2)+'\n')
    print('All six per-arm phase streams locked; every augmented episode agrees with independent literal rule.',counts)

if __name__=='__main__':main()
