import hashlib,json,os
import numpy as np
from model import ROOT,Model,core
from experiment import SIZE,atomic
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))})
rows=[]
for p in sorted((ROOT/'results'/SIZE).glob('seed*/*/training.json')):
    meta=json.loads(p.read_text());cp=p.parent/'last.npz';m=Model.load(cp)
    assert m.stats()['observations']==meta['targets'],p
    assert m.parameter_count==meta['parameters'],p
    assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all() and np.isfinite(m.ymean()).all(),p
    assert np.isfinite(m.means()).all(),p
    rows.append(dict(profile=meta['profile'],seed=meta['seed'],targets=meta['targets'],sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),checkpoint_matches_progress=True,finite=True));m.close()
atomic(ROOT/'results'/f'artifact_checks_{SIZE}.json',rows);print(SIZE,len(rows),'checkpoints verified')
