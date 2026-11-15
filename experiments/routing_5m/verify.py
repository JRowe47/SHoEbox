"""Neutral-runtime parity, policy restoration, data and continuation checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.routing_5m.run import install,configure,ARMS
from experiments.plateau_100m.common import Model,runtime,atomic
from experiments.scale_100m.train import file_hash

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
meta=json.loads((a.data/'manifest.json').read_text())
for entry in meta['files'].values():
    assert file_hash(a.data/entry['file'])==entry['sha256']
    assert sum(d['stop']-d['start']-1 for d in entry['documents'])==entry['targets']
train,dev=[meta['files'][k]['documents'] for k in ('train','development')]
for field in ('id','text_sha256'):
    assert not {d[field] for d in train}&{d[field] for d in dev}
x=np.fromfile(a.data/'train.bin',np.uint8)[:10001]
base=runtime._library();expected={}
for arm in 'ABCD':
    with Model(parameters=5_000_000,seed=7,routing_arm=arm,semantic_laces=False,router='scan') as m:
        configure(m,2_000_000,arm);loss=m.run(x[:8192],x[1:8193]);m.finish()
        expected[arm]=(loss,hashlib.sha256(m.dumps()).hexdigest())
checks={}
for arm in ARMS:
    install(a.work/'runtime',arm)
    with Model(parameters=5_000_000,seed=7,routing_arm=ARMS[arm][0],semantic_laces=False,router='scan') as m:
        configure(m,2_000_000,arm);loss=m.run(x[:8192],x[1:8193]);m.finish()
        state=hashlib.sha256(m.dumps()).hexdigest()
        if arm in expected:assert expected[arm]==(loss,state)
        m.run(x[:511],x[1:512]);saved=m.dumps()
        path=a.work/'roundtrip.shoe';path.write_bytes(saved)
        with Model.load(path) as loaded:
            aa=m.run(x[511:1024],x[512:1025]);bb=loaded.run(x[511:1024],x[512:1025])
            assert aa==bb and m.dumps()==loaded.dumps()
        checks[arm]=dict(neutral_parity=arm in expected,continuation=True,nll=loss,sha256=state)
(a.work/'roundtrip.shoe').unlink()
atomic(a.work/'verification.json',dict(data_hashes=True,document_disjoint=True,checks=checks))
print(json.dumps(checks,indent=2))
