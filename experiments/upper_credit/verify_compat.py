"""Run in separate processes against old/new runtime roots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.source_root.resolve()))
from shoe_multiscale import Model
raw=a.checkpoint.read_bytes()
with Model.loads(raw) as m:
    before=m.dumps();assert before==raw
    rng=np.random.default_rng(299);data=rng.integers(0,256,4098,dtype=np.uint8)
    loss=m.run(data[:-1],data[1:]);after=m.dumps()
    r=dict(original_sha256=hashlib.sha256(raw).hexdigest(),roundtrip_exact=True,
           loss=loss,continuation_sha256=hashlib.sha256(after).hexdigest())
    a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
