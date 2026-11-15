import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import Model,build,install,atomic
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();work=a.base/'diagnostic_40m'
build(work/'runtime_followup')
data=np.fromfile(work/'data/train.bin',np.uint8)[:8193];parent=a.base/'run_20m_40m/latest.shoe'
with Model.load(parent) as m:
    m.reset_context(clear_memory=True);loss=m.run(data[:-1],data[1:]);m.finish();expected=hashlib.sha256(m.dumps()).hexdigest()
_,lib=install(work/'runtime_followup')
with Model.load(parent) as m:
    m.reset_context(clear_memory=True);observed=m.run(data[:-1],data[1:]);m.finish();actual=hashlib.sha256(m.dumps()).hexdigest()
    assert loss==observed and expected==actual
    m.reset_context(clear_memory=True);m.run(data[:511],data[1:512]);out=np.zeros((4,2));m._check(lib.diag_validate_gradient(m._alive(),out.ctypes.data))
    assert np.all(out[:,0]<2e-6)
atomic(work/'followup_verification.json',dict(baseline_parity=True,baseline_state_sha256=actual,gradient_vs_real_flush_maximum_errors=out[:,0].tolist(),gradient_maximum_values=out[:,1].tolist()))
print(json.dumps(dict(ready=True,baseline_parity=True,gradient_vs_real_flush_maximum_errors=out[:,0].tolist())),flush=True)
