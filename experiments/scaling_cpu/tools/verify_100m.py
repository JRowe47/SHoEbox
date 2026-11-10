"""Build the actual packed 99.73M allocation and exercise training/evaluation."""
from pathlib import Path
import json,time,resource
import numpy as np
from shoebox_runtime import Architecture,Model,evaluate
from shoebox_runtime.model import atomic_json,initialize,views

root=Path(__file__).resolve().parents[1]
c=json.loads((root/'configs/proposed_100m.json').read_text())
a=Architecture(c['width'],c['experts'])
data=np.fromfile(root/'data/comparison/calibration.bin',np.uint8)[:8193]
start=time.perf_counter()
with Model(a,seed=11) as m:
    assert m.parameter_count==99_727_616
    initial=m.params['W'][a.offsets[0]:a.offsets[1],192:,192:].copy()
    t=time.perf_counter();loss=m.run(data[:-1],data[1:],mode='periodic',lr=.002,lp=.002/3,route=.002);seconds=time.perf_counter()-t
    assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all()
    assert np.any(initial!=m.params['W'][a.offsets[0]:a.offsets[1],192:,192:])
    fp=m.fingerprint();probe=evaluate(m,data[:513],diagnostics=True)
    assert m.fingerprint()==fp and len(probe['diagnostics'])==18
    totals=m.regional_totals()
    for n,counts,execs in zip(a.experts,totals['expert_counts'],totals['executions']):assert len(counts)==n and sum(counts)==execs
    result=dict(parameters=m.parameter_count,architecture=a.as_dict(),targets=m.observations,training_nll=loss,
        training_seconds=seconds,smoke_bytes_per_second=8192/seconds,finite_parameters=True,upper_coordinates_learn=True,
        frozen_evaluation_preserved_state=True,probe=probe,wall_seconds=time.perf_counter()-start,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
atomic_json(root/'reports/full_100m_smoke.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('probe','architecture')},indent=2))
