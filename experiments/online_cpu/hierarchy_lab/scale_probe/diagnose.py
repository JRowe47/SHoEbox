import json,os,time
import numpy as np
from model import Model,legacy,ROOT,core,evaluate
size='1m' if core.E==12 else '5m'
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))})
source=ROOT/'sources'/size/'last.npz'
train=np.frombuffer((ROOT/'data/train.bin').read_bytes(),np.uint8)
val=np.frombuffer((ROOT/'data/validation.bin').read_bytes(),np.uint8)
a=Model.load(source);b=legacy.Model.load(source)
rates=dict(mode='periodic',lr=.008,lp=.008/3,route=.002)
x=a.run(train[:4096],train[1:4097],**rates);y=b.run(train[:4096],train[1:4097],**rates)
assert x==y and a.fingerprint()==b.fingerprint(),(x,y)
a.close();b.close()
m=Model.load(source)
out={'equivalence':True,'size':size,'frozen':evaluate(m,val[:65537],True)}
before=m.lace_metrics();m.lib.diagnostic_enable(m.handle,1)
m.run(train[:65536],train[1:65537],**rates)
after=m.lace_metrics();out['training_diagnostics']=m.diagnostics();out['credit_delta']={r:{k:after[r][k]-before[r][k] for k in before[r]} for r in before}
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results'/f'diagnosis_{size}.json').write_text(json.dumps(out,indent=2))
print(size,'equivalence passed',out['frozen']['nll'])
for r in ['A1','B1','L3','F4','G5','Y']:
 d=out['frozen']['diagnostics'][r]
 print(r,'experts',round(out['frozen']['effective_experts'][r],2),'input',d['input'],'output',d['output'],'ctx',d['context_to_bottom_norm'],'mask',d['context_energy_in_router_mask'],'sat',d['saturated_h_fraction'])
