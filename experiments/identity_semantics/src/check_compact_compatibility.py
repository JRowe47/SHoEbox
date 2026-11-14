"""Real Stage-6 v42 pending checkpoint: import equality and resumed learning."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-runtime',type=Path);p.add_argument('--runtime',type=Path)
p.add_argument('--worker',choices=('baseline','audit','production'));p.add_argument('--folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.worker:
 sys.path.insert(0,str(a.runtime));from runtime import fresh_model,Model,core
 raw=b'The river runs through the valley. The king crossed the bridge. '
 x=np.frombuffer((raw*100)[:2049],np.uint8);checkpoint=a.folder/'v42.npz'
 if a.worker=='baseline':
  m=fresh_model(seed=7,profile='pipeline');m.run(x[:1024],x[1:1025],mode='periodic',lr=.04,lp=.04/3);m.save(checkpoint)
 else:
  m=Model.load(checkpoint);m.save(a.folder/(a.worker+'_before.npz'))
  sys.path.insert(0,str(ROOT/'runtime'));from memory_inspection import diagnostic_ranges
  (a.folder/'ranges.json').write_text(json.dumps(diagnostic_ranges(m)))
 nll,logits=m.run(x[1024:-1],x[1025:],mode='periodic',lr=.04,lp=.04/3,logits=True)
 m.save(a.folder/(a.worker+'_after.npz'));np.save(a.folder/(a.worker+'_logits.npy'),logits);m.close();sys.exit(0)
sys.path.insert(0,str(ROOT/'runtime'));from memory_inspection import decode_pending,normalize_observations
with tempfile.TemporaryDirectory(prefix='shoebox-v42-') as tmp:
 folder=Path(tmp)
 for name in ('baseline','audit','production'):
  runtime=a.baseline_runtime if name=='baseline' else ROOT/'runtime'
  env={**os.environ,'SHOEBOX_INLINE_DIAGNOSTICS':'1' if name=='audit' else '0','OPENBLAS_NUM_THREADS':'1'}
  subprocess.run([sys.executable,__file__,'--worker',name,'--runtime',str(runtime),'--folder',tmp,'--output',str(a.output)],env=env,check=True)
 def read(path):
  with np.load(path) as z:return {k:z[k].copy() for k in z.files}
 initial=read(folder/'v42.npz');old=read(folder/'baseline_after.npz');ranges=json.loads((folder/'ranges.json').read_text());rows=[]
 for name in ('audit','production'):
  before=read(folder/(name+'_before.npz'));after=read(folder/(name+'_after.npz'))
  for key in initial:
   if key not in ('version','bank_pipeline_state'):np.testing.assert_array_equal(initial[key],before[key])
  left=decode_pending(initial['bank_pipeline_state'],initial['flat'],int(initial['experts'][0]));right=decode_pending(before['bank_pipeline_state'],before['flat'],int(initial['experts'][0]))
  assert left['header']==right['header']
  for bank in left['banks']:np.testing.assert_array_equal(left['banks'][bank],right['banks'][bank])
  oldlogits=np.load(folder/'baseline_logits.npy');newlogits=np.load(folder/(name+'_logits.npy'))
  maxlogit=float(abs(oldlogits-newlogits).max());maxweight=float(abs(old['flat']-after['flat']).max())
  assert maxlogit<1e-4 and maxweight<1e-5
  aa=normalize_observations(old,ranges);bb=normalize_observations(after,ranges)
  mismatches=[k for k in aa if k not in ('version','bank_pipeline_state') and not np.array_equal(aa[k],bb[k])]
  rows.append(dict(build=name,initial_import_bitwise=True,continuation_logit_max_abs=maxlogit,
                   continuation_parameter_max_abs=maxweight,bitwise_operational_state=not mismatches,different_sections=mismatches))
 a.output.write_text(json.dumps(dict(baseline='fbf7c816446289bcd987a85028d7cab59389b7ed',checkpoint_version=42,warmup=1024,targets=1024,runs=rows),indent=2)+'\n')
 print(json.dumps(rows))
