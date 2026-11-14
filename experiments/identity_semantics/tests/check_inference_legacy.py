"""Check exact old checkpoint import and bounded cross-build continuation drift."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
CODE=r'''
import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from shoebox_reference import fresh_model,Model
folder=Path(sys.argv[2]);which=sys.argv[3]
x=np.frombuffer(b'The river crosses the valley and the king follows. '*40,np.uint8)
variants=[{},dict(profile='event',frames='lightweight-frames-v1'),dict(profile='epochs',pipeline_workers=2,bank_epochs=dict(epoch_bytes=8,lag_epochs=1),prediction=dict(compact=True,slow_stride=8))]
for i,kw in enumerate(variants):
 if which=='old':
  m=fresh_model(seed=7,**kw);m.run(x[:259],x[1:260],mode='periodic');m.save(folder/f'v{i}.npz');(folder/f'initial{i}.txt').write_text(m.fingerprint())
 else:
  m=Model.load(folder/f'v{i}.npz');assert m.fingerprint()==(folder/f'initial{i}.txt').read_text(),'import changed state'
 z=m.run(x[259:1284],x[260:1285],mode='periodic',logits=True)[1]
 events=m.content_events()
 arrays=dict(logits=z,flat=m.flat,extra=m.extra_flat,publications=m.publications())
 for r,v in events.items():
  for k,value in v.items():arrays[r+'_'+k]=np.asarray(value)
 if which=='old':np.savez(folder/f'expected{i}.npz',**arrays)
 else:
  with np.load(folder/f'expected{i}.npz') as expected:
   drift={}
   for k,value in arrays.items():
    if np.issubdtype(value.dtype,np.floating):
     delta=float(np.max(np.abs(value-expected[k]))) if value.size else 0
     np.testing.assert_allclose(value,expected[k],rtol=0,atol=1e-5,err_msg=k)
     if delta:drift[k]=delta
    else:np.testing.assert_array_equal(value,expected[k],err_msg=k)
   print(json.dumps(dict(profile=i,import_bitwise_exact=True,continuation_targets=1025,max_absolute_drift=max(drift.values(),default=0),drift=drift,identities_and_clocks_exact=True)))
 m.close()
print(which+': v40/v45/v46 compatibility completed')
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
 with tempfile.TemporaryDirectory() as d:
  for root,which in ((a.baseline,'old'),(ROOT,'new')):
   subprocess.run([sys.executable,'-c',CODE,str(root),d,which],check=True,env={**os.environ,'SHOEBOX_EXPERTS':'12'})
if __name__=='__main__':main()
