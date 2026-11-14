"""Resume a real stage-5 checkpoint under the new runtime without policy migration."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline-runtime',type=Path);p.add_argument('--runtime',type=Path)
 p.add_argument('--worker',choices=['old','new']);p.add_argument('--folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.worker:
  import numpy as np
  sys.path.insert(0,str(a.runtime));from runtime import Model,fresh_model
  raw=b'The river runs through the valley. The king crossed the bridge. '*34
  x=np.frombuffer(raw[:2049],np.uint8);checkpoint=a.folder/'stage5.npz'
  if a.worker=='old':
   m=fresh_model(seed=7,profile='event');m.run(x[:1024],x[1:1025],mode='periodic',lr=.04,lp=.04/3);m.save(checkpoint)
  else:m=Model.load(checkpoint)
  before=m.fingerprint();nll,logits=m.run(x[1024:-1],x[1025:],mode='periodic',lr=.04,lp=.04/3,logits=True)
  np.savez(a.folder/(a.worker+'.npz'),logits=logits,flat=m.flat,extra=m.extra_flat)
  a.output.write_text(json.dumps(dict(before=before,after=m.fingerprint(),nll=nll)));m.close();return
 import numpy as np
 with tempfile.TemporaryDirectory(prefix='shoebox-stage5-compat-') as temp:
  folder=Path(temp);rows={}
  for name,runtime in [('old',a.baseline_runtime),('new',ROOT/'runtime')]:
   output=folder/(name+'.json');subprocess.run([sys.executable,__file__,'--worker',name,'--runtime',str(runtime),'--folder',temp,'--output',str(output)],check=True)
   rows[name]=json.loads(output.read_text())
  assert rows['old']==rows['new'],rows
  with np.load(folder/'old.npz') as old,np.load(folder/'new.npz') as new:
   for k in old.files:np.testing.assert_array_equal(old[k],new[k])
 a.output.write_text(json.dumps(dict(stage5_base='fd3edf42819b880b6cf10fdb0fdde576c474d98a',warmup=1024,targets=1024,bitwise_logits=True,bitwise_parameters=True,bitwise_full_state=True,**rows),indent=2)+'\n')
if __name__=='__main__':main()
