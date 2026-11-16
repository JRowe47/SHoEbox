"""Compare historical constructors, snapshots and continuation to pinned main."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

WORKER=r'''
import hashlib,json
from shoe_multiscale import Model
cases=[('v1',dict(memory_profile='legacy')),('v2',dict(memory_profile='addressed_v2')),
 ('v3',dict(memory_profile='addressed_v3')),('v4',dict(memory_profile='addressed_v4')),
 ('v7',dict(core='full',routing_arm='A')),('v8',dict(routing_arm='B')),('v9c',dict(routing_arm='C')),
 ('v9d',dict(routing_arm='D')),('v10',dict(core='full')),
 ('v11',dict(language_credit=[1,1,2,3])),('v12',dict(recall_credit=[0,0,1,1])),
 ('v17',dict(route_lr={'period':17})),('v18',dict(loops={'steps':[0,0,1,1]})),
 ('v19',dict(core='minimal'))]
rows=[];data=b'Exact history across pending constituents and training boundaries. '*70
for name,options in cases:
 with Model(parameters=500000,seed=19,**options) as model:
  model.train(data[:518]);before=model.dumps()
  with Model.loads(before) as resumed:
   loss=resumed.train(data[517:4098]);resumed.finish()
   rows.append(dict(policy=name,version=model.stats()['checkpoint_version'],
    before_sha256=hashlib.sha256(before).hexdigest(),
    after_sha256=hashlib.sha256(resumed.dumps()).hexdigest(),loss=loss))
print(json.dumps(rows))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=[]
 for root in (a.old,Path(__file__).resolve().parents[2]):
  r=subprocess.run([sys.executable,'-c',WORKER],cwd=root,capture_output=True,text=True,check=True)
  rows.append(json.loads(r.stdout))
 assert rows[0]==rows[1],rows
 a.out.write_text(json.dumps(dict(exact=True,rows=rows[1],correctness_only_discarded_input_bytes=len(rows[1])*2*4097),indent=2)+'\n')
 print('14 historical constructors/checkpoints and resumed updates are byte-identical.')
if __name__=='__main__':main()
