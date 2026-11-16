"""Verify loops-off arithmetic against the pre-loop source in another checkout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

WORKER='''
import hashlib,json,inspect
from shoe_multiscale import Model
data=(b'Exact historical continuation through completed constituents. '*100)[:4098]
rows=[]
for options in ({},{'routing_arm':'D'},{'route_lr':{'period':17},'language_credit':[1,2,3,4],'recall_credit':[1,1,1,1]}):
    selected=dict(options)
    if 'core' in inspect.signature(Model).parameters:selected['core']='full'
    with Model(**selected) as model:
        model.train(data[:514]);initial=model.dumps()
        with Model.loads(initial) as resumed:
            loss=resumed.train(data);resumed.finish()
            rows.append(dict(options=options,before=hashlib.sha256(initial).hexdigest(),
                             after=hashlib.sha256(resumed.dumps()).hexdigest(),loss=loss))
print(json.dumps(rows))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();rows=[]
    for root in (args.old,Path(__file__).resolve().parents[2]):
        result=subprocess.run([sys.executable,'-c',WORKER],cwd=root,check=True,capture_output=True,text=True)
        rows.append(json.loads(result.stdout))
    assert rows[0]==rows[1],rows
    args.out.write_text(json.dumps(dict(identical=True,base_commit='e2a2e93',cases=rows[0]),indent=2)+'\n')
    print('All three old/new checkpoint and 4097-target continuations are byte-identical.')


if __name__=='__main__':main()
