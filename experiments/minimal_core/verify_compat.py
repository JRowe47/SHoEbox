"""Verify seven historical policies against a pre-minimal source checkout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

WORKER = '''
import hashlib,inspect,json
from shoe_multiscale import Model
profiles=[('v7',{'routing_arm':'A'}),('v9',{'routing_arm':'C'}),('v10',{}),
          ('v11',{'language_credit':[1,1,2,3]}),('v12',{'recall_credit':[0,0,1,1]}),
          ('v17',{'route_lr':{'period':17}}),('v18',{'loops':{'steps':[0,0,1,1]}})]
has_core='core' in inspect.signature(Model).parameters
rows=[]
data=(b'Exact compatibility through pending ordered constituents and memory. '*80)
for name,options in profiles:
    if has_core:options['core']='full'
    with Model(parameters=500000,seed=19,**options) as m:
        m.train(data[:518]);before=m.dumps()
        version=m.stats()['checkpoint_version']
        assert version==int(name[1:]),(name,version)
        with Model.loads(before) as resumed:
            loss=resumed.train(data[517:4098]);resumed.finish()
            rows.append(dict(policy=name,before_sha256=hashlib.sha256(before).hexdigest(),
                after_sha256=hashlib.sha256(resumed.dumps()).hexdigest(),loss=loss))
print(json.dumps(rows))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    roots=(a.old,Path(__file__).resolve().parents[2])
    rows=[]
    for root in roots:
        result=subprocess.run([sys.executable,'-c',WORKER],cwd=root,check=True,capture_output=True,text=True)
        rows.append(json.loads(result.stdout))
    assert rows[0]==rows[1],rows
    old_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.old,text=True).strip()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(identical=True,old_commit=old_commit,cases=rows[0]),indent=2)+'\n')
    print('All seven historical checkpoint policies and resumed updates are byte-identical.')


if __name__=='__main__':main()
