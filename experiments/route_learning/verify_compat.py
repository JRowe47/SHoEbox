"""Compare frozen old/new public runtimes against a real old checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

WORKER='''
import hashlib,json,sys
from shoe_multiscale import Model
with Model.load(sys.argv[1]) as m:
    before=hashlib.sha256(m.dumps()).hexdigest()
    data=(b'Exact historical continuation across optimizer revisions. '*80)[:4098]
    loss=m.train(data)
    print(json.dumps(dict(before=before,after=hashlib.sha256(m.dumps()).hexdigest(),loss=loss)))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();new=Path(__file__).resolve().parents[2]
    results=[]
    for source in (a.old,new):
        result=subprocess.run([sys.executable,'-c',WORKER,str(a.checkpoint.resolve())],cwd=source,
                              check=True,capture_output=True,text=True)
        results.append(json.loads(result.stdout))
    assert results[0]==results[1],results
    a.out.write_text(json.dumps(dict(identical=True,old_source=str(a.old),new_source=str(new),result=results[0]),indent=2))
    print(json.dumps(results[0]))

if __name__=='__main__':main()
