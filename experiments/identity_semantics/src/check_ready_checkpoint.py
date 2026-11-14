"""Resume actual Action-2 checkpoint bytes in the reusable execution runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--load',action='store_true')
    p.add_argument('--profile',default='reference');p.add_argument('--heads',type=int,default=0)
    a=p.parse_args()
    if a.worker:
        sys.path[:0]=[str(a.repo/'experiments/identity_semantics/runtime'),str(a.repo/'experiments/identity_semantics/src')]
        import numpy as np
        from runtime import fresh_model,Model
        from memory_inspection import capture_memory,memory_fingerprint,read_memory_index
        from measure_current_baseline import fixtures
        raw=fixtures('natural',9217)[0][0];data=np.frombuffer(raw,np.uint8)
        if a.load:m=Model.load(a.checkpoint)
        else:
            m=fresh_model(seed=7,profile=a.profile,event={'context_heads':a.heads} if a.heads else None)
            m.run(data[:769],data[1:770],mode='periodic',lr=.04,lp=.04/3)
            m.save(a.checkpoint)
        before=memory_fingerprint(capture_memory(m));digest=hashlib.sha256();regional=hashlib.sha256()
        for lo,hi in ([(i,i+1) for i in range(769,802)]+[(802,897),(897,2049),(2049,8961)]):
            _,logits=m.run(data[lo:hi],data[lo+1:hi+1],mode='periodic',lr=.04,lp=.04/3,logits=True)
            digest.update(logits.tobytes())
            regional.update(json.dumps(read_memory_index(m),sort_keys=True).encode())
        m.finish_document(mode='periodic',lr=.04,lp=.04/3)
        after=memory_fingerprint(capture_memory(m))
        m.reset_context();m.run(data[8961:9217],data[8962:9218],mode='periodic',lr=.04,lp=.04/3)
        reset=memory_fingerprint(capture_memory(m));m.close()
        a.output.write_text(json.dumps(dict(before=before,after=after,reset=reset,logits=digest.hexdigest(),regional=regional.hexdigest()))+'\n');return
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':'12','SHOEBOX_INLINE_DIAGNOSTICS':'0'}
    result={'checks':[]}
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint=Path(tmp)/'legacy.npz';out=Path(tmp)/'row.json'
        for profile,heads in (('reference',0),('event',1),('event',2),('pipeline',1)):
            rows=[]
            for load,repo in ((False,a.baseline),(True,a.repo)):
                cmd=[sys.executable,__file__,'--worker','--repo',str(repo),'--checkpoint',str(checkpoint),
                     '--output',str(out),'--profile',profile,'--heads',str(heads)]
                if load:cmd.append('--load')
                subprocess.run(cmd,env=env,check=True);rows.append(json.loads(out.read_text()))
            if rows[0]!=rows[1]:raise AssertionError((profile,heads,rows))
            result['checks'].append(dict(profile=profile,heads=heads,continuation_bytes=8192,exact=True,hashes=rows[0]))
            a.output.write_text(json.dumps(result,indent=2)+'\n');print(profile,heads,'legacy continuation exact',flush=True)
    result['completed']=True;a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
