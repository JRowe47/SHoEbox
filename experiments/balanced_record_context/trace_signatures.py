"""Registered signature trace, executed only after the final evaluation lock."""
import json,sys,time
from pathlib import Path
from .backend import Model
from .tasks import cases
from .diagnose import evaluate
from .run import sha,write

def main():
    w=Path(sys.argv[1]);out=w/'signature_traces.json';assert not out.exists()
    lock=json.loads((w/'evaluation.lock.json').read_text())
    for path,digest in lock.items():assert sha(w/path)==digest
    rows=[];start=time.process_time()
    for seed in (1,7,19):
        row=dict(seed=seed,arms={})
        for arm in ('e8','a6','a8'):
            with Model.loads((w/'a8'/str(seed)/(arm+'.qr')).read_bytes(),arm) as m:
                row['arms'][arm]={c:evaluate(m,cases('dev',32,condition=c),trace_state=True) for c in ('pad0','diverse16','diverse64','diverse256','compete16','overwrite16')}
        rows.append(row)
    write(out,dict(rows=rows,cpu_seconds=time.process_time()-start,scope='First 32 worlds, four orders; teacher-fed signature diagnosis, zero fits.'))
    print('Registered signature traces complete')

if __name__=='__main__':main()
