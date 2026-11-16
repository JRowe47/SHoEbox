"""Check every actual fitted snapshot/cursor without extending selected research."""
import argparse,hashlib,json,time
from pathlib import Path
from .backend import Model
from .stream import advance
from .run import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rows=[];start=time.process_time();totals=dict(input_bytes=0,padding_bytes=0,targets=0,local_projection_steps=0,local_presence_steps=0,gate_targets=0,local_training_windows=0)
    for path in sorted(a.runs.glob('*/*.qr')):
        raw=path.read_bytes();cursor=json.loads(path.with_suffix('.cursor.json').read_text());policy='a1' if path.name.startswith('a1') else 'e8'
        with Model.loads(raw,policy) as x,Model.loads(raw,policy) as y:
            before=x.inventory();s,l=advance(x,cursor,1000);t,k=advance(y,cursor,371)
            with Model.loads(y.dumps(),policy) as z:
                u,j=advance(z,t,629)
                assert s==u and x.dumps()==z.dumps(),path
                after=x.inventory()
                totals['input_bytes']+=2000;totals['padding_bytes']+=2*(s['padding_bytes']-cursor['padding_bytes']);totals['targets']+=2*(s['targets']-cursor['targets'])
                for name in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):totals[name]+=2*(after[name]-before[name])
                rows.append(dict(file=str(path.relative_to(a.runs)),sha256=hashlib.sha256(raw).hexdigest(),cursor_sha256=hashlib.sha256(path.with_suffix('.cursor.json').read_bytes()).hexdigest(),exact=True,continuation_bytes=2000))
        assert path.read_bytes()==raw
    write(a.out,dict(rows=rows,count=len(rows),all_exact=True,discarded_correctness_updates=totals,cpu_seconds=time.process_time()-start))
    print(len(rows),'fitted checkpoint/cursor pairs exact')
if __name__=='__main__':main()
