"""Python correctness call/argument and comparison accounting, including children.
Native consumed bytes come from native counters; Python API arguments are a
separate attempted-work ledger and are never added to native counts twice.
"""
import atexit,collections,json,os,sys
from pathlib import Path

def install():
    dest=os.environ.get('A11_PYTHON_EVENT_DIR')
    if not dest or os.environ.get('A11_PYTHON_DISABLE'):return
    rows={}
    def profile(frame,event,arg):
        if event!='call':return
        file=frame.f_code.co_filename;name=frame.f_code.co_name
        if not ('/experiments/' in file or '/multiscale/' in file or 'shoebox_reference' in file or '/unittest/case.py' in file or '/numpy/testing/' in file or name in ('array_equal','allclose','isclose')):return
        if '/eight_row_affine/' in file and name in ('profile','install','finish'):return
        key=f'{file}:{frame.f_code.co_firstlineno}:{name}'
        row=rows.setdefault(key,dict(calls=0,bytes_arguments={},array_elements={},episode_bytes=0,episode_count=0))
        row['calls']+=1
        for k,v in frame.f_locals.items():
            if isinstance(v,(bytes,bytearray)):
                row['bytes_arguments'][k]=row['bytes_arguments'].get(k,0)+len(v)
            elif type(v).__module__.split('.')[0] in ('numpy','torch') and hasattr(v,'shape'):
                n=1
                for size in v.shape:n*=size
                row['array_elements'][k]=row['array_elements'].get(k,0)+int(n)
        cases=frame.f_locals.get('cases')
        if isinstance(cases,(list,tuple)) and cases and hasattr(cases[0],'data'):
            row['episode_count']+=len(cases);row['episode_bytes']+=sum(len(ep.data) for ep in cases)
    sys.setprofile(profile)
    def finish():
        sys.setprofile(None)
        Path(dest,f'{os.getpid()}.json').write_text(json.dumps(dict(command=sys.argv,rows=rows),indent=2)+'\n')
    atexit.register(finish)
