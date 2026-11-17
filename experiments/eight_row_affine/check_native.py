"""Original seven native A9 microtest suites, counted before execution."""
import sys
from . import backend
from experiments.row_parallel_affine import backend as a10
from experiments.selective_record_key_credit import check_native as old

def main():
    execution=sys.argv[1]
    if execution in ('a10','a11'):
        selected=a10 if execution=='a10' else backend
        head=selected.header();old.header=lambda policy:head if policy=='a9' else backend.scalar.header(policy)
    elif execution!='scalar':raise ValueError(execution)
    # AVX2 must also be available in sanitizer builds; same fixed kernel.
    original=old.subprocess.run
    def run(cmd,*args,**kwargs):
        if cmd[0]=='g++' and execution=='a11':cmd=[cmd[0],'-mavx2',*cmd[1:]]
        return original(cmd,*args,**kwargs)
    old.subprocess.run=run;old.main()
if __name__=='__main__':main()
