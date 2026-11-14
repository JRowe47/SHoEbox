"""Inspect version payloads and live records between chunks; never instrument run()."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model
from memory_inspection import capture_memory,compare_memory,read_memory_index

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--targets',type=int,default=4096);p.add_argument('--chunk',type=int,default=512)
    p.add_argument('--workers',type=int,choices=(1,2),default=1)
    a=p.parse_args();x=np.frombuffer(a.corpus.read_bytes()[:a.warmup+a.targets+1],np.uint8)
    if len(x)!=a.warmup+a.targets+1:raise ValueError('Insufficient corpus')
    m=fresh_model(seed=7,profile='pipeline',pipeline_workers=a.workers)
    try:
        m.run(x[:a.warmup],x[1:a.warmup+1],mode='periodic',lr=.04,lp=.04/3)
        before=capture_memory(m,include_stream=False);points=[];seconds=0
        for lo in range(a.warmup,a.warmup+a.targets,a.chunk):
            hi=min(lo+a.chunk,a.warmup+a.targets);start=time.perf_counter()
            loss=m.run(x[lo:hi],x[lo+1:hi+1],mode='periodic',lr=.04,lp=.04/3)
            seconds+=time.perf_counter()-start
            points.append(dict(bytes=hi-a.warmup,training_seconds=seconds,nll=loss,memory=m.inspect_pipeline()))
        result=dict(protocol=__doc__,points=points,bytes_per_second=a.targets/seconds,
                    fitted_changes=compare_memory(before,capture_memory(m,include_stream=False)),live=read_memory_index(m))
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:m.close()
if __name__=='__main__':main()
