"""Isolated, pinned-CPU, interleaved full-256-byte inference cost."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from shoe_multiscale import Model as Public
from .backend import Model
from .tasks import cases
from .run import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});root=a.runs/'7'
    models={name:Model.loads((root/(name+'.qr')).read_bytes(),name) for name in ('e8','a2','a3')}
    models={'native':Model('e8',seed=7,kind='native'),**models}
    public,receipt=Public.import_e8_research((root/'e8.qr').read_bytes());models['public_e8']=public
    data=b''.join(c.ep.data for condition in ('pad0','repeat16','diverse16','compete16','overwrite16','diverse256') for c in cases('dev',16,condition=condition))
    rows={name:dict(cpu_seconds=[],checksums=[]) for name in models}
    try:
        names=list(models)
        for repeat in range(3):
            for name in names[repeat:]+names[:repeat]:
                base=models[name]
                frozen=base.inference_copy(cold=False) if name=='public_e8' else base.clone()
                with frozen as m:
                    reset=(lambda:m.reset_context(clear_memory=True)) if name=='public_e8' else m.reset
                    reset();m.benchmark(data,1024);reset();start=time.process_time();checksum=m.benchmark(data,65536)
                    rows[name]['cpu_seconds'].append(time.process_time()-start);rows[name]['checksums'].append(checksum)
        assert rows['e8']['checksums']==rows['public_e8']['checksums']
        for r in rows.values():r['bytes_per_cpu_second']=65536/float(np.median(r['cpu_seconds']))
        for r in rows.values():r['relative_native']=r['bytes_per_cpu_second']/rows['native']['bytes_per_cpu_second'];r['gate_pass']=r['relative_native']>=.8
        write(a.out,dict(rows=rows,cpu=cpu,repetitions=3,bytes_per_repeat=65536,warmup_per_repeat=1024,
            total_inference_bytes=5*3*(65536+1024),timing_padding_bytes=5*3*(sum(data[i%len(data)]==9 for i in range(65536))+sum(data[i%len(data)]==9 for i in range(1024))),
            data_sha256=hashlib.sha256(data).hexdigest(),public_import=receipt,public_e8_exact=True,
            checkpoint_hashes={name:hashlib.sha256((root/(name+'.qr')).read_bytes()).hexdigest() for name in ('e8','a2','a3')},
            native_control='fresh untrained v19 498367 parameters; zero fitting bytes',
            scope='Three interleaved full-distribution repeats on one pinned CPU after all other local research jobs stopped; compilation, cloning and warmup excluded from intervals.'))
        print({k:r['relative_native'] for k,r in rows.items()})
    finally:
        for m in models.values():m.close()
if __name__=='__main__':main()
