"""Isolated public/research E8/native parity and full-distribution throughput."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from shoe_multiscale import Model
from experiments.joint_cold_gate.adapter import Adapter
from experiments.joint_readout.adapter import Adapter as Native
from experiments.joint_cold_gate.tasks import dataset,balanced_offsets


def measure(root,name,retained=False):
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')))
    suffix='.retained.qr' if retained else '.qr'
    raw=(root/f'exp01e8-{name}/7'/('local_gate'+suffix)).read_bytes()
    native_raw=(root/f'exp01e8-{name}/7'/('native_all'+suffix)).read_bytes()
    public,receipt=Model.import_e8_research(raw)
    models=dict(native=Native.loads(native_raw),research=Adapter.loads(raw),public=public)
    rows={key:dict(cpu_seconds=[],checksums=[]) for key in models}
    try:
        names=list(models)
        for repeat in range(3):
            for key in names[repeat:]+names[:repeat]:
                base=models[key]
                frozen=base.inference_copy(cold=False) if key=='public' else base.clone()
                with frozen as m:
                    reset=(lambda:m.reset_context(clear_memory=True)) if key=='public' else m.reset
                    reset();m.benchmark(data,1024);reset()
                    start=time.process_time();checksum=m.benchmark(data,65536)
                    rows[key]['cpu_seconds'].append(time.process_time()-start)
                    rows[key]['checksums'].append(checksum)
        assert rows['public']['checksums']==rows['research']['checksums']
        for row in rows.values():row['bytes_per_cpu_second']=65536/float(np.median(row['cpu_seconds']))
        for row in rows.values():row['relative_native']=row['bytes_per_cpu_second']/rows['native']['bytes_per_cpu_second']
        return dict(capacity=name,retained=retained,cpu=cpu,rows=rows,
                    public_gate_pass=rows['public']['relative_native']>=.8,
                    stream_sha256=hashlib.sha256(data).hexdigest(),import_receipt=receipt,
                    native_checkpoint_sha256=hashlib.sha256(native_raw).hexdigest())
    finally:
        for model in models.values():model.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rows=[]
    for name,retained in [('formal',False),('1m',False),('5m',False),('formal',True)]:
        row=measure(a.workspace,name,retained);rows.append(row)
        print(name,retained,row['rows']['public']['relative_native'],flush=True)
    a.out.write_text(json.dumps(dict(rows=rows,repetitions=3,bytes_per_repetition=65536,
        warmup_bytes_per_repetition=1024,total_inference_input_bytes=4*3*3*(65536+1024),
        all_pass=all(r['public_gate_pass'] for r in rows),
        scope='Pinned isolated native full-256 distribution per byte; interleaved median CPU; no compilation/cloning/warmup in timed intervals.'),indent=2)+'\n')
if __name__=='__main__':main()
