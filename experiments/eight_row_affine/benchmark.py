"""The five and only five registered timing arms; write-once raw measurements."""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
from .backend import Model as Candidate
from experiments.row_parallel_affine.backend import Model as A10
from experiments.selective_record_key_credit.backend import Model as Scalar

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    lock=json.loads((a.work/'pre_timing.lock.json').read_text());assert lock['passed']
    for name,digest in lock['source_hashes'].items():assert sha(root/name)==digest,name
    for name,digest in lock['evidence_hashes'].items():assert sha(a.work/name)==digest,name
    assert not (a.work/'timing_started.json').exists(),'Timing cannot be repeated'
    import subprocess
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    committed=subprocess.check_output(['git','show','HEAD:reports/exp0305a11/pre_timing.lock.json'],cwd=root)
    assert committed==(a.work/'pre_timing.lock.json').read_bytes(),'Commit passing lock before timing'
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    assert not any(os.environ.get(key) for key in ('A11_EVENT_DIR','A11_INSTRUMENT_ROOT','A11_PYTHON_EVENT_DIR'))
    for name in lock['required_control_processes']:
        assert json.loads((a.work/name/'process.json').read_text())['exit_code']==0
    data=(a.work/'workload/timing.bin').read_bytes();workload=json.loads((a.work/'workload/lock.json').read_text())
    assert hashlib.sha256(data).hexdigest()==workload['stream_sha256']
    source=a.prior/'a9/7';e8=(source/'e8.qr').read_bytes();a9=(source/'a9.qr').read_bytes()
    models={'native':Scalar('e8',seed=7,kind='native'),'e8':Scalar.loads(e8,'e8'),
        'scalar_a9':Scalar.loads(a9,'a9'),'a10':A10.loads(a9,'a9'),'a11':Candidate.loads(a9,'a9')}
    production=json.loads((a.work/'production_retry/result.json').read_text())['libraries']
    for name,library_name in [('native','e8'),('e8','e8'),('scalar_a9','scalar'),('a10','a10'),('a11','a11')]:
        assert sha(Path(models[name].lib._name))==production[library_name]['sha256']
    names=list(models);rows={name:dict(cpu_seconds=[],checksums=[],warmup_checksums=[]) for name in names}
    started=dict(pre_timing_commit=head,source_lock_sha256=sha(a.work/'pre_timing.lock.json'),cpu=cpu,
        checkpoints={'e8':hashlib.sha256(e8).hexdigest(),'a9':hashlib.sha256(a9).hexdigest()},
        workload=workload['stream_sha256'],arms=names,repeats=3,bytes_per_repeat=65536,warmup_per_repeat=1024,
        all_other_research_jobs_finished=True,research_fitting_bytes=0)
    (a.work/'timing_started.json').write_text(json.dumps(started,indent=2)+'\n')
    raw=a.work/'timing_raw.jsonl'
    try:
        with raw.open('x') as f:
            for repeat in range(3):
                for name in names[repeat:]+names[:repeat]:
                    with models[name].clone() as m:
                        m.reset();warmup=m.benchmark(data,1024);m.reset()
                        start=time.process_time_ns();checksum=m.benchmark(data,65536);duration=time.process_time_ns()-start
                    measurement=dict(repeat=repeat,arm=name,cpu_nanoseconds=duration,cpu_seconds=duration/1e9,
                        checksum=checksum,checksum_hex=checksum.hex(),warmup_checksum=warmup,warmup_checksum_hex=warmup.hex(),input_bytes=65536,warmup_bytes=1024)
                    f.write(json.dumps(measurement)+'\n');f.flush();os.fsync(f.fileno())
                    r=rows[name];r['cpu_seconds'].append(duration/1e9);r['checksums'].append(checksum.hex());r['warmup_checksums'].append(warmup.hex())
        exact=all(rows['scalar_a9'][key]==rows[name][key] for name in ('a10','a11') for key in ('checksums','warmup_checksums'))
        for r in rows.values():r['median_cpu_seconds']=statistics.median(r['cpu_seconds']);r['bytes_per_cpu_second']=65536/r['median_cpu_seconds']
        for r in rows.values():r['relative_native']=rows['native']['median_cpu_seconds']/r['median_cpu_seconds']
        candidate=rows['a11'];speed=candidate['relative_native']>=.8
        no_slowdown=candidate['median_cpu_seconds']<=rows['scalar_a9']['median_cpu_seconds']
        no_a10_slowdown=candidate['median_cpu_seconds']<=rows['a10']['median_cpu_seconds']
        result=dict(**started,rows=rows,exact_checksums=exact,native_speed_pass=speed,no_scalar_slowdown=no_slowdown,no_a10_slowdown=no_a10_slowdown,
            execution_pass=exact and speed and no_slowdown and no_a10_slowdown,accuracy_scored=False,
            total_input_bytes=5*3*(65536+1024),timed_bytes=5*3*65536,warmup_bytes=5*3*1024,
            padding_bytes=15*sum(data[i%len(data)]==9 for n in (65536,1024) for i in range(n)),
            nuisance_bytes=15*sum(data[i%len(data)]>=160 for n in (65536,1024) for i in range(n)),
            execution_does_not_promote_a9=True)
        (a.work/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        for m in models.values():m.close()
if __name__=='__main__':main()
