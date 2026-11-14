"""Isolated Action-1 versus Action-2 production measurements and exact parity.

Uses the Action-1 fixture, export scope, learning policy and API chunks. All
copies, quality evaluation and memory inspection remain outside timed calls.
Run alone on the measured host. Temporary model/data files are never published.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PROFILES=('reference','event','compact_serial','compact_threaded')

def worker(a):
    sys.path[:0]=[str(a.repo/'experiments/identity_semantics/runtime'),str(a.repo/'experiments/identity_semantics/src')]
    import measure_current_baseline as baseline
    original=baseline.inventory
    def inventory(model):
        result=original(model)
        if hasattr(model.lib,'indexed_storage_status'):
            from memory_inspection import read_indexed_storage
            result['indexed_storage']=read_indexed_storage(model)
        return result
    baseline.inventory=inventory
    from types import SimpleNamespace
    baseline.worker(SimpleNamespace(profile=a.profile,workload=a.workload,targets=a.targets,warmup=1024,
        chunk=512,validation_bytes=1024,seed=7,lr=.04,cpu=a.cpu,worker_cpu=a.worker_cpu,
        data_revision=baseline.PIN,output=a.output))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--targets',type=int,default=4096)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cpu',type=int,default=0);p.add_argument('--worker-cpu',type=int,default=1)
    p.add_argument('--profile',choices=PROFILES,default='reference')
    p.add_argument('--profiles',nargs='+',default=list(PROFILES),choices=PROFILES)
    p.add_argument('--workload',default='natural',choices=('repeated','natural','documents'))
    p.add_argument('--workloads',nargs='+',default=['repeated','natural','documents'])
    a=p.parse_args()
    if a.worker:return worker(a)
    if not a.baseline:p.error('--baseline must point to the Action-1 checkout')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'SHOEBOX_EXPERTS':'12','SHOEBOX_INLINE_DIAGNOSTICS':'0','SHOEBOX_KERNEL':'completion'}
    if env.get('SHOEBOX_CXX_FLAGS'):p.error('Production flags only')
    result={'protocol':__doc__,'baseline_revision':subprocess.check_output(['git','-C',str(a.baseline),'rev-parse','HEAD'],text=True).strip(),
            'runs':[],'comparisons':[],'all_exact':True}
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'row.json'
        for workload in a.workloads:
            for repeat in range(a.repeats):
                order=a.profiles[repeat%len(a.profiles):]+a.profiles[:repeat%len(a.profiles)]
                for profile in order:
                    for variant in (('before','after') if repeat%2==0 else ('after','before')):
                        repo=a.baseline if variant=='before' else a.repo
                        subprocess.run([sys.executable,__file__,'--worker','--repo',str(repo),'--output',str(out),
                            '--profile',profile,'--workload',workload,'--targets',str(a.targets),
                            '--cpu',str(a.cpu),'--worker-cpu',str(a.worker_cpu)],env=env,check=True)
                        row=json.loads(out.read_text())
                        keep=('seconds','process_cpu_seconds','bytes_per_second','train_nll','heldout_nll',
                              'fitted_parameter_count','logits_sha256','continuation_sha256','quality_curve','data',
                              'native_library_sha256','training_peak_rss_kib')
                        short={k:row[k] for k in keep}
                        short.update(variant=variant,profile=profile,workload=workload,repeat=repeat,targets=a.targets,
                                     final_memory=row['memory_curve'][-1]['inventory'])
                        result['runs'].append(short);save()
                        print(f'{variant} {workload} {profile} {repeat+1}: {row["bytes_per_second"]:.1f} B/s',flush=True)
            for profile in a.profiles:
                groups={v:[r for r in result['runs'] if (r['variant'],r['profile'],r['workload'])==(v,profile,workload)] for v in ('before','after')}
                exact=all(len({r[k] for rows in groups.values() for r in rows})==1 for k in
                          ('logits_sha256','continuation_sha256','heldout_nll','fitted_parameter_count'))
                before,after=(statistics.median(r['bytes_per_second'] for r in groups[v]) for v in ('before','after'))
                result['comparisons'].append(dict(workload=workload,profile=profile,before_bps=before,after_bps=after,
                    speedup=after/before,exact=exact,before_rates=[r['bytes_per_second'] for r in groups['before']],
                    after_rates=[r['bytes_per_second'] for r in groups['after']]))
                result['all_exact'] &= exact
                save()
    if not result['all_exact']:raise AssertionError('Baseline arithmetic or logical continuation mismatch')
    result['completed']=True;save()

if __name__=='__main__':main()
