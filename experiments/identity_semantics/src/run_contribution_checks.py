"""Separated regional interventions on fixed checkpoints and a reserved split."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model,core

BASE_COMMIT='f9eeb8a2c2fe6eb31a4e36954f2705904a893ecf'
KINDS=('neural_output','content_descriptor','relay_context','retained_input',
       'semantic_residual','whole_source','expert_transform')
MEMORY=('cold_adaptive','cold_frozen','warm_frozen','warm_adaptive')
TRACE_KEYS=('event_hash','publications','neural_executions','span_sum','max_span')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def score(m,data):
    blocks=[];start=time.perf_counter()
    for i in range(0,len(data)-1,4096):
        n=min(4096,len(data)-1-i)
        nll=m.run(data[i:i+n],data[i+1:i+n+1])
        if not np.isfinite(nll):raise RuntimeError('Nonfinite intervention loss')
        blocks.append(dict(start=i,targets=n,nll=nll))
    return sum(b['targets']*b['nll'] for b in blocks)/(len(data)-1),blocks,time.perf_counter()-start


def measure(job):
    os.sched_setaffinity(0,{job['core']})
    data=np.fromfile(job['heldout'],np.uint8)[32768:]
    original=Model.load(job['checkpoint']);source=original.fingerprint()
    m=original.evaluation_copy(memory=job['memory'],diagnostic=True)
    m.configure_contribution(job['kind'],job['region'])
    fitted=m.composition_state(fitted=True).copy();flat=m.flat.copy();extra=m.extra_flat.copy()
    before=m.evaluation_memory_info()
    nll,blocks,seconds=score(m,data)
    after=m.evaluation_memory_info()
    assert np.array_equal(fitted,m.composition_state(fitted=True)), 'Fitted dictionary/semantic mutation'
    assert np.array_equal(flat,m.flat) and np.array_equal(extra,m.extra_flat), 'Neural parameter mutation'
    assert original.fingerprint()==source, 'Source checkpoint state mutated'
    if job['memory'].endswith('frozen'):assert before['store_digest']==after['store_digest']
    row={k:job[k] for k in ('seed','kind','region','memory')}
    row.update(nll=nll,targets=len(data)-1,blocks=blocks,seconds=seconds,
               trace=m.contribution_metrics(),memory_before=before,memory_after=after,
               persistent=m.persistent_metrics(),composition=m.composition_metrics(),
               fitted_state_unchanged=True,source_state_unchanged=True)
    if job['kind']=='none':
        production=original.evaluation_copy(memory=job['memory'])
        standard,_,_=score(production,data)
        assert nll==standard, (nll,standard)
        assert production.fingerprint()==m.fingerprint(), 'Disabled control state mismatch'
        row['production_nll']=standard;row['production_parity']=True
        production.close()
    m.close();original.close()
    return row


def execute(jobs,workers,work):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(measure,job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            name=f"s{row['seed']}_{row['memory']}_{row['kind']}_{row['region']}"
            write(work/(name+'.json'),row)
            print(f"{len(rows)}/{len(jobs)} {name} NLL={row['nll']:.9f} {row['seconds']:.1f}s",flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,nargs='+',default=[ROOT/f'results/relay_full_relays_seed{s}/last.npz' for s in (1,2)])
    p.add_argument('--heldout',type=Path,default=ROOT/'data/heldout.bin')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--output',type=Path,default=ROOT/'results/contribution_checks.json')
    p.add_argument('--work',type=Path,default=ROOT/'results/contribution_tmp')
    a=p.parse_args();cpus=sorted(os.sched_getaffinity(0))
    if a.workers<1:p.error('--workers must be positive')
    workers=min(a.workers,len(cpus));manifest=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
    assert sha(a.heldout)==manifest['heldout_sha256'],'Wrong heldout split'
    sources=[];baseline_jobs=[]
    for seed,path in enumerate(a.checkpoints,1):
        path=path.resolve();sources.append(dict(seed=seed,checkpoint_sha256=sha(path)))
        for memory in MEMORY:
            baseline_jobs.append(dict(seed=seed,checkpoint=str(path),heldout=str(a.heldout.resolve()),
                                      memory=memory,kind='none',region=None,core=cpus[len(baseline_jobs)%workers]))
    baselines=execute(baseline_jobs,workers,a.work)
    lookup={(r['seed'],r['memory']):r for r in baselines}
    for seed,path in enumerate(a.checkpoints,1):
        record=path.with_name('training.json')
        if record.exists():
            trained=json.loads(record.read_text())
            expected=trained['test_nll']
            assert abs(lookup[seed,'cold_adaptive']['nll']-expected)<2e-12
            sources[seed-1].update(expected_test_nll=expected,training_targets=trained['tokens'],parameters=trained['parameters'])
    jobs=[];skipped=[]
    for seed,path in enumerate(a.checkpoints,1):
        base=lookup[seed,'cold_adaptive']
        for kind in KINDS:
            for region,name in enumerate(core.NAMES):
                if kind=='semantic_residual' and region<2:continue
                if kind=='relay_context' and base['composition'][name]['singleton_relays']==0:
                    skipped.append(dict(seed=seed,kind=kind,region=region,reason='No relay publications in this stream'));continue
                jobs.append(dict(seed=seed,checkpoint=str(path.resolve()),heldout=str(a.heldout.resolve()),
                                 memory='cold_adaptive',kind=kind,region=region,core=cpus[len(jobs)%workers]))
        # Memory-sensitive output paths under each alternative memory contract.
        for memory in MEMORY[1:]:
            for region in (15,17):
                jobs.append(dict(seed=seed,checkpoint=str(path.resolve()),heldout=str(a.heldout.resolve()),
                                 memory=memory,kind='neural_output',region=region,core=cpus[len(jobs)%workers]))
    measured=execute(jobs,workers,a.work)
    compact=[]
    for row in sorted(measured,key=lambda r:(r['seed'],r['memory'],r['kind'],r['region'])):
        base=lookup[row['seed'],row['memory']]
        changed=[name for name in core.NAMES if any(row['trace'][name][key]!=base['trace'][name][key] for key in TRACE_KEYS)]
        if row['kind']!='whole_source':assert not changed, (row['kind'],changed)
        r=row['region'];name=core.NAMES[r]
        compact.append({k:row[k] for k in ('seed','memory','kind','region','nll','targets','seconds','fitted_state_unchanged','source_state_unchanged','memory_before','memory_after')}|
                       dict(name=name,baseline_nll=base['nll'],delta_nll=row['nll']-base['nll'],
                            event_stream_preserved=not changed,changed_event_regions=changed,
                            intervention_applications=row['trace'][name]['intervention_applications'],
                            block_delta_nll=[b['nll']-c['nll'] for b,c in zip(row['blocks'],base['blocks'])]))
    for source,path in zip(sources,a.checkpoints):assert sha(path)==source['checkpoint_sha256']
    result=dict(base_commit=BASE_COMMIT,parameters=1031936,heldout_sha256=sha(a.heldout),
                validation_offset=32768,test_targets=78771,sources=sources,
                protocol='Frozen neural/semantic parameters and dictionaries; fresh stream; one intervention per copy; no retraining',
                baselines=sorted(baselines,key=lambda r:(r['seed'],r['memory'])),interventions=compact,skipped=skipped)
    write(a.output,result)
    print(f"Completed {len(compact)} interventions and {len(baselines)} baseline parity checks: {a.output}",flush=True)


if __name__=='__main__':main()
