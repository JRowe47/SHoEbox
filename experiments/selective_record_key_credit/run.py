"""Write-once fresh development comparison. Audit access is a separate prerequisite."""
import argparse,hashlib,json,os,resource,time,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from .backend import Model,ROOT,library
from .tasks import manifest,cases,worlds
from .stream import new_cursor,advance
from .diagnose import evaluate

ARMS=('e8','a8','a9')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    # Lock all runtime/research dependencies, not just direct imports. No data/binaries.
    paths=[ROOT/'shoe_multiscale.py', *ROOT.glob('multiscale/**/*.py'), *ROOT.glob('multiscale/**/*.hpp'), *ROOT.glob('multiscale/**/*.cpp'), *ROOT.glob('multiscale/**/*.inc'), *ROOT.glob('experiments/**/*.py'), *ROOT.glob('experiments/**/*.hpp'), *ROOT.glob('experiments/**/*.cpp'), *ROOT.glob('experiments/**/*.inc'), ROOT/'reports/exp0305a8/A9_REGISTRATION.md', ROOT/'reports/exp0305a9/A9_F1_REGISTRATION.md', ROOT/'reports/exp0305a9/EXECUTION_LOCK.md', ROOT/'reports/exp0305a9/DIAGNOSIS_PLAN.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.is_file() and not (p.parent.name=='selective_record_key_credit' and p.name in ('report.py','write_report.py','verify_evidence.py','analyze_matching.py','trace_geometry.py','trace_retention.py','conflicts.py','account_correctness.py'))}

def fit(m,cursor,budget,folder,stem):
    initial=m.inventory();cpu=time.process_time();wall=time.perf_counter();curve=[];total_loss=0
    while cursor['bytes']<budget:
        amount=min(500000,budget-cursor['bytes']);cursor,loss=advance(m,cursor,amount);total_loss+=loss
        curve.append(dict(bytes=cursor['bytes'],cpu_seconds=time.process_time()-cpu,nll=total_loss/max(1,cursor['targets'])))
    result=dict(cursor=cursor,initial_inventory=initial,inventory=m.inventory(),cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,curve=curve)
    raw=m.dumps();(folder/(stem+'.qr')).write_bytes(raw);write(folder/(stem+'.cursor.json'),cursor)
    result['checkpoint_sha256']=hashlib.sha256(raw).hexdigest();result['cursor_sha256']=sha(folder/(stem+'.cursor.json'))
    return result

def job(args):
    seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)]});out=Path(folder)/str(seed);out.mkdir()
    row=dict(seed=seed,arms={});start=time.process_time()
    for arm in ARMS:
        policy=arm;extra=7670
        with Model(policy,parameters=500000-extra,seed=seed,kind='native') as base:
            warm=fit(base,new_cursor(True,False,arm),250000,out,arm+'.warmup')
            with Model(policy,seed=seed,kind='paired_local') as m:
                m.adopt(base);m.reset();r=fit(m,new_cursor(False,True,arm),2000000,out,arm);r['warmup']=warm
                cpu=time.process_time();r['development']={}
                for condition in ('pad0','repeat16','diverse16','compete16','overwrite16','repeat64','repeat256','diverse64','diverse256'):
                    r['development'][condition]=evaluate(m,cases('dev',128,condition=condition))
                r['binary_regression']=evaluate(m,cases('dev',128,vocab=2))
                r['evaluation_cpu_seconds']=time.process_time()-cpu
                row['arms'][arm]=r;write(out/'results.json',row)
                print(seed,arm,{c:round(v['exact'],4) for c,v in r['development'].items()},flush=True)
    row['worker_cpu_seconds']=time.process_time()-start;row['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(out/'results.json',row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prefit',type=Path,required=True);a=p.parse_args()
    pre=json.loads(a.prefit.read_text());assert pre['passed'] and pre['source_hashes']==sources()
    committed=subprocess.check_output(['git','show','HEAD:reports/exp0305a9/prefit/prefit.json'],cwd=ROOT)
    assert committed==a.prefit.read_bytes(),'Passing pre-fit receipt must be committed before fitting'
    prefit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert pre['world_file_sha256']==sha(a.prefit.parent/'pre_fit_worlds.json')
    assert pre['amendment']=='A9-F1'
    assert pre['f1_allocation_sha256']==sha(a.prefit.parent/'f1_allocation.json')
    a.out.mkdir(parents=True,exist_ok=False)
    plan=dict(utc=datetime.now(timezone.utc).isoformat(),seeds=[1,7,19],arms=ARMS,parameters=500000,warmup=250000,continuation=2000000,
        fitting_bytes=20250000,fixture='three distinct values; balanced changed key by world+epoch',order_seeds=[10801,10802],folds=manifest(),source_hashes=sources(),cpus=sorted(os.sched_getaffinity(0))[:3],audit_opened=False)
    plan['prefit_sha256']=sha(a.prefit)
    plan['amendment']='A9-F1';plan['f1_allocation_sha256']=pre['f1_allocation_sha256']
    plan['prefit_commit']=prefit_commit
    assert pre['stream_sha256']==sha(a.prefit.parent/'streams.json')
    plan['per_arm_stream_digests']=pre['per_arm_stream_digests'];plan['stream_sha256']=pre['stream_sha256']
    write(a.out/'plan.json',plan)
    write(a.out/'binary_dev_worlds.json',[json.loads(w.group_key) for w in worlds('dev',128,vocab=2)])
    write(a.out/'worlds.json',{p:[json.loads(w.group_key) for w in worlds(p,n)] for p,n in [('train',512),('dev',128),('audit',256)]})
    library('e8');library('a8');library('a9')
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(job,[(s,plan,str(a.out)) for s in plan['seeds']]))
    assert plan['source_hashes']==sources(),'source changed during fit'
    write(a.out/'results.json',dict(plan=plan,rows=rows))
    write(a.out/'selection.json',dict(locked=True,audit_opened=False,checkpoints={str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.suffix in ('.qr','.json') and p.name not in ('selection.json',)}))
if __name__=='__main__':main()
