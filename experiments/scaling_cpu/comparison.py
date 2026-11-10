"""Calibrate, launch, resume and summarize matched SHoEbox scaling runs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from shoebox_runtime import Architecture,Model,evaluate
from shoebox_runtime.model import atomic_json,fixed_transports
from prepare_data import verify,sha

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'configs/comparison.json'

def config():return json.loads(CONFIG.read_text())
def runtime_hash():
    paths=sorted(p for p in (ROOT/'shoebox_runtime').rglob('*') if p.suffix in ('.py','.cpp','.inc'))
    paths += [Path(__file__),ROOT/'configs/legacy_policy.json',ROOT/'configs/legacy_fixed_64.npz']
    h=hashlib.sha256()
    for p in paths:h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
def architecture(arm):
    item=config()['arms'][arm];a=Architecture(item['width'],item['experts'])
    if a.parameter_count!=item['parameters']:raise ValueError('Parameter budget changed')
    return a
def pin(cpu):
    if cpu is not None:
        if cpu not in os.sched_getaffinity(0):raise ValueError(f'CPU {cpu} is unavailable')
        os.sched_setaffinity(0,{cpu})

def schedule(pos,n,peak,horizon):
    c=config();f=(pos+n/2)/horizon
    warm=min(1.,max(.1,f/c['warmup_fraction']))
    cool=1. if f<=c['cooldown_start_fraction'] else max(c['cooldown_floor'],1-(1-c['cooldown_floor'])*(f-c['cooldown_start_fraction'])/(1-c['cooldown_start_fraction']))
    mature=np.clip((f-c['maturation_start_fraction'])/(c['cooldown_start_fraction']-c['maturation_start_fraction']),0.,1.)
    formation=1.-.9*mature
    a1=1. if cool==1 else 1.-.9*(1-cool)/(1-c['cooldown_floor'])
    lr=peak*warm*cool
    return dict(lr=float(lr),lp=float(lr*c['local_lr_ratio']),route=float(.002*formation*cool),
                prototype_scale=float(formation*cool),center_rate=float(.001*formation*cool),mean_rate=float(.001*formation*cool),
                task_route_lr_scale=float(1./cool),a1_prototype_scale=float(a1),a1_decoder_scale=float(a1))

def apply_schedule(m,s,peak):
    m.configure_adaptation(prototype_scale=s['prototype_scale'],center_rate=s['center_rate'],mean_rate=s['mean_rate'])
    m.configure_training_policy(np.zeros(18,np.int32),s['task_route_lr_scale'])
    # Global prototype schedule already follows cooldown, so avoid applying
    # the same cooldown factor twice through the optional A1 link.
    m.configure_a1_learning(dict(prototype_scale=s['a1_prototype_scale'],follow_cooldown=0,reference_lr=peak,decoder_scale=s['a1_decoder_scale']))
    return {k:s[k] for k in ('lr','lp','route')}

def load_fixed(directory,width):
    path=Path(directory)/f'fixed_d{width}.npz'
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def prepare_fixed(data_dir,work):
    data_dir=Path(data_dir);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    meta=verify(data_dir);input_hash=meta['splits']['calibration']['sha256']
    data=np.fromfile(data_dir/'calibration.bin',np.uint8)[:100001]
    for d in sorted({v['width'] for v in config()['arms'].values()}):
        path=work/f'fixed_d{d}.npz';record=work/f'fixed_d{d}.json'
        if path.exists() and record.exists():
            old=json.loads(record.read_text())
            if old['sha256']!=sha(path) or old['calibration_sha256']!=input_hash:raise ValueError('Fixed calibration does not match data')
            continue
        fixed=fixed_transports(d)
        # Equal, training-only centering for all expert-count arms at a width.
        fixed['address_center'][:]=0
        with Model(Architecture(d,2),seed=0,fixed=fixed,config={'gain':0.,'no_returns':1.,'wake_threshold':0.}) as m:
            total=np.zeros((2,d),np.float64);count=0
            for start in range(0,len(data)-1,64):
                stop=min(len(data)-1,start+64);m.run(data[start:stop],data[start+1:stop+1]);total+=m.queries();count+=1
            fixed['address_center'][:]=total/count
        with path.open('wb') as f:np.savez_compressed(f,**fixed)
        atomic_json(record,dict(width=d,sha256=sha(path),calibration_sha256=input_hash,calibration_targets=100000,
                               method='Frozen two-expert reference, seed 0, no recall contribution, sample EC queries every 64 bytes; parameters never learn.'))

def calibration_arm(data_dir,work,arm,cpu=None):
    pin(cpu);c=config();meta=verify(data_dir);work=Path(work);dest=work/f'calibration_{arm}.json'
    inputs=dict(runtime_sha256=runtime_hash(),config_sha256=sha(CONFIG),calibration_sha256=meta['splits']['calibration']['sha256'],validation_sha256=meta['splits']['validation']['sha256'],fixed_sha256=sha(work/f'fixed_d{architecture(arm).width}.npz'))
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['inputs']!=inputs:raise ValueError('Existing calibration is incompatible')
        return old
    data=np.fromfile(Path(data_dir)/'calibration.bin',np.uint8)[100000:100000+c['pilot_targets']+1]
    val=np.fromfile(Path(data_dir)/'validation.bin',np.uint8)[:c['pilot_validation_targets']+1]
    results=[]
    for lr in c['lr_candidates']:
        with Model(architecture(arm),seed=c['pilot_seed'],fixed=load_fixed(work,architecture(arm).width)) as m:
            start=time.perf_counter();train_seconds=0.;history=[];scores=[]
            initial=evaluate(m,val)['nll']
            m.observe_updates()
            for pos in range(0,c['pilot_targets'],10000):
                n=min(10000,c['pilot_targets']-pos);s=schedule(pos,n,lr,c['pilot_targets']);rates=apply_schedule(m,s,lr)
                t=time.perf_counter();loss=m.run(data[pos:pos+n],data[pos+1:pos+n+1],mode='periodic',**rates);train_seconds+=time.perf_counter()-t
                history.append(dict(targets=pos+n,nll=loss,lr=s['lr']))
                if pos+n in (400000,450000,500000):
                    ev=evaluate(m,val);scores.append({**ev,'evaluation_targets':ev['targets'],'targets':pos+n})
            if not np.isfinite(m.flat).all() or not np.isfinite(m.extra_flat).all():raise FloatingPointError('Nonfinite pilot parameters')
            observed=m.update_totals()
            score=float(np.mean([x['nll'] for x in scores if x['targets']>=450000]))
            item=dict(peak_lr=lr,score=score,initial_nll=initial,validation=scores,history=history,
                      actual_update_totals=observed.tolist(),training_seconds=train_seconds,bytes_per_second=c['pilot_targets']/train_seconds,
                      wall_seconds=time.perf_counter()-start,final_parameters_finite=True)
            results.append(item);print(json.dumps(dict(event='pilot',arm=arm,peak_lr=lr,score=score,bytes_per_second=item['bytes_per_second'])),flush=True)
    selected=min(results,key=lambda r:r['score'])
    result=dict(arm=arm,inputs=inputs,pilot_targets=c['pilot_targets'],pilot_seed=c['pilot_seed'],selected_peak_lr=selected['peak_lr'],selection='Mean validation at 450K and 500K; test not opened.',trials=results)
    atomic_json(dest,result);return result

def calibrate(data_dir,work,workers=3):
    prepare_fixed(data_dir,work);cpus=sorted(os.sched_getaffinity(0));workers=min(workers,len(cpus),len(config()['arms']))
    if workers<1:raise ValueError('workers must be positive')
    arms=list(config()['arms'])
    def job(pair):
        i,arm=pair;log=Path(work)/f'calibration_{arm}.log'
        with log.open('w') as output:
            subprocess.run([sys.executable,str(__file__),'calibrate-arm','--data',str(data_dir),'--work',str(work),'--arm',arm,'--cpu',str(cpus[i%workers])],stdout=output,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(job,enumerate(arms)))
    results={arm:json.loads((Path(work)/f'calibration_{arm}.json').read_text()) for arm in arms}
    selection=dict(runtime_sha256=runtime_hash(),config_sha256=sha(CONFIG),data_manifest_sha256=sha(Path(data_dir)/'manifest.json'),
                   selected_peak_lr={arm:r['selected_peak_lr'] for arm,r in results.items()},
                   calibration_sha256={arm:sha(Path(work)/f'calibration_{arm}.json') for arm in arms},
                   fixed_sha256={str(d):sha(Path(work)/f'fixed_d{d}.npz') for d in sorted({architecture(arm).width for arm in arms})},
                   status='Short pilot selection; full 25M comparison has not been run.',test_opened=False)
    atomic_json(Path(work)/'selection.json',selection);print(json.dumps(selection,indent=2))

def protocol(data_dir,work,arm,seed):
    c=config();meta=verify(data_dir);selection=json.loads((Path(work)/'selection.json').read_text())
    if selection['runtime_sha256']!=runtime_hash() or selection['config_sha256']!=sha(CONFIG) or selection['data_manifest_sha256']!=sha(Path(data_dir)/'manifest.json'):raise ValueError('Calibration and requested protocol differ')
    if seed not in c['seeds']:raise ValueError('Use a preregistered comparison seed')
    for name,digest in selection['calibration_sha256'].items():
        if sha(Path(work)/f'calibration_{name}.json')!=digest:raise ValueError('Calibration record changed')
    d=architecture(arm).width
    if selection['fixed_sha256'][str(d)]!=sha(Path(work)/f'fixed_d{d}.npz'):raise ValueError('Fixed transforms changed')
    result=dict(experiment=c['name'],arm=arm,seed=seed,architecture=architecture(arm).as_dict(),parameters=architecture(arm).parameter_count,
                horizon=c['targets'],peak_lr=selection['selected_peak_lr'][arm],config_sha256=sha(CONFIG),runtime_sha256=runtime_hash(),
                train_sha256=meta['splits']['train']['sha256'],validation_sha256=meta['splits']['validation']['sha256'],
                fixed_sha256=selection['fixed_sha256'][str(d)],selection_sha256=sha(Path(work)/'selection.json'),test_policy=c['test_policy'])
    return result

def run(data_dir,work,output,arm,seed,resume=False,max_targets=None,cpu=None):
    pin(cpu);c=config();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=c['targets'] if max_targets is None else min(c['targets'],max_targets)
    if target<0 or target%c['chunk_targets']:raise ValueError('Pause targets must be a nonnegative multiple of chunk_targets, preserving the exact schedule')
    p=protocol(data_dir,work,arm,seed);protocol_hash=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    last=output/'last.npz';initial=output/'initial.npz'
    if last.exists():
        if not resume:raise ValueError('Run exists; use --resume')
        with np.load(last,allow_pickle=False) as z:record=json.loads(str(z['metadata_json']))
        if record['protocol_hash']!=protocol_hash:raise ValueError('Refusing to resume with changed data, runtime, schedule or selection')
        m=Model.load(last)
        if m.observations!=record['targets']:raise ValueError('Checkpoint target counter mismatch')
    else:
        if resume:raise ValueError('No checkpoint to resume')
        if initial.exists():raise ValueError('Partial initialization exists; select a different output directory')
        m=Model(architecture(arm),seed=seed,fixed=load_fixed(work,architecture(arm).width))
        record=dict(protocol_hash=protocol_hash,targets=0,history=[],validation=[],best=None,training_seconds=0.,run_wall_seconds=0.,status='initialized')
        atomic_json(output/'protocol.json',p);m.save(initial,record)
    if target<record['targets']:m.close();raise ValueError('Requested target precedes checkpoint')
    train=np.memmap(Path(data_dir)/'train.bin',mode='r',dtype=np.uint8)
    val=np.fromfile(Path(data_dir)/'validation.bin',np.uint8)
    wall_start=time.perf_counter();previous_wall=record.get('run_wall_seconds',0.)
    m.observe_updates(record.get('actual_update_totals'))
    def checkpoint(evaluation=False):
        record['targets']=m.observations
        record['actual_update_totals']=m.update_totals().tolist()
        record['credit_totals']=m.credit_totals()
        if evaluation:
            before=m.fingerprint();t=time.perf_counter();ev=evaluate(m,val)
            if m.fingerprint()!=before:raise RuntimeError('Evaluation mutated training state')
            seen=None
            if m.observations>=131072:seen=evaluate(m,train[:131073])['nll']
            item={**ev,'targets':m.observations,'evaluation_targets':ev['targets'],'seen_training_probe_nll':seen,'evaluation_seconds':time.perf_counter()-t}
            record['validation'].append(item)
            if record['best'] is None or ev['nll']<record['best']['nll']:
                record['best']=dict(targets=m.observations,nll=ev['nll']);m.save(output/'best.npz',record)
        record['run_wall_seconds']=previous_wall+time.perf_counter()-wall_start
        record['status']='completed' if m.observations==c['targets'] else 'paused'
        m.save(last,record);atomic_json(output/'metrics.json',record)
        atomic_json(output/'progress.json',dict(arm=arm,seed=seed,targets=m.observations,horizon=c['targets'],status=record['status'],best=record['best'],training_seconds=record['training_seconds']))
    try:
        if not record['validation']:checkpoint(evaluation=True)
        while m.observations<target:
            pos=m.observations;n=min(c['chunk_targets'],target-pos,c['validation_interval']-pos%c['validation_interval'])
            s=schedule(pos,n,p['peak_lr'],c['targets']);rates=apply_schedule(m,s,p['peak_lr'])
            t=time.perf_counter();loss=m.run(train[pos:pos+n],train[pos+1:pos+n+1],mode='periodic',**rates);seconds=time.perf_counter()-t
            if not np.isfinite(m.flat).all() or not np.isfinite(m.extra_flat).all():raise FloatingPointError('Nonfinite parameters')
            record['training_seconds']+=seconds;record['targets']=m.observations
            record['history'].append(dict(targets=m.observations,nll=loss,chunk_targets=n,seconds=seconds,schedule=s))
            atomic_json(output/'progress.json',dict(arm=arm,seed=seed,targets=m.observations,horizon=c['targets'],status='running',training_seconds=record['training_seconds'],last_nll=loss))
            if m.observations%c['validation_interval']==0 or m.observations==target:
                checkpoint(evaluation=True)
                if m.observations%5_000_000==0 or m.observations==target:
                    fp=m.fingerprint();probe=val[:c['diagnostic_targets']+1]
                    observed=evaluate(m,probe,diagnostics=True,route_trace_stride=64)
                    mean_y=evaluate(m,probe,mean_y=True)['nll'];no_recall=evaluate(m,probe,block_recall=True)['nll']
                    prior=sorted(output.glob('diagnostics_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
                    churn=None
                    if prior:
                        old=json.loads(prior[-1].read_text())['validation_probe']['routing_trace']
                        churn=np.mean(np.asarray(old)!=np.asarray(observed['routing_trace']),axis=0).tolist()
                    if m.fingerprint()!=fp:raise RuntimeError('Diagnostics mutated training state')
                    atomic_json(output/f'diagnostics_{m.observations}.json',dict(targets=m.observations,validation_probe=observed,
                        mean_y_delta_nll=mean_y-observed['nll'],blocked_recall_delta_nll=no_recall-observed['nll'],
                        routing_churn_from_previous_probe=churn,training_regional_totals=m.regional_totals(),
                        actual_update_totals=m.update_totals().tolist(),credit_totals=m.credit_totals()))
                if m.observations==20_000_000:m.save(output/'stable_20m.npz',record)
                print(json.dumps(dict(event='checkpoint',arm=arm,seed=seed,targets=m.observations,validation_nll=record['validation'][-1]['nll'],best=record['best'])),flush=True)
        record['last_invocation_wall_seconds']=time.perf_counter()-wall_start
        record['final_fingerprint']=m.fingerprint();checkpoint(evaluation=False)
        return record
    finally:m.close()

def launch(data_dir,work,output,workers=3,resume=False,dry_run=False):
    c=config();cpus=sorted(os.sched_getaffinity(0));workers=min(workers,len(cpus))
    if workers<1:raise ValueError('workers must be positive')
    jobs=[(arm,seed) for arm in c['arms'] for seed in c['seeds']]
    if dry_run:
        for arm,seed in jobs:print(json.dumps(protocol(data_dir,work,arm,seed)))
        return
    # A fixed worker owns one core and executes its queue sequentially.
    queues=[jobs[i::workers] for i in range(workers)]
    def worker(i):
        for arm,seed in queues[i]:
            dest=Path(output)/f'{arm}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(__file__),'run','--data',str(data_dir),'--work',str(work),'--output',str(dest),'--arm',arm,'--seed',str(seed),'--cpu',str(cpus[i])]
            if resume and (dest/'last.npz').exists():cmd+=['--resume']
            with (dest/'console.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(worker,range(workers)))
    summarize(output)

def summarize(output):
    rows=[]
    for path in sorted(Path(output).glob('*/metrics.json')):
        r=json.loads(path.read_text());p=json.loads((path.parent/'protocol.json').read_text())
        scores={v['targets']:v['nll'] for v in r['validation']}
        final=[scores[t] for t in (23_000_000,24_000_000,25_000_000) if t in scores]
        rows.append(dict(arm=p['arm'],seed=p['seed'],targets=r['targets'],status=r['status'],
                         final_window_nll=float(np.mean(final)) if len(final)==3 else None,
                         endpoint_nll=r['validation'][-1]['nll'] if r['validation'] else None,
                         best=r['best'],training_seconds=r['training_seconds'],run_wall_seconds=r.get('run_wall_seconds')))
    grouped={}
    for arm in config()['arms']:
        items=[r for r in rows if r['arm']==arm and r['final_window_nll'] is not None and r['status']=='completed']
        if {r['seed'] for r in items}==set(config()['seeds']):
            grouped[arm]=dict(mean_final_window_nll=float(np.mean([r['final_window_nll'] for r in items])),
                mean_endpoint_nll=float(np.mean([r['endpoint_nll'] for r in items])),total_training_seconds=sum(r['training_seconds'] for r in items))
    paired={}
    control={r['seed']:r for r in rows if r['arm']=='control_5m' and r['final_window_nll'] is not None}
    for arm in grouped:
        items=[r for r in rows if r['arm']==arm and r['seed'] in control]
        if arm!='control_5m' and len(items)==len(config()['seeds']):
            deltas={str(r['seed']):r['final_window_nll']-control[r['seed']]['final_window_nll'] for r in items}
            paired[arm]=dict(by_seed=deltas,mean_delta_nll=float(np.mean(list(deltas.values()))))
    result=dict(runs=rows,complete_arm_means=grouped,paired_deltas_against_control=paired,test_evaluated=False)
    atomic_json(Path(output)/'summary.json',result);print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('calibrate','calibrate-arm','run','launch','summary','prepare-fixed'):
        q=sub.add_parser(name);q.add_argument('--data',type=Path,default=ROOT/'data/comparison');q.add_argument('--work',type=Path,default=ROOT/'calibration');q.add_argument('--output',type=Path,default=ROOT/'runs')
        if name in ('calibrate','launch'):q.add_argument('--workers',type=int,default=3)
        if name in ('calibrate-arm','run'):q.add_argument('--arm',choices=list(config()['arms']),required=True);q.add_argument('--cpu',type=int)
        if name=='run':q.add_argument('--seed',type=int,required=True);q.add_argument('--max-targets',type=int)
        if name in ('run','launch'):q.add_argument('--resume',action='store_true')
        if name=='launch':q.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.command=='calibrate':calibrate(a.data,a.work,a.workers)
    elif a.command=='calibrate-arm':calibration_arm(a.data,a.work,a.arm,a.cpu)
    elif a.command=='prepare-fixed':prepare_fixed(a.data,a.work)
    elif a.command=='run':run(a.data,a.work,a.output,a.arm,a.seed,a.resume,a.max_targets,a.cpu)
    elif a.command=='launch':launch(a.data,a.work,a.output,a.workers,a.resume,a.dry_run)
    elif a.command=='summary':summarize(a.output)

if __name__=='__main__':main()
