"""Matched 1M-reference timing, paired recall interventions and continuation."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model
from validate_content_experts import TRAIN,HELD

def score(model,data,ablation=0,diagnostic=False):
    m=model.evaluation_copy(memory='warm_frozen',diagnostic=diagnostic)
    try:
        m.ablate(ablation);before=m.evaluation_memory_info()['store_digest'];weights=m.flat.copy()
        loss,logits=m.run(data[:-1],data[1:],logits=True)
        assert before==m.evaluation_memory_info()['store_digest']
        np.testing.assert_array_equal(weights,m.flat);assert np.isfinite(logits).all()
        return dict(nll=float(loss),paired=m.paired_memory_metrics()),logits
    finally:m.close()

def repeatable_timing(args):
    """Restore the identical warmed state before each timed training segment."""
    warmup=1024;data=np.frombuffer(TRAIN*((max(args.targets,warmup)+1)//len(TRAIN)+1),np.uint8)
    models={name:fresh_model(seed=7) for name in ('independent_transitions','paired_episodes')}
    models['independent_transitions'].configure_paired_memory(False)
    result=dict(seed=7,cpu=args.cpu,warmup_targets=warmup,targets_per_round=args.targets,repeats=args.repeats,
                protocol='Each timed segment starts from an exact copy of its original warmed state; alternating order. Copying, compilation and warmup excluded.',
                train_sha256=hashlib.sha256(data).hexdigest(),lr=.04,lp=.04/3,runs={})
    try:
        for name,m in models.items():
            m.run(data[:warmup],data[1:warmup+1],mode='periodic',lr=.04,lp=.04/3)
            result['runs'][name]=dict(parameters=m.parameter_count,rounds=[])
        for repeat in range(args.repeats):
            for name in list(models)[::1 if repeat%2==0 else -1]:
                copy=models[name].snapshot_copy()
                try:
                    start=time.perf_counter();cpu=time.process_time()
                    loss=copy.run(data[:args.targets],data[1:args.targets+1],mode='periodic',lr=.04,lp=.04/3)
                    seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
                    rounds=result['runs'][name]['rounds']
                    if rounds:assert float(loss)==rounds[0]['nll']
                    row=dict(nll=float(loss),seconds=seconds,cpu_seconds=cpu_seconds,
                             bytes_per_second=args.targets/seconds,cpu_bytes_per_second=args.targets/cpu_seconds)
                    rounds.append(row);print(name,repeat+1,row,flush=True)
                finally:copy.close()
        for row in result['runs'].values():
            for field in ('bytes_per_second','cpu_bytes_per_second'):
                row['median_'+field]=float(np.median([v[field] for v in row['rounds']]))
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        for m in models.values():m.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=int,default=1024);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--cpu',type=int,default=max(os.sched_getaffinity(0)))
    p.add_argument('--repeatable-timing',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results/paired_memory_validation.json');args=p.parse_args()
    if min(args.targets,args.repeats)<1:p.error('positive targets and repeats required')
    os.sched_setaffinity(0,{args.cpu});warmup=1024
    if args.repeatable_timing:return repeatable_timing(args)
    data=np.frombuffer(TRAIN*((max(args.targets,warmup)+1)//len(TRAIN)+1),np.uint8)
    models={name:fresh_model(seed=7) for name in ('independent_transitions','paired_episodes')}
    models['independent_transitions'].configure_paired_memory(False)
    result=dict(seed=7,cpu=args.cpu,warmup_targets=warmup,targets_per_round=args.targets,repeats=args.repeats,
                lr=.04,lp=.04/3,train_sha256=hashlib.sha256(data).hexdigest(),heldout_sha256=hashlib.sha256(HELD).hexdigest(),
                workload='Repeated-sentence mechanism/timing smoke and separate synthetic held-out prose.',runs={})
    def save():
        args.output.parent.mkdir(exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for name,m in models.items():
            m.run(data[:warmup],data[1:warmup+1],mode='periodic',lr=.04,lp=.04/3)
            result['runs'][name]=dict(parameters=m.parameter_count,rounds=[])
        for repeat in range(args.repeats):
            for name in list(models)[::1 if repeat%2==0 else -1]:
                m=models[name];start=time.perf_counter();cpu=time.process_time()
                loss=m.run(data[:args.targets],data[1:args.targets+1],mode='periodic',lr=.04,lp=.04/3)
                seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
                assert np.isfinite(loss) and np.isfinite(m.flat).all()
                row=dict(nll=float(loss),seconds=seconds,cpu_seconds=cpu_seconds,
                         bytes_per_second=args.targets/seconds,cpu_bytes_per_second=args.targets/cpu_seconds)
                result['runs'][name]['rounds'].append(row);print(name,repeat+1,row,flush=True);save()
        for name,m in models.items():
            m.finish_document(mode='periodic',lr=.04,lp=.04/3);row=result['runs'][name]
            for field in ('bytes_per_second','cpu_bytes_per_second'):
                row['median_'+field]=float(np.median([v[field] for v in row['rounds']]))
            row.update(paired=m.paired_memory_metrics(),memory=m.evaluation_memory_info(),
                       paired_snapshot_bytes=int(m.paired_memory_state().size),
                       local=m.local_credit_metrics(),content=m.content_expert_metrics(),neural=m.neural_metrics(),
                       lace=m.lace_metrics(),messages=m.message_metrics(),completion=m.completion_metrics(),teaching=m.relay_teaching_status())
            for field,key in [('content','missing_producers'),('local','missing_producers'),('neural','missing_live_dependencies'),
                              ('lace','missing_parent'),('lace','overwritten_credit'),('messages','missing_feedback_producers'),
                              ('completion','expired_semantic_handles'),('completion','semantic_capacity_prunes'),('teaching','expired_producers')]:
                assert all(v[key]==0 for v in row[field].values()),(name,field,key)
            if m.paired_memory:
                assert row['paired']['admissions']>0 and row['paired']['hits']>0 and row['paired']['utility_credit']>0
            fingerprint=m.fingerprint();held=np.frombuffer(HELD,np.uint8)
            row['heldout'],production=score(m,held)
            _,diagnostic=score(m,held,diagnostic=True);np.testing.assert_array_equal(production,diagnostic)
            row['diagnostic_parity']=True
            if m.paired_memory:
                for label,sample in (('heldout',held),('familiar',data[:513])):
                    row[label],base=score(m,sample);row[label+'_no_returns'],removed=score(m,sample,3)
                    row[label+'_recall_delta_nll']=row[label+'_no_returns']['nll']-row[label]['nll']
                    row[label+'_recall_logit_rms']=float(np.sqrt(np.mean((base.astype(float)-removed)**2)))
            assert m.fingerprint()==fingerprint
            snapshot=m.snapshot_copy()
            try:
                a=m.run(data[:64],data[1:65],mode='periodic',logits=True)[1]
                b=np.concatenate([snapshot.run(data[i:i+16],data[i+1:i+17],mode='periodic',logits=True)[1] for i in range(0,64,16)])
                np.testing.assert_array_equal(a,b);assert m.fingerprint()==snapshot.fingerprint()
                row['exact_chunked_continuation']=True
            finally:snapshot.close()
            save()
    finally:
        for m in models.values():m.close()
    print(args.output,flush=True)

if __name__=='__main__':main()
