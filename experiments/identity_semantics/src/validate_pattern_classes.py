"""Matched class-recognition cost, frozen downstream effects and continuation."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model
from validate_content_experts import TRAIN,HELD,generation

def frozen_score(model,data,classes=None,diagnostic=False):
    m=model.evaluation_copy(diagnostic=diagnostic)
    try:
        if classes is not None:m.configure_pattern_classes(classes)
        fitted=m.pattern_state(fitted=True).copy();weights=m.flat.copy()
        loss,logits=m.run(data[:-1],data[1:],logits=True)
        m.finish_document()
        np.testing.assert_array_equal(fitted,m.pattern_state(fitted=True))
        np.testing.assert_array_equal(weights,m.flat)
        assert np.isfinite(logits).all()
        return dict(nll=float(loss),classes=m.pattern_metrics(),completion=m.completion_metrics()),logits
    finally:m.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=int,default=1024);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',type=Path,default=ROOT/'results/pattern_class_validation.json')
    args=p.parse_args()
    if min(args.targets,args.repeats)<1:p.error('positive targets and repeats required')
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    warmup=1024;data=np.frombuffer(TRAIN*((max(args.targets,warmup)+1)//len(TRAIN)+1),np.uint8)
    models={name:fresh_model(seed=7) for name in ('exact_bindings','pattern_classes')}
    models['exact_bindings'].configure_pattern_classes(False)
    result=dict(seed=7,cpu=cpu,warmup_targets=warmup,targets_per_round=args.targets,repeats=args.repeats,
                lr=.04,lp=.04/3,train_sha256=hashlib.sha256(data).hexdigest(),
                heldout_sha256=hashlib.sha256(HELD).hexdigest(),
                workload='Repeated-sentence mechanism/timing smoke and separate synthetic held-out prose.',runs={})
    def checkpoint():
        args.output.parent.mkdir(exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for name,m in models.items():
            m.run(data[:warmup],data[1:warmup+1],mode='periodic',lr=.04,lp=.04/3)
            result['runs'][name]=dict(parameters=m.parameter_count,rounds=[])
        for repeat in range(args.repeats):
            for name in list(models)[::1 if repeat%2==0 else -1]:
                m=models[name];start=time.perf_counter();cpu_start=time.process_time()
                loss=m.run(data[:args.targets],data[1:args.targets+1],mode='periodic',lr=.04,lp=.04/3)
                seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu_start
                assert np.isfinite(loss) and np.isfinite(m.flat).all()
                row=dict(nll=float(loss),seconds=seconds,cpu_seconds=cpu_seconds,
                         bytes_per_second=args.targets/seconds,cpu_bytes_per_second=args.targets/cpu_seconds)
                result['runs'][name]['rounds'].append(row);print(name,repeat+1,row,flush=True);checkpoint()
        for name,m in models.items():
            m.finish_document(mode='periodic',lr=.04,lp=.04/3)
            row=result['runs'][name]
            for field in ('bytes_per_second','cpu_bytes_per_second'):row['median_'+field]=float(np.median([v[field] for v in row['rounds']]))
            row.update(patterns=m.pattern_metrics(),composition=m.composition_metrics(),
                       content=m.content_expert_metrics(),local=m.local_credit_metrics(),neural=m.neural_metrics(),
                       lace=m.lace_metrics(),messages=m.message_metrics(),completion=m.completion_metrics(),teaching=m.relay_teaching_status())
            for field,key in [('content','missing_producers'),('local','missing_producers'),('neural','missing_live_dependencies'),
                              ('lace','missing_parent'),('lace','overwritten_credit'),('messages','missing_feedback_producers'),
                              ('completion','expired_semantic_handles'),('completion','semantic_capacity_prunes'),('teaching','expired_producers')]:
                assert all(v[key]==0 for v in row[field].values()),(name,field,key)
            if m.pattern_classes:
                assert sum(v['promoted'] for v in row['patterns'].values())>0
                assert sum(v['class_hits'] for v in row['patterns'].values())>0
            fingerprint=m.fingerprint();held=np.frombuffer(HELD,np.uint8)
            row['heldout'],production=frozen_score(m,held)
            _,diagnostic=frozen_score(m,held,diagnostic=True);np.testing.assert_array_equal(production,diagnostic)
            row['diagnostic_parity']=True
            if m.pattern_classes:
                row['class_ablation_heldout'],ablated=frozen_score(m,held,classes=False)
                row['class_ablation_delta_nll']=row['class_ablation_heldout']['nll']-row['heldout']['nll']
                row['class_ablation_logit_rms']=float(np.sqrt(np.mean((production.astype(float)-ablated)**2)))
                # Repeat-pattern contribution is reported separately from unseen prose.
                familiar=data[:513];row['familiar'],base=frozen_score(m,familiar)
                row['class_ablation_familiar'],removed=frozen_score(m,familiar,classes=False)
                row['familiar_delta_nll']=row['class_ablation_familiar']['nll']-row['familiar']['nll']
                row['familiar_logit_rms']=float(np.sqrt(np.mean((base.astype(float)-removed)**2)))
            row['generation']=generation(m);assert m.fingerprint()==fingerprint
            snapshot=m.snapshot_copy()
            try:
                a=m.run(data[:64],data[1:65],mode='periodic',logits=True)[1]
                b=np.concatenate([snapshot.run(data[i:i+16],data[i+1:i+17],mode='periodic',logits=True)[1] for i in range(0,64,16)])
                np.testing.assert_array_equal(a,b);assert m.fingerprint()==snapshot.fingerprint()
                row['exact_chunked_continuation']=True
            finally:snapshot.close()
            checkpoint()
    finally:
        for m in models.values():m.close()
    print(args.output,flush=True)

if __name__=='__main__':main()
