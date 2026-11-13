"""Matched constituent-expert smoke, throughput, frozen contributions and generation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model

TRAIN=(b'The river runs through the valley. The king crossed the bridge. '
       b'A bird flew over the water. The queen opened the gate. ')
HELD=(b'The traveler followed a narrow road beside the river. At the bridge, '
      b'a woman stopped to watch the birds. Beyond the gate stood a quiet garden. '
      b'After the rain, the valley filled with soft light and the children went outside. ')*4

def frozen_score(model,kind=None,region=None):
    m=model.evaluation_copy(diagnostic=kind is not None)
    try:
        if kind:m.configure_contribution(kind,region)
        before=m.flat.copy();fitted=m.composition_state(fitted=True).copy()
        data=np.frombuffer(HELD,np.uint8);loss,logits=m.run(data[:-1],data[1:],logits=True)
        assert np.isfinite(loss) and np.isfinite(logits).all()
        np.testing.assert_array_equal(before,m.flat)
        np.testing.assert_array_equal(fitted,m.composition_state(fitted=True))
        return float(loss),logits
    finally:m.close()

def generation(model):
    m=model.evaluation_copy();rng=np.random.default_rng(17);prompt=b'The river '
    try:
        before=m.flat.copy();last=m.run(np.frombuffer(prompt,np.uint8),logits=True)[1][-1]
        allowed=np.array([10,*range(32,127)]);output=[]
        for _ in range(96):
            scaled=last[allowed].astype(np.float64)/.8;prob=np.exp(scaled-scaled.max());prob/=prob.sum()
            byte=int(rng.choice(allowed,p=prob));output.append(byte)
            last=m.run(np.array([byte],np.uint8),np.array([0],np.uint8),logits=True)[1][-1]
            assert np.isfinite(last).all()
        np.testing.assert_array_equal(before,m.flat)
        return dict(prompt=prompt.decode(),text=bytes(output).decode(),temperature=.8,seed=17,
                    sampling='printable ASCII plus newline; 96 bytes')
    finally:m.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets',type=int,default=2048)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output',type=Path,default=ROOT/'results/content_expert_validation.json')
    args=parser.parse_args()
    if args.targets<1 or args.repeats<1:parser.error('positive targets/repeats required')
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    data=np.frombuffer(TRAIN*((max(args.targets,2048)+1)//len(TRAIN)+1),np.uint8)
    models={name:fresh_model(seed=7) for name in ('previous_policy','content_experts')}
    models['previous_policy'].configure_content_experts(False)
    result=dict(seed=7,cpu=cpu,warmup_targets=2048,targets_per_round=args.targets,repeats=args.repeats,
                lr=.04,lp=.04/3,train_sha256=hashlib.sha256(data).hexdigest(),
                heldout_sha256=hashlib.sha256(HELD).hexdigest(),
                workload='Repeated-sentence mechanism smoke and separate synthetic held-out prose; no trained checkpoint or corpus saved.',runs={})
    try:
        for name,m in models.items():
            m.run(data[:2048],data[1:2049],mode='periodic',lr=.04,lp=.04/3)
            result['runs'][name]=dict(parameters=m.parameter_count,rounds=[])
        for repeat in range(args.repeats):
            for name in list(models)[::1 if repeat%2==0 else -1]:
                m=models[name];start=time.perf_counter();cpu_start=time.process_time()
                loss=m.run(data[:args.targets],data[1:args.targets+1],mode='periodic',lr=.04,lp=.04/3)
                seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu_start
                assert np.isfinite(loss) and np.isfinite(m.flat).all()
                row=dict(nll=float(loss),seconds=seconds,bytes_per_second=args.targets/seconds,
                         cpu_seconds=cpu_seconds)
                row['cpu_bytes_per_second']=args.targets/row['cpu_seconds']
                result['runs'][name]['rounds'].append(row)
                print(name,repeat+1,row,flush=True)
        for name,m in models.items():
            m.finish_document(mode='periodic',lr=.04,lp=.04/3)
            row=result['runs'][name];row['median_bytes_per_second']=float(np.median([r['bytes_per_second'] for r in row['rounds']]))
            row['median_cpu_bytes_per_second']=float(np.median([r['cpu_bytes_per_second'] for r in row['rounds']]))
            row.update(content_experts=m.content_expert_metrics(),local_credit=m.local_credit_metrics(),neural=m.neural_metrics(),lace=m.lace_metrics(),
                       messages=m.message_metrics(),completion=m.completion_metrics(),teaching=m.relay_teaching_status())
            for field,key in [('content_experts','missing_producers'),('local_credit','missing_producers'),('neural','missing_live_dependencies'),
                              ('lace','missing_parent'),('lace','overwritten_credit'),
                              ('messages','missing_feedback_producers'),('completion','expired_semantic_handles'),
                              ('teaching','expired_producers')]:
                assert all(v[key]==0 for v in row[field].values()),(name,field,key)
            if name=='content_experts':
                assert sum(v['updates'] for v in row['content_experts'].values())>0
                assert sum(v['neural_updates'] for v in row['local_credit'].values())>0
                assert sum(v['feedback_deliveries'] for v in row['local_credit'].values())>0
            before=m.fingerprint();row['heldout_nll'],production=frozen_score(m)
            _,diagnostic=frozen_score(m,'none');np.testing.assert_array_equal(production,diagnostic)
            row['diagnostic_parity']=True
            row['contribution_delta_nll']={region:frozen_score(m,'content_expert',region)[0]-row['heldout_nll'] for region in ('L3','F4','G5')}
            row['generation']=generation(m);assert before==m.fingerprint()
            snapshot=m.snapshot_copy()
            try:
                expected=m.run(data[:128],data[1:129],mode='periodic',logits=True)[1]
                actual=np.concatenate([snapshot.run(data[i:i+32],data[i+1:i+33],mode='periodic',logits=True)[1] for i in range(0,128,32)])
                np.testing.assert_array_equal(expected,actual);assert m.fingerprint()==snapshot.fingerprint()
                row['exact_chunked_continuation']=True
            finally:snapshot.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(args.output,flush=True)
    finally:
        for m in models.values():m.close()

if __name__=='__main__':main()
