"""Run reference contracts, matched frozen interventions and the regression suite.

The default is a complete acceptance run. --mechanisms-only is an explicit
partial diagnostic and never produces an overall passed acceptance status.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model
from reference_acceptance import collect,check_integrity,check_training,check_contribution,require
from validate_content_experts import TRAIN,HELD


def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def fitted_digest(m):
    digest=hashlib.sha256()
    for value in (m.flat,m.extra_flat,m.composition_state(fitted=True),m.pattern_state(fitted=True)):
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluate(model,data,*,kind=None,region=None,classes=None,memory='cold_frozen',no_returns=False):
    diagnostic=kind is not None
    m=model.evaluation_copy(memory=memory,diagnostic=diagnostic)
    try:
        if kind is not None:m.configure_contribution(kind,region)
        if classes is not None:m.configure_pattern_classes(classes)
        if no_returns:m.ablate(3)
        before=fitted_digest(m);store=m.evaluation_memory_info()['store_digest']
        nll,logits=m.run(data[:-1],data[1:],logits=True)
        m.finish_document()
        require(np.isfinite(nll) and np.isfinite(logits).all(),'Nonfinite frozen evaluation')
        require(before==fitted_digest(m),'Frozen evaluation changed fitted parameters')
        require(store==m.evaluation_memory_info()['store_digest'],'Frozen evaluation changed episodic store')
        metrics=collect(m);check_integrity(metrics)
        return dict(nll=float(nll),fitted_unchanged=True,trace=m.contribution_metrics() if diagnostic else {},
                    acceptance=metrics['acceptance'],patterns=metrics['pattern_classes'],
                    paired=metrics['paired_memory']),logits
    finally:m.close()


def contribution(model,data,label,*,kind=None,region=None,classes=None,memory='cold_frozen',no_returns=False):
    source=model.fingerprint()
    baseline,base=evaluate(model,data,memory=memory,kind='none' if kind is not None else None)
    changed,logits=evaluate(model,data,kind=kind,region=region,classes=classes,memory=memory,no_returns=no_returns)
    if kind is not None:
        applications=changed['trace'][region]['intervention_applications']
        identical=all(baseline['trace'][r]['event_hash']==changed['trace'][r]['event_hash'] for r in baseline['trace'])
    elif classes is not None:
        applications=sum(v['observed'] for v in baseline['patterns'].values());identical=False
    else:
        applications=baseline['paired']['hits'];identical=False
    row=dict(label=label,kind=kind,region=region,memory=memory,baseline_nll=baseline['nll'],
             intervention_nll=changed['nll'],delta_nll=changed['nll']-baseline['nll'],
             logit_rms=float(np.sqrt(np.mean((logits.astype(float)-base)**2))),applications=applications,
             events_identical=identical,fitted_unchanged=True,source_unchanged=model.fingerprint()==source)
    # Channel hooks preserve occurrence identity and primary boundaries. Class
    # removal intentionally changes completion and recognition decisions.
    return check_contribution(row,require_effect=True,preserve_events=kind is not None)


def generate(model,prompt,seed=17,*,diagnostic=False,count=64):
    m=model.evaluation_copy(memory='cold_frozen',diagnostic=diagnostic)
    try:
        if diagnostic:m.configure_contribution('none')
        before=fitted_digest(m);rng=np.random.default_rng(seed)
        logits=m.run(np.frombuffer(prompt,np.uint8),logits=True)[1][-1]
        allowed=np.array([10,*range(32,127)]);output=[]
        for _ in range(count):
            require(np.isfinite(logits).all(),'Nonfinite generation logits')
            scaled=logits[allowed].astype(float)/.8
            probabilities=np.exp(scaled-scaled.max());probabilities/=probabilities.sum()
            byte=int(rng.choice(allowed,p=probabilities));output.append(byte)
            logits=m.run(np.array([byte],np.uint8),np.array([0],np.uint8),logits=True)[1][-1]
        require(before==fitted_digest(m),'Generation changed fitted parameters')
        check_integrity(collect(m))
        return dict(prompt=prompt.decode(),seed=seed,temperature=.8,bytes=count,
                    sampling='ASCII 32-126 plus newline',text=bytes(output).decode(),
                    longest_identical_run=max((len(x.group(0)) for x in re.finditer(r'(.)\1*',bytes(output).decode(),re.DOTALL)),default=0))
    finally:m.close()


def run_regressions(path):
    log=path.with_name(path.stem+'_tests.txt')
    with log.open('w') as out:
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
                              stdout=out,stderr=subprocess.STDOUT)
    text=log.read_text();counts=re.findall(r'Ran (\d+) tests? in ([\d.]+)s',text)
    require(result.returncode==0 and counts and int(counts[-1][0])>0,
            'Regression suite failed or discovered no tests; see '+str(log))
    return dict(passed=int(counts[-1][0]),seconds=float(counts[-1][1]),log=log.name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=int,default=4096)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--mechanisms-only',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results/reference_acceptance.json')
    args=p.parse_args()
    if args.targets<4096:p.error('Acceptance requires at least 4096 training targets')
    os.sched_setaffinity(0,{args.cpu})
    result=dict(status='running',seed=7,cpu=args.cpu,targets=args.targets,lr=.04,lp=.04/3,
                scope='1M/5M deterministic mechanism acceptance; separate synthetic held-out prose',
                regressions='not_run',sanitizer='not_run')
    save(args.output,result)
    model=None
    try:
        model=fresh_model(seed=7)
        data=np.frombuffer(TRAIN*((args.targets+1)//len(TRAIN)+1),np.uint8)
        result['parameters']=model.parameter_count
        result['train_sha256']=hashlib.sha256(data[:args.targets+1]).hexdigest()
        result['heldout_sha256']=hashlib.sha256(HELD).hexdigest()
        start=time.perf_counter();cpu=time.process_time()
        loss=model.run(data[:args.targets],data[1:args.targets+1],mode='periodic',lr=.04,lp=.04/3)
        seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
        model.finish_document(mode='periodic',lr=.04,lp=.04/3)
        require(np.isfinite(loss) and np.isfinite(model.flat).all(),'Nonfinite training')
        result['training']=dict(nll=float(loss),seconds=seconds,cpu_seconds=cpu_seconds,
                                bytes_per_second=args.targets/seconds)
        result['metrics']=collect(model);result['contracts']=check_training(result['metrics'])
        print('Training/credit/primary-clock contracts passed',flush=True);save(args.output,result)
        source=model.fingerprint();held=np.frombuffer(HELD,np.uint8)
        baseline,logits=evaluate(model,held)
        diagnostic,other=evaluate(model,held,kind='none')
        np.testing.assert_array_equal(logits,other)
        require(baseline['acceptance']==diagnostic['acceptance'],'Diagnostic observation mismatch')
        result['heldout_nll']=baseline['nll'];result['diagnostic_parity']=True
        result['contributions']=[]
        for kind,region in (('content_expert','L3'),('content_expert','F4'),('content_expert','G5'),('feedback_channel','G5')):
            row=contribution(model,held,'heldout',kind=kind,region=region)
            result['contributions'].append(row);print(kind,region,row['delta_nll'],flush=True);save(args.output,result)
        result['contributions'].append(contribution(model,held,'heldout',classes=False))
        # Recall may not activate on unseen prose: report it without treating a
        # zero-hit control as evidence. Require the mechanism on a familiar cue.
        cold,_=evaluate(model,held,memory='warm_frozen')
        result['heldout_memory_hits']=cold['paired']['hits']
        result['contributions'].append(contribution(model,data[:513],'familiar',memory='warm_frozen',no_returns=True))
        result['generation']=[]
        for prompt in (b'The traveler followed ',b'Beyond the garden '):
            require(prompt not in TRAIN,'Generation prompt overlaps training fixture')
            row=generate(model,prompt);other=generate(model,prompt,diagnostic=True)
            require(row==other,'Generation is not deterministic across production/diagnostics')
            row['diagnostic_reproduction']=True;result['generation'].append(row)
        require(model.fingerprint()==source,'Evaluation changed the trained checkpoint')
        result['source_unchanged']=True
        snapshot=model.snapshot_copy()
        try:
            model.reset_acceptance_metrics()
            a=model.run(data[:128],data[1:129],mode='periodic',logits=True)[1]
            b=np.concatenate([snapshot.run(data[i:i+32],data[i+1:i+33],mode='periodic',logits=True)[1] for i in range(0,128,32)])
            np.testing.assert_array_equal(a,b)
            require(model.fingerprint()==snapshot.fingerprint(),'Checkpoint continuation differs')
            require(model.acceptance_metrics()==snapshot.acceptance_metrics(),'Chunked counters differ')
            result['exact_chunked_continuation']=True
        finally:snapshot.close()
        model.close();model=None
        save(args.output,result)
        if args.mechanisms_only:
            result['status']='mechanisms_passed';result['complete_acceptance']=False
        else:
            print('Running full native regression suite',flush=True)
            result['regressions']=run_regressions(args.output);save(args.output,result)
            log=args.output.with_name(args.output.stem+'_sanitizer.txt')
            with log.open('w') as out:
                sanitized=subprocess.run([sys.executable,str(ROOT/'tests/run_completion_sanitizer.py')],stdout=out,stderr=subprocess.STDOUT)
            require(sanitized.returncode==0,'Sanitizer failed; see '+str(log))
            result['sanitizer']=dict(passed=True,log=log.name)
            result['status']='passed';result['complete_acceptance']=True
        save(args.output,result);print(args.output,flush=True)
    except Exception as error:
        result.update(status='failed',complete_acceptance=False,error=f'{type(error).__name__}: {error}')
        save(args.output,result);raise
    finally:
        if model is not None:model.close()


if __name__=='__main__':main()
