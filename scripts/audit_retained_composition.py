"""Reconstruct training, frozen components, exports and held-out predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.composition_train import source_hashes
from experiments.text_regions.publication_data import make_batch,batch_digest
from experiments.text_regions.retained_composition import build,tables,evaluate,evaluate_revision,ingest,query,EDGE
from experiments.text_regions.retained_composition_train import load,validation
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from evaluate_retained_composition import conditions

SEEDS=(20261120,20261121,20261122)
COUNTERS=('expert_executions','source_occurrences','external_events')


@contextmanager
def cached_source_memory():
    """Reuse immutable inference state only for exactly identical observations.

    Each unique stream is recomputed in this independent audit process. Cache
    keys include the model instance and publication policy; continuing streams
    always execute. This avoids repeating producer work for query-only controls.
    """
    import experiments.text_regions.retained_composition as runtime
    original=runtime.ingest;cache={}
    def cached(model,batch,state=None,policy='learned',segment=16):
        if state is not None:return original(model,batch,state,policy,segment)
        digest=hashlib.sha256()
        for value in (batch.inputs,batch.available,batch.reset):
            digest.update(str((value.shape,value.dtype)).encode());digest.update(value.numpy().tobytes())
        key=(id(model),policy,segment,digest.hexdigest())
        if key not in cache:cache[key]=original(model,batch,state,policy,segment)
        return cache[key]
    runtime.ingest=cached
    try:yield
    finally:runtime.ingest=original


def exact(a,b):
    assert a.keys()==b.keys()
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def audit(root,seed,evaluations):
    lock=json.loads((root/'protocol_lock.json').read_text())
    assert lock['seeds']==list(SEEDS) and lock['steps']==160 and lock['test_count']==512
    assert hashlib.sha256((ROOT/'docs/SHoEbox_Retained_Composition_Protocol.md').read_bytes()).hexdigest()==lock['protocol_sha256']
    for name,sha in lock['parents'].items():assert hashlib.sha256((root/'parents'/name).read_bytes()).hexdigest()==sha
    path=root/str(seed)
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    model,_,best=load(path/'best.pt');p=final['protocol']
    assert p==best['protocol']==initial['protocol'] and p['source_hashes']==source_hashes()
    assert final['step']==160 and initial['step']==0 and len(final['history'])==160
    assert p['composer_sha256']==lock['parents']['composer.pt']
    assert p['producer_sha256']==lock['parents'][f'producer_{seed}.pt']
    exact(initial['weights'],build(root/'parents/composer.pt',root/f'parents/producer_{seed}.pt',seed).state_dict())
    edge_name='connections.edges.'+EDGE+'.value.weight'
    for checkpoint in (best,final):
        changed=[k for k,v in initial['weights'].items() if not torch.equal(v,checkpoint['weights'][k])]
        assert changed==[edge_name],changed
    totals=dict.fromkeys(COUNTERS,0);choices=[];hashes=[]
    for step,row in enumerate(final['history'],1):
        kind='recall' if step%2 else 'correction';gap=(0,4,8,16)[((step-1)//2)%4]
        assert (row['step'],row['kind'],row['gap'])==(step,kind,gap)
        start=(step-1)*p['batch'];b=make_batch('train',start,p['batch'],seed,gap,kind)
        assert batch_digest(b)==row['batch_sha256'];hashes.append(row['batch_sha256'])
        tb=tables('train',start,p['batch'],seed,b.answers)
        assert tuple(tuple(k) for k in row['keys'])==tb.keys
        events=b.inputs.numel();n=p['batch']
        expected=dict(expert_executions=12*events+264*n,source_occurrences=events+10*n,external_events=events+5*n)
        for k in COUNTERS:assert row[k]==expected[k];totals[k]+=row[k]
        if 'validation' in row:
            records=row['validation'];key=(min(r['accuracy'] for r in records),sum(r['accuracy'] for r in records)/3,
                                           -sum(r['nats'] for r in records)/3)
            choices.append((key,-step))
    chosen=max(choices);assert best['step']==-chosen[1] and best['best']==list(chosen[0])==final['best']
    assert all(final['totals'][k]==totals[k] for k in COUNTERS)
    assert final['totals']['training_seconds']==sum(r['training_seconds'] for r in final['history'])
    assert json.loads(json.dumps(final['history']))==json.loads((path/'history.json').read_text())
    key,scores=validation(model,seed,p['validation_count'])
    assert list(key)==best['best'] and scores==best['history'][-1]['validation']
    record=dict(seed=seed,all_passed=True,selected_step=best['step'],validation_key=list(key),
        parameters=model.parameter_summary(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_component_weights_exact=True,training_batches=len(hashes),batch_hashes=hashes,
        training_seconds=final['totals']['training_seconds'],**totals)
    if evaluations:
        saved=json.loads((path/'evaluation.json').read_text());groups=0
        for name,kwargs in conditions().items():
            actual=evaluate(model,'test',seed,**kwargs)
            assert json.loads(json.dumps(actual))==saved[name],name
            if kwargs.get('query_control','intact')=='intact':assert actual['exact_memory_preserved']
            groups+=1
            print(json.dumps(dict(seed=seed,verified=name)),flush=True)
        for name in ('no_edge','reset_memory','shuffled_memory','no_exchange','forced_always'):
            assert all(saved[name][k]==saved['correction_16'][k] for k in COUNTERS)
        assert evaluate_revision(model,'test',seed)==saved['revision_after_queries'];groups+=3
        print(json.dumps(dict(seed=seed,verified='revision_after_queries')),flush=True)
        last,_,_=load(path/'last.pt');assert json.loads(json.dumps(evaluate(last,'test',seed)))==saved['final_endpoint'];groups+=1
        exported=torch.load(path/'inference.pt',map_location='cpu',weights_only=True)
        assert exported['selected_step']==best['step'];exact(exported['weights'],model.state_dict())
        exported_model=RegionLanguageModel(GraphConfig.from_dict(exported['config']));exported_model.load_state_dict(exported['weights'])
        with torch.no_grad():
            b=make_batch('test',0,8,seed,16);tb=tables('test',0,8,seed,b.answers)
            a,_=ingest(model,b);z,_=ingest(exported_model,b)
            ra=query(model,tb.first,tb.second,a);rz=query(exported_model,tb.first,tb.second,z)
            torch.testing.assert_close(ra['logits'],rz['logits'],rtol=0,atol=0)
            for x,y in zip(ra['state'].tensors(),rz['state'].tensors()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        record.update(prediction_groups=groups,export_exact=True,controls_equal_execution=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'artifacts/retained_composition')
    p.add_argument('--evaluations',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args();torch.set_num_threads(1)
    suffix='evaluation_audit' if a.evaluations else 'training_audit'
    if a.seed:
        with cached_source_memory():result=audit(a.root,a.seed,a.evaluations)
        (a.root/str(a.seed)/(suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    def run(seed):
        command=[sys.executable,__file__,'--root',str(a.root),'--seed',str(seed)]
        if a.evaluations:command.append('--evaluations')
        with (a.root/str(seed)/(suffix+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads((a.root/str(seed)/(suffix+'.json')).read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(run,SEEDS))
    result=dict(all_passed=all(r['all_passed'] for r in records),runs=records,
                prediction_groups=sum(r.get('prediction_groups',0) for r in records))
    (a.root/(suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_passed=result['all_passed'],prediction_groups=result['prediction_groups'])))


if __name__=='__main__':main()
