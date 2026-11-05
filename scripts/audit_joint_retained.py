"""Independently regenerate data, selection, predictions and frozen weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.composition_train import source_hashes
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.publication_data import make_batch,batch_digest
from experiments.text_regions.retained_composition import tables,evaluate,evaluate_revision,ingest,query
from experiments.text_regions.retained_composition_train import validation
from experiments.text_regions.joint_retained import allowed,publication_metrics,COUNTERS
from experiments.text_regions.joint_retained_train import load
from evaluate_retained_composition import conditions
from audit_retained_composition import cached_source_memory,exact


def audit(path,evaluations=True):
    lock=json.loads((path.parent/'protocol_lock.json').read_text())
    assert hashlib.sha256((ROOT/'docs/SHoEbox_Joint_Retention_Protocol.md').read_bytes()).hexdigest()==lock['protocol_sha256']
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    model,_,best=load(path/'best.pt');p=final['protocol'];mode=p['mode'];seed=p['seed']
    assert p==best['protocol']==initial['protocol'] and p['source_hashes']==source_hashes()
    assert seed in lock['seeds'] and mode in lock['modes'] and p['batch']==lock['batch']
    assert (p['producer_lr'],p['edge_lr'],p['segment'],p['validate_every'],p['validation_count'])==(.0003,.001,16,20,128)
    parent_path=path.parent/'parents'/f'{seed}.pt'
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['parent_sha256']==lock['parents'][str(seed)]
    exact(initial['weights'],torch.load(parent_path,map_location='cpu',weights_only=True)['weights'])
    assert final['step']==lock['steps']==p['steps'] and initial['step']==0
    assert len(final['history'])==final['step']+1
    changes={}
    for label,checkpoint in (('best',best),('last',final)):
        changed=[k for k,v in initial['weights'].items() if not torch.equal(v,checkpoint['weights'][k])]
        assert all(allowed(k,mode) for k in changed),changed
        changes[label]=changed
    totals=dict.fromkeys(COUNTERS,0);choices=[];hashes=[]
    for step,row in enumerate(final['history']):
        assert row['step']==step
        if step:
            kind='recall' if step%2 else 'correction';gap=(0,4,8,16)[((step-1)//2)%4]
            assert (row['kind'],row['gap'])==(kind,gap)
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
    chosen=max(choices)
    assert best['step']==-chosen[1] and best['best']==list(chosen[0])==final['best']
    assert all(final['totals'][k]==totals[k] for k in COUNTERS)
    assert final['totals']['training_seconds']==sum(r.get('training_seconds',0.) for r in final['history'])
    assert json.loads(json.dumps(final['history']))==json.loads((path/'history.json').read_text())
    key,scores=validation(model,seed,p['validation_count'])
    assert list(key)==best['best'] and scores==best['history'][-1]['validation']
    record=dict(all_passed=True,seed=seed,mode=mode,selected_step=best['step'],validation_key=list(key),
        parameters=model.parameter_summary(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        composer_weights_exact=True,changed_tensors=changes,training_batches=len(hashes),batch_hashes=hashes,
        training_seconds=final['totals']['training_seconds'],**totals)
    if evaluations:
        saved=json.loads((path/'evaluation.json').read_text());groups=0
        def same(name,actual,expected):
            assert json.loads(json.dumps(actual))==expected,name
            print(json.dumps(dict(seed=seed,mode=mode,verified=name)),flush=True)
        for name,kwargs in conditions().items():
            actual=evaluate(model,'test',seed,**kwargs);same(name,actual,saved[name]);groups+=1
            if kwargs.get('query_control','intact')=='intact':assert actual['exact_memory_preserved']
        for name in ('no_edge','reset_memory','shuffled_memory','no_exchange','forced_always'):
            assert all(saved[name][k]==saved['correction_16'][k] for k in COUNTERS)
        same('revision',evaluate_revision(model,'test',seed),saved['revision_after_queries']);groups+=3
        for gap in (16,64):
            same(f'publication_{gap}',publication_metrics(model,'test',seed,gap=gap),saved[f'publication_{gap}'])
            assert saved[f'publication_{gap}']['hold_pairs']==saved[f'publication_{gap}']['exact_hold_pairs']
        start_model,_,_=load(path/'initial.pt')
        for name,kwargs in conditions().items():
            if name in saved['initial']:
                same('initial/'+name,evaluate(start_model,'test',seed,**kwargs),saved['initial'][name]);groups+=1
        last,_,_=load(path/'last.pt');same('final',evaluate(last,'test',seed),saved['final_endpoint']);groups+=1
        exported=torch.load(path/'inference.pt',map_location='cpu',weights_only=True)
        assert exported['selected_step']==best['step'];exact(exported['weights'],model.state_dict())
        export_model=RegionLanguageModel(GraphConfig.from_dict(exported['config']));export_model.load_state_dict(exported['weights'])
        with torch.no_grad():
            b=make_batch('test',0,8,seed,16);tb=tables('test',0,8,seed,b.answers)
            a,_=ingest(model,b);z,_=ingest(export_model,b)
            ra=query(model,tb.first,tb.second,a);rz=query(export_model,tb.first,tb.second,z)
            torch.testing.assert_close(ra['logits'],rz['logits'],rtol=0,atol=0)
            for x,y in zip(ra['state'].tensors(),rz['state'].tensors()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        record.update(prediction_groups=groups,export_exact=True,controls_equal_execution=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
    p.add_argument('--training-only',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    with cached_source_memory():result=audit(a.path,not a.training_only)
    (a.path/('training_audit.json' if a.training_only else 'audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,prediction_groups=result.get('prediction_groups',0))))
