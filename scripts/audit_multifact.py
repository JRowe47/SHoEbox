"""Independently reconstruct training provenance, selection, and predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.multifact_train import load
from experiments.text_regions.multifact import trainable,ingest,query,table_batches
from experiments.text_regions.multifact_data import make_batch,batch_digest,KINDS
from experiments.text_regions.multifact_eval import evaluate,publication_metrics
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.state import GraphState
from experiments.text_regions.composition_train import source_hashes
from evaluate_multifact import conditions


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    model,_,best=load(a.run/'best.pt');_,_,last=load(a.run/'last.pt');initial,_,zero=load(a.run/'initial.pt')
    protocol=last['protocol'];assert protocol['source_hashes']==source_hashes()
    seed=protocol['seed'];assert last['step']==protocol['steps']
    progress_path=a.run/'audit_progress.json'
    contract=dict(best=hashlib.sha256((a.run/'best.pt').read_bytes()).hexdigest(),
        last=hashlib.sha256((a.run/'last.pt').read_bytes()).hexdigest(),
        evaluation=hashlib.sha256((a.run/'evaluation.json').read_bytes()).hexdigest(),
        source_hashes=source_hashes(),auditor=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    progress=json.loads(progress_path.read_text()) if a.resume and progress_path.exists() else {}
    if progress.get('contract')!=contract:progress=dict(contract=contract,reproduced=[])
    def remember(name):
        if name not in progress['reproduced']:progress['reproduced'].append(name)
        temporary=progress_path.with_suffix('.tmp');temporary.write_text(json.dumps(progress,indent=2)+'\n');temporary.replace(progress_path)
    validations=[r for r in last['history'] if 'validation' in r]
    def key(row):
        fields=[f for v in row['validation'] for f in v['fields']]
        return (min(f['accuracy'] for f in fields),sum(f['accuracy'] for f in fields)/len(fields),
                -sum(f['nats'] for f in fields)/len(fields))
    selected=max(validations,key=key);assert selected['step']==best['step']
    # Evaluate selected validation once again; selection never consults test data.
    for kind,stored in zip(KINDS,selected['validation']):
        keyname='validation:'+kind
        if keyname not in progress['reproduced']:
            same(evaluate(model,'validation',seed,protocol['validation_count'],16,kind),stored);remember(keyname)
    for checkpoint in (last,best):
        for name,value in checkpoint['weights'].items():
            if not trainable(name):assert torch.equal(value,zero['weights'][name]),name
    counters=dict(expert_executions=0,external_events=0,source_occurrences=0)
    ids=set()
    for row in last['history'][1:]:
        step=row['step'];start=(step-1)*protocol['batch']
        kind=KINDS[(step-1)%5];gap=protocol['gaps'][((step-1)//5)%4]
        batch=make_batch('train',start,protocol['batch'],seed,gap,kind)
        assert row['batch_sha256']==batch_digest(batch);assert row['kind']==kind and row['gap']==gap
        same(row['table_keys'],[tb.keys for tb in table_batches(batch,'train',start,seed)])
        b,t,_=batch.inputs.shape
        expected=dict(expert_executions=b*(48*t+3*264),external_events=b*(t+15),source_occurrences=b*(4*t+30))
        for k,v in expected.items():assert row[k]==v;counters[k]+=v
        assert not ids.intersection(batch.identities);ids.update(batch.identities)
    for k,v in counters.items():assert v==last['totals'][k]
    evaluation=json.loads((a.run/'evaluation.json').read_text());count=evaluation['recall']['count'];cache={}
    lock=a.run.parent/'protocol_lock.json'
    if lock.exists():assert count==json.loads(lock.read_text())['test_count']
    groups=0
    for name,kw in conditions(count).items():
        if name in progress['reproduced']:
            groups+=3+3*int(evaluation[name]['live']);continue
        kw=dict(kw);split=kw.pop('split','test');kw['batch_size']=evaluation[name]['batch_size']
        actual=evaluate(model,split,seed,cache=cache,**kw)
        same(actual,evaluation[name]);groups+=3+3*int(actual['live'])
        if actual['query_control'] in ('intact','no_edge'):assert actual['exact_memories_preserved']
        remember(name)
        print(json.dumps(dict(reproduced=name,fields=groups)),flush=True)
    for kind in KINDS:
        if 'initial:'+kind not in progress['reproduced']:
            same(evaluate(initial,'test',seed,count,16,kind,batch_size=evaluation['initial'][kind]['batch_size']),evaluation['initial'][kind]);remember('initial:'+kind)
        groups+=3
    if 'publication' not in progress['reproduced']:
        same(publication_metrics(model,seed,count),evaluation['publication']);remember('publication')
    for row in evaluation['publication'].values():assert row['hold_pairs']==row['exact_hold_pairs']
    for control in ('no_exchange','no_edge','reset','shuffle','always','prefer_stale_left'):
        assert evaluation[control]['expert_executions']==evaluation['shared_right']['expert_executions']
    for erased in ('erase_left','erase_right'):
        assert evaluation[erased]['expert_executions']==evaluation['recall']['expert_executions']
    for split,n in (('validation',protocol['validation_count']),('test',count),('independent',count)):
        cohort=set(make_batch(split,0,n,seed).identities);assert len(cohort)==n and not cohort.intersection(ids)
        ids.update(cohort)
    exported=torch.load(a.run/'inference.pt',map_location='cpu',weights_only=True)
    restored=RegionLanguageModel(GraphConfig.from_dict(exported['config']));restored.load_state_dict(exported['weights'])
    state=GraphState.from_dict(exported['state']);batch=make_batch('test',17,4,seed,16,'shared_right')
    with torch.no_grad():recreated,_=ingest(restored,batch)
    same(recreated.to_dict(),exported['state'])
    tb=table_batches(batch,'test',17,seed)[2]
    with torch.no_grad():r=query(restored,tb.first,tb.second,state,torch.full((4,),2,dtype=torch.long))
    same(r['logits'],exported['logits']);same(r['state'].to_dict(),exported['query_state'])
    result=dict(all_passed=True,seed=seed,selected_step=best['step'],training_batches=last['step'],
        prediction_groups=groups,composer_weights_exact=True,export_exact=True,controls_equal_execution=True,
        parameters=model.parameter_summary(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        distinct_checked_ids=len(ids),**last['totals'])
    (a.run/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
