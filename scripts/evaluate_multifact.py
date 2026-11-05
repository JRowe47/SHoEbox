"""Evaluate locked multi-source retention conditions and export inference state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.multifact_train import load
from experiments.text_regions.multifact_eval import evaluate,publication_metrics
from experiments.text_regions.multifact_data import KINDS,make_batch
from experiments.text_regions.multifact import ingest,query,table_batches
from experiments.text_regions.composition_train import atomic_save,source_hashes


def conditions(count=512):
    rows={k:dict(kind=k) for k in KINDS}
    rows.update({'withheld_'+k:dict(kind=k,source_control='withheld') for k in KINDS[1:]})
    rows.update({k:dict(split='independent',source_control=k) for k in ('left_only','right_only','erase_left','erase_right')})
    rows.update(long_recall=dict(gap=64),long_correction=dict(gap=64,kind='shared_right'))
    controls=dict(k1=dict(rounds=1),k4=dict(rounds=4),no_exchange=dict(feedback=False),
        no_edge=dict(query_control='no_edge'),reset=dict(query_control='reset_memory'),
        shuffle=dict(query_control='shuffle_memory'),always=dict(policy='always'),repeated=dict(requests=3),
        prefer_stale_left=dict(shared_policy='left'))
    rows.update({k:dict(kind='shared_right',**v) for k,v in controls.items()})
    rows['prefer_stale_right']=dict(kind='shared_left',shared_policy='right')
    rows.update({'live_'+k:dict(kind=k,live=True) for k in KINDS[1:]})
    rows['live_withheld']=dict(kind='shared_right',live=True,source_control='withheld')
    return {k:dict(count=count,batch_size=count,**v) for k,v in rows.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--count',type=int,default=512)
    p.add_argument('--restart',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    model,_,saved=load(a.run/'best.pt');seed=saved['protocol']['seed'];result={};cache={}
    assert saved['protocol']['source_hashes']==source_hashes(),'Checkpoint production source changed.'
    contract=dict(checkpoint_sha256=hashlib.sha256((a.run/'best.pt').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=saved['protocol']['source_hashes'],seed=seed,count=a.count)
    contract_path=a.run/'evaluation_contract.json'
    compatible=contract_path.exists() and json.loads(contract_path.read_text())==contract
    if not a.restart and compatible:
        for filename in ('evaluation.json','evaluation.partial.json'):
            if (a.run/filename).exists():result=json.loads((a.run/filename).read_text());break
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    def checkpoint():
        temporary=a.run/'evaluation.partial.tmp';temporary.write_text(json.dumps(result)+'\n')
        temporary.replace(a.run/'evaluation.partial.json')
    for name,kw in conditions(a.count).items():
        if name in result:
            print(json.dumps(dict(condition=name,resumed=True)),flush=True);continue
        kw=dict(kw);split=kw.pop('split','test')
        result[name]=evaluate(model,split,seed,cache=cache,**kw)
        checkpoint()
        print(json.dumps(dict(condition=name,accuracy=[x['accuracy'] for x in result[name]['fields']])),flush=True)
    initial,_,_=load(a.run/'initial.pt')
    result.setdefault('initial',{})
    for kind in KINDS:
        if kind not in result['initial']:
            result['initial'][kind]=evaluate(initial,'test',seed,a.count,16,kind,batch_size=a.count);checkpoint()
    if 'publication' not in result:result['publication']=publication_metrics(model,seed,a.count);checkpoint()
    (a.run/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.run/'evaluation.partial.json').unlink(missing_ok=True)
    batch=make_batch('test',17,4,seed,16,'shared_right');tb=table_batches(batch,'test',17,seed)[2]
    with torch.no_grad():
        state,_=ingest(model,batch)
        r=query(model,tb.first,tb.second,state,torch.full((4,),2,dtype=torch.long))
    atomic_save(a.run/'inference.pt',dict(schema='shoebox-multifact-inference-v1',config=model.cfg.to_dict(),
        weights=model.state_dict(),state=state.to_dict(),seed=seed,logits=r['logits'],query_state=r['state'].to_dict(),
        interface=dict(addresses=['shared','left_private','right_private'],owners=['F','L','G','R'])))
    print(json.dumps(dict(complete=True,conditions=len(conditions(a.count)),selected_step=saved['step'])),flush=True)


if __name__=='__main__':main()
