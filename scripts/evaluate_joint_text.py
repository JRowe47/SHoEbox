"""Measure natural language and retained capabilities at the selected checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_text import load,evaluate_text
from experiments.text_regions.multifact_eval import evaluate,publication_metrics,memory_tensors
from experiments.text_regions.multifact import ingest,query,table_batches,workspace
from experiments.text_regions.multifact_data import make_batch,KINDS
from experiments.text_regions import byte_stream
from experiments.text_regions.data import split_bytes,OrderedByteStreams
from experiments.text_regions.composition_train import source_hashes,atomic_save
from evaluate_multifact import conditions


@torch.no_grad()
def interleave(model,seed,count,text):
    batch=make_batch('test',0,count,seed,16,'shared_right')
    state,budget=ingest(model,batch);before=[t.clone() for t in memory_tensors(state)]
    values=torch.tensor(list(text),dtype=torch.long)[None].expand(count,-1)
    r=byte_stream.forward(model,values,state=workspace(state))
    exact=all(torch.equal(a,b) for a,b in zip(before,memory_tensors(r['state'])))
    fields=[]
    for field,tb in enumerate(table_batches(batch,'test',0,seed)):
        answer=query(model,tb.first,tb.second,r['state'],torch.full((count,),field,dtype=torch.long))
        predictions=answer['logits'][:,-1].argmax(-1)
        fields.append(dict(predictions=predictions.tolist(),targets=tb.answers.tolist(),
            accuracy=float(predictions.eq(tb.answers).double().mean())))
    return dict(count=count,text_hex=text.hex(),fields=fields,exact_memory=exact,
        interleaved_expert_executions=r['expert_executions'],interleaved_source_occurrences=r['source_occurrences'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--count',type=int,default=512);p.add_argument('--text-chunks',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(1)
    model,_,_,saved=load(a.run/'best.pt');seed=saved['protocol']['seed']
    assert saved['protocol']['source_hashes']==source_hashes(),'Production source changed.'
    raw=a.data.read_bytes();assert hashlib.sha256(raw).hexdigest()==saved['protocol']['data_sha256']
    train,val,test=split_bytes(raw)
    contract=dict(best_sha256=hashlib.sha256((a.run/'best.pt').read_bytes()).hexdigest(),
        last_sha256=hashlib.sha256((a.run/'last.pt').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(raw).hexdigest(),count=a.count,text_chunks=a.text_chunks)
    partial=a.run/'evaluation.partial.json'
    result=json.loads(partial.read_text()) if partial.exists() else dict(contract=contract,memory={},text={})
    if result['contract']!=contract:raise ValueError('Evaluation contract changed; use a new run directory.')
    def save():
        temporary=partial.with_suffix('.tmp');temporary.write_text(json.dumps(result)+'\n');temporary.replace(partial)
    cache={}
    for name,kw in conditions(a.count).items():
        if name in result['memory']:continue
        kw=dict(kw);split=kw.pop('split','test')
        result['memory'][name]=evaluate(model,split,seed,cache=cache,**kw);save()
        print(json.dumps(dict(condition=name,accuracy=[f['accuracy'] for f in result['memory'][name]['fields']])),flush=True)
    text_conditions=dict(intact={},k1=dict(rounds=1),k4=dict(rounds=4),
        no_exchange=dict(feedback=False),reset_chunks=dict(reset_chunks=True))
    for name,kw in text_conditions.items():
        if name in result['text']:continue
        result['text'][name]=evaluate_text(model,test,8,16,a.text_chunks,**kw);save()
        print(json.dumps(dict(text_condition=name,nats=result['text'][name]['nats'])),flush=True)
    if 'initial' not in result['text']:
        initial,_,_,_=load(a.run/'initial.pt')
        result['text']['initial']=evaluate_text(initial,test,8,16,a.text_chunks);save()
    # The budget-end checkpoint is a separately declared diagnostic. It never
    # replaces validation selection, even if its test score is better.
    if 'final' not in result:result['final']=dict(memory={})
    final,_,_,last=load(a.run/'last.pt')
    result['final']['step']=last['step']
    for kind in KINDS:
        if kind not in result['final']['memory']:
            result['final']['memory'][kind]=evaluate(final,'test',seed,a.count,16,kind,batch_size=a.count);save()
    if 'text' not in result['final']:
        result['final']['text']=evaluate_text(final,test,8,16,a.text_chunks);save()
    # Reference entropy is fitted using the complete training split only.
    counts=torch.bincount(torch.tensor(list(train)),minlength=256).double()+1
    logprob=(counts/counts.sum()).log()
    targets=torch.tensor([t for row in result['text']['intact']['rows'] for batch in row['targets'] for t in batch])
    result['unigram_nats']=float(-logprob[targets].mean())
    if 'publication' not in result:result['publication']=publication_metrics(model,seed,a.count);save()
    if 'interleaved' not in result:
        result['interleaved']=interleave(model,seed,a.count,train[:32]);save()
    # Interleave actual text generation with retained facts in one live state.
    if 'live_text' not in result or not (a.run/'inference.pt').exists():
        batch=make_batch('test',23,1,seed,16,'shared_right')
        with torch.no_grad():state,_=ingest(model,batch)
        before=[t.clone() for t in memory_tensors(state)]
        session=byte_stream.start(model,b'ROMEO:\n',seed=seed,state=workspace(state))
        initial_session=session.to_dict();tokens,session,cost=byte_stream.generate(model,session,128)
        held=all(torch.equal(x,y) for x,y in zip(before,memory_tensors(session.state)))
        answers=[]
        with torch.no_grad():
            for field,tb in enumerate(table_batches(batch,'test',23,seed)):
                r=query(model,tb.first,tb.second,session.state,torch.tensor([field]))
                answers.append(dict(prediction=int(r['logits'][0,-1].argmax()),target=int(tb.answers[0])))
        result['live_text']=dict(exact_memory=held,answers=answers,generated_bytes=128,expert_executions=cost,
            prompt='ROMEO:\n',seed=seed,temperature=.8,sample_hex=bytes(tokens[0].tolist()).hex());save()
        atomic_save(a.run/'inference.pt',dict(schema='shoebox-joint-text-inference-v1',config=model.cfg.to_dict(),
            weights=model.state_dict(),initial_session=initial_session,final_session=session.to_dict(),tokens=tokens))
        (a.run/'sample.bin').write_bytes(bytes(tokens[0].tolist()))
        (a.run/'sample.txt').write_text((b'ROMEO:\n'+bytes(tokens[0].tolist())).decode('utf-8',errors='replace'))
    (a.run/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,selected_step=saved['step'])),flush=True)


if __name__=='__main__':main()
