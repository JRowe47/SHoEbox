"""Evaluate both paired arms on new text positions and retained-fact episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_stability import load, primary_conditions
from experiments.text_regions.joint_text import evaluate_text
from experiments.text_regions.multifact_eval import evaluate, memory_tensors
from experiments.text_regions.multifact import ingest, query, table_batches, workspace
from experiments.text_regions.multifact_data import make_batch
from experiments.text_regions import byte_stream
from experiments.text_regions.data import split_bytes
from experiments.text_regions.composition_train import source_hashes, atomic_save
from evaluate_joint_text import interleave


def fresh_text(test):
    """Eight new windows in the second half, disjoint from old primary targets.

    The previous study tested offsets 1..512 in each of eight equal shards.
    Use offsets 1024 and 3072 in each of the last four original shards. Each
    1024-byte window becomes an independent row; evaluate its first 512 targets.
    """
    old_edges=[len(test)*i//8 for i in range(9)]
    starts=[old_edges[i]+offset for i in range(4,8) for offset in (1024,3072)]
    if any(s+1024>old_edges[5+i//2] for i,s in enumerate(starts)):
        raise ValueError('Test corpus is too small for the locked fresh windows.')
    old_positions={base+j for base in old_edges[:-1] for j in range(1,513)}
    new_positions={base+j for base in starts for j in range(1,513)}
    if old_positions & new_positions or len(new_positions)!=4096:
        raise ValueError('Fresh test targets overlap old targets or each other.')
    return b''.join(test[s:s+1024] for s in starts), dict(
        test_relative_starts=starts,window_bytes=1024,target_offsets=[1,512],
        previous_primary_overlap=0,target_bytes=4096)


def conditions(count):
    rows={name:dict(kind=kind,source_control=control) for name,kind,control in primary_conditions()}
    for name in ('left_only','right_only','erase_left','erase_right'):
        rows[name]=dict(split='independent',source_control=name)
    rows.update(long_recall=dict(gap=64),long_correction=dict(kind='shared_right',gap=64))
    for name,kw in dict(k1=dict(rounds=1),k4=dict(rounds=4),no_exchange=dict(feedback=False),
        no_edge=dict(query_control='no_edge'),reset=dict(query_control='reset_memory'),
        shuffle=dict(query_control='shuffle_memory')).items():
        rows[name]=dict(kind='shared_right',**kw)
    return {name:dict(count=count,batch_size=count,**kw) for name,kw in rows.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    a=p.parse_args();torch.set_num_threads(1)
    model,_,_,saved=load(a.run/'best.pt');seed=saved['protocol']['seed']
    assert saved['protocol']['source_hashes']==source_hashes(),'Production source changed.'
    raw=a.data.read_bytes();train,_,test=split_bytes(raw);text,layout=fresh_text(test)
    assert hashlib.sha256(raw).hexdigest()==saved['protocol']['data_sha256']
    count=512
    contract=dict(best_sha256=hashlib.sha256((a.run/'best.pt').read_bytes()).hexdigest(),
        last_sha256=hashlib.sha256((a.run/'last.pt').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=source_hashes(),data_sha256=hashlib.sha256(raw).hexdigest(),count=count,
        text_layout=layout,text_sha256=hashlib.sha256(text).hexdigest(),text_chunks=32)
    partial=a.run/'evaluation.partial.json'
    result=json.loads(partial.read_text()) if partial.exists() else dict(
        contract=contract,selected_step=saved['step'],memory={},text={},final=dict(memory={}),initial={})
    if result['contract']!=contract:raise ValueError('Evaluation contract changed.')
    def save():
        temporary=partial.with_suffix('.tmp');temporary.write_text(json.dumps(result)+'\n');temporary.replace(partial)
    cache={}
    model.eval()
    for name,kw in conditions(count).items():
        if name in result['memory']:continue
        kw=dict(kw);split=kw.pop('split','test')
        result['memory'][name]=evaluate(model,split,seed,cache=cache,**kw);save()
        print(json.dumps(dict(memory=name,accuracy=[f['accuracy'] for f in result['memory'][name]['fields']])),flush=True)
    del cache
    for name,kw in dict(intact={},k1=dict(rounds=1),reset_chunks=dict(reset_chunks=True)).items():
        if name in result['text']:continue
        result['text'][name]=evaluate_text(model,text,8,16,32,**kw);save()
        print(json.dumps(dict(text=name,nats=result['text'][name]['nats'])),flush=True)
    for label,filename in (('initial','initial.pt'),('final','last.pt')):
        target,_,_,s=load(a.run/filename);target.eval();result[label]['step']=s['step']
        if 'text' not in result[label]:
            result[label]['text']=evaluate_text(target,text,8,16,32);save()
        if label=='final':
            for name,kind,control in primary_conditions():
                if name in result[label]['memory']:continue
                result[label]['memory'][name]=evaluate(target,'test',seed,count,16,kind,
                    source_control=control,batch_size=count);save()
        del target
    counts=torch.bincount(torch.tensor(list(train)),minlength=256).double()+1
    logprob=(counts/counts.sum()).log()
    targets=torch.tensor([v for row in result['text']['intact']['rows'] for batch in row['targets'] for v in batch])
    result['unigram_nats']=float(-logprob[targets].mean())
    if 'interleaved' not in result:
        result['interleaved']=interleave(model,seed,count,text[:32]);save()
    if 'live_text' not in result or not (a.run/'inference.pt').exists():
        batch=make_batch('test',23,1,seed,16,'shared_right')
        with torch.no_grad():state,_=ingest(model,batch)
        before=[v.clone() for v in memory_tensors(state)]
        session=byte_stream.start(model,b'ROMEO:\n',seed=seed,state=workspace(state))
        initial_session=session.to_dict()
        tokens,session,cost=byte_stream.generate(model,session,128)
        exact=all(torch.equal(x,y) for x,y in zip(before,memory_tensors(session.state)))
        answers=[]
        with torch.no_grad():
            for field,tb in enumerate(table_batches(batch,'test',23,seed)):
                r=query(model,tb.first,tb.second,session.state,torch.tensor([field]))
                answers.append(dict(prediction=int(r['logits'][0,-1].argmax()),target=int(tb.answers[0])))
        result['live_text']=dict(exact_memory=exact,answers=answers,prompt='ROMEO:\n',seed=seed,
            temperature=.8,generated_bytes=128,expert_executions=cost,sample_hex=bytes(tokens[0].tolist()).hex())
        atomic_save(a.run/'inference.pt',dict(schema='shoebox-joint-stability-inference-v1',
            config=model.cfg.to_dict(),weights=model.state_dict(),initial_session=initial_session,
            final_session=session.to_dict(),tokens=tokens))
        (a.run/'sample.bin').write_bytes(bytes(tokens[0].tolist()))
        (a.run/'sample.txt').write_text((b'ROMEO:\n'+bytes(tokens[0].tolist())).decode('utf8',errors='replace'))
        save()
    result['targets']=dict(text=result['text']['intact']['nats']<4,
        memory=min(f['accuracy'] for name,_,_ in primary_conditions() for f in result['memory'][name]['fields'])>=.9,
        recurrence=sum(f['accuracy'] for f in result['memory']['shared_right']['fields'])/3-
                   sum(f['accuracy'] for f in result['memory']['k1']['fields'])/3>=.05,
        ownership=result['live_text']['exact_memory'] and result['interleaved']['exact_memory'])
    (a.run/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,selected_step=saved['step'],targets=result['targets'])),flush=True)


if __name__=='__main__':main()
