"""Reconstruct training data, guarded selection, held-out outputs and live export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_text import load,evaluate_text,selection,chunk_digest
from experiments.text_regions.multifact_eval import evaluate,publication_metrics,memory_tensors
from experiments.text_regions.multifact_data import make_batch,batch_digest,KINDS
from experiments.text_regions.multifact import table_batches,query,ingest,workspace
from experiments.text_regions import byte_stream
from experiments.text_regions.data import OrderedByteStreams,split_bytes
from experiments.text_regions.composition_train import source_hashes
from audit_multifact import same
from evaluate_multifact import conditions
from evaluate_joint_text import interleave


def text_same(a,b):
    same({k:v for k,v in a.items() if k!='seconds'},{k:v for k,v in b.items() if k!='seconds'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    a=p.parse_args();torch.set_num_threads(1)
    model,_,_,best=load(a.run/'best.pt');_,_,_,last=load(a.run/'last.pt');initial,_,_,zero=load(a.run/'initial.pt')
    protocol=last['protocol'];assert protocol['source_hashes']==source_hashes()
    assert last['step']==protocol['steps'];seed=protocol['seed']
    raw=a.data.read_bytes();assert hashlib.sha256(raw).hexdigest()==protocol['data_sha256']
    train,val,test=split_bytes(raw);streams=OrderedByteStreams.from_bytes(train,protocol['batch'])
    validated=[r for r in last['history'] if 'text_validation' in r]
    chosen=max(validated,key=lambda r:selection(r['text_validation'],r['memory_validation']))
    assert chosen['step']==best['step']
    counters=dict.fromkeys(('expert_executions','external_events','source_occurrences','text_bytes'),0)
    for row in last['history'][1:]:
        step=row['step'];offset=(step-1)*protocol['memory_batch']
        kind=KINDS[(step-1)%5];gap=protocol['gaps'][((step-1)//5)%4]
        batch=make_batch('train',offset,protocol['memory_batch'],seed,gap,kind)
        assert row['memory']['batch_sha256']==batch_digest(batch)
        same(row['memory']['table_keys'],[tb.keys for tb in table_batches(batch,'train',offset,seed)])
        chunk=streams.next_chunk(protocol['context']);assert row['text_batch_sha256']==chunk_digest(chunk)
        b,t,_=batch.inputs.shape;nb=chunk.values.numel()
        expected=dict(expert_executions=b*(48*t+3*264)+nb*168,
            external_events=b*(t+15)+nb,source_occurrences=b*(4*t+30)+nb*3,text_bytes=nb)
        for k,v in expected.items():assert row[k]==v;counters[k]+=v
    same(streams.state_dict(),last['streams'])
    for k,v in counters.items():assert last['totals'][k]==v
    # Inference tensors must match the actual selected checkpoint, and all
    # major trainable subsystems must have participated in joint learning.
    changed=[k for k in best['weights'] if not torch.equal(best['weights'][k],zero['weights'][k])]
    participation={prefix:any(k.startswith(prefix) for k in changed) for prefix in
        ('hat.','nodes.P.','nodes.Q.','readout.','nodes.F.','nodes.L.','nodes.G.','nodes.R.')}
    assert all(participation.values()),'A required trainable subsystem did not change.'
    result=json.loads((a.run/'evaluation.json').read_text());contract=result['contract']
    assert contract['best_sha256']==hashlib.sha256((a.run/'best.pt').read_bytes()).hexdigest()
    assert contract['last_sha256']==hashlib.sha256((a.run/'last.pt').read_bytes()).hexdigest()
    progress_path=a.run/'audit_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    audit_contract=dict(best=contract['best_sha256'],source=source_hashes(),
        evaluation=hashlib.sha256((a.run/'evaluation.json').read_bytes()).hexdigest(),
        auditor=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if progress.get('contract')!=audit_contract:progress=dict(contract=audit_contract,reproduced=[])
    def remember(name):
        progress['reproduced'].append(name)
        temporary=progress_path.with_suffix('.tmp');temporary.write_text(json.dumps(progress,indent=2)+'\n')
        temporary.replace(progress_path)
        print(json.dumps(dict(reproduced=name)),flush=True)
    for kind,stored in zip(KINDS,chosen['memory_validation']):
        name='validation:'+kind
        if name not in progress['reproduced']:
            same(evaluate(model,'validation',seed,protocol['validation_count'],16,kind),stored);remember(name)
    text=evaluate_text(model,val,protocol['batch'],protocol['context'],protocol['validation_chunks'])
    same({k:v for k,v in text.items() if k not in ('rows','seconds')},
         {k:v for k,v in chosen['text_validation'].items() if k!='seconds'})
    cache={}
    for name,kw in conditions(contract['count']).items():
        label='memory:'+name
        if label in progress['reproduced']:continue
        kw=dict(kw);split=kw.pop('split','test')
        actual=evaluate(model,split,seed,cache=cache,**kw);same(actual,result['memory'][name]);remember(label)
    for name,stored in result['text'].items():
        label='text:'+name
        if label in progress['reproduced']:continue
        actual=evaluate_text(initial if name=='initial' else model,test,stored['batch'],stored['context'],stored['chunks'],
            stored['rounds'],stored['feedback'],stored['reset_chunks'])
        text_same(actual,stored);remember(label)
    final_model,_,_,_=load(a.run/'last.pt')
    for kind,stored in result['final']['memory'].items():
        label='final:'+kind
        if label not in progress['reproduced']:
            same(evaluate(final_model,'test',seed,contract['count'],16,kind,batch_size=contract['count']),stored);remember(label)
    if 'final:text' not in progress['reproduced']:
        stored=result['final']['text']
        text_same(evaluate_text(final_model,test,stored['batch'],stored['context'],stored['chunks']),stored);remember('final:text')
    for name in ('no_exchange','no_edge','reset','shuffle','always'):
        assert result['memory'][name]['expert_executions']==result['memory']['shared_right']['expert_executions']
    for name in ('no_exchange','reset_chunks'):
        assert result['text'][name]['expert_executions']==result['text']['intact']['expert_executions']
    if 'publication' not in progress['reproduced']:
        same(publication_metrics(model,seed,contract['count']),result['publication']);remember('publication')
    if 'interleaved' not in progress['reproduced']:
        actual=interleave(model,seed,contract['count'],bytes.fromhex(result['interleaved']['text_hex']))
        same(actual,result['interleaved']);assert actual['exact_memory'];remember('interleaved')
    exported=torch.load(a.run/'inference.pt',map_location='cpu',weights_only=True)
    same(exported['weights'],best['weights'])
    batch=make_batch('test',23,1,seed,16,'shared_right')
    with torch.no_grad():initial_state,_=ingest(model,batch)
    recreated=byte_stream.start(model,result['live_text']['prompt'],seed=seed,
        temperature=result['live_text']['temperature'],state=workspace(initial_state))
    same(recreated.to_dict(),exported['initial_session'])
    session=byte_stream.ByteSession.from_dict(exported['initial_session'])
    left,middle,_=byte_stream.generate(model,session,41)
    middle=byte_stream.ByteSession.from_dict(middle.to_dict())
    right,final,_=byte_stream.generate(model,middle,87)
    same(torch.cat((left,right),1),exported['tokens']);same(final.to_dict(),exported['final_session'])
    raw=bytes(exported['tokens'][0].tolist())
    assert (a.run/'sample.bin').read_bytes()==raw and raw.hex()==result['live_text']['sample_hex']
    assert len(raw)==result['live_text']['generated_bytes']
    assert (a.run/'sample.txt').read_text()==(result['live_text']['prompt'].encode('utf-8')+raw).decode('utf-8',errors='replace')
    assert result['live_text']['exact_memory']
    for before,after in zip(memory_tensors(session.state),memory_tensors(final.state)):same(before,after)
    batch=make_batch('test',23,1,seed,16,'shared_right');answers=[]
    with torch.no_grad():
        for field,tb in enumerate(table_batches(batch,'test',23,seed)):
            r=query(model,tb.first,tb.second,final.state,torch.tensor([field]))
            answers.append(dict(prediction=int(r['logits'][0,-1].argmax()),target=int(tb.answers[0])))
    same(answers,result['live_text']['answers'])
    summary=dict(all_passed=True,selected_step=best['step'],training_batches=last['step'],
        training_data_exact=True,guarded_selection_exact=True,parameters=model.parameter_summary(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        changed_subsystems=participation,export_and_resume_exact=True,
        reproduced_groups=len(progress['reproduced']),**last['totals'])
    (a.run/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
