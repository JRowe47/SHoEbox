"""Audit paired data, selection, task scales, predictions, and live continuation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_stability import load, primary_conditions
from experiments.text_regions.joint_text import evaluate_text, selection, chunk_digest
from experiments.text_regions.multifact_eval import evaluate, memory_tensors
from experiments.text_regions.multifact_data import make_batch, batch_digest, KINDS
from experiments.text_regions.multifact import table_batches, ingest, workspace
from experiments.text_regions import byte_stream
from experiments.text_regions.data import OrderedByteStreams, split_bytes
from experiments.text_regions.composition_train import source_hashes
from audit_multifact import same
from evaluate_joint_stability import fresh_text


def without_timing(row):
    return {k:v for k,v in row.items() if k!='seconds'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--parents',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    model,_,_,best=load(a.run/'best.pt');_,_,_,last=load(a.run/'last.pt');_,_,_,initial=load(a.run/'initial.pt')
    protocol=last['protocol'];seed=protocol['seed'];assert protocol['source_hashes']==source_hashes()
    assert last['step']==protocol['steps']
    parent_dir=a.parents or a.run.parents[2]/'parents'
    candidates=[path for path in parent_dir.rglob('*.pt') if hashlib.sha256(path.read_bytes()).hexdigest()==protocol['parent_sha256']]
    assert len(candidates)==1,'Exactly one hash-matched parent checkpoint required.'
    parent=torch.load(candidates[0],weights_only=True,map_location='cpu')
    for key in ('weights','optimizer','streams','graph_state','reference','torch_rng'):
        same(parent[key],initial[key])
    other_path=a.run.parent/('sum' if protocol['mode']=='group_clip' else 'group_clip')/'last.pt'
    other=torch.load(other_path,weights_only=True,map_location='cpu')
    other_initial=torch.load(other_path.with_name('initial.pt'),weights_only=True,map_location='cpu')
    for key in ('weights','optimizer','streams','graph_state','reference'):
        same(initial[key],other_initial[key])
    raw=a.data.read_bytes();assert hashlib.sha256(raw).hexdigest()==protocol['data_sha256']
    train,val,test=split_bytes(raw);fresh,layout=fresh_text(test)
    streams=OrderedByteStreams.from_bytes(train,protocol['batch']);streams.load_state_dict(initial['streams'])
    validation_fields=0
    for row in last['history']:
        for score in row.get('memory_validation',[]):
            b=make_batch('validation',0,score['count'],seed,score['gap'],score['kind'],score['source_control'])
            for field,tb in enumerate(table_batches(b,'validation',0,seed)):
                f=score['fields'][field]
                assert f['targets']==tb.answers.tolist()
                assert len(f['predictions'])==len(f['targets'])==score['count']
                assert f['accuracy']==sum(x==y for x,y in zip(f['predictions'],f['targets']))/len(f['targets'])
                validation_fields+=1
    chosen=max((r for r in last['history'] if 'text_validation' in r),
               key=lambda r:selection(r['text_validation'],r['memory_validation']))
    assert chosen['step']==best['step']
    totals=dict.fromkeys(('expert_executions','external_events','source_occurrences','text_bytes'),0)
    for row,control in zip(last['history'][1:],other['history'][1:]):
        step=row['step'];offset=(step-1)*protocol['memory_batch'];kind=KINDS[(step-1)%5]
        gap=protocol['gaps'][((step-1)//5)%4]
        b=make_batch('train',offset,protocol['memory_batch'],seed,gap,kind)
        assert row['memory']['batch_sha256']==batch_digest(b)==control['memory']['batch_sha256']
        keys=[tb.keys for tb in table_batches(b,'train',offset,seed)]
        same(keys,row['memory']['table_keys']);same(keys,control['memory']['table_keys'])
        chunk=streams.next_chunk(protocol['context'])
        assert row['text_batch_sha256']==chunk_digest(chunk)==control['text_batch_sha256']
        n,t,_=b.inputs.shape;nb=chunk.values.numel()
        expected=dict(expert_executions=n*(48*t+3*264)+nb*168,
            external_events=n*(t+15)+nb,source_occurrences=n*(4*t+30)+nb*3,text_bytes=nb)
        for k,v in expected.items():assert row[k]==v==control[k];totals[k]+=v
        for group in row['gradient_groups'].values():
            for task in ('text','memory'):
                norm=group[task+'_norm'];scale=group[task+'_scale']
                assert math.isfinite(norm) and norm>=0
                expected=min(1.,protocol['gradient_cap']/max(norm,1e-300)) if protocol['mode']=='group_clip' else 1.
                assert scale==expected
        assert row['gradient_groups']['producer']['text_norm']==0
    assert len(last['history'])==len(other['history'])==protocol['steps']+1
    same(streams.state_dict(),last['streams'])
    for key,value in totals.items():assert last['totals'][key]==value
    result=json.loads((a.run/'evaluation.json').read_text());contract=result['contract']
    assert contract['best_sha256']==hashlib.sha256((a.run/'best.pt').read_bytes()).hexdigest()
    assert contract['last_sha256']==hashlib.sha256((a.run/'last.pt').read_bytes()).hexdigest()
    assert contract['evaluator_sha256']==hashlib.sha256((ROOT/'scripts/evaluate_joint_stability.py').read_bytes()).hexdigest()
    assert contract['text_layout']==layout
    # Recompute every recorded memory accuracy and its ground-truth table.
    scored=0
    for collection in (result['memory'],result['final']['memory']):
        for name,row in collection.items():
            split=row['split'];b=make_batch(split,0,row['count'],seed,row['gap'],row['kind'],row['source_control'])
            for field,tb in enumerate(table_batches(b,split,0,seed)):
                f=row['fields'][field];assert f['targets']==tb.answers.tolist()
                assert len(f['predictions'])==len(f['targets'])==row['count']
                assert f['accuracy']==sum(x==y for x,y in zip(f['predictions'],f['targets']))/len(f['targets'])
                scored+=1
    # A full held-out correction, withheld update, and matched-budget control
    # are rerun from the selected weights. Remaining arrays are rescored above.
    reproduced=[];model.eval()
    weakest=min(chosen['memory_validation'],key=lambda r:min(f['accuracy'] for f in r['fields']))
    actual=dict(condition=weakest['condition'],**evaluate(model,'validation',seed,weakest['count'],
        weakest['gap'],weakest['kind'],source_control=weakest['source_control'],batch_size=weakest['batch_size']))
    # Original checkpoints can cross CPU hosts during recovery. Require the
    # complete discrete result to match; permit only FP32 loss-reduction noise.
    validation_nll_differences=[]
    for observed,recorded in zip(actual['fields'],weakest['fields']):
        validation_nll_differences.append(abs(observed['nats']-recorded['nats']))
        assert math.isclose(observed['nats'],recorded['nats'],rel_tol=1e-6,abs_tol=1e-6)
        same({k:v for k,v in observed.items() if k!='nats'},
             {k:v for k,v in recorded.items() if k!='nats'})
    same({k:v for k,v in actual.items() if k!='fields'},
         {k:v for k,v in weakest.items() if k!='fields'})
    for name in ('shared_right','withheld_left_private','no_exchange'):
        stored=result['memory'][name]
        actual=evaluate(model,stored['split'],seed,stored['count'],stored['gap'],stored['kind'],
            source_control=stored['source_control'],query_control=stored['query_control'],
            rounds=stored['rounds'],feedback=stored['feedback'],batch_size=stored['batch_size'])
        same(actual,stored);reproduced.append(name)
        print(json.dumps(dict(reproduced=name)),flush=True)
    actual=evaluate_text(model,fresh,8,16,8)
    same(actual['rows'],result['text']['intact']['rows'][:8])
    # Decode stored cross-entropies independently of aggregate metrics.
    for row in [*result['text'].values(),result['initial']['text'],result['final']['text']]:
        values=[v for chunk in row['rows'] for batch in chunk['nll'] for v in batch]
        assert abs(sum(values)/len(values)-row['nats'])<1e-10
    exported=torch.load(a.run/'inference.pt',weights_only=True,map_location='cpu');same(exported['weights'],best['weights'])
    batch=make_batch('test',23,1,seed,16,'shared_right')
    with torch.no_grad():state,_=ingest(model,batch)
    session=byte_stream.start(model,b'ROMEO:\n',seed=seed,state=workspace(state))
    same(session.to_dict(),exported['initial_session'])
    left,mid,_=byte_stream.generate(model,session,41)
    # Go through actual disk serialization, not only an in-memory dict.
    session_path=a.run/'audit_session.pt';torch.save(mid.to_dict(),session_path)
    restored=byte_stream.ByteSession.from_dict(torch.load(session_path,weights_only=True))
    right,end,_=byte_stream.generate(model,restored,87)
    same(torch.cat((left,right),1),exported['tokens']);same(end.to_dict(),exported['final_session'])
    session_path.unlink()
    for x,y in zip(memory_tensors(session.state),memory_tensors(end.state)):same(x,y)
    assert (a.run/'sample.bin').read_bytes()==bytes(exported['tokens'][0].tolist())
    for name in ('no_exchange','no_edge','reset','shuffle'):
        assert result['memory'][name]['expert_executions']==result['memory']['shared_right']['expert_executions']
    assert result['text']['reset_chunks']['expert_executions']==result['text']['intact']['expert_executions']
    primary_floor=min(f['accuracy'] for name,_,_ in primary_conditions() for f in result['memory'][name]['fields'])
    assert result['targets']['memory']==(primary_floor>=.9)
    summary=dict(all_passed=True,selected_step=best['step'],parent_state_exact=True,paired_training_exact=True,
        training_batches=last['step'],recorded_memory_fields_rescored=scored,model_conditions_reproduced=reproduced,
        validation_fields_rescored=validation_fields,weakest_selected_validation_reproduced=weakest['condition'],
        selected_validation_predictions_exact=True,selected_validation_nll_max_abs_difference=max(validation_nll_differences),
        text_prefix_bytes_reproduced=1024,all_text_nll_recomputed=True,fresh_test_positions=layout,
        guarded_selection_exact=True,export_and_resume_exact=True,
        parameters=model.parameter_summary(),primary_memory_min=primary_floor,**last['totals'])
    (a.run/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
