"""Audit qualified continuation, preserved optimizer state and inference export."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from experiments.text_regions.composition import encode_sources,answer_query
from experiments.text_regions.composition_continue import completed_updates,evaluate_progress,gate_failures
from experiments.text_regions.composition_data import batch_from_keys,evaluation_keys
from experiments.text_regions.composition_train import load_checkpoint,source_hashes
from experiments.text_regions.model import RegionLanguageModel


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/second_lookup'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    run=args.root/'continuation'
    parent=torch.load(args.root/'parent.pt',weights_only=True)
    initial=torch.load(run/'initial.pt',weights_only=True)
    last=torch.load(run/'last.pt',weights_only=True)
    selection=json.loads((run/'selection.json').read_text())
    assert selection['finished_step']==last['step']
    assert selection['optimizer_step']==completed_updates(last)
    for key,value in last['counters'].items():assert selection[key]==value,key
    for field in ('weights','optimizer','rng','torch_rng'):assert equal(parent[field],initial[field]),field
    assert initial['protocol']['optimizer_step_offset']==completed_updates(parent)
    assert last['protocol']['source_hashes']==source_hashes()
    assert last['protocol']['parent_sha256']==hashlib.sha256((args.root/'parent.pt').read_bytes()).hexdigest()
    updates=[r for r in last['history'] if r['phase_step']>0]
    assert [r['phase_step'] for r in updates]==list(range(1,last['step']+1))
    assert all(r['step']==initial['protocol']['optimizer_step_offset']+r['phase_step'] for r in updates)
    for key in ('expert_executions','source_occurrences'):
        assert last['counters'][key]==sum(r[key] for r in updates)
    assert last['counters']['episodes']==last['step']*last['protocol']['batch']
    checked=[r for r in last['history'] if 'validation' in r]
    qualified=[r for r in checked if not gate_failures(r['validation'],last['protocol']['threshold'])]
    chosen=min(qualified or checked,key=lambda r:r['validation']['nats_per_answer'])
    assert selection['qualified_checkpoint']==bool(qualified)
    assert selection['selected_checkpoint']==('best.pt' if qualified else 'best_candidate.pt')
    selected=run/selection['selected_checkpoint']
    assert selection['selected_sha256']==hashlib.sha256(selected.read_bytes()).hexdigest()
    model,_,_,saved=load_checkpoint(selected)
    assert saved['step']==chosen['phase_step']
    streak=0;mastered=None
    for row in checked:
        failures=gate_failures(row['validation'],last['protocol']['threshold'])
        assert failures==row['gate_failures']
        streak=0 if failures else streak+1
        if streak>=last['protocol']['patience'] and mastered is None:mastered=row['phase_step']
    assert mastered==selection['mastered_step']
    metrics={s:evaluate_progress(model,s) for s in ('validation','test')}
    assert metrics['validation']==chosen['validation']
    result=json.loads((run/'test.json').read_text())
    assert result['checkpoint_sha256']==selection['selected_sha256']
    assert result['controls']['k2']['correct']==metrics['test']['metrics']['overall']['correct']
    for name,row in result['controls'].items():
        assert row['episodes']==592==len(row['predictions'])==len(row['targets'])
        assert row['correct']==sum(a==b for a,b in zip(row['predictions'],row['targets']))
        assert row['accuracy']==row['correct']/592
        assert row['expert_executions']==592*(96+84*int(name[1]))
        assert row['source_occurrences']==592*11
        if name in ('k1','k2','k4'):assert len(set(map(tuple,row['keys'])))==592
    inference=RegionLanguageModel(model.cfg)
    weights={k:v for k,v in saved['weights'].items() if not k.startswith(('source_reconstruction.','regional_readout.'))}
    inference.load_state_dict(weights)
    batch=batch_from_keys(evaluation_keys('test',64))
    with torch.no_grad():
        a=answer_query(model,batch,encode_sources(model,batch)['state'],2)
        b=answer_query(inference,batch,encode_sources(inference,batch)['state'],2)
    torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
    torch.save({'schema':'shoebox-region-inference-v1','config':model.cfg.to_dict(),
                'weights':weights,'source_checkpoint_sha256':selection['selected_sha256']},run/'inference.pt')
    output={'initial_weights_optimizer_and_rng_preserved':True,'selected_phase_step':saved['step'],
            'selected_optimizer_step':completed_updates(saved),'finished_phase_step':last['step'],
            'qualified_validation_checkpoint':bool(qualified),'mastered_step':mastered,
            'final_validation_gate_failures':checked[-1]['gate_failures'],
            'test_gate_failures':gate_failures(metrics['test'],last['protocol']['threshold']),
            'checked_prediction_groups':len(result['controls']),
            'inference_parameters':sum(p.numel() for p in inference.parameters()),
            'training_heads_removed_with_exact_logits':True,'progress':metrics}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
