"""Audit acquisition selection, source budgets and completed held-out results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.memory_acquisition import acquisition_cases,delayed_cases,pack_cases
from experiments.text_regions.data import split_bytes


def compact(evaluation):
    result={}
    for name,row in evaluation['measurements'].items():
        result[name]={k:v for k,v in row.items() if k in ('accuracy','nll','answer_nll','targets',
            'answers','cases','complete_set_accuracy','answer_switch_fraction','value_margin','groups','owners')}
        if 'rows' in row:
            counts={}
            for case in row['rows']:
                for answer,prediction in zip(case['answers'],case['predictions']):
                    group=counts.setdefault(chr(answer),dict(targets=0,correct=0,predictions={}))
                    group['targets']+=1;group['correct']+=answer==prediction
                    key=chr(prediction);group['predictions'][key]=group['predictions'].get(key,0)+1
            result[name]['per_value']=counts
    return result


def stream_cohorts(evaluation_path):
    groups={}
    for split in ('test','long_delay'):
        path=evaluation_path.parent/(evaluation_path.stem+'_'+split+'_traces.jsonl')
        if not path.exists():continue
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        predicates=dict(native_acquisition=lambda r:r['cycle']==-1,
            later_exact=lambda r:r['kind']=='exact' and r['cycle']>=0,
            corrected_exact=lambda r:r['kind']=='exact' and r['correction'],
            exact_beyond_96=lambda r:r['kind']=='exact' and r['delay']>96,
            exact_with_interference=lambda r:r['kind']=='exact' and r['intervening_writes']>0)
        groups[split]={}
        for name,predicate in predicates.items():
            selected=[r for r in rows if predicate(r)]
            groups[split][name]=dict(targets=len(selected),
                accuracy=sum(r['prediction']==r['answer'] for r in selected)/len(selected) if selected else None,
                nll=sum(r['nll'] for r in selected)/len(selected) if selected else None)
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/memory_acquisition')
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    a=p.parse_args();torch.set_num_threads(1)
    raw=a.data.read_bytes();train,_,_=split_bytes(raw)
    result=dict(decision=json.loads((a.root/'decision.json').read_text()),
        parent=compact(json.loads((a.root/'parent_evaluation.json').read_text())),arms={})
    result['parent']['stream_cohorts']=stream_cohorts(a.root/'parent_evaluation.json')
    for directory in sorted(a.root.iterdir()):
        if not (directory/'protocol.json').exists():continue
        protocol=json.loads((directory/'protocol.json').read_text())
        rows=[json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
        saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
        best=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        initial=json.loads((directory/'initial.json').read_text())
        assert [r['step'] for r in rows]==list(range(1,saved['step']+1))
        opportunities=[(0,initial['acquisition']['nll'])]+[(r['step'],r['validation']['acquisition']['nll'])
            for r in rows if 'validation' in r]
        selected=min(opportunities,key=lambda x:x[1])[0]
        assert selected==best['step'],(directory,selected,best['step'])
        assert saved['raw_sha256']==hashlib.sha256(raw).hexdigest()
        assert all(bool(torch.isfinite(v).all()) for v in saved['weights'].values() if v.is_floating_point())
        p=protocol['protocol'];data=TaskStreamConfig.from_dict(protocol['data_config'])
        first_document=p.get('initial_document',0)
        cursor=first_document
        memory_bytes=padded=0
        for row in rows:
            cases=acquisition_cases(data,split='train',seed=p['data_seed'],start=cursor,
                count=p['cases'],distractor_bytes=train)
            gap=p.get('gaps',[0])[(row['step']-1)%len(p.get('gaps',[0]))]
            if gap:cases=delayed_cases(cases,gap,train)
            packed=pack_cases(cases);cursor+=p['cases']
            memory_bytes+=int(packed[3].sum());padded+=packed[0].numel()
            assert row['next_document']==cursor
            assert row['totals']['memory_bytes']==memory_bytes
            assert row['totals']['memory_padded_positions']==padded
            assert row['totals']['memory_targets']==(cursor-first_document)*len(data.values)
            assert row['totals']['language_bytes']==row['step']*p['language_batch']*p['language_context']
        assert saved['next_document']==cursor
        assert saved['totals']==rows[-1]['totals']
        same_weights=all(torch.equal(v,saved['weights'][k]) for k,v in best['weights'].items())
        arm=dict(completed_updates=saved['step'],planned_maximum=p['steps'],selected_update=best['step'],
            selected_and_final_weights_equal=same_weights,totals=saved['totals'],
            validation=[dict(step=0,**initial)]+[dict(step=r['step'],**r['validation']) for r in rows if 'validation'in r],
            protocol=p,finite_weights=True,reconstructed_training_budgets=True)
        for filename,tag in [('evaluation.json','selected_test'),('final_evaluation.json','final_test')]:
            path=directory/filename
            if path.exists():
                evaluation=json.loads(path.read_text())
                checkpoint=directory/('best.pt' if tag=='selected_test' else 'last.pt')
                assert evaluation['identity']['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                arm[tag]=compact(evaluation)
                arm[tag]['stream_cohorts']=stream_cohorts(path)
                expected={'language',*(f'gap_{gap}' for gap in evaluation['identity']['gaps'])}
                if not evaluation['identity']['skip_streams']:expected.update(('test','long_delay'))
                arm[tag]['complete']=expected.issubset(evaluation['measurements'])
        if same_weights and 'selected_test' in arm:arm['final_test']=arm['selected_test']
        result['arms'][directory.name]=arm
    (a.root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{k:arm[k] for k in ('completed_updates','selected_update','totals')}
                      for name,arm in result['arms'].items()}))


if __name__=='__main__':main()
