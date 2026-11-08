"""Reconstruct the paired phase's budgets, selection and held-out evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.memory_selectivity import selective_cases
from experiments.text_regions.memory_acquisition import pack_cases
from experiments.text_regions.paired_recall import paired_batch
from scripts.run_paired_recall import MODES
from scripts.summarize_memory_selectivity import evaluation


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/paired_recall')
    p.add_argument('--parent',type=Path,default=ROOT/'artifacts/memory_selectivity/selective/best.pt')
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    a=p.parse_args(); torch.set_num_threads(1)
    parent=torch.load(a.parent,map_location='cpu',weights_only=True)
    directory=a.root/'train'
    last=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
    best=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    protocol=last['protocol']; data=TaskStreamConfig.from_dict(last['data_config'])
    assert last['config']==best['config']==parent['config']
    assert last['data_config']==best['data_config']==parent['data_config']
    assert protocol['parent_sha256']==digest(a.parent)
    assert protocol['generator_sha256']==digest(ROOT/'experiments/text_regions/paired_recall.py')
    assert protocol['base_generator_sha256']==digest(ROOT/'experiments/text_regions/memory_selectivity.py')
    assert last['raw_sha256']==digest(a.data)==parent['raw_sha256']
    train,_,_=split_bytes(a.data.read_bytes())
    rows=[json.loads(x) for x in (directory/'history.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,protocol['steps']+1))
    assert last['step']==protocol['steps']
    initial=read(directory/'initial.json')
    opportunities=[dict(step=0,**initial)]+[dict(step=r['step'],**r['validation']) for r in rows if 'validation'in r]
    selected=min(opportunities,key=lambda r:r['selection_score'])
    assert selected['step']==best['step'] and selected['selection_score']==best['best']
    cursor=protocol['initial_document']
    assert cursor==parent['next_document']
    totals=dict(memory_targets=0,memory_bytes=0,memory_padded_positions=0,executed_positions=0,
                language_bytes=0,by_mode={mode:0 for mode in ('paired','single','correction')})
    selected_totals=None
    for row in rows:
        step=row['step']-1; mode=MODES[step%len(MODES)]; gap=protocol['gaps'][step%len(protocol['gaps'])]
        options=dict(split='train',seed=protocol['data_seed'],start=cursor,count=1,gap=gap,distractor_bytes=train)
        if mode=='paired':
            batch=paired_batch(data,**options)
            targets=len(batch['answers']); memory_bytes=batch['full'].numel(); padded=memory_bytes
            executed=batch['history'].numel()+batch['query'].numel()
        else:
            options['count']=len(data.values) if mode=='single' else 1
            packed=pack_cases(selective_cases(data,mode=mode,**options))
            targets=len(packed[2]); memory_bytes=int(packed[3].sum()); padded=packed[0].numel(); executed=padded
        totals['memory_targets']+=targets; totals['memory_bytes']+=memory_bytes
        totals['memory_padded_positions']+=padded; totals['executed_positions']+=executed
        totals['language_bytes']+=protocol['language_batch']*protocol['language_context']
        totals['by_mode'][mode]+=targets; cursor+=options['count']
        assert row['mode']==mode and row['gap']==gap and row['next_document']==cursor
        assert row['totals']==totals
        if row['step']==best['step']: selected_totals=json.loads(json.dumps(totals))
    if best['step']==0:
        selected_totals={k:({name:0 for name in value} if isinstance(value,dict) else 0)
                         for k,value in totals.items()}
    assert last['next_document']==cursor and last['totals']==totals and best['totals']==selected_totals
    for checkpoint in (last,best):
        assert all(bool(torch.isfinite(v).all()) for v in checkpoint['weights'].values() if v.is_floating_point())
    comparisons=dict(parent=evaluation(a.root/'parent_test.json',a.parent),
                     selected=evaluation(a.root/'selected_test.json',directory/'best.pt'))
    identities=[v['identity'] for v in comparisons.values()]
    for key in ('single_cases','selective_cases','documents','data_seed','start','raw_sha256','evaluator_sha256','generator_sha256'):
        assert len({v[key] for v in identities})==1
    assert identities[0]['data_seed']==1031 and identities[0]['start']==30000
    diagnostics={name:read(a.root/f'{name}_diagnostic.json') for name in ('parent','selected')}
    for name,checkpoint in (('parent',a.parent),('selected',directory/'best.pt')):
        identity=diagnostics[name]['identity']
        assert identity['checkpoint_sha256']==digest(checkpoint)
        assert identity['diagnostic_sha256']==digest(ROOT/'scripts/diagnose_paired_recall.py')
        assert identity['generator_sha256']==protocol['generator_sha256']
        assert set(diagnostics[name]['gaps'])=={'0','32'}
    result=dict(completed_updates=last['step'],selected_update=best['step'],
        selected_and_final_weights_equal=all(torch.equal(v,last['weights'][k]) for k,v in best['weights'].items()),
        parameters=sum(v.numel() for v in best['weights'].values()),finite_weights=True,
        reconstructed_training_budgets=True,protocol=protocol,totals=totals,selected_totals=selected_totals,
        initial_document=protocol['initial_document'],next_document=cursor,validation=opportunities,
        comparisons=comparisons,diagnostics=diagnostics,
        production_equivalence=read(a.root/'production_equivalence.json'),
        executed_position_saving=1-totals['executed_positions']/totals['memory_padded_positions'])
    (a.root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('completed_updates','selected_update','totals','executed_position_saving')}))


if __name__=='__main__': main()
