"""Reconstruct source budgets, validation selection and held-out comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.memory_acquisition import acquisition_cases, delayed_cases, pack_cases
from experiments.text_regions.memory_selectivity import MODES, training_cases, summarize_values
from scripts.summarize_memory_acquisition import stream_cohorts


def read(path):
    return json.loads(path.read_text())


def evaluation(path, checkpoint):
    result = read(path)
    assert result['identity']['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = {'language', *(f'single_{g}' for g in (0,8,24,64,128)),
        *(f'{mode}_{g}' for mode in MODES[1:] for g in (0,24,64)),
        *(f'query_pairs_{g}' for g in (0,24,64))}
    if not result['identity']['skip_streams']:
        expected.update(('test','long_delay'))
    assert expected == set(result['measurements'])
    measurements = {name:summarize_values(row) if 'rows' in row else row
                    for name,row in result['measurements'].items()}
    for gap in (0,24,64):
        items = []
        for row in result['measurements'][f'correction_{gap}']['rows']:
            previous = row['query']['previous_answer']
            items.extend((answer,prediction,nll,previous)
                for answer,prediction,nll in zip(row['answers'],row['predictions'],row['nll'])
                if answer != previous)
        measurements[f'changed_correction_{gap}'] = dict(targets=len(items),
            accuracy=sum(a == p for a,p,n,old in items)/len(items),
            nll=sum(n for a,p,n,old in items)/len(items),
            obsolete_answer_fraction=sum(p == old for a,p,n,old in items)/len(items))
    return dict(identity=result['identity'], complete=True, measurements=measurements,
                stream_cohorts=stream_cohorts(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/memory_selectivity')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    a = p.parse_args(); torch.set_num_threads(1)
    raw = a.data.read_bytes(); train,_,_ = split_bytes(raw)
    acquired = torch.load(a.root/'acquired_parent.pt',map_location='cpu',weights_only=True)
    selection = read(a.root/'selection.json')
    assert selection['selected'] == min(selection['candidates'],key=lambda row:row['validation_nll'])
    assert selection['selected']['checkpoint_sha256'] == hashlib.sha256((a.root/'retention_parent.pt').read_bytes()).hexdigest()
    result = dict(selection=selection,
        parent=evaluation(a.root/'parent_evaluation.json', a.root/'acquired_parent.pt'), arms={})
    for name in ('retention','retention_low','selective'):
        directory = a.root/name
        last = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
        protocol = last['protocol']; data = TaskStreamConfig.from_dict(last['data_config'])
        assert last['config'] == best['config'] == acquired['config']
        assert last['data_config'] == best['data_config'] == acquired['data_config']
        assert last['raw_sha256'] == hashlib.sha256(raw).hexdigest()
        parent = a.root/'retention_parent.pt' if name == 'selective' else a.root/'acquired_parent.pt'
        assert protocol['parent_sha256'] == hashlib.sha256(parent.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1,last['step']+1))
        assert last['step'] == protocol['steps']
        initial = read(directory/'initial.json')
        score = (lambda v:v['selection_score']) if name == 'selective' else (lambda v:v['acquisition']['nll'])
        opportunities = [(0,score(initial))]+[(r['step'],score(r['validation'])) for r in rows if 'validation' in r]
        selected = min(opportunities, key=lambda row:row[1])[0]
        assert selected == best['step']
        cursor = protocol['initial_document']
        totals = dict(memory_targets=0,memory_bytes=0,memory_padded_positions=0,language_bytes=0)
        if name == 'selective':
            totals['by_mode'] = {mode:0 for mode in MODES}
        for row in rows:
            step = row['step']-1
            if name != 'selective':
                cases = acquisition_cases(data, split='train', seed=protocol['data_seed'], start=cursor,
                    count=protocol['cases'], distractor_bytes=train)
                gap = protocol['gaps'][step % len(protocol['gaps'])]
                cases = delayed_cases(cases,gap,train)
                cursor += protocol['cases']
            else:
                cases,mode,gap,cursor = training_cases(data,step=step,start=cursor,
                    seed=protocol['data_seed'],gaps=protocol['gaps'],distractor_bytes=train)
                assert row['mode'] == mode
            assert row['gap'] == gap
            packed = pack_cases(cases)
            totals['memory_targets'] += len(packed[2])
            totals['memory_bytes'] += int(packed[3].sum())
            totals['memory_padded_positions'] += packed[0].numel()
            totals['language_bytes'] += protocol['language_batch']*protocol['language_context']
            if name == 'selective':
                totals['by_mode'][mode] += len(packed[2])
            assert row['next_document'] == cursor and row['totals'] == totals
        assert last['next_document'] == cursor and last['totals'] == totals
        for saved in (last,best):
            assert all(bool(torch.isfinite(v).all()) for v in saved['weights'].values() if v.is_floating_point())
        same = all(torch.equal(v,last['weights'][key]) for key,v in best['weights'].items())
        result['arms'][name] = dict(completed_updates=last['step'],selected_update=best['step'],
            selected_and_final_weights_equal=same,totals=totals,selected_totals=best['totals'],
            protocol=protocol,finite_weights=True,reconstructed_training_budgets=True,
            validation=[dict(step=0,**initial)]+[dict(step=r['step'],**r['validation']) for r in rows if 'validation' in r],
            selected_test=evaluation(directory/'evaluation.json',directory/'best.pt'))
    first,second = (result['arms'][name] for name in ('retention','retention_low'))
    assert first['totals'] == second['totals']
    assert {k:v for k,v in first['protocol'].items() if k != 'lr'} == {
        k:v for k,v in second['protocol'].items() if k != 'lr'}
    result['matched_retention_schedules'] = True
    identities = [result['parent']['identity']]+[arm['selected_test']['identity'] for arm in result['arms'].values()]
    for key in ('single_cases','selective_cases','documents','data_seed','start','raw_sha256','evaluator_sha256','generator_sha256'):
        assert len({identity[key] for identity in identities}) == 1
    (a.root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{k:arm[k] for k in ('completed_updates','selected_update','totals')}
        for name,arm in result['arms'].items()}))


if __name__ == '__main__':
    main()
