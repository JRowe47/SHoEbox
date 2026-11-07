"""Audit paired exposure, selection and completed held-out measurements."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/connected_recall')
    p.add_argument('--out', type=Path, default=ROOT/'experiments/text_regions/results/connected_recall/summary.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    evaluation = json.loads((a.root/'evaluation.json').read_text())
    # Separate evaluation roots allow independent arms to run concurrently
    # without racing on a shared cache or aggregate result file.
    for arm in ('connected','cut_16'):
        path = a.root/f'eval_{arm}'/'evaluation.json'
        if path.exists():
            evaluation.update(json.loads(path.read_text()))
    histories = {arm: [json.loads(line) for line in (a.root/arm/'history.jsonl').read_text().splitlines()]
                 for arm in ('connected','cut_16')}
    expected = None; arms = {}
    for arm, history in histories.items():
        saved = torch.load(a.root/arm/'last.pt', map_location='cpu', weights_only=True)
        selected = torch.load(a.root/arm/'best.pt', map_location='cpu', weights_only=True)
        protocol = saved['protocol']
        assert [row['step'] for row in history] == list(range(1, protocol['steps']+1))
        assert saved['step'] == protocol['steps']
        assert all(bool(t.isfinite().all()) for t in saved['weights'].values() if t.is_floating_point())
        initial = json.loads((a.root/arm/'initial.json').read_text())
        candidates = [(initial['score'],0)] + [(row['validation']['score'], row['step'])
                                              for row in history if 'validation' in row]
        score, step = min(candidates)
        assert selected['step'] == step and selected['best'] == score
        assert evaluation[arm+'_best']['step'] == selected['step']
        assert evaluation[arm+'_last']['step'] == saved['step']
        comparable = {k:v for k,v in protocol.items() if k != 'cut_at_segments'}
        if expected is None: expected = comparable
        else: assert expected == comparable
        for source in ('language','memory'):
            total = saved['totals'][source]
            assert total['bytes'] == protocol['steps']*protocol['batch']*protocol['context']
            assert total == history[-1]['totals'][source]
        effective_key = 'within_window' if arm == 'connected' else 'within_segment'
        arms[arm] = dict(selected_step=step, final_step=saved['step'], validation_score=score,
            totals=saved['totals'], peak_process_mib=max(row['peak_process_mib'] for row in history),
            effective_exact_credit={group: dict(count=row[effective_key], answers=row['answers'],
                fraction=row[effective_key]/row['answers'] if row['answers'] else None)
                for group,row in saved['totals']['memory']['coverage'].items()})
    for left,right in zip(histories['connected'], histories['cut_16']):
        assert left['memory']['coverage'] == right['memory']['coverage']
        for source in ('language','memory'):
            for key in ('bytes','answers','experts','updates'):
                assert left['totals'][source][key] == right['totals'][source][key]
    metrics = {}
    for name in ('parent','connected_best','connected_last','cut_16_best','cut_16_last'):
        row = evaluation[name]
        metrics[name] = dict(step=row['step'], signature=row['signature'],
            language=row['language'], ordinary=row['test'], long_delay=row['long_delay'],
            acquisition={k:v for k,v in row['acquisition'].items() if k != 'rows'},
            delayed_values={k:v for k,v in row['delayed'].items() if k != 'rows'})
    result = dict(protocol=expected, arms=arms, evaluation=metrics,
        activation_checkpoint=json.loads((a.root/'activation_comparison.json').read_text()),
        checks=dict(paired_protocol=True, identical_exposure=True, complete_fixed_budget=True,
                    selection_reproduced=True, finite_final_weights=True))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name:dict(language=row['language']['nll'],answer=row['ordinary']['answer_nll'],
        acquisition=row['acquisition']['accuracy'], complete=row['acquisition']['complete_set_accuracy'])
        for name,row in metrics.items()}), flush=True)


if __name__ == '__main__':
    main()
