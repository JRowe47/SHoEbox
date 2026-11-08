"""Compare completed pilot/control runs only when training protocols match."""
import argparse
import json
from pathlib import Path


def compare(pilot, control):
    for record in (pilot, control):
        if not record['complete']:
            raise ValueError('Only completed runs are comparable.')
    if pilot['protocol']['local_only'] or not control['protocol']['local_only']:
        raise ValueError('Need the hierarchy followed by its local-only control.')
    a, b = (dict(r['protocol']) for r in (pilot, control))
    a.pop('local_only'); b.pop('local_only')
    if a != b or pilot['trained_targets'] != control['trained_targets']:
        raise ValueError('Corpus, code, update count and training settings must match.')
    p, c = pilot['evaluation'][-1], control['evaluation'][-1]
    if not p['full'] or not c['full'] or p['targets'] != c['targets']:
        raise ValueError('Need matching full held-out evaluations.')
    return dict(updates=pilot['updates'], trained_targets=pilot['trained_targets'], heldout_targets=p['targets'],
                pilot_nll=p['nll'], local_nll=c['nll'], pilot_minus_local_nll=p['nll']-c['nll'],
                upper_swap_delta_nll=pilot['upper_swap_delta_nll'],
                hierarchy_gate=p['nll'] < c['nll'] and pilot['upper_swap_delta_nll'] > .01,
                pilot_bytes_per_second=pilot['training_bytes_per_second'],
                local_bytes_per_second=control['training_bytes_per_second'],
                pilot_cost=pilot['parameters'], local_cost=control['parameters'],
                baselines=pilot['baselines'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot', type=Path); p.add_argument('control', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = compare(json.loads(args.pilot.read_text()), json.loads(args.control.read_text()))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('pilot_cost', 'local_cost')}))


if __name__ == '__main__':
    main()
