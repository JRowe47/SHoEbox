"""Compare identical hierarchy checkpoints continued with joint/cyclic training."""
import argparse
from collections import Counter
import json
from pathlib import Path


def compare(joint, cycle):
    for run in (joint, cycle):
        if not run['complete'] or run['protocol']['local_only']:
            raise ValueError('Need two completed full-hierarchy runs.')
    a, b = (dict(run['protocol']) for run in (joint, cycle))
    schedule_a, schedule_b = a.pop('regional_cycle'), b.pop('regional_cycle')
    if schedule_a['mode'] != 'joint' or schedule_b['mode'] != 'cycle':
        raise ValueError('Expected joint baseline followed by cycle experiment.')
    if a != b or joint['trained_targets'] != cycle['trained_targets']:
        raise ValueError('Initial checkpoint, data, code and exposure budgets must match.')
    ja, ca = joint['evaluation'][-1], cycle['evaluation'][-1]
    if not ja['full'] or not ca['full'] or ja['targets'] != ca['targets']:
        raise ValueError('Need matching full validation evaluations.')
    return dict(
        format='shoebox-regional-cycle-comparison-1', protocol=a,
        additional_updates=joint['updates'], additional_targets=joint['trained_targets'],
        lifetime_updates=joint['lifetime_updates'], lifetime_targets=joint['lifetime_targets'],
        heldout_targets=ja['targets'], cycle_schedule=schedule_b,
        joint_nll=ja['nll'], cycle_nll=ca['nll'], cycle_minus_joint_nll=ca['nll']-ja['nll'],
        joint_span_nll=ja['span_nll'], cycle_span_nll=ca['span_nll'],
        joint_decoder_swap_delta=joint['upper_swap_delta_nll'],
        cycle_decoder_swap_delta=cycle['upper_swap_delta_nll'],
        joint_all_upper_swap_delta=joint['upper_all_swap_delta_nll'],
        cycle_all_upper_swap_delta=cycle['upper_all_swap_delta_nll'],
        joint_bytes_per_second=joint['training_bytes_per_second'],
        cycle_bytes_per_second=cycle['training_bytes_per_second'],
        phases=dict(Counter(r['phase'] for r in cycle['history'])),
        attenuated_updates=sum(r['local_scale'] != 1. for r in cycle['history']),
        improved=ca['nll'] < ja['nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('joint', type=Path)
    parser.add_argument('cycle', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--initial-audit', type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.joint.read_text()), json.loads(args.cycle.read_text()))
    if args.initial_audit:
        result['initial_audit'] = json.loads(args.initial_audit.read_text())
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
