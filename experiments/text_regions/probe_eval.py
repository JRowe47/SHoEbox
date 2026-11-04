"""Score a saved probe model on fresh episodes without resuming training."""
import argparse
import json
from pathlib import Path
import torch
from .probe_train import evaluate_probe, load_probe, parameter_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--gaps', type=int, nargs='+')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.episodes, args.batch, args.threads) < 1 or (args.gaps and min(args.gaps) < 0):
        parser.error('Counts must be positive and gaps nonnegative.')
    torch.set_num_threads(args.threads)
    model, _, _, saved = load_probe(args.checkpoint, restore_rng=False)
    protocol = saved['protocol']
    gaps = args.gaps if args.gaps is not None else [protocol['gap']]
    result = {'checkpoint_step': saved['step'], 'training_protocol': protocol,
              'parameters': parameter_summary(model), 'evaluations': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for gap in dict.fromkeys(gaps):
        evaluation = evaluate_probe(model, protocol['kind'], gap=gap, count=args.episodes,
                                    batch_size=args.batch, segment=protocol['segment'],
                                    seed=protocol['seed'], split=args.split,
                                    query_every=protocol['query_every'])
        result['evaluations'].append(evaluation)
        args.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'gap': gap, 'split': args.split,
                          'controls': {key: {field: value for field, value in row.items()
                                             if field not in ('predictions', 'episode_ids')}
                                       for key, row in evaluation['controls'].items()}}), flush=True)


if __name__ == '__main__':
    main()
