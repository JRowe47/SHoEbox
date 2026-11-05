"""Read-only check of source retention in a composition checkpoint.

Run from the repository root with ``python -m
experiments.text_regions.results.composition.retention_diagnostic``.
The existing training-only reconstruction head is scored without fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from experiments.text_regions.composition import encode_sources, answer_query
from experiments.text_regions.composition_data import batch_from_keys, evaluation_keys
from experiments.text_regions.composition_train import load_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', default='test', choices=('train', 'validation', 'test', 'all'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, _, _, saved = load_checkpoint(args.checkpoint)
    if not hasattr(model, 'source_reconstruction'):
        parser.error('Checkpoint has no source reconstruction head.')
    model.eval()
    keys = evaluation_keys(args.split)
    scores = {}
    for offset in range(0, len(keys), 64):
        batch = batch_from_keys(keys[offset:offset+64])
        prefix = encode_sources(model, batch)
        query = answer_query(model, batch, prefix['state'], 2)
        for stage, result in (('prefix', prefix), ('query_k2', query)):
            for node, target in (('B', batch.first), ('C', batch.second)):
                logits = model.source_reconstruction(result['state'].ports.latest[node].payload)
                correct = logits.reshape(-1, 4, 4).argmax(-1).eq(target-65)
                row = scores.setdefault(stage+'_'+node,
                                        dict(entries=0, correct_entries=0, tables=0, exact_tables=0))
                row['entries'] += correct.numel()
                row['correct_entries'] += int(correct.sum())
                row['tables'] += batch.size
                row['exact_tables'] += int(correct.all(-1).sum())
    for row in scores.values():
        row['entry_accuracy'] = row['correct_entries']/row['entries']
        row['table_accuracy'] = row['exact_tables']/row['tables']
    result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  checkpoint_step=saved['step'], split=args.split, unique_queries=len(keys),
                  note='Existing local head, no fitting. Source tables recur across queries; '
                       'these are retained entries, not independent new source tables.', scores=scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
