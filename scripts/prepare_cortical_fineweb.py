"""Download a small, auditable FineWeb byte corpus through the official viewer.

The training budget counts next-byte targets, with one unscored prefix byte
per independent stream. Validation starts in a different source document.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(rows, *, train_targets, batch, validation_targets, validation_batch):
    if train_targets % batch or validation_targets % validation_batch:
        raise ValueError('Target budgets must divide evenly across batch rows.')
    if min(train_targets, batch, validation_targets, validation_batch) < 1:
        raise ValueError('Budgets and batch sizes must be positive.')
    sizes = {'train': train_targets+batch,
             'validation': validation_targets+validation_batch}
    payloads, records = {}, {}
    cursor = 0
    for split, needed in sizes.items():
        content, ledger = bytearray(), []
        while len(content) < needed:
            if cursor == len(rows):
                raise ValueError('Not enough complete source rows for requested budgets.')
            item = rows[cursor]
            cursor += 1
            if item.get('truncated_cells'):
                raise ValueError('Viewer truncated a source cell; refuse partial source data.')
            row = item['row']
            encoded = row['text'].encode('utf-8') + b'\n\n'
            used = min(len(encoded), needed-len(content))
            start = len(content)
            content.extend(encoded[:used])
            ledger.append({'row_idx': item['row_idx'], 'id': row['id'],
                           'url': row.get('url'), 'dump': row.get('dump'),
                           'source_text_sha256': sha(row['text'].encode('utf-8')),
                           'source_text_bytes': len(encoded)-2,
                           'offset': start, 'bytes_used': used})
        payloads[split] = bytes(content)
        records[split] = {'file': split+'.bin', 'bytes': needed,
                          'sha256': sha(content), 'documents': ledger}
    train_ids = {r['id'] for r in records['train']['documents']}
    val_ids = {r['id'] for r in records['validation']['documents']}
    train_hashes = {r['source_text_sha256'] for r in records['train']['documents']}
    val_hashes = {r['source_text_sha256'] for r in records['validation']['documents']}
    if train_ids & val_ids or train_hashes & val_hashes:
        raise ValueError('Training and validation source documents overlap.')
    return payloads, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-targets', type=int, default=100000)
    p.add_argument('--batch', type=int, default=200)
    p.add_argument('--validation-targets', type=int, default=4096)
    p.add_argument('--validation-batch', type=int, default=32)
    p.add_argument('--rows-json', type=Path)
    p.add_argument('--metadata-json', type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error('Choose a new output directory.')
    endpoint = 'https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(
        dict(dataset='HuggingFaceFW/fineweb', config='sample-10BT', split='train', offset=0, length=100))
    source = (args.rows_json.read_bytes() if args.rows_json else
              urllib.request.urlopen(endpoint, timeout=60).read())
    metadata = json.loads(args.metadata_json.read_bytes() if args.metadata_json else
                          urllib.request.urlopen('https://huggingface.co/api/datasets/HuggingFaceFW/fineweb', timeout=60).read())
    response = json.loads(source)
    if response.get('partial'):
        raise ValueError('Refuse a partial viewer response.')
    payloads, records = prepare(response['rows'], train_targets=args.train_targets,
                                batch=args.batch, validation_targets=args.validation_targets,
                                validation_batch=args.validation_batch)
    manifest = {'dataset': 'HuggingFaceFW/fineweb', 'config': 'sample-10BT', 'split': 'train',
                'dataset_url': 'https://huggingface.co/datasets/HuggingFaceFW/fineweb',
                'viewer_endpoint': endpoint, 'dataset_revision_observed': metadata['sha'],
                'revision_note': 'Viewer rows are served from main; byte hashes pin the actual downloaded slice.',
                'viewer_response_sha256': sha(source),
                'prepared_utc': datetime.now(timezone.utc).isoformat(),
                'train_targets': args.train_targets, 'batch': args.batch,
                'validation_targets': args.validation_targets, 'validation_batch': args.validation_batch,
                'preprocessing': 'UTF-8 text; two newline bytes between source documents; fixed byte truncation.',
                'split_policy': 'Validation starts after the final training source document; no shared source IDs or text hashes.',
                'files': records}
    args.out.mkdir(parents=True)
    for name, data in payloads.items():
        (args.out/(name+'.bin')).write_bytes(data)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({name: {'bytes': len(data), 'sha256': sha(data)} for name, data in payloads.items()}))


if __name__ == '__main__':
    main()
