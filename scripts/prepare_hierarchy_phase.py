"""Prepare fresh training documents while retaining the exact validation set."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(rows, previous, out, *, train_targets=524288, batch=8):
    previous, out = Path(previous), Path(out)
    parent = json.loads((previous/'manifest.json').read_text())
    if train_targets < 1 or batch < 1 or train_targets % batch:
        raise ValueError('Training targets must divide into positive independent streams.')
    for split in ('train', 'validation'):
        if sha((previous/(split+'.bin')).read_bytes()) != parent['files'][split]['sha256']:
            raise ValueError('Parent corpus checksum mismatch.')
    historical = parent.get('historical_training_documents', [])+parent['files']['train']['documents']
    excluded = historical+parent['files']['validation']['documents']+parent.get('reserved_validation_documents', [])
    ids = {r['id'] for r in excluded}
    hashes = {r.get('source_sha256', r.get('source_text_sha256')) for r in excluded}
    content, documents = bytearray(), []
    duplicates = 0
    for item in sorted(rows, key=lambda r: r['row_idx']):
        if item.get('truncated_cells'):
            raise ValueError('Refuse truncated source text.')
        row = item['row']
        digest = sha(row['text'].encode('utf-8'))
        if row['id'] in ids or digest in hashes:
            duplicates += 1
            continue
        encoded = row['text'].encode('utf-8')+b'\n\n'
        n = min(len(encoded), train_targets+batch-len(content))
        if not n:
            break
        ids.add(row['id']); hashes.add(digest)
        documents.append(dict(id=row['id'], row_idx=item['row_idx'], source_sha256=digest,
                              offset=len(content), bytes=n))
        content.extend(encoded[:n])
    if len(content) != train_targets+batch:
        raise ValueError('Insufficient fresh source bytes; fetch more complete documents.')
    manifest = dict(format='shoebox-hierarchy-data-1', dataset=parent['dataset'],
                    train_targets=train_targets, batch=batch, validation_targets=parent['validation_targets'],
                    validation_document_cap=parent['validation_document_cap'],
                    parent_train_sha256=parent['files']['train']['sha256'],
                    historical_training_documents=historical,
                    reserved_validation_documents=parent.get('reserved_validation_documents', []),
                    duplicate_or_reserved_documents_skipped=duplicates,
                    preprocessing=parent['preprocessing'],
                    files=dict(train=dict(sha256=sha(content), bytes=len(content), documents=documents),
                               validation=deepcopy(parent['files']['validation'])))
    out.mkdir(parents=True, exist_ok=False)
    (out/'train.bin').write_bytes(content)
    (out/'validation.bin').write_bytes((previous/'validation.bin').read_bytes())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows-json', nargs='+', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-targets', type=int, default=524288)
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()
    rows = []
    for path in args.rows_json:
        data = json.loads(path.read_text())
        if data.get('partial'):
            raise ValueError('Refuse a partial viewer response.')
        rows.extend(data['rows'])
    result = prepare(rows, args.previous, args.out, train_targets=args.train_targets, batch=args.batch)
    print(json.dumps(dict(train_targets=result['train_targets'], documents=len(result['files']['train']['documents']),
                          validation_sha256=result['files']['validation']['sha256'])))


if __name__ == '__main__':
    main()
