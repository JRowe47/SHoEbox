"""Make an auditable pilot corpus from cached official FineWeb viewer responses.

Validation uses many separate documents, each capped at 4096 targets. Every
validation document begins in fresh recurrent state during evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(rows, out, train_targets=194560, batch=8, validation_targets=65536, document_cap=4096, reserved=()):
    if train_targets < 1 or batch < 1 or train_targets % batch or validation_targets < 1 or document_cap < 1:
        raise ValueError('Invalid corpus budget.')
    unique, ids, hashes = [], {r['id'] for r in reserved}, {r['source_text_sha256'] for r in reserved}
    for item in sorted(rows, key=lambda r: r['row_idx']):
        if item.get('truncated_cells'):
            raise ValueError('Refuse truncated source text.')
        row = item['row']; encoded = row['text'].encode('utf-8')+b'\n\n'
        digest = sha(row['text'].encode('utf-8'))
        if row['id'] in ids or digest in hashes:
            continue
        ids.add(row['id']); hashes.add(digest)
        unique.append((item, encoded, digest))
    heldout, count = [], 0
    while unique and count < validation_targets:
        item, encoded, digest = unique.pop()
        n = min(len(encoded)-1, document_cap, validation_targets-count)
        heldout.append((item, encoded[:n+1], digest))
        count += n
    if count < validation_targets or (validation_targets >= 65536 and len(heldout) < 16):
        raise ValueError('Need at least 64k validation targets spanning 16+ source documents.')
    train = bytearray(); train_docs = []
    for item, encoded, digest in unique:
        n = min(len(encoded), train_targets+batch-len(train))
        if n <= 0:
            break
        train_docs.append(dict(id=item['row']['id'], row_idx=item['row_idx'], source_sha256=digest,
                               offset=len(train), bytes=n))
        train.extend(encoded[:n])
    if len(train) != train_targets+batch:
        raise ValueError('Insufficient disjoint training documents for requested budget.')
    validation = bytearray(); val_docs = []
    for item, encoded, digest in reversed(heldout):
        val_docs.append(dict(id=item['row']['id'], row_idx=item['row_idx'], source_sha256=digest,
                            offset=len(validation), bytes=len(encoded)))
        validation.extend(encoded)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'train.bin').write_bytes(train)
    (out/'validation.bin').write_bytes(validation)
    manifest = dict(format='shoebox-hierarchy-data-1', dataset='HuggingFaceFW/fineweb',
                    train_targets=train_targets, batch=batch, validation_targets=count,
                    validation_document_cap=document_cap,
                    reserved_validation_documents=list(reserved),
                    preprocessing='UTF-8 text plus two newlines. Validation documents scored separately.',
                    files=dict(train=dict(sha256=sha(train), bytes=len(train), documents=train_docs),
                               validation=dict(sha256=sha(validation), bytes=len(validation), documents=val_docs)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows-json', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-targets', type=int, default=194560)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--validation-targets', type=int, default=65536)
    p.add_argument('--reserve-manifest', type=Path, help='Also keep the old continuity documents out of both splits.')
    args = p.parse_args()
    rows = []
    for path in args.rows_json:
        data = json.loads(path.read_text())
        if data.get('partial'):
            raise ValueError('Refuse a partial viewer response.')
        rows.extend(data['rows'])
    reserved = (json.loads(args.reserve_manifest.read_text())['files']['validation']['documents']
                if args.reserve_manifest else ())
    result = prepare(rows, args.out, args.train_targets, args.batch, args.validation_targets, reserved=reserved)
    print(json.dumps({k: result[k] for k in ('train_targets', 'validation_targets')}))


if __name__ == '__main__':
    main()
