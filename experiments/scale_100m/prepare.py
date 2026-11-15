"""Prepare exactly 20M unique FineWeb next-byte targets with a pinned dev split."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--previous-data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    previous = json.loads((a.previous_data / 'manifest.json').read_text())
    dev = previous['files']['development']
    dev_bytes = (a.previous_data / dev['file']).read_bytes()
    assert sha(dev_bytes) == dev['sha256']
    seen_ids = {d['id'] for d in dev['documents']}
    seen_hashes = {d['text_sha256'] for d in dev['documents']}

    def fetch(offset):
        path = a.cache / f'rows_{offset:06d}.json'
        if not path.exists():
            query = urllib.parse.urlencode(dict(dataset='HuggingFaceFW/fineweb',
                config='sample-10BT', split='train', offset=offset, length=100))
            with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?' + query, timeout=45) as r:
                body = r.read()
            result = json.loads(body)
            assert 'rows' in result and not result.get('partial')
            path.write_bytes(body)
        result = json.loads(path.read_bytes())
        assert not result.get('partial')
        return result['rows']

    data, docs, skipped = bytearray(), [], []
    remaining, offset = 20_000_000, 0
    while remaining:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            batches = list(pool.map(fetch, range(offset, offset + 400, 100)))
        offset += 400
        for batch in batches:
            for item in batch:
                r = item['row']
                raw = r['text'].encode('utf-8')
                digest = sha(raw)
                if item.get('truncated_cells') or r['id'] in seen_ids or digest in seen_hashes or len(raw) < 2:
                    skipped.append(item['row_idx'])
                    continue
                seen_ids.add(r['id']); seen_hashes.add(digest)
                body = (raw + b'\n\n')[:remaining + 1]
                start = len(data); data.extend(body)
                n = len(body) - 1; remaining -= n
                docs.append(dict(row_idx=item['row_idx'], id=r['id'], text_sha256=digest,
                    source_bytes=len(raw), start=start, stop=len(data), targets=n))
                if not remaining:
                    break
            if not remaining:
                break
        print(json.dumps(dict(downloaded_rows=offset, training_targets=20_000_000 - remaining)), flush=True)
    train = dict(file='train.bin', bytes=len(data), targets=20_000_000,
                 sha256=sha(data), documents=docs)
    if previous['files'].get('train', {}).get('targets') == 20_000_000:
        assert train['sha256'] == previous['files']['train']['sha256'], 'Source rows changed from the pinned 20M sample'
    assert not {d['id'] for d in docs} & {d['id'] for d in dev['documents']}
    assert not {d['text_sha256'] for d in docs} & {d['text_sha256'] for d in dev['documents']}
    (a.out / 'train.bin').write_bytes(data)
    (a.out / 'development.bin').write_bytes(dev_bytes)
    manifest = dict(dataset='HuggingFaceFW/fineweb', config='sample-10BT', split='train',
        previous_observed_revision=previous.get('observed_revision', previous.get('previous_observed_revision')),
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        source='Official datasets-server row API; cached source rows reused when available.',
        revision_note='Row API serves main; exact document and binary SHA-256 values pin the sample.',
        preprocessing='UTF-8 text plus two newlines. Final train document truncated to exact target budget.',
        split_policy='Preserve the prior D development documents; exclude their IDs and full-text hashes from training.',
        files=dict(train=train, development=dev), skipped_rows=skipped)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'documents'} | {'documents': len(v['documents'])}
                      for k, v in manifest['files'].items()}), flush=True)


if __name__ == '__main__':
    main()
