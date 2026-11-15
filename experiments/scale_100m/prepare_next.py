"""Prepare new FineWeb documents while excluding all prior train/dev documents."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import urllib.error


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous-data', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--targets', type=int, default=20_000_000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.cache.mkdir(parents=True, exist_ok=True)
    previous = [json.loads((d/'manifest.json').read_text()) for d in a.previous_data]
    dev = previous[0]['files']['development']
    dev_raw = (a.previous_data[0]/dev['file']).read_bytes()
    assert sha(dev_raw) == dev['sha256']
    for manifest in previous:
        assert manifest['files']['development']['sha256'] == dev['sha256']
    all_previous = [d for manifest in previous for split in manifest['files'].values() for d in split['documents']]
    seen_ids = {d['id'] for d in all_previous}; seen_hashes = {d['text_sha256'] for d in all_previous}
    first_row = 1 + max(d['row_idx'] for d in all_previous)
    offset = (first_row//100)*100

    def fetch(offset):
        path = a.cache/f'rows_{offset:06d}.json'
        if not path.exists():
            query = urllib.parse.urlencode(dict(dataset='HuggingFaceFW/fineweb', config='sample-10BT',
                split='train', offset=offset, length=100))
            for attempt in range(12):
                try:
                    time.sleep(2)
                    with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+query, timeout=45) as response:
                        body = response.read()
                    result = json.loads(body)
                    assert result.get('rows') and not result.get('partial')
                    path.write_bytes(body)
                    break
                except Exception as error:
                    if attempt == 11:
                        raise
                    if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                        retry_after = error.headers.get('Retry-After','60')
                        delay = max(10,float(retry_after)) if retry_after.isdigit() else 60
                    else:
                        delay = min(60,5*(attempt+1))
                    print(json.dumps(dict(retrying_source_offset=offset,attempt=attempt+1,wait_seconds=delay)),flush=True)
                    time.sleep(delay)
        result = json.loads(path.read_bytes())
        assert result.get('rows') and not result.get('partial')
        return result['rows']

    data, docs, skipped = bytearray(), [], []
    remaining = a.targets
    while remaining:
        with ThreadPoolExecutor(max_workers=1) as pool:
            batches = list(pool.map(fetch, range(offset, offset+400, 100)))
        offset += 400
        for batch in batches:
            for item in batch:
                if item['row_idx'] < first_row:
                    continue
                row = item['row']; raw = row['text'].encode('utf-8'); digest = sha(raw)
                if item.get('truncated_cells') or len(raw)<2 or row['id'] in seen_ids or digest in seen_hashes:
                    skipped.append(item['row_idx']); continue
                seen_ids.add(row['id']); seen_hashes.add(digest)
                body = (raw+b'\n\n')[:remaining+1]
                start = len(data); data.extend(body); n = len(body)-1; remaining -= n
                docs.append(dict(row_idx=item['row_idx'], id=row['id'], text_sha256=digest,
                    source_bytes=len(raw), start=start, stop=len(data), targets=n))
                if not remaining:
                    break
            if not remaining:
                break
        print(json.dumps(dict(next_row=offset, prepared_targets=a.targets-remaining)), flush=True)
    assert not {d['id'] for d in docs} & {d['id'] for d in all_previous}
    assert not {d['text_sha256'] for d in docs} & {d['text_sha256'] for d in all_previous}
    assert sum(d['targets'] for d in docs) == a.targets
    (a.out/'train.bin').write_bytes(data); (a.out/'development.bin').write_bytes(dev_raw)
    manifest = dict(dataset='HuggingFaceFW/fineweb', config='sample-10BT', split='train',
        prepared_utc=datetime.now(timezone.utc).isoformat(), source='Official datasets-server row API',
        revision_note='Row API serves main; complete-text and binary SHA-256 hashes pin this sample.',
        preprocessing='UTF-8 text plus two newlines; final document truncated to exact target allowance.',
        split_policy='Fresh source documents; exclude all previous training and development IDs and complete-text hashes.',
        first_source_row=first_row, last_source_row=docs[-1]['row_idx'],
        excluded_previous_train_sha256=[m['files']['train']['sha256'] for m in previous],
        prior_manifest_sha256=[sha((d/'manifest.json').read_bytes()) for d in a.previous_data],
        files=dict(train=dict(file='train.bin', bytes=len(data), targets=a.targets, sha256=sha(data), documents=docs),
            development=dev), skipped_rows=skipped)
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(completed=True, targets=a.targets, documents=len(docs),
        first_source_row=first_row, last_source_row=docs[-1]['row_idx'], train_sha256=sha(data),
        development_sha256=dev['sha256'])), flush=True)


if __name__ == '__main__':
    main()
