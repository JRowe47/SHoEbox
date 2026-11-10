"""Restore the exact published byte splits from pinned FineWeb documents."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,gzip,hashlib,json,os,re
from bisect import bisect_right
from prepare_data import verify,sha

ROOT=Path(__file__).resolve().parent


def fetch_file(job):
    revision,source,documents=job
    if not re.fullmatch(r'[0-9a-f]{40}',revision) or not re.fullmatch(r'sample/10BT/\d{3}_00000\.parquet',source):
        raise ValueError('Invalid pinned source')
    import fsspec
    import pyarrow.parquet as pq
    url=f'https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/{revision}/{source}'
    wanted={d['row_idx']:d for d in documents};result={}
    with fsspec.open(url,'rb',block_size=1_048_576,client_kwargs={'trust_env':True},cache_type='readahead') as stream:
        parquet=pq.ParquetFile(stream);offsets=[0]
        for i in range(parquet.num_row_groups):offsets.append(offsets[-1]+parquet.metadata.row_group(i).num_rows)
        if any(row<0 or row>=offsets[-1] for row in wanted):raise ValueError('Row index outside pinned source')
        groups=sorted({bisect_right(offsets,row)-1 for row in wanted})
        for group in groups:
            rows=parquet.read_row_group(group,columns=['id','text'],use_threads=False).to_pylist()
            for index,row in enumerate(rows,offsets[group]):
                if index not in wanted:continue
                d=wanted[index];raw=row['text'].encode('utf-8')
                if row['id']!=d['id'] or hashlib.sha256(raw).hexdigest()!=d['source_sha256']:raise ValueError('Pinned document differs')
                result[d['id']]=raw+b'\n\n'
    if len(result)!=len(wanted):raise ValueError('Missing pinned documents')
    return result


def restore(manifest_path,output,workers=3):
    if workers<1:raise ValueError('workers must be positive')
    raw=Path(manifest_path).read_bytes()
    if str(manifest_path).endswith('.gz'):raw=gzip.decompress(raw)
    manifest=json.loads(raw);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() and (output/'manifest.json').read_bytes()!=raw:raise ValueError('Existing manifest differs')
    missing=[]
    for name,split in manifest['splits'].items():
        if name not in ('calibration','train','validation','test'):raise ValueError('Invalid split name')
        path=output/(name+'.bin')
        if path.exists():
            if path.stat().st_size!=split['targets']+1 or sha(path)!=split['sha256']:raise ValueError(f'Existing split differs: {name}')
        else:missing.append(name)
    sources={}
    for name in missing:
        for d in manifest['splits'][name]['documents']:sources.setdefault(d['source_file'],[]).append(d)
    jobs=[(manifest['revision'],source,docs) for source,docs in sources.items()]
    payloads={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group in pool.map(fetch_file,jobs):payloads.update(group)
    for name in missing:
        split=manifest['splits'][name];blob=bytearray()
        for d in split['documents']:
            if len(blob)!=d['start']:raise ValueError('Invalid document span')
            offset=d.get('source_byte_offset',0);segment=payloads[d['id']][offset:offset+d['bytes_used']]
            if len(segment)!=d['bytes_used'] or hashlib.sha256(segment).hexdigest()!=d['segment_sha256']:raise ValueError('Segment hash mismatch')
            blob.extend(segment)
        if len(blob)!=split['targets']+1 or hashlib.sha256(blob).hexdigest()!=split['sha256']:raise ValueError('Split hash mismatch')
        path=output/(name+'.bin');pending=path.with_name(path.name+'.pending')
        with pending.open('wb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
    pending=output/'manifest.json.pending';pending.write_bytes(raw);os.replace(pending,output/'manifest.json')
    verify(output)
    return dict(restored=missing,verified=list(manifest['splits']),test_evaluated=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'datasets/comparison_manifest.json.gz')
    p.add_argument('--output',type=Path,default=ROOT/'data/comparison');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();print(json.dumps(restore(a.manifest,a.output,a.workers),indent=2))


if __name__=='__main__':main()
