"""Prepare exact, disjoint FineWeb byte streams for the scaling comparison."""
from pathlib import Path
import argparse,hashlib,json,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from shoebox_runtime.model import atomic_json

REVISION='9bb295ddab0e05d785b879661af7260fed5140fc'
SEED=2026111031
SPLITS={'calibration':1_000_000,'train':25_000_000,'validation':2_000_000,'test':4_000_000}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identities(obj,ids,hashes):
    if isinstance(obj,dict):
        if 'id' in obj and 'source_sha256' in obj:ids.add(obj['id']);hashes.add(obj['source_sha256'])
        if 'ids' in obj and isinstance(obj['ids'],list):ids.update(obj['ids'])
        if 'source_sha256' in obj and isinstance(obj['source_sha256'],list):hashes.update(obj['source_sha256'])
        for v in obj.values():identities(v,ids,hashes)
    elif isinstance(obj,list):
        for v in obj:identities(v,ids,hashes)

def fetch(index):
    import fsspec
    import pyarrow.parquet as pq
    path=f'sample/10BT/{index:03d}_00000.parquet'
    url=f'https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/{REVISION}/{path}'
    records=[]
    with fsspec.open(url,'rb',block_size=1_048_576,client_kwargs={'trust_env':True},cache_type='readahead') as f:
        p=pq.ParquetFile(f)
        groups=sorted(np.random.default_rng(SEED+index).choice(np.arange(5,p.num_row_groups-5),2,replace=False).tolist())
        offsets=np.cumsum([0]+[p.metadata.row_group(g).num_rows for g in range(p.num_row_groups)])
        for group in groups:
            for row_index,row in enumerate(p.read_row_group(group,columns=['id','text','dump'],use_threads=False).to_pylist(),int(offsets[group])):
                raw=row['text'].encode('utf-8')
                records.append(dict(id=row['id'],source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),source_file=path,row_idx=row_index,crawl=row['dump'],payload=raw+b'\n\n'))
    print(json.dumps(dict(event='fetched',file=path,row_groups=groups,documents=len(records))),flush=True)
    return records,dict(source_file=path,row_groups=groups,documents=len(records))

def verify(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    used_ids=set();used_hashes=set()
    for name,info in manifest['splits'].items():
        path=directory/(name+'.bin')
        if path.stat().st_size!=info['targets']+1 or sha(path)!=info['sha256']:raise ValueError(f'Data integrity failure: {name}')
        ids={d['id'] for d in info['documents']};hashes={d['source_sha256'] for d in info['documents']}
        if len(ids)!=len(info['documents']) or len(hashes)!=len(info['documents']) or ids&used_ids or hashes&used_hashes:raise ValueError('Document overlap')
        used_ids|=ids;used_hashes|=hashes
        raw=path.read_bytes();offset=0
        for d in info['documents']:
            if d['start']!=offset:raise ValueError('Noncontiguous document spans')
            segment=raw[offset:offset+d['bytes_used']]
            if hashlib.sha256(segment).hexdigest()!=d['segment_sha256']:raise ValueError('Document segment hash mismatch')
            offset+=d['bytes_used']
        if offset!=len(raw):raise ValueError('Unaccounted bytes')
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/comparison'))
    p.add_argument('--exclude',type=Path,action='append',default=[]);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():verify(a.output);print('Existing data verified');return
    ids=set();hashes=set();provenance=[]
    for path in a.exclude:
        identities(json.loads(path.read_text()),ids,hashes);provenance.append(dict(name=path.name,sha256=sha(path)))
    initial_ids=set(ids);initial_hashes=set(hashes)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:groups=list(pool.map(fetch,[0,2,5,8,11,12]))
    records=[d for group,_ in groups for d in group]
    order=np.random.default_rng(SEED+100).permutation(len(records));cursor=0;splits={}
    for name,targets in SPLITS.items():
        blob=bytearray();documents=[]
        while len(blob)<targets+1 and cursor<len(order):
            d=records[int(order[cursor])];cursor+=1
            if d['id'] in ids or d['source_sha256'] in hashes:continue
            n=min(len(d['payload']),targets+1-len(blob));segment=d['payload'][:n]
            documents.append({**{k:v for k,v in d.items() if k!='payload'},'start':len(blob),'bytes_used':n,'source_byte_offset':0,'segment_sha256':hashlib.sha256(segment).hexdigest()})
            blob.extend(segment);ids.add(d['id']);hashes.add(d['source_sha256'])
        if len(blob)!=targets+1:raise ValueError('Fetched pool is too small; no split manifest committed')
        path=a.output/(name+'.bin');path.write_bytes(blob)
        splits[name]=dict(targets=targets,sha256=sha(path),documents=documents,crawl_counts=dict(Counter(d['crawl'] for d in documents)))
        print(json.dumps(dict(event='prepared',split=name,targets=targets,documents=len(documents))),flush=True)
    manifest=dict(dataset='HuggingFaceFW/fineweb',revision=REVISION,seed=SEED,splits=splits,source_pools=[meta for _,meta in groups],
                  exclusions=dict(document_ids=len(initial_ids),text_hashes=len(initial_hashes),sources=provenance),
                  boundary_policy='Continuous stream per split, shuffled whole documents, only each final document clipped; the entire source document is excluded from later splits.',
                  test_policy='Test bytes are reserved; calibration and training commands do not evaluate this split.',
                  created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(a.output/'manifest.json',manifest);verify(a.output)

if __name__=='__main__':main()
