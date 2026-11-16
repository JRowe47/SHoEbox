"""Pin fresh document-grouped FineWeb bytes before any language fitting."""
import argparse,hashlib,json,urllib.parse,urllib.request
from pathlib import Path
from datetime import datetime,timezone
SALT=b'SHoEbox-EXP01E-language-v1\0'
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'manifest.json').exists():raise FileExistsError('pinned corpus exists')
 need={'train':1000000,'dev':65536,'test':65536};data={k:bytearray() for k in need};docs={k:[] for k in need};seen=set();offset=20000
 while any(need.values()):
  cache=a.out/f'rows_{offset}.json'
  if not cache.exists():
   old=Path('/workspace/scratch/7eed2913f791')/f'fineweb_rows_{offset}.json'
   if old.exists():cache.write_bytes(old.read_bytes())
   else:
    q=urllib.parse.urlencode(dict(dataset='HuggingFaceFW/fineweb',config='sample-10BT',split='train',offset=offset,length=100))
    with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+q,timeout=45) as r:cache.write_bytes(r.read())
  rows=json.loads(cache.read_bytes());assert not rows.get('partial')
  for item in rows['rows']:
   r=item['row'];raw=r['text'].encode();h=sha(raw)
   if item.get('truncated_cells') or h in seen or len(raw)<2:continue
   seen.add(h);bucket=int.from_bytes(hashlib.sha256(SALT+bytes.fromhex(h)).digest()[:8],'big')%10
   fold='train' if bucket<6 else 'dev' if bucket<8 else 'test'
   if not need[fold]:continue
   body=(raw+b'\n\n')[:need[fold]+1];start=len(data[fold]);data[fold].extend(body);n=len(body)-1;need[fold]-=n
   docs[fold].append(dict(id=r['id'],row=item['row_idx'],text_sha256=h,start=start,stop=len(data[fold]),targets=n))
  print(offset,need,flush=True);offset+=100
 files={}
 for f,body in data.items():
  (a.out/f'{f}.bin').write_bytes(body);files[f]=dict(file=f'{f}.bin',sha256=sha(body),bytes=len(body),targets=sum(d['targets'] for d in docs[f]),documents=docs[f])
 assert all(not {d['id'] for d in docs[x]}&{d['id'] for d in docs[y]} for x,y in [('train','dev'),('train','test'),('dev','test')])
 manifest=dict(utc=datetime.now(timezone.utc).isoformat(),dataset='HuggingFaceFW/fineweb',config='sample-10BT',source='official datasets-server row API (main); exact text and byte hashes pin observed contents',salt=SALT.decode(),files=files,preprocessing='UTF-8 plus two newlines; truncate final document per fold; reset at every document, never score across documents')
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
