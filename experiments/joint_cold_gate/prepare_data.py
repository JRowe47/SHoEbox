"""Fresh grouped corpus using pinned source rows, excluding opened E1 audit."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
SALT=b'SHoEbox-EXP01E8-language-v1\0'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'manifest.json').exists():raise FileExistsError('pinned corpus exists')
 old=json.loads((a.source/'manifest.json').read_text());excluded={d['text_sha256'] for d in old['files']['test']['documents']};need={'train':1000000,'dev':65536,'test':65536};data={k:bytearray() for k in need};docs={k:[] for k in need};seen=set()
 for f in sorted(a.source.glob('rows_*.json')):
  for item in json.loads(f.read_bytes())['rows']:
   r=item['row'];raw=r['text'].encode();h=sha(raw)
   if item.get('truncated_cells') or h in seen or h in excluded or len(raw)<2:continue
   seen.add(h);bucket=int.from_bytes(hashlib.sha256(SALT+bytes.fromhex(h)).digest()[:8],'big')%10;fold='train' if bucket<6 else 'dev' if bucket<8 else 'test'
   if not need[fold]:continue
   body=(raw+b'\n\n')[:need[fold]+1];start=len(data[fold]);data[fold].extend(body);n=len(body)-1;need[fold]-=n
   docs[fold].append(dict(id=r['id'],row=item['row_idx'],text_sha256=h,start=start,stop=len(data[fold]),targets=n))
 if any(need.values()):raise ValueError(('insufficient pinned rows',need))
 files={}
 for f,body in data.items():(a.out/f'{f}.bin').write_bytes(body);files[f]=dict(file=f'{f}.bin',sha256=sha(body),bytes=len(body),targets=sum(d['targets'] for d in docs[f]),documents=docs[f])
 assert all(not {d['text_sha256'] for d in docs[x]}&{d['text_sha256'] for d in docs[y]} for x,y in [('train','dev'),('train','test'),('dev','test')])
 (a.out/'manifest.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),dataset=old['dataset'],config=old['config'],source='pinned original row bodies',salt=SALT.decode(),excluded_opened_audit_text_hashes=sorted(excluded),files=files,preprocessing=old['preprocessing']),indent=2))
if __name__=='__main__':main()
