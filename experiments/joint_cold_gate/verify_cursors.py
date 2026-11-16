"""Discarded-copy continuation checks against real fitted checkpoints."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .stream import advance
from .tasks import dataset
from .resume import language_advance,load_task,task_advance
from experiments.joint_readout.adapter import Adapter as Old
from experiments.joint_readout.stream import advance as old_advance
from experiments.joint_readout.regressions import Corpus
from experiments.query_readout.run import write

def verify_binding(path,cls):
 raw=path.read_bytes();cursor=json.loads(path.with_suffix('.cursor.json').read_text());fn=advance if cls is Adapter else old_advance;cases=dataset(2048,'train')
 with cls.loads(raw) as x,cls.loads(raw) as y:
  whole,_=fn(x,cursor,1000,cases);partial,_=fn(y,cursor,371,cases)
  with cls.loads(y.dumps()) as z:
   resumed,_=fn(z,partial,629,cases);assert x.dumps()==z.dumps();assert whole==resumed

def verify_language(path,cls,corpus):
 raw=path.read_bytes();cursor=json.loads(path.with_suffix('.cursor.json').read_text())
 with cls.loads(raw) as x,cls.loads(raw) as y:
  whole=language_advance(x,cursor,1000,corpus);partial=language_advance(y,cursor,371,corpus)
  with cls.loads(y.dumps()) as z:
   resumed=language_advance(z,partial,629,corpus);assert x.dumps()==z.dumps();assert whole==resumed

def verify_task(path,cls,seed,task):
 cursor=json.loads(path.with_suffix('.cursor.json').read_text());x,h=load_task(path,cls,seed);y,j=load_task(path,cls,seed)
 with x,y:
  whole=task_advance(x,h,cursor,1000,task);partial=task_advance(y,j,cursor,371,task)
  with cls.loads(y.dumps()) as z:
   resumed=task_advance(z,j,partial,629,task);assert x.dumps()==z.dumps();assert whole==resumed
   if h:
    for k in h.arrays:np.testing.assert_array_equal(h.arrays[k],j.arrays[k])

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--previous',type=Path);a=p.parse_args();start=time.process_time();rows=json.loads(a.previous.read_text())['rows'] if a.previous else [];previous=len(rows);verified={r['checkpoint']:r['sha256'] for r in rows}
 paths=[(a.workspace/'exp01e8-pilot/binding.qr',Adapter,'binding'),(a.workspace/'exp01e8-pilot/retained.qr',Adapter,'retention')]
 for size in ('formal','1m','5m'):
  for path in sorted((a.workspace/f'exp01e8-{size}').glob('*/*.qr')):
   arm=path.name.split('.')[0];cls=Old if arm in ('native_all','frozen_d3') else Adapter
   paths.append((path,cls,'retention' if '.retained.' in path.name else 'binding'))
 for path,cls,kind in paths:
  if verified.get(str(path.relative_to(a.workspace)))==hashlib.sha256(path.read_bytes()).hexdigest():continue
  if kind=='binding':verify_binding(path,cls)
  else:verify_language(path,cls,Corpus(a.workspace/'exp01e-data'))
  rows.append(dict(checkpoint=str(path.relative_to(a.workspace)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),type=kind,exact=True))
 for path in sorted((a.workspace/'exp01e8-regressions').glob('*/*/*.checkpoint')):
  if verified.get(str(path.relative_to(a.workspace)))==hashlib.sha256(path.read_bytes()).hexdigest():continue
  arm=path.parent.parent.name;seed=int(path.parent.name);cls=Adapter if arm=='local_gate' else Old;task=path.stem.rsplit('-',1)[0]
  if task=='language':verify_language(path,cls,Corpus(a.workspace/'exp01e7-data'))
  else:verify_task(path,cls,seed,task)
  rows.append(dict(checkpoint=str(path.relative_to(a.workspace)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),type=task,exact=True))
 write(a.out,dict(scope='1,000 next updates versus 371 + checkpoint reload + 629, on discarded copies only',rows=rows,cpu_seconds=time.process_time()-start,new_verified_pairs=len(rows)-previous,previous_verified_pairs=previous,correctness_only_discarded_input_bytes=2000*(len(rows)-previous)))
 print('verified',len(rows),'checkpoint/cursor pairs')
if __name__=='__main__':main()
