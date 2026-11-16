"""Exact continuation of selected real trained checkpoints and outer cursors."""
import argparse,hashlib,importlib,json
from pathlib import Path
from experiments.query_readout.run import write
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 targets=[(f'exp01e-formal/{s}/gated_all','joint_readout') for s in (1,7,19)]+[(f'exp01e-{size}/7/gated_all','joint_readout') for size in ('1m','5m')]+[('exp01e-switch-pilot/binding','joint_switch'),('exp01e-local-gate-pilot/binding','joint_local_gate'),('exp01e-request-gate-pilot/binding','joint_request_gate')]
 for stem,module in targets:
  cls=importlib.import_module(f'experiments.{module}.adapter').Adapter;cases=importlib.import_module(f'experiments.{module}.tasks').dataset(2048,'train');advance=importlib.import_module('experiments.joint_request_gate.stream' if module=='joint_request_gate' else 'experiments.joint_readout.stream').advance
  path=a.workspace/(stem+'.qr');raw=path.read_bytes();cursor=json.loads((a.workspace/(stem+'.cursor.json')).read_text())
  with cls.loads(raw) as x,cls.loads(raw) as y:
   whole,_=advance(x,cursor,1000,cases);partial,_=advance(y,cursor,371,cases)
   with cls.loads(y.dumps()) as z:
    resumed,_=advance(z,partial,629,cases);assert x.dumps()==z.dumps();assert whole==resumed
   rows.append(dict(checkpoint=stem+'.qr',module=module,sha256=hashlib.sha256(raw).hexdigest(),cursor_sha256=hashlib.sha256((a.workspace/(stem+'.cursor.json')).read_bytes()).hexdigest(),exact_1000_vs_371_reload_629=True))
 write(a.out,dict(scope='correctness-only updates; discarded copies, not research fits',rows=rows));print('verified',len(rows),'real checkpoint/cursor pairs')
if __name__=='__main__':main()
