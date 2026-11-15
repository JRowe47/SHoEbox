import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from common import Model,install,score,atomic,file_hash,DocumentChunks,take_documents

ARMS={
 'baseline':{},
 'quarter_all':dict(fast_rate=.0002,slow_rate=.002,query_rate=.0002,head_rate=.004),
 'quad_all':dict(fast_rate=.0032,slow_rate=.032,query_rate=.0032,head_rate=.064),
 'quad_fast':dict(fast_rate=.0032),
 'quarter_head':dict(head_rate=.004),
 'quad_slow':dict(slow_rate=.032),
 'no_attraction':dict(route_rate=0.),
 'freeze_bias':{},
 'no_aux':dict(auxiliary_weight=0.,reconstruction_weight=0.),
 'zero_bias':{},
 'upper_aux_tenth':{}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--cpu',type=int,default=0)
 p.add_argument('--targets',type=int,default=1_000_000);p.add_argument('--resume',action='store_true');a=p.parse_args()
 os.sched_setaffinity(0,{a.cpu});_,lib=install(a.runtime);a.out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((a.base/'diagnostic_40m/protocol.json').read_text())
 data=np.fromfile(a.base/'diagnostic_40m/data/train.bin',np.uint8)
 dev=np.fromfile(a.base/'data/development.bin',np.uint8)
 stream=DocumentChunks(meta['train_documents'],7);devdocs=meta['development_documents'];probe=take_documents(devdocs,65536)
 fresh=meta['fresh_screen_documents']
 checkpoint=a.out/'latest.shoe' if a.resume else a.base/'run_20m_40m/latest.shoe'
 r=json.loads((a.out/'results.json').read_text()) if a.resume else dict(arm=a.arm,changes=ARMS[a.arm],targets=0,training_seconds=0.,metrics=[],parent_sha256=meta['parent_sha256'])
 model=Model.load(checkpoint)
 if not a.resume:model.finish()
 lib.diag_policy(int(a.arm not in {'freeze_bias','zero_bias'}),0. if a.arm=='no_aux' else .1)
 if a.arm=='zero_bias' and not a.resume:model._check(lib.diag_set_bias(model._alive(),0.))
 if a.arm=='upper_aux_tenth':lib.diag_aux_levels(1.,1.,.1,.1)
 model.configure(**ARMS[a.arm]);r['learning_rates']=model.stats()['learning_rates']
 r['balance_updates']=a.arm not in {'freeze_bias','zero_bias'};r['sdr_weight']=0. if a.arm=='no_aux' else .1
 r['zero_initial_bias']=a.arm=='zero_bias';r['upper_contrastive_factor']=.1 if a.arm=='upper_aux_tenth' else 1.
 done=r['targets'];window=0.;wn=0;start=time.perf_counter()
 r.setdefault('start_wall_seconds',time.time())
 while done<a.targets:
    boundary=min(a.targets,((done//250000)+1)*250000)
    begin,end,reset=stream.take(done,boundary-done)
    tick=time.perf_counter()
    if reset:model.reset_context(clear_memory=True)
    loss=model.run(data[begin:end],data[begin+1:end+1]);elapsed=time.perf_counter()-tick
    r['training_seconds']+=elapsed;done+=end-begin;window+=loss*(end-begin);wn+=end-begin
    if done==boundary:
        evaluation=score(model,dev,probe)
        r['metrics'].append(dict(additional_targets=done,training_seconds=r['training_seconds'],train_nll=window/wn,probe_nll=evaluation['nll'],routing=model.routing_stats()))
        r['targets']=done;atomic(a.out/'results.json',r)
        print(json.dumps(dict(arm=a.arm,targets=done,probe_nll=evaluation['nll'],train_nll=window/wn)),flush=True)
        window=0.;wn=0
 model.save(a.out/'latest.shoe')
 r['development']=score(model,dev,devdocs);r['fresh']=score(model,data,fresh)
 if a.targets>=3_000_000:r['audit']=score(model,data,meta['fresh_audit_documents'])
 r['stats']=model.stats();r['checkpoint_sha256']=file_hash(a.out/'latest.shoe');r['phase_wall_seconds']=time.perf_counter()-start
 atomic(a.out/'results.json',r);model.close()
 print(json.dumps(dict(completed=True,arm=a.arm,targets=done,development=r['development']['nll'],fresh=r['fresh']['nll'])),flush=True)

if __name__=='__main__':main()
