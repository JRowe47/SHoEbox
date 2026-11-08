"""Verify the completed endpoint and summarize matched operator profiles."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shoebox.checkpoint import load
from shoebox.profile_wide import probe
from shoebox.train_fineweb import atomic_json, source_digest
from shoebox.transport import map_tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--tests-log',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    read=lambda name:json.loads((args.run/name).read_text())
    s=read('results.json')
    model,state,_,checkpoint=load(args.run/'last.pt')
    model.set_execution_backend('optimized')
    assert s['complete'] and s['trained_targets']==1000 and s['updates']==8
    assert checkpoint['step']==8 and checkpoint['metadata']['complete']
    assert checkpoint['metadata']['trained_targets']==1000
    assert checkpoint['metadata']['cursor']==state.event==250
    assert state.batch==4 and state.memory.writes==250
    assert state.signature==model.cfg.to_dict()==s['config']
    assert s['source_sha256']==source_digest()
    assert checkpoint['metadata']['protocol']['source_sha256']==s['source_sha256']
    raw=(args.run/'profile_train.bin').read_bytes()
    assert len(raw)==1004 and hashlib.sha256(raw).hexdigest()==s['data_sha256']
    assert len(s['history'])==len(s['port_history'])==8
    assert [r['trained_targets'] for r in s['history']]==[128,256,384,512,640,768,896,1000]
    assert all(r['finite'] and r['gradient_rms']>0 for r in s['temporal_gradients'])
    def finite(t):
        assert bool(torch.isfinite(t).all())
        return t
    map_tensors(checkpoint,finite)
    assert all(frame.distribution is None for history in state.ports.values() for frame in history)
    assert state.storage_bytes()==s['final_state_bytes']
    data=torch.tensor(list(raw),dtype=torch.long).reshape(4,-1)
    final=probe(model,data[:,:33])
    assert final['nll']==s['final_probe']['nll']
    assert final['ports']==s['final_probe']['ports']
    with torch.no_grad():
        continued=model(data[:,:1],state=state)
    assert continued['state'].event==251 and torch.isfinite(continued['logits']).all()
    tests=args.tests_log.read_text()
    count=int(re.search(r'Ran (\d+) tests',tests).group(1))
    assert count>=17 and tests.rstrip().endswith('OK')
    for name,key in [('history.jsonl','history'),('ports.jsonl','port_history')]:
        recorded=[json.loads(line) for line in (args.run/name).read_text().splitlines()]
        assert recorded==s[key]
    audit={'passed':True,'tests_passed':count,'parameters':sum(p.numel() for p in model.parameters()),
           'trained_targets':1000,'updates':8,'state_events':state.event,'state_batch':state.batch,
           'hippocampal_writes':state.memory.writes,'source_sha256':source_digest(),
           'data_sha256':s['data_sha256'],'data_bytes':len(raw),'exact_reloaded_probe':True,
           'finite_weights_optimizer_state':True,'continued_event':continued['state'].event}
    atomic_json(args.run/'endpoint_audit.json',audit)
    ref,opt=read('benchmark_reference.json'),read('benchmark_optimized.json')
    x=torch.load(args.run/'benchmark_reference_logits.pt',weights_only=True)
    y=torch.load(args.run/'benchmark_optimized_logits.pt',weights_only=True)
    difference=(x-y).abs()
    comparison={'max_logit_difference':float(difference.max()),'mean_logit_difference':float(difference.mean()),
                'greedy_predictions_identical':bool((x.argmax(-1)==y.argmax(-1)).all()),
                'speedup':opt['bytes_per_second']/ref['bytes_per_second'],
                'peak_rss_reduction_percent':100*(1-opt['peak_rss_mib']/ref['peak_rss_mib']),
                'saved_storage_reduction_percent':100*(1-opt['saved_unique_storage_bytes']/ref['saved_unique_storage_bytes'])}
    assert comparison['max_logit_difference']<1e-5 and comparison['greedy_predictions_identical']
    atomic_json(args.run/'benchmark_comparison.json',comparison)
    combined={}
    for backend in ['reference','optimized']:
        profile=read(f'operators_{backend}.json')
        sums=defaultdict(lambda:{'calls':0,'self_cpu_seconds':0.,'positive_self_allocated_bytes':0})
        for row in profile['operators']:
            total=sums[row['operator']]
            total['calls']+=row['calls']
            total['self_cpu_seconds']+=row['self_cpu_seconds']
            total['positive_self_allocated_bytes']+=max(0,row['self_cpu_memory_bytes'])
        rows=[dict(operator=name,**row) for name,row in sums.items()]
        combined[backend+'_top_self']=sorted(rows,key=lambda row:row['self_cpu_seconds'],reverse=True)
        combined[backend+'_top_allocation']=sorted(rows,key=lambda row:row['positive_self_allocated_bytes'],reverse=True)
        combined[backend+'_total_self_cpu_seconds']=profile['total_self_cpu_seconds']
        combined[backend+'_positive_self_allocated_bytes']=profile['positive_self_allocated_bytes']
    atomic_json(args.run/'operator_comparison.json',combined)
    print(json.dumps({'audit':audit,'comparison':comparison,
                      'top_self':combined['optimized_top_self'][:8],
                      'top_allocation':combined['optimized_top_allocation'][:6]},indent=2))


if __name__=='__main__':main()
