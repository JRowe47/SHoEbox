"""Audit the fusion endpoint, parallel kernel, timings and allocation profiles."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shoebox.checkpoint import load
from shoebox.fused_kron import fused_final,native_status
from shoebox.profile_wide import probe
from shoebox.train_fineweb import source_digest,atomic_json
from shoebox.transport import map_tensors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--tests-log',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    read=lambda name:json.loads((args.run/name).read_text())
    s=read('results.json')
    model,state,_,checkpoint=load(args.run/'last.pt')
    model.set_execution_backend('fused')
    assert native_status()['available'] and s['native']['available']
    assert s['complete'] and s['trained_targets']==1000 and s['updates']==8
    assert s['source_sha256']==source_digest()==checkpoint['metadata']['protocol']['source_sha256']
    assert checkpoint['step']==8 and checkpoint['metadata']['execution_backend']=='fused'
    assert state.event==state.memory.writes==checkpoint['metadata']['cursor']==250 and state.batch==4
    raw=(args.run/'profile_train.bin').read_bytes()
    assert len(raw)==1004 and hashlib.sha256(raw).hexdigest()==s['data_sha256']
    def finite(t):
        assert bool(torch.isfinite(t).all())
        return t
    map_tensors(checkpoint,finite)
    assert state.storage_bytes()==s['final_state_bytes']
    data=torch.tensor(list(raw),dtype=torch.long).reshape(4,-1)
    final=probe(model,data[:,:33])
    assert final['nll']==s['final_probe']['nll'] and final['ports']==s['final_probe']['ports']
    with torch.no_grad():continued=model(data[:,:1],state=state)
    assert continued['state'].event==251 and torch.isfinite(continued['logits']).all()
    for name,key in [('history.jsonl','history'),('ports.jsonl','port_history')]:
        assert [json.loads(line) for line in (args.run/name).read_text().splitlines()]==s[key]
    tests=args.tests_log.read_text()
    count=int(re.search(r'Ran (\d+) tests',tests).group(1))
    assert count==23 and tests.rstrip().endswith('OK') and 'skipped' not in tests.lower()

    # Exercise real parallel_for execution, with disjoint head-state ownership.
    torch.manual_seed(816)
    k=F.normalize(torch.randn(20,9,2,3,dtype=torch.float64),dim=-1).requires_grad_()
    v=F.normalize(torch.randn(20,9,2,5,dtype=torch.float64),dim=-1).requires_grad_()
    g=(-torch.rand_like(k)*.2).requires_grad_()
    beta=torch.rand(20,9,2,dtype=torch.float64,requires_grad=True)
    write=torch.rand_like(beta,requires_grad=True)
    previous=torch.randn(20,2,3,5,dtype=torch.float64,requires_grad=True)
    inputs=(k,v,g,beta,write,previous)
    a=fused_final(*inputs);upstream=torch.randn_like(a)
    ga=torch.autograd.grad(a,inputs,upstream)
    torch.set_num_threads(2)
    b=fused_final(*inputs);gb=torch.autograd.grad(b,inputs,upstream)
    torch.set_num_threads(1)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=0,rtol=0)
    old,new=read('benchmark_optimized.json'),read('benchmark_fused.json')
    x=torch.load(args.run/'benchmark_optimized_logits.pt',weights_only=True)
    y=torch.load(args.run/'benchmark_fused_logits.pt',weights_only=True)
    difference=(x-y).abs()
    assert float(difference.max())<1e-5
    assert bool((x.argmax(-1)==y.argmax(-1)).all())
    summary={'complete':True,'tests_passed':count,'endpoint_reload_exact':True,'parallel_value_gradient_exact':True,
             'source_sha256':s['source_sha256'],'data_sha256':s['data_sha256'],
             'parameters':s['parameters'],'trained_targets':1000,'updates':8,
             'initial_probe_nll':s['initial_probe']['nll'],'final_probe_nll':s['final_probe']['nll'],
             'training_seconds':s['training_seconds'],'training_bytes_per_second':s['training_bytes_per_second'],
             'training_peak_rss_mib':s['peak_rss_mib'],'state_bytes':s['final_state_bytes'],
             'matched':{'baseline':old,'fused':new,
                        'speedup':new['bytes_per_second']/old['bytes_per_second'],
                        'peak_rss_reduction_percent':100*(1-new['peak_rss_mib']/old['peak_rss_mib']),
                        'saved_storage_reduction_percent':100*(1-new['saved_unique_storage_bytes']/old['saved_unique_storage_bytes']),
                        'max_logit_difference':float(difference.max()),'mean_logit_difference':float(difference.mean()),
                        'greedy_predictions_identical':True},'native':native_status()}
    for backend in ('optimized','fused'):
        profile=read('operators_'+backend+'.json')
        sums=defaultdict(lambda:{'calls':0,'self_seconds':0.,'positive_allocation_bytes':0})
        for row in profile['operators']:
            name=row['operator']
            if name.startswith('shoebox_minkron_'):name='minkron::'+name.split('::')[-1]
            total=sums[name]
            total['calls']+=row['calls']
            total['self_seconds']+=row['self_cpu_seconds']
            total['positive_allocation_bytes']+=max(0,row['self_cpu_memory_bytes'])
        operators=sorted([dict(operator=name,**row) for name,row in sums.items()],key=lambda r:r['self_seconds'],reverse=True)
        summary[backend+'_operators']={'total_self_seconds':profile['total_self_cpu_seconds'],
                                      'positive_allocation_bytes':profile['positive_self_allocated_bytes'],
                                      'top_self':operators[:12],
                                      'native':[r for r in operators if r['operator'].startswith('minkron::')]}
    atomic_json(args.run/'fusion_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('native','optimized_operators','fused_operators')},indent=2))
    print(json.dumps(summary['fused_operators'],indent=2))


if __name__=='__main__':main()
