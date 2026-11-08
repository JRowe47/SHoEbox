"""Endpoint reload, learned route/port diagnostics and an isolated operator profile."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shoebox import CorticalModel, cortical_wide
from shoebox.checkpoint import load
from shoebox.cpu_run import stream, routing_summary
from shoebox.diagnostics import PortMonitor
from shoebox.train_fineweb import atomic_json, evaluate, source_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('Choose a new output file.')
    torch.set_num_threads(1)
    m,state,_,checkpoint=load(args.run/'last.pt')
    m.set_execution_backend('cpu').eval()
    expected=json.loads((args.run/'results.json').read_text())
    assert expected['source_sha256']==source_digest()
    validation,checksum=stream(args.validation,expected['final_validation']['targets'],expected['batch'])
    assert checksum==expected['validation_sha256']
    actual=evaluate(m,validation,context=expected['context'])
    for name in ('nll','bits_per_byte','accuracy','targets'):
        assert actual[name]==expected['final_validation'][name],(name,actual,expected['final_validation'])

    torch.manual_seed(m.cfg.seed)
    initial=CorticalModel(m.cfg).set_execution_backend('cpu').eval()
    torch.manual_seed(m.cfg.seed)
    dense=CorticalModel(cortical_wide())
    base=dense.state_dict();other=initial.state_dict()
    assert all(torch.equal(value,other[name]) for name,value in base.items())
    digest=hashlib.sha256()
    for name,value in base.items():digest.update(name.encode());digest.update(value.numpy().tobytes())
    del dense
    with torch.no_grad():
        before=initial(validation[:,:32])
        after=m(validation[:,:32])
    old=torch.stack([r['selected'] for r in before['routing']])
    new=torch.stack([r['selected'] for r in after['routing']])
    changed=float((old!=new).any(-1).float().mean())
    initial_routes=routing_summary(before['routing'],validation.shape[0])
    final_routes=routing_summary(after['routing'],validation.shape[0])
    del initial,before,after

    monitor=PortMonitor(m,tap_events=(0,15,31))
    result=m(validation[:,:32],targets=validation[:,1:33],observer=monitor)
    taps=list(monitor.taps.items())
    router=list(m.graph_router.named_parameters())
    loss=torch.nn.functional.cross_entropy(result['logits'][:,-1],validation[:,32])
    gradients=torch.autograd.grad(loss,[v for _,v in taps]+[v for _,v in router],allow_unused=True)
    temporal=[{'region':name,'event':event,'lag':31-event,
               'task_gradient_l1':0. if g is None else float(g.abs().sum())}
              for ((name,event),_),g in zip(taps,gradients[:len(taps)])]
    route_grad={name:0. if g is None else float(g.abs().sum()) for (name,_),g in zip(router,gradients[len(taps):])}
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)
    ports=monitor.snapshot()
    del result,monitor,gradients,taps,router

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False) as profiler:
        result=m(validation[:,:1],targets=validation[:,1:2],state=state)
        result['loss'].backward()
    operators=[{'operator':x.key,'calls':x.count,'self_cpu_seconds':x.self_cpu_time_total/1e6}
               for x in profiler.key_averages()]
    operators.sort(key=lambda r:r['self_cpu_seconds'],reverse=True)
    report={'reload_exact':True,'source_sha256':source_digest(),'regional_initialization_sha256':digest.hexdigest(),
            'initial_regional_weights_equal':True,'validation':actual,
            'changed_deterministic_route_fraction':changed,'initial_routes':initial_routes,'final_routes':final_routes,
            'temporal_gradients':temporal,'last_prediction_router_task_gradients':route_grad,'ports':ports,
            'operator_profile':{'scope':'one external observation across four streams, forward/backward, no optimizer',
                                'operators':operators,'total_operator_calls':sum(r['calls'] for r in operators),
                                'self_cpu_seconds':sum(r['self_cpu_seconds'] for r in operators)}}
    atomic_json(args.out,report)
    print(json.dumps({'reload_exact':True,'changed_deterministic_route_fraction':changed,
                      'temporal_gradients':temporal,'top_operators':operators[:6]}),flush=True)


if __name__=='__main__':main()
