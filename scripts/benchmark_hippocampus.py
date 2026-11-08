"""Actual sparse-service work at 64-512 mock regions; CPU actor simulation."""
import argparse
from dataclasses import fields,is_dataclass
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.text_regions.hippocampus import Hippocampus
from experiments.text_regions.hippocampus_config import HippocampusConfig
from experiments.text_regions.hippocampus_interface import EngramHandle,LocalAnchor,BindingOffer
from experiments.text_regions.hippocampus_runtime import RecallFabric,FabricState,Envelope
from experiments.text_regions.ports import tensor_list
from experiments.text_regions.node import RecurrentNode


class MockOwner(nn.Module):
    encode_anchor=RecurrentNode.encode_anchor
    read_anchor=RecurrentNode.read_anchor
    def __init__(self):
        super().__init__()
        self.anchor_encoder=nn.Linear(32,32)
        self.anchor_query=nn.Linear(32,32)
        self.anchor_reader=nn.Linear(128,32)


def storage(tensors):
    return sum({(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes() for t in tensors}.values())


def metadata_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen or torch.is_tensor(value): return 0
    seen.add(id(value)); total=sys.getsizeof(value)
    if is_dataclass(value): return total+sum(metadata_bytes(getattr(value,f.name),seen) for f in fields(value))
    if isinstance(value,dict): return total+sum(metadata_bytes(k,seen)+metadata_bytes(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)): return total+sum(metadata_bytes(v,seen) for v in value)
    return total


def run(regions,shards):
    torch.manual_seed(404+regions)
    config=HippocampusConfig(enabled=True,shards=shards,maintenance_budget=2)
    graph=SimpleNamespace(width=32,hippocampus=config,regions={f'R{i}':(f'N{i}',) for i in range(regions)})
    hub=Hippocampus(graph); nn.init.normal_(hub.compose.weight,std=.02)
    owners=nn.ModuleDict({f'N{i}':MockOwner() for i in range(regions)})
    state=hub.initial_state(1,'cpu',torch.float32)
    # One capsule per registered owner, allocated lazily. No all-to-all edges.
    offers=[]
    with torch.no_grad():
        for i in range(regions):
            key=torch.nn.functional.normalize(torch.randn(32),dim=0); code,_=hub.separate(key)
            handle=EngramHandle(0,f'R{i}',f'N{i}',-1,0,0,1)
            anchor=LocalAnchor(handle,torch.randn(32),key,key,'external',1,1)
            offers.append(BindingOffer((0,0,f'N{i}'),0,f'N{i}',anchor,key,code,key,
                torch.tensor(.9),torch.tensor(.2),torch.tensor([0.,0.,1.,0.]),1))
        state=hub.commit(state,(tuple(offers),))
    record=dict(regions=regions,shards=shards,allocated_anchors=sum(map(len,state.rows[0].anchors.values())),
        allocated_bindings=sum(map(len,state.rows[0].bindings.values())),persistent_tensor_bytes=storage(tensor_list(state)),persistent_metadata_estimate_bytes=metadata_bytes(state),
        model_weights_bytes=storage(list(hub.parameters())+list(owners.parameters())),optimizer_bytes=0,
        backend='lazy immutable one-record pages; mock owners; single-process CPU')
    for label,load in (('fixed_load',8),('increasing_load',regions//8)):
        latencies=[]; work=[]; fabric=RecallFabric(config); queue=FabricState(); active_bytes=[]
        def handler(payload,snapshot):
            key,requester=payload
            start=time.perf_counter()
            with torch.no_grad():
                value,receipts,metrics=hub.query(snapshot,key[None,None],owners,{requester:graph.regions[requester][0]},
                    torch.ones(1,1,dtype=torch.bool),regions=(requester,))
            latencies.append((time.perf_counter()-start)*1000)
            work.extend(metrics); active_bytes.append(storage([value,key]))
            return value
        for name in graph.regions: fabric.register(name,handler)
        started=time.perf_counter()
        for j in range(load):
            requester=f'R{j%regions}'; source=offers[(j*37)%regions]
            queue,status=fabric.submit(queue,Envelope((requester,0),requester,0,2,(source.key,requester)))
        while queue.queues:
            queue,_=fabric.service(queue,0,state)
        elapsed=time.perf_counter()-started
        sorted_latency=sorted(latencies)
        record[label]=dict(requests=load,p50_ms=statistics.median(latencies),p95_ms=sorted_latency[min(len(latencies)-1,int(.95*len(latencies)))],
            throughput_requests_s=load/elapsed,max_work={k:max(x[k] for x in work) for k in work[0]},
            queue_capacity=config.queue_capacity,dropped=queue.dropped,deferred=queue.deferred,expired=queue.expired,
            max_return_and_cue_bytes=max(active_bytes))
    # One real backward through bounded requests. Cold owner weights do not
    # allocate gradient tensors. Capture unique saved training tensor storage.
    saved=[]
    def pack(t): saved.append(t); return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        value,_,_=hub.query(state,offers[0].key[None,None],owners,{'R1':'N1'},torch.ones(1,1,dtype=torch.bool),regions=('R1',))
        value.square().sum().backward()
    record['one_query_saved_tensor_bytes']=storage(saved)
    record['one_query_gradient_bytes']=storage([p.grad for p in (*hub.parameters(),*owners.parameters()) if p.grad is not None])
    record['owners_with_gradient']=sum(any(p.grad is not None for p in owner.parameters()) for owner in owners.values())
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True); a=parser.parse_args()
    torch.set_num_threads(1)
    results=[run(n,s) for n,s in ((64,4),(128,8),(256,16),(512,32))]
    for row in results:
        for mode in ('fixed_load','increasing_load'):
            work=row[mode]['max_work']
            assert work['shards']<=2 and work['posting_reads']<=128 and work['candidates']<=32 and work['readbacks']<=8
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{'regions':r['regions'],'latency_ms':r['fixed_load']['p50_ms'],'work':r['fixed_load']['max_work']} for r in results]))

if __name__=='__main__': main()
