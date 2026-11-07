"""Small matched CPU probe of optional recurrent-state trace overhead."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.graph_credit import make_graph_eligibility,graph_window_forward
from experiments.text_regions.state_credit import make_state_graph_eligibility,state_window_forward
from experiments.text_regions.task_stream import TaskChunk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=ROOT/'artifacts/state_credit/parent.pt')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/state_credit/trace_benchmark.json')
    a=p.parse_args();torch.set_num_threads(1)
    saved=torch.load(a.parent,weights_only=True,map_location='cpu')
    cfg=GraphConfig.from_dict(saved['config']);torch.manual_seed(54)
    x=torch.randint(0,256,(2,8));y=torch.full_like(x,-100)
    chunk=TaskChunk(x,y,y,y.ne(-100),torch.zeros_like(x,dtype=torch.bool),y,y)
    rows={}
    for label,factory,forward in [('publication',make_graph_eligibility,graph_window_forward),
                                  ('state',make_state_graph_eligibility,state_window_forward)]:
        model=RegionLanguageModel(cfg);model.load_state_dict(saved['weights']);times=[]
        for repeat in range(4):
            model.zero_grad(set_to_none=True);state=model.initial_state(2);credit=factory(model,2)
            start=time.perf_counter()
            result,_,receipts=forward(model,chunk,state,credit,truncate_every=4)
            result['logits'].square().mean().backward()
            for receipt in receipts:
                receipt.apply(model.publisher.policy);receipt.apply_payload(model.publisher.payload_adapter)
                if hasattr(receipt,'apply_state'):receipt.apply_state(model.publisher.policy)
            elapsed=time.perf_counter()-start
            if repeat:times.append(elapsed)
        rows[label]=dict(seconds=times,median_seconds=statistics.median(times),credit_bytes=credit.storage_bytes())
    rows['protocol']=dict(batch=2,context=8,truncate_every=4,threads=1,repeats=3,warmups=1,optimizer=False)
    rows['state_over_publication']=rows['state']['median_seconds']/rows['publication']['median_seconds']
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':
    main()
