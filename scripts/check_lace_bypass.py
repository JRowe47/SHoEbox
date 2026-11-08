"""Compare one unchanged production update with the head-installed bypass arm."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_paired_recall import PairedRecallRun
from scripts.run_lace_separation import LaceRun
from scripts.run_memory_selectivity import write_json
from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing
from experiments.text_regions.ports import tensor_list

def main():
    torch.set_num_threads(1);root=ROOT/'artifacts/lace_separation'
    parent=torch.load(ROOT/'artifacts/hippocampus_binding/ca1/bounded/last.pt',weights_only=True,map_location='cpu')
    raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes();saved=[]
    for bypass in (False,True):
        # AdamW loading may alias saved moment tensors on the same device.
        # Independent production arms deserialize independently as well.
        parent=torch.load(ROOT/'artifacts/hippocampus_binding/ca1/bounded/last.pt',weights_only=True,map_location='cpu')
        run=LaceRun(parent,raw,'A') if bypass else PairedRecallRun(parent,raw,steps=32,eval_every=32,validation_cases=8)
        enable_associative_checkpointing(run.model,segment=16,min_positions=512)
        row=run.update()
        weights={n:p.detach().clone() for n,p in run.model.named_parameters() if '.lace_head.' not in n}
        moments={n:{k:v.clone() if torch.is_tensor(v) else v for k,v in run.optimizer.state[p].items()}
                 for n,p in run.model.named_parameters() if '.lace_head.' not in n}
        state=[x.clone() for x in tensor_list(run.language_state)]
        saved.append(dict(weights=weights,moments=moments,state=state,stream=run.streams.state_dict(),rng=torch.get_rng_state().clone(),
                          next_document=run.next_document,loss=row['memory_nll']))
    a,b=saved
    assert a['loss']==b['loss'] and a['stream']==b['stream'] and a['next_document']==b['next_document']
    assert torch.equal(a['rng'],b['rng'])
    for n,v in a['weights'].items():
        assert torch.equal(v,b['weights'][n]),n
        for k,x in a['moments'][n].items():
            y=b['moments'][n][k];assert torch.equal(x,y) if torch.is_tensor(x) else x==y,(n,k)
    assert len(a['state'])==len(b['state']) and all(torch.equal(x,y) for x,y in zip(a['state'],b['state']))
    result=dict(exact=True,parameter_tensors=len(a['weights']),state_tensors=len(a['state']),next_document=a['next_document'],memory_nll=a['loss'])
    write_json(root/'bypass_trajectory.json',result);print(json.dumps(result))
if __name__=='__main__':main()
