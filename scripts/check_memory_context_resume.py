"""Exact AdamW/state resume and inherited-moment migration checks."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.hippocampus_config import HippocampusConfig
from experiments.text_regions.memory_context import MemoryContextConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.state import GraphState
from experiments.text_regions.memory_context_checkpoint import migrate_memory_context
from experiments.text_regions.test_hippocampus import same
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(87)
    config=GraphConfig.tiny(rounds=2,hippocampus=HippocampusConfig(enabled=True,cue_clock='event',ca1_policy='bounded'),
                            memory_context=MemoryContextConfig(enabled=True))
    model=RegionLanguageModel(config).double();optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    x=torch.tensor([[1,4,2],[3,7,5]])
    result=model(x,targets=x.roll(-1,1));result['loss'].backward();optimizer.step()
    state=result['state'].detach()
    stream=io.BytesIO();torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),state=state.to_dict()),stream)
    stream.seek(0);saved=torch.load(stream,weights_only=True)
    restored=RegionLanguageModel(config).double();restored.load_state_dict(saved['weights'])
    resumed=torch.optim.AdamW(restored.parameters(),lr=.003);resumed.load_state_dict(saved['optimizer'])
    final=[]
    for m,o,s in ((model,optimizer,state),(restored,resumed,GraphState.from_dict(saved['state']))):
        o.zero_grad(set_to_none=True);r=m(x.flip(1),targets=x,state=s);r['loss'].backward();o.step()
        final.append((r['logits'].detach(),r['state'].to_dict(),copy.deepcopy(o.state_dict())))
    test=unittest.TestCase()
    same(test,final[0],final[1]);same(test,model.state_dict(),restored.state_dict())
    parent=torch.load(a.parent,weights_only=True,map_location='cpu')
    migrated,opt,missing=migrate_memory_context(parent)
    old=RegionLanguageModel(GraphConfig.from_dict(parent['config']));old.load_state_dict(parent['weights'])
    ids=parent['optimizer']['param_groups'][0]['params']
    new=dict(migrated.named_parameters());count=0
    for (name,param),ident in zip(old.named_parameters(),ids):
        same(test,parent['weights'][name],migrated.state_dict()[name])
        if ident in parent['optimizer']['state']:
            same(test,parent['optimizer']['state'][ident],opt.state[new[name]]);count+=1
    report=dict(exact_resume=True,weights_identical=True,optimizer_identical=True,state_identical=True,
                inherited_optimizer_tensors=count,new_parameter_tensors=len(missing),new_parameters=sum(p.numel() for p in migrated.parameters()))
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
