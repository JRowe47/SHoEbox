"""Fresh held-out CA1 continuation evaluation and same-weight controller controls."""
import argparse
from dataclasses import replace
import hashlib,json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.memory_selectivity import selective_cases,summarize_values
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language
from scripts.evaluate_hippocampus import evaluate_pairs
from scripts.run_memory_selectivity import write_json


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--part',choices=('pairs','retention','language'),required=True)
 p.add_argument('--policy',choices=('legacy','bounded'));p.add_argument('--blocked',action='store_true')
 a=p.parse_args();torch.set_num_threads(1)
 saved=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
 raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']:raise ValueError('Corpus mismatch.')
 cfg=GraphConfig.from_dict(saved['config']);h=cfg.hippocampus
 cfg=replace(cfg,hippocampus=replace(h,ca1_policy=a.policy or h.ca1_policy,
                                   control='blocked' if a.blocked else 'intact'))
 model=RegionLanguageModel(cfg);model.load_state_dict(saved['weights']);model.eval()
 data=TaskStreamConfig.from_dict(saved['data_config']);test=split_bytes(raw)[2]
 identity=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),step=saved['step'],
    seed=1699,start=160000,pair_templates=8,single_templates=16,gaps=[0,32,64],
    ca1_policy=cfg.hippocampus.ca1_policy,blocked=a.blocked,part=a.part,
    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 if a.out.exists():
  result=json.loads(a.out.read_text())
  if result['identity']!=identity:raise ValueError('Evaluation cache mismatch.')
 else:result=dict(identity=identity,measurements={})
 def record(name,value):
  result['measurements'][name]=value;a.out.parent.mkdir(parents=True,exist_ok=True)
  write_json(a.out,result);print(json.dumps({'measurement':name}),flush=True)
 with torch.no_grad():
  if a.part=='pairs':
   for gap in (0,32,64):
    name=f'pairs_{gap}'
    if name not in result['measurements']:
     record(name,evaluate_pairs(model,data,test,seed=1699,start=160000,count=8,gap=gap))
  elif a.part=='retention':
   for mode,count in (('single',16),('correction',8)):
    name=mode+'_64'
    if name not in result['measurements']:
     cases=selective_cases(data,split='test',seed=1699,start=160000,count=count,
                          mode=mode,gap=64,distractor_bytes=test)
     record(name,summarize_values(evaluate_values(model,cases,context=32,cases_per_batch=8)))
  elif 'language' not in result['measurements']:
   record('language',evaluate_language(model,test,batch=4,context=32,chunks=8))

if __name__=='__main__':main()
