"""Held-out balanced recall, delay transfer, and full ordinary task streams."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents
from experiments.text_regions.memory_acquisition import acquisition_cases, delayed_cases
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language, evaluate_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--cases', type=int, default=32)
    p.add_argument('--documents', type=int, default=8)
    p.add_argument('--data-seed', type=int, default=683)
    p.add_argument('--gaps', type=int, nargs='+', default=[0,8,24,64,128])
    p.add_argument('--skip-streams', action='store_true')
    a=p.parse_args(); torch.set_num_threads(1)
    saved=torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    raw=a.data.read_bytes(); _,_,test=split_bytes(raw)
    if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']:
        raise ValueError('Evaluation corpus differs from checkpoint.')
    cfg=GraphConfig.from_dict(saved['config']); data=TaskStreamConfig.from_dict(saved['data_config'])
    model=RegionLanguageModel(cfg); model.load_state_dict(saved['weights']); model.eval()
    cases=acquisition_cases(data, split='test', seed=a.data_seed, start=10000,
        count=a.cases, distractor_bytes=test)
    identity=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        step=saved['step'], cases=a.cases, documents=a.documents, data_seed=a.data_seed,
        test_start=10000, gaps=a.gaps, raw_sha256=saved['raw_sha256'], skip_streams=a.skip_streams,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acquisition_sha256=hashlib.sha256((ROOT/'experiments/text_regions/memory_acquisition.py').read_bytes()).hexdigest())
    if a.out.exists():
        result=json.loads(a.out.read_text())
        if result['identity']!=identity:
            raise ValueError('Evaluation cache identity mismatch.')
    else:
        result=dict(identity=identity, measurements={})
    def record(name, value):
        result['measurements'][name]=value
        a.out.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.out.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)+'\n'); tmp.replace(a.out)
        print(json.dumps(dict(measurement=name,summary={k:v for k,v in value.items()
            if k not in ('rows','groups','write_rates','retention','mean_demand')})),flush=True)
    if 'language' not in result['measurements']:
        record('language',evaluate_language(model,test,batch=4,context=32,chunks=8))
    for gap in a.gaps:
        name=f'gap_{gap}'
        if name not in result['measurements']:
            record(name,evaluate_values(model,delayed_cases(cases,gap,test),context=32,cases_per_batch=8))
    if not a.skip_streams:
        for split in ('test','long_delay'):
            if split in result['measurements']:
                continue
            documents=make_documents(data,split=split,seed=a.data_seed,count=a.documents,
                start=10000,distractor_bytes=test)
            metrics,traces=evaluate_memory(model,data,a.data_seed,a.documents,32,split,documents=documents)
            (a.out.parent/(a.out.stem+'_'+split+'_traces.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in traces))
            record(split,metrics)


if __name__=='__main__':
    main()
