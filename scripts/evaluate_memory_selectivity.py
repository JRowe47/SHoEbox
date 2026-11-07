"""Fresh held-out retention, competing facts and ordinary-stream transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.data import split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents
from experiments.text_regions.memory_selectivity import MODES, selective_cases, summarize_values
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language, evaluate_memory


def paired_queries(older, newer, values='0123'):
    """Both queries must work on one unchanged two-fact history."""
    size = len(values)
    if len(older['rows']) != len(newer['rows']) or len(older['rows']) % size:
        raise ValueError('Paired cohorts differ.')
    hits = count = 0
    for base in range(0, len(older['rows']), size):
        for a in range(size):
            for b in range(size):
                if a == b:
                    continue
                left, right = older['rows'][base+b], newer['rows'][base+a]
                if left['document'] != right['document']:
                    raise ValueError('Unmatched query intervention.')
                hits += (left['predictions'][a] == ord(values[a])
                         and right['predictions'][b] == ord(values[b]))
                count += 1
    return dict(pairs=count, both_correct=hits, accuracy=hits/count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--single-cases', type=int, default=32)
    p.add_argument('--selective-cases', type=int, default=16)
    p.add_argument('--documents', type=int, default=8)
    p.add_argument('--data-seed', type=int, default=917)
    p.add_argument('--start', type=int, default=20000)
    p.add_argument('--skip-streams', action='store_true')
    p.add_argument('--part', choices=('full','single','pairs','revision','streams'), default='full',
                   help='Independent evaluation cohort for parallel execution')
    a = p.parse_args(); torch.set_num_threads(1)
    if min(a.single_cases, a.selective_cases, a.documents) < 1 or a.start < 0:
        p.error('Positive cohort sizes and nonnegative document start required.')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    raw = a.data.read_bytes(); _,_,test = split_bytes(raw)
    if hashlib.sha256(raw).hexdigest() != saved['raw_sha256']:
        raise ValueError('Evaluation corpus differs from checkpoint.')
    cfg = GraphConfig.from_dict(saved['config']); data = TaskStreamConfig.from_dict(saved['data_config'])
    model = RegionLanguageModel(cfg); model.load_state_dict(saved['weights']); model.eval()
    identity = dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        step=saved['step'], single_cases=a.single_cases, selective_cases=a.selective_cases,
        documents=a.documents, data_seed=a.data_seed, start=a.start, skip_streams=a.skip_streams, part=a.part,
        raw_sha256=saved['raw_sha256'],
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256((ROOT/'experiments/text_regions/memory_selectivity.py').read_bytes()).hexdigest())
    if a.out.exists():
        result = json.loads(a.out.read_text())
        if result['identity'] != identity:
            raise ValueError('Evaluation cache identity mismatch.')
    else:
        result = dict(identity=identity, measurements={})
    def record(name, value):
        result['measurements'][name] = value
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); tmp.replace(a.out)
        summary = summarize_values(value) if 'rows' in value else value
        print(json.dumps(dict(measurement=name, summary=summary)), flush=True)
    if a.part in ('full','streams') and 'language' not in result['measurements']:
        record('language', evaluate_language(model,test,batch=4,context=32,chunks=8))
    for mode in MODES:
        part = 'single' if mode == 'single' else ('revision' if mode == 'correction' else 'pairs')
        if a.part not in ('full',part):
            continue
        for gap in ([0,8,24,64,128] if mode == 'single' else [0,24,64]):
            name = f'{mode}_{gap}'
            if name not in result['measurements']:
                cases = selective_cases(data, split='test', seed=a.data_seed, start=a.start,
                    count=a.single_cases if mode == 'single' else a.selective_cases,
                    mode=mode, gap=gap, distractor_bytes=test)
                record(name, evaluate_values(model,cases,context=32,cases_per_batch=8))
    for gap in (0,24,64):
        if a.part not in ('full','pairs'):
            continue
        name = f'query_pairs_{gap}'
        if name not in result['measurements']:
            record(name, paired_queries(result['measurements'][f'older_{gap}'],
                                        result['measurements'][f'newer_{gap}'], data.values))
    if not a.skip_streams and a.part in ('full','streams'):
        for split in ('test','long_delay'):
            if split in result['measurements']:
                continue
            documents = make_documents(data, split=split, seed=a.data_seed, count=a.documents,
                start=a.start, distractor_bytes=test)
            metrics, traces = evaluate_memory(model,data,a.data_seed,a.documents,32,split,documents=documents)
            (a.out.parent/(a.out.stem+'_'+split+'_traces.jsonl')).write_text(
                ''.join(json.dumps(row)+'\n' for row in traces))
            record(split, metrics)


if __name__ == '__main__':
    main()
