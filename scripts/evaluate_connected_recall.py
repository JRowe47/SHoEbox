"""Held-out language/recall and balanced source-value interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.data import split_bytes
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.graph_mixed_train import evaluate_language, evaluate_memory
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents
from experiments.text_regions.value_recall import value_cases, evaluate_values
from scripts.evaluate_state_credit import exact_cohort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/connected_recall')
    p.add_argument('--parent', type=Path, default=ROOT/'artifacts/state_credit/state/best.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--checkpoints', nargs='+', default=['parent','connected_best','connected_last','cut_16_best','cut_16_last'])
    a = p.parse_args()
    torch.set_num_threads(1)
    raw = a.data.read_bytes(); _, _, test = split_bytes(raw)
    parent = torch.load(a.parent, map_location='cpu', weights_only=True)
    if hashlib.sha256(raw).hexdigest() != parent['raw_sha256']:
        raise ValueError('Evaluation corpus mismatch.')
    cfg = GraphConfig.from_dict(parent['config']); data = TaskStreamConfig.from_dict(parent['data_config'])
    protocol = dict(data_seed=583, count=8, context=32, language_targets=1024,
                    acquisition='all native immediate exact queries, all four observed values',
                    delayed='first exact query beyond 96 bytes per long-delay document, all four values')
    documents = {split: make_documents(data, split=split, seed=583, count=8, distractor_bytes=test)
                 for split in ('test','long_delay')}
    cases = dict(acquisition=value_cases(documents['test'], data.values, acquisition=True),
                 delayed=value_cases(documents['long_delay'], data.values, minimum_delay=97, one_per_document=True))
    paths = {'parent': a.parent}
    paths.update({arm+'_'+tag: a.root/arm/(tag+'.pt') for arm in ('connected','cut_16') for tag in ('best','last')})
    a.root.mkdir(parents=True, exist_ok=True)
    output_path = a.root/'evaluation.json'; cache_path = a.root/'evaluation_cache.json'
    output = json.loads(output_path.read_text()) if output_path.exists() else {}
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    source = hashlib.sha256()
    for directory in ('text_regions', 'hat_minkron'):
        for path in sorted((ROOT/'experiments'/directory).glob('*.py')):
            if path.name.startswith('test_'):
                continue
            source.update(str(path.relative_to(ROOT)).encode()); source.update(path.read_bytes())
    def write(path, value):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)
    for name in a.checkpoints:
        saved = torch.load(paths[name], map_location='cpu', weights_only=True)
        if saved['config'] != parent['config'] or saved['raw_sha256'] != parent['raw_sha256']:
            raise ValueError('Checkpoint graph/corpus mismatch.')
        digest = hashlib.sha256(json.dumps(dict(protocol=protocol, config=cfg.to_dict(),
            data=data.to_dict(), corpus=parent['raw_sha256'], source_sha256=source.hexdigest(), schema=1,
            evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            values_sha256=hashlib.sha256((ROOT/'experiments/text_regions/value_recall.py').read_bytes()).hexdigest()),
            sort_keys=True).encode())
        for key, value in sorted(saved['weights'].items()):
            digest.update(repr((key, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        signature = digest.hexdigest(); row = cache.setdefault(signature, {})
        model = RegionLanguageModel(cfg); model.load_state_dict(saved['weights']); model.eval()
        for measurement in ('language','test','long_delay','acquisition','delayed'):
            if measurement in row:
                continue
            if measurement == 'language':
                result = evaluate_language(model, test, batch=4, context=32, chunks=8)
            elif measurement in cases:
                result = evaluate_values(model, cases[measurement], context=32)
            else:
                result, traces = evaluate_memory(model, data, 583, 8, 32, measurement,
                                                documents=documents[measurement])
                result['exact_beyond_96'] = exact_cohort(traces)
                (a.root/f'{signature}_{measurement}_traces.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
            row[measurement] = result; write(cache_path, cache)
            print(json.dumps(dict(checkpoint=name, measurement=measurement)), flush=True)
        output[name] = dict(step=saved['step'], signature=signature, protocol=protocol, **row)
        write(output_path, output)


if __name__ == '__main__':
    main()
