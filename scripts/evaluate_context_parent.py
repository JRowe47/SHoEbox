"""Score the common starting weights on the exact new held-out protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.context_balance import ContextBalancedRun
from experiments.text_regions.graph_mixed_train import evaluate_memory
from experiments.text_regions.task_stream import TaskStreamConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/context_balance')
    parser.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    a = parser.parse_args()
    torch.set_num_threads(1)
    spec = json.loads((a.root/'group_clip/protocol.json').read_text())
    p = spec['protocol']
    identity = dict(parent_sha256=hashlib.sha256((a.root/'parent.pt').read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())
    if identity['parent_sha256'] != p['parent_sha256'] or identity['raw_sha256'] != spec['raw_sha256']:
        raise ValueError('Common parent or corpus differs from the study.')
    path = a.root/'parent_evaluation.json'
    record = json.loads(path.read_text()) if path.exists() else dict(identity=identity, measurements={})
    if record['identity'] != identity:
        raise ValueError('Existing parent evaluation has a different protocol.')
    run = ContextBalancedRun(a.data.read_bytes(), GraphConfig.from_dict(spec['config']),
        TaskStreamConfig.from_dict(spec['data_config']), p, a.root/'parent.pt')
    for key in ('language', 'test', 'long_delay', 'legacy_test'):
        if key in record['measurements']:
            continue
        if key == 'language':
            metric = run.language_metrics(run.test_raw, p['eval_chunks']*2)
        elif key == 'legacy_test':
            legacy = TaskStreamConfig(entities='ABCD', queries_per_cycle=4, delays=(2,48), interference=True)
            metric, _ = evaluate_memory(run.model, legacy, 283, 8, p['context'], 'test')
        else:
            metric, traces = run.memory_metrics(key, p['eval_count']*2)
            (a.root/f'parent_{key}_traces.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
        record['measurements'][key] = metric
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(record,indent=2)+'\n'); temp.replace(path)
        print(json.dumps(dict(parent_evaluation=key)), flush=True)


if __name__ == '__main__':
    main()
