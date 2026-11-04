"""Same-weight query-only source interventions with paired full-factorial samples."""
from collections import Counter, defaultdict
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1]/'SHoEbox')
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/views/views/checkpoint.pt'))
parser.add_argument('--count', type=int, default=512)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--out', type=Path, default=Path(__file__).with_suffix(''))
args = parser.parse_args()
if args.count < 1 or args.count % 64 or min(args.batch, args.threads) < 1:
    parser.error('Use a positive multiple of64 episodes and positive batch/threads.')
sys.path.insert(0, str(args.repo))
from experiments.text_regions.learning import RegionLearningModel
from experiments.text_regions.view_data import make_view_batch
from experiments.text_regions.view_train import tensor_fingerprint

torch.set_num_threads(args.threads)
checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.repo/args.checkpoint
saved = torch.load(checkpoint, weights_only=True)
model = RegionLearningModel.from_config(saved['learner_config'])
model.load_state_dict(saved['weights']);model.eval()
before = tensor_fingerprint(model)
original_cfg = model.cfg
protocol = saved['protocol']
args.out.mkdir(parents=True, exist_ok=True)
report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'model_tensor_sha256': before, 'checkpoint_step': saved['step'],
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'command': 'python '+' '.join(sys.argv), 'torch_version': str(torch.__version__),
          'seed': protocol['seed'], 'cohort': 'independent', 'query_type': 'shared',
          'count': args.count, 'batch_size': args.batch, 'threads': args.threads,
          'gap': protocol['gap'], 'query_every': protocol['query_every'],
          'scoring': 'argmax over256 bytes at final query; same identities/factors/templates in every condition',
          'conditions': {
              'B_only': 'Only B observations available; A/C entirely absent.',
              'B_final_query_AC': 'Same B facts; A/C receive only the final shared-query marker.',
              'B_all_queries_AC': 'Same B facts; A/C receive only the shared-query marker at each supervised query occurrence.'},
          'results': {}}
expected_ids = None
expected_calls = None
with (args.out/'predictions.jsonl').open('w') as raw, torch.no_grad():
    for policy in ('original', 'evidence_gated'):
        model.model.cfg = replace(original_cfg, evidence_gated_ports=policy == 'evidence_gated')
        report['results'][policy] = {}
        for condition in report['conditions']:
            started = time.perf_counter()
            counts, correct = Counter(), Counter()
            confusion = defaultdict(Counter)
            template_counts, template_correct = Counter(), Counter()
            crossed_counts, crossed_correct = Counter(), Counter()
            query_correct = Counter()
            identities, answers, predictions, template_ids = [], [], [], []
            calls = sources = 0
            ce = 0.
            for start in range(0, args.count, args.batch):
                batch = make_view_batch('independent', start, min(args.batch, args.count-start),
                                        seed=protocol['seed'], gap=protocol['gap'], query_every=protocol['query_every'],
                                        query_type='shared', control='b_only')
                available = {view: mask.clone() for view, mask in batch.availability.items()}
                if condition == 'B_final_query_AC':
                    for view in ('A', 'C'):
                        available[view][:, -1] = True
                elif condition == 'B_all_queries_AC':
                    for view in ('A', 'C'):
                        available[view] = batch.targets != -100
                # Only availability changes; no labels, factor metadata, or
                # private observations are passed into the model.
                result = model(batch.inputs, reset=batch.reset, availability=available)
                predicted = result['logits'].argmax(-1)
                good = predicted[:, -1] == batch.answers
                calls += result['expert_executions'];sources += result['source_occurrences']
                ce += float(F.cross_entropy(result['logits'][:, -1], batch.answers, reduction='sum'))
                for index in (batch.targets[0] != -100).nonzero().flatten().tolist():
                    query_correct[str(index)] += int((predicted[:, index] == batch.answers).sum())
                for value, prediction, template, hit in zip(batch.answers.tolist(), predicted[:, -1].tolist(),
                                                           batch.template_ids['B'].tolist(), good.tolist()):
                    label = chr(value)
                    counts[label] += 1;correct[label] += int(hit)
                    template_counts[str(template)] += 1;template_correct[str(template)] += int(hit)
                    crossed = label+'/'+str(template)
                    crossed_counts[crossed] += 1;crossed_correct[crossed] += int(hit)
                    confusion[label][chr(prediction) if prediction in b'ABCD' else 'OTHER'] += 1
                identities.extend(batch.identities);answers.extend(batch.answers.tolist())
                predictions.extend(predicted[:, -1].tolist());template_ids.extend(batch.template_ids['B'].tolist())
            if expected_ids is None:
                expected_ids, expected_calls = identities, calls
            assert identities == expected_ids and calls == expected_calls
            score = {'accuracy': sum(correct.values())/args.count, 'nats_per_answer': ce/args.count,
                     'by_answer': {key: {'count': count, 'correct': correct[key], 'accuracy': correct[key]/count}
                                   for key, count in counts.items()},
                     'by_B_template': {key: {'count': count, 'correct': template_correct[key], 'accuracy': template_correct[key]/count}
                                       for key, count in template_counts.items()},
                     'by_answer_B_template': {key: {'count': count, 'correct': crossed_correct[key], 'accuracy': crossed_correct[key]/count}
                                              for key, count in crossed_counts.items()},
                     'confusion': {key: dict(value) for key, value in confusion.items()},
                     'query_accuracy': {key: value/args.count for key, value in query_correct.items()},
                     'expert_executions': calls, 'source_occurrences': sources,
                     'episode_sha256': hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
                     'seconds': time.perf_counter()-started}
            report['results'][policy][condition] = score
            raw.write(json.dumps({'policy': policy, 'condition': condition, 'episode_ids': identities,
                                  'answers': answers, 'predictions': predictions, 'B_templates': template_ids})+'\n')
            raw.flush()
            assert tensor_fingerprint(model) == before
            report['model_tensors_unchanged'] = True
            (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'policy': policy, 'condition': condition, **score}), flush=True)
