"""Continue joint learning with balanced retention, selection and revision."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.state import GraphState
from experiments.text_regions.data import OrderedByteStreams, split_bytes
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.task_memory_train import byte_loss
from experiments.text_regions.memory_acquisition import pack_cases, prefix_forward
from experiments.text_regions.memory_selectivity import MODES, selective_cases, training_cases, summarize_values
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language
from experiments.text_regions.mixed_gradients import clip_mixed_gradients


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class SelectivityRun:
    def __init__(self, parent, raw, *, lr=.003, steps=48, gaps=(0,4,8,16,32),
                 language_weight=.25, eval_every=16, validation_cases=4):
        if (parent.get('kind') != 'balanced_acquisition' or min(steps, eval_every, validation_cases) < 1
                or lr <= 0 or language_weight < 0 or not gaps or min(gaps) < 0):
            raise ValueError('Require an acquisition parent and valid positive budgets.')
        digest = hashlib.sha256(raw).hexdigest()
        if digest != parent['raw_sha256']:
            raise ValueError('Corpus differs from parent.')
        self.cfg = GraphConfig.from_dict(parent['config'])
        self.data_cfg = TaskStreamConfig.from_dict(parent['data_config'])
        self.train, self.validation, _ = split_bytes(raw)
        old = parent['protocol']
        self.protocol = dict(lr=lr, steps=steps, gaps=list(gaps), language_weight=language_weight,
            eval_every=eval_every, validation_cases=validation_cases, validation_start=1000,
            validation_gaps=[0,32], data_seed=old['data_seed'], language_batch=old['language_batch'],
            language_context=old['language_context'], initial_document=parent['next_document'],
            parent_step=parent['step'], modes=list(MODES), syntax_weight=.1, global_clip=1.,
            selection='Minimum unweighted mean NLL over four modes by two gaps; earlier ties win',
            generator_sha256=hashlib.sha256((ROOT/'experiments/text_regions/memory_selectivity.py').read_bytes()).hexdigest())
        self.raw_sha256 = digest
        self.model = RegionLanguageModel(self.cfg)
        self.model.load_state_dict(parent['weights'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(.9,.95), weight_decay=.01)
        self.optimizer.load_state_dict(parent['optimizer'])
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.streams = OrderedByteStreams.from_bytes(self.train, old['language_batch'])
        self.streams.load_state_dict(parent['language_stream'])
        self.language_state = GraphState.from_dict(parent['language_state'])
        torch.set_rng_state(parent['rng'])
        self.step = 0
        self.next_document = parent['next_document']
        self.best = float('inf')
        self.totals = dict(memory_targets=0, memory_bytes=0, memory_padded_positions=0,
                           language_bytes=0, by_mode={mode:0 for mode in MODES})
        self.val_cases = {f'{mode}_{gap}': selective_cases(self.data_cfg, split='validation',
            seed=old['data_seed'], start=1000, count=validation_cases, mode=mode, gap=gap,
            distractor_bytes=self.validation) for mode in MODES for gap in (0,32)}

    def validate(self):
        cohorts = {key:summarize_values(evaluate_values(self.model, cases, context=32, cases_per_batch=8))
                   for key,cases in self.val_cases.items()}
        score = sum(row['nll'] for row in cohorts.values())/len(cohorts)
        language = evaluate_language(self.model, self.validation, batch=4, context=32, chunks=2)
        return dict(selection_score=score, cohorts=cohorts, language=language)

    def update(self):
        started = time.perf_counter()
        p = self.protocol
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        language = self.streams.next_chunk(p['language_context'])
        result = self.model(language.values, state=self.language_state,
                            reset=language.reset, targets=language.targets)
        language_nll = float(result['ce'].detach())
        (p['language_weight']*result['loss']).backward()
        self.language_state = result['state'].detach()
        del result
        cases, mode, gap, next_document = training_cases(self.data_cfg, step=self.step,
            start=self.next_document, seed=p['data_seed'], gaps=p['gaps'], distractor_bytes=self.train)
        packed = pack_cases(cases)
        result = prefix_forward(self.model, packed)
        logits = result['logits'][:, -1]
        primary = F.cross_entropy(logits, packed[2])
        syntax = byte_loss(result['logits'], packed[1])
        loss = (primary + p['syntax_weight']*syntax + self.cfg.auxiliary_loss_weight*result['aux']
                + self.cfg.hat_balance_weight*result['hat_aux'])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite memory objective.')
        loss.backward()
        norm, recovered = clip_mixed_gradients(self.model.parameters(), p['global_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite gradient norm.')
        self.optimizer.step()
        self.step += 1
        self.next_document = next_document
        self.totals['memory_targets'] += packed[2].numel()
        self.totals['memory_bytes'] += int(packed[3].sum())
        self.totals['memory_padded_positions'] += packed[0].numel()
        self.totals['language_bytes'] += language.values.numel()
        self.totals['by_mode'][mode] += packed[2].numel()
        return dict(step=self.step, next_document=self.next_document, mode=mode, gap=gap,
            language_nll=language_nll, memory_nll=float(primary.detach()), syntax_nll=float(syntax.detach()),
            train_accuracy=float((logits.argmax(-1) == packed[2]).float().mean()),
            gradient_norm=float(norm), fp64_norm_recovered=recovered,
            totals=json.loads(json.dumps(self.totals)), seconds=time.perf_counter()-started,
            peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

    def state_dict(self):
        return dict(schema=1, kind='selective_recall', config=self.cfg.to_dict(),
            data_config=self.data_cfg.to_dict(), protocol=self.protocol, raw_sha256=self.raw_sha256,
            weights=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            language_state=self.language_state.to_dict(), language_stream=self.streams.state_dict(),
            rng=torch.get_rng_state(), step=self.step, next_document=self.next_document,
            best=self.best, totals=self.totals)

    def restore(self, saved):
        for key in ('schema','kind','config','data_config','protocol','raw_sha256'):
            if saved[key] != self.state_dict()[key]:
                raise ValueError(f'Resume mismatch: {key}')
        self.model.load_state_dict(saved['weights']); self.optimizer.load_state_dict(saved['optimizer'])
        self.language_state = GraphState.from_dict(saved['language_state'])
        self.streams.load_state_dict(saved['language_stream']); torch.set_rng_state(saved['rng'])
        self.step, self.next_document, self.best, self.totals = (
            saved[k] for k in ('step','next_document','best','totals'))

    def save(self, path):
        temporary = path.with_suffix('.tmp')
        torch.save(self.state_dict(), temporary); temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--steps', type=int, default=48)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--gaps', type=int, nargs='+', default=[0,4,8,16,32])
    p.add_argument('--language-weight', type=float, default=.25)
    p.add_argument('--eval-every', type=int, default=16)
    p.add_argument('--validation-cases', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    stop = a.steps if a.stop_after is None else a.stop_after
    if not 1 <= stop <= a.steps:
        p.error('Stopping point must lie inside the declared budget.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()):
        p.error('A new experiment requires an empty directory.')
    torch.set_num_threads(1)
    run = SelectivityRun(torch.load(a.parent, map_location='cpu', weights_only=True), a.data.read_bytes(),
        lr=a.lr, steps=a.steps, gaps=a.gaps, language_weight=a.language_weight,
        eval_every=a.eval_every, validation_cases=a.validation_cases)
    run.protocol['parent_sha256'] = hashlib.sha256(a.parent.read_bytes()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        run.restore(torch.load(a.out/'last.pt', map_location='cpu', weights_only=True))
        if not (a.out/'best.pt').exists():
            raise ValueError('Resume requires best.pt.')
    else:
        initial = run.validate(); run.best = initial['selection_score']
        run.save(a.out/'best.pt'); run.save(a.out/'last.pt')
        write_json(a.out/'initial.json', initial)
        write_json(a.out/'protocol.json', dict(protocol=run.protocol, config=run.cfg.to_dict(),
            data_config=run.data_cfg.to_dict(), raw_sha256=run.raw_sha256,
            parameters=sum(p.numel() for p in run.model.parameters())))
        print(json.dumps(dict(stage='initial', validation=initial)), flush=True)
    if stop < run.step:
        raise ValueError('Stopping point precedes saved step.')
    while run.step < stop:
        row = run.update()
        if run.step % a.eval_every == 0 or run.step == a.steps:
            row['validation'] = run.validate()
            if row['validation']['selection_score'] < run.best:
                run.best = row['validation']['selection_score']; run.save(a.out/'best.pt')
        run.save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
