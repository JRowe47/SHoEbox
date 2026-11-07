"""Memory-focused continuation with balanced ordinary acquisition prefixes."""
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
from experiments.text_regions.memory_acquisition import acquisition_cases, delayed_cases, pack_cases, prefix_forward
from experiments.text_regions.value_recall import evaluate_values
from experiments.text_regions.graph_mixed_train import evaluate_language
from experiments.text_regions.mixed_gradients import clip_mixed_gradients


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=ROOT/'artifacts/connected_recall/connected/best.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--steps', type=int, default=96)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--language-batch', type=int, default=4)
    p.add_argument('--language-context', type=int, default=32)
    p.add_argument('--language-weight', type=float, default=.25)
    p.add_argument('--gaps', type=int, nargs='+', default=[0])
    p.add_argument('--carry-cursor', action='store_true', help='Continue an acquisition parent at its next source positions')
    p.add_argument('--eval-every', type=int, default=16)
    p.add_argument('--seed', type=int, default=107)
    p.add_argument('--data-seed', type=int, default=683)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    stop = a.steps if a.stop_after is None else a.stop_after
    if min(a.steps, a.cases, a.language_batch, a.language_context, a.eval_every) < 1 or not 1 <= stop <= a.steps:
        p.error('Positive budgets and valid stopping step required.')
    if a.lr <= 0 or a.language_weight < 0 or min(a.gaps) < 0:
        p.error('Positive learning rate; nonnegative language weight and gaps required.')
    if not a.resume and a.out.exists() and any(a.out.iterdir()):
        p.error('A new experiment requires an empty directory.')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    raw = a.data.read_bytes()
    train, validation, _ = split_bytes(raw)
    parent = torch.load(a.parent, map_location='cpu', weights_only=True)
    raw_digest = hashlib.sha256(raw).hexdigest()
    if raw_digest != parent['raw_sha256']:
        raise ValueError('Corpus differs from parent.')
    cfg = GraphConfig.from_dict(parent['config'])
    data_config = TaskStreamConfig.from_dict(parent['data_config'])
    if not data_config.acquisition_queries:
        raise ValueError('Ordinary grammar must include native acquisition questions.')
    protocol = {k:getattr(a,k) for k in ('lr','steps','cases','language_batch','language_context',
        'language_weight','eval_every','seed','data_seed')}
    protocol.update(syntax_weight=.1, global_clip=1., optimizer='parent AdamW moments',
        parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
        selection='Minimum validation balanced acquisition NLL at 0 and every eval interval, earlier ties win',
        validation_cases=16, memory='All four source values, new base documents each update, full connected prefix',
        inference='Unmodified graph; no forced publication, owner labels, or answer inputs')
    if a.gaps != [0]:
        protocol['gaps'] = a.gaps
        protocol['selection'] = 'Minimum validation balanced recall NLL averaged over declared gaps; earlier ties win'
        protocol['validation_cases'] *= len(a.gaps)
    if a.carry_cursor:
        if parent.get('kind') != 'balanced_acquisition':
            raise ValueError('Cursor continuation requires an acquisition parent.')
        for key in ('data_seed','language_batch','language_context'):
            if parent['protocol'][key] != protocol[key]:
                raise ValueError(f'Carried source setting differs: {key}')
        protocol.update(carry_cursor=True, initial_document=parent['next_document'], parent_step=parent['step'])
    model = RegionLanguageModel(cfg); model.load_state_dict(parent['weights'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(.9,.95), weight_decay=.01)
    optimizer.load_state_dict(parent['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = a.lr
    streams = OrderedByteStreams.from_bytes(train, a.language_batch)
    language_state = model.initial_state(a.language_batch)
    step, next_document, best = 0, 0, float('inf')
    if a.carry_cursor:
        streams.load_state_dict(parent['language_stream'])
        language_state = GraphState.from_dict(parent['language_state'])
        next_document = parent['next_document']
        torch.set_rng_state(parent['rng'])
    totals = dict(memory_targets=0, memory_bytes=0, memory_padded_positions=0, language_bytes=0)
    native_cases = acquisition_cases(data_config, split='validation', seed=a.data_seed,
        start=0, count=16, distractor_bytes=validation)
    val_cases = [case for gap in a.gaps for case in delayed_cases(native_cases,gap,validation)]
    a.out.mkdir(parents=True, exist_ok=True)

    def save(path):
        temporary = path.with_suffix('.tmp')
        torch.save(dict(schema=1, kind='balanced_acquisition', config=cfg.to_dict(),
            data_config=data_config.to_dict(), protocol=protocol, raw_sha256=raw_digest,
            weights=model.state_dict(), optimizer=optimizer.state_dict(),
            language_state=language_state.to_dict(), language_stream=streams.state_dict(),
            rng=torch.get_rng_state(), step=step, next_document=next_document, best=best, totals=totals), temporary)
        temporary.replace(path)

    def validate():
        result = evaluate_values(model, val_cases, context=32, cases_per_batch=8)
        language = evaluate_language(model, validation, batch=4, context=32, chunks=2)
        return dict(acquisition={k:v for k,v in result.items() if k != 'rows'}, language=language)

    if a.resume:
        saved = torch.load(a.out/'last.pt', map_location='cpu', weights_only=True)
        for key, value in dict(kind='balanced_acquisition', protocol=protocol, config=cfg.to_dict(),
                               data_config=data_config.to_dict(), raw_sha256=raw_digest).items():
            if saved[key] != value:
                raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        language_state = GraphState.from_dict(saved['language_state'])
        streams.load_state_dict(saved['language_stream']); torch.set_rng_state(saved['rng'])
        step, next_document, best, totals = (saved[k] for k in ('step','next_document','best','totals'))
        if not (a.out/'best.pt').exists():
            raise ValueError('Resume requires best.pt.')
    else:
        initial = validate(); best = initial['acquisition']['nll']
        save(a.out/'best.pt'); save(a.out/'last.pt')
        write_json(a.out/'initial.json', initial)
        write_json(a.out/'protocol.json', dict(protocol=protocol, config=cfg.to_dict(),
            data_config=data_config.to_dict(), parameters=sum(p.numel() for p in model.parameters()),
            parent=str(a.parent.resolve().relative_to(ROOT)), raw_sha256=raw_digest))
        print(json.dumps(dict(stage='initial', validation=initial)), flush=True)
    if stop < step:
        raise ValueError('Stopping point precedes saved step.')
    started = time.perf_counter()
    while step < stop:
        model.train(); optimizer.zero_grad(set_to_none=True)
        language = streams.next_chunk(a.language_context)
        result = model(language.values, state=language_state, reset=language.reset, targets=language.targets)
        language_loss = result['loss']; language_nll = float(result['ce'].detach())
        (a.language_weight * language_loss).backward()
        language_state = result['state'].detach()
        del result, language_loss
        cases = acquisition_cases(data_config, split='train', seed=a.data_seed, start=next_document,
            count=a.cases, distractor_bytes=train)
        gap = a.gaps[step % len(a.gaps)]
        if gap:
            cases = delayed_cases(cases,gap,train)
        packed = pack_cases(cases)
        result = prefix_forward(model, packed)
        logits = result['logits'][:, -1]
        primary = F.cross_entropy(logits, packed[2])
        syntax = byte_loss(result['logits'], packed[1])
        memory_loss = primary + .1*syntax + cfg.auxiliary_loss_weight*result['aux'] + cfg.hat_balance_weight*result['hat_aux']
        if not bool(torch.isfinite(memory_loss)):
            raise RuntimeError('Nonfinite acquisition objective.')
        memory_loss.backward()
        norm, recovered = clip_mixed_gradients(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite gradient norm.')
        optimizer.step(); step += 1; next_document += a.cases
        totals['memory_targets'] += packed[2].numel()
        totals['memory_bytes'] += int(packed[3].sum())
        totals['memory_padded_positions'] += packed[0].numel()
        totals['language_bytes'] += language.values.numel()
        row = dict(step=step, next_document=next_document, language_nll=language_nll,
            memory_nll=float(primary.detach()), syntax_nll=float(syntax.detach()),
            train_accuracy=float((logits.argmax(-1) == packed[2]).float().mean()),
            gradient_norm=float(norm), fp64_norm_recovered=recovered,
            totals=dict(totals), seconds=time.perf_counter()-started,
            peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if a.gaps != [0]:
            row['gap'] = gap
        del result, logits, primary, syntax, memory_loss
        if step % a.eval_every == 0 or step == a.steps:
            row['validation'] = validate()
            if row['validation']['acquisition']['nll'] < best:
                best = row['validation']['acquisition']['nll']; save(a.out/'best.pt')
        save(a.out/'last.pt')
        with (a.out/'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
