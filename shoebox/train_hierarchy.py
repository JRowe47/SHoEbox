"""Matched hierarchy/local training, document-disjoint evaluation, exact resume.

python -m shoebox.train_hierarchy --data data/hierarchy --out artifacts/pilot
python -m shoebox.train_hierarchy --data data/hierarchy --out artifacts/local --local-only
Default B8/T128, 1024 updates = 1,048,576 observed next-byte targets.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import torch
from torch.nn import functional as F
from .hierarchy import HierarchyModel
from .hierarchy_config import HierarchyConfig
from .checkpoint import save, load, HIERARCHY_FORMAT
from .train_fineweb import atomic_json, optimizer_for, source_digest
from .fused_kron import native_status


def load_data(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['format'] != 'shoebox-hierarchy-data-1':
        raise ValueError('Use scripts/prepare_hierarchy_data.py for the document-aware corpus.')
    payloads = {}
    for name in ('train', 'validation'):
        raw = (folder/(name+'.bin')).read_bytes()
        spec = manifest['files'][name]
        if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Corpus checksum/size mismatch.')
        payloads[name] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    for key in ('id', 'source_sha256'):
        if ({r[key] for r in manifest['files']['train']['documents']} &
                {r[key] for r in manifest['files']['validation']['documents']}):
            raise ValueError('Training and validation documents overlap.')
    batch = manifest['batch']
    train = payloads['train'].reshape(batch, -1)
    if train.numel()-batch != manifest['train_targets']:
        raise ValueError('Training target count mismatch.')
    docs = [payloads['validation'][r['offset']:r['offset']+r['bytes']]
            for r in manifest['files']['validation']['documents']]
    if sum(d.numel()-1 for d in docs) != manifest['validation_targets']:
        raise ValueError('Validation target count mismatch.')
    return train, docs, manifest


@torch.no_grad()
def evaluate(model, docs, *, context=128, batch=32, intervention=None, limit=None):
    training = model.training
    model.eval()
    nll, correct, count, rows = 0., 0, 0, 0
    start = time.perf_counter()
    try:
        for offset in range(0, len(docs), batch):
            group = docs[offset:offset+batch]
            maximum = max(d.numel()-1 for d in group)
            values = torch.zeros(len(group), maximum, dtype=torch.long)
            targets = torch.full_like(values, -100)
            for i, d in enumerate(group):
                values[i, :d.numel()-1], targets[i, :d.numel()-1] = d[:-1], d[1:]
            # All rows start in separate documents with fresh states. Padding
            # never becomes a scored target or carries into another document.
            state = None
            permutation = torch.arange(len(group)).roll(1)
            for cursor in range(0, maximum, context):
                x, y = values[:, cursor:cursor+context], targets[:, cursor:cursor+context]
                if limit is not None:
                    valid = (y != -100).flatten().nonzero().flatten()
                    remaining = limit-count
                    if remaining <= 0:
                        break
                    if valid.numel() > remaining:
                        y = y.clone().contiguous()
                        y.view(-1)[valid[remaining:]] = -100
                scored = int((y != -100).sum())
                if not scored:
                    continue
                out = model(x if intervention != 'input' else x[permutation], state=state,
                            context_permutation=permutation if intervention == 'upper' else None)
                state = out['state']
                nll += float(F.cross_entropy(out['logits'].flatten(0, 1), y.flatten(), reduction='sum'))
                correct += int(((out['logits'].argmax(-1) == y) & (y != -100)).sum())
                count += scored
                rows += sum(out['work']['region_rows'].values())
            if limit is not None and count >= limit:
                break
    finally:
        model.train(training)
    seconds = time.perf_counter()-start
    return dict(nll=nll/count, accuracy=correct/count, targets=count, seconds=seconds,
                scored_bytes_per_second=count/seconds, region_rows_including_padding=rows,
                intervention=intervention)


def count_baselines(counts, docs):
    # Add-one smoothing on the exact observed training pairs, including repeats.
    unigram = counts.sum(0)+1
    unigram = unigram/unigram.sum()
    bigram = counts+1
    bigram = bigram/bigram.sum(-1, keepdim=True)
    records = {}
    for name, probabilities in (('unigram', unigram), ('bigram', bigram)):
        loss, correct, count = 0., 0, 0
        for d in docs:
            x, y = d[:-1], d[1:]
            p = probabilities.expand(x.numel(), -1) if probabilities.ndim == 1 else probabilities[x]
            loss -= float(p.gather(1, y[:, None]).log().sum())
            correct += int((p.argmax(-1) == y).sum()); count += y.numel()
        records[name] = dict(nll=loss/count, accuracy=correct/count, targets=count, smoothing='add-one')
    return records


@torch.no_grad()
def routing_diagnostic(model, docs, steps=32):
    """Small fresh-state probe, separate from scored evaluation and throughput."""
    group = [d[:steps] for d in docs if d.numel() >= steps][:8]
    if not group:
        return {}
    training = model.training; model.eval()
    try:
        result = model(torch.stack(group), trace=True)
        selected = {}
        for name, ids in result['experts'].items():
            # [events,batch,three layers,k], retaining actual sparse selections.
            selected[name] = [int(ids[:, :, layer].unique().numel()) for layer in range(ids.shape[2])]
        return dict(work=result['work'], unique_experts_per_layer=selected,
                    experts_per_bank=model.cfg.experts_per_layer,
                    selected_per_invocation=model.cfg.active_groups*model.cfg.experts_per_group,
                    regional_routing='fixed complete paths; no regional balancing loss')
    finally:
        model.train(training)


@torch.no_grad()
def sample(model, prompt='The ', count=64, seed=731, temperature=.8):
    training = model.training; model.eval()
    try:
        generator = torch.Generator().manual_seed(seed)
        values = torch.tensor([list(prompt.encode())], dtype=torch.long)
        if values.numel() == 0 or temperature <= 0:
            raise ValueError('Need a nonempty prompt and positive temperature.')
        result = model(values); output = bytearray(prompt.encode())
        for _ in range(count):
            value = torch.multinomial((result['logits'][:, -1]/temperature).softmax(-1), 1, generator=generator)
            output.append(int(value[0, 0]))
            result = model(value, state=result['state'])
        return output.decode('utf-8', errors='replace')
    finally:
        model.train(training)


def train_update(model, optimizer, x, y, state, future_targets):
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    out = model(x, y, state, future_targets=future_targets)
    forward = time.perf_counter()-start
    start = time.perf_counter(); out['loss'].backward(); backward = time.perf_counter()-start
    start = time.perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
    optimizer.step(); update = time.perf_counter()-start
    seconds = forward+backward+update
    return out['state'].detach(), dict(nll=float(out['ce'].detach()),
            mid_nll=float(out['span_losses']['mid'].detach()), high_nll=float(out['span_losses']['high'].detach()),
            span_targets=out['span_targets'], gradient_norm=float(norm),
            forward_seconds=forward, backward_seconds=backward, optimizer_seconds=update,
            training_seconds=seconds, training_bytes_per_second=x.numel()/seconds,
            region_executions_per_byte=sum(out['work']['region_rows'].values())/x.numel())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1024)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--warmup-updates', type=int, default=16)
    p.add_argument('--validate-every', type=int, default=64)
    p.add_argument('--checkpoint-every', type=int, default=32)
    p.add_argument('--log-every', type=int, default=16)
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-updates', type=int, help='Pause after this many additional updates, retaining the full protocol.')
    p.add_argument('--continuity', type=Path, help='Optional old 4k raw validation slice; separate continuity metric.')
    args = p.parse_args()
    if min(args.updates, args.context, args.threads, args.validate_every, args.checkpoint_every, args.log_every) < 1 or args.lr <= 0 or args.warmup_updates < 0:
        p.error('Counts and learning rate must be positive.')
    if args.max_updates is not None and args.max_updates < 1:
        p.error('--max-updates must be positive.')
    if args.out.exists() and not args.resume:
        p.error('Choose a new output directory.')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    train, validation, manifest = load_data(args.data)
    continuity_sha = None
    if args.continuity:
        old = json.loads((args.continuity.parent/'manifest.json').read_text())['files']['validation']
        continuity_sha = hashlib.sha256(args.continuity.read_bytes()).hexdigest()
        if continuity_sha != old['sha256']:
            raise ValueError('Continuity data checksum mismatch.')
        for key, old_key in (('id', 'id'), ('source_sha256', 'source_text_sha256')):
            if ({r[key] for r in manifest['files']['train']['documents']} &
                    {r[old_key] for r in old['documents']}):
                raise ValueError('Continuity documents overlap training; reserve them during preparation.')
    if (train.shape[1]-1) % args.context:
        p.error('Training stream length must divide into complete credit windows.')
    digest = source_digest()
    protocol = dict(source_sha256=digest, train_sha256=manifest['files']['train']['sha256'],
                    validation_sha256=manifest['files']['validation']['sha256'],
                    batch=train.shape[0], context=args.context, updates=args.updates, lr=args.lr,
                    warmup_updates=args.warmup_updates, local_only=args.local_only, threads=args.threads)
    protocol['continuity_sha256'] = continuity_sha
    step, cursor, epoch, elapsed = 0, 0, 0, 0.
    history, evaluations, state = [], [], None
    counts = torch.zeros(256, 256, dtype=torch.float64)
    if args.resume:
        model, state, _, checkpoint = load(args.resume)
        if checkpoint['format'] != HIERARCHY_FORMAT or checkpoint['metadata']['protocol'] != protocol:
            raise ValueError('Resume requires the same model, corpus, source and training protocol.')
        optimizer = optimizer_for(model, args.lr, .01); optimizer.load_state_dict(checkpoint['optimizer'])
        old = checkpoint['metadata']
        step, cursor, epoch, elapsed = checkpoint['step'], old['cursor'], old['epoch'], old['elapsed']
        history, evaluations, counts = old['history'], old['evaluations'], old['counts']
    else:
        cfg = HierarchyConfig(local_only=args.local_only)
        torch.manual_seed(cfg.seed)
        model = HierarchyModel(cfg)
        optimizer = optimizer_for(model, args.lr, .01)
    model.set_execution_backend('cpu')
    kernel = native_status()
    if not kernel['available']:
        raise RuntimeError(kernel)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/'data_manifest.json', manifest)
    atomic_json(args.out/'config.json', model.cfg.to_dict())
    (args.out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    started = time.perf_counter(); first_step = step

    def summary(complete=False):
        return dict(format=HIERARCHY_FORMAT, complete=complete, protocol=protocol,
                    parameters=model.parameter_cost(), updates=step, trained_targets=step*train.shape[0]*args.context,
                    unique_corpus_targets=manifest['train_targets'], epoch=epoch, history=history, evaluation=evaluations,
                    training_seconds=elapsed, training_bytes_per_second=step*train.shape[0]*args.context/elapsed if elapsed else 0,
                    wall_seconds_this_session=time.perf_counter()-started,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    runtime_state_bytes=state.storage_bytes() if state else 0, torch=str(torch.__version__), native=kernel)

    def snapshot(complete=False):
        save(args.out/'last.pt', model, state, optimizer, step=step,
             metadata=dict(protocol=protocol, cursor=cursor, epoch=epoch, elapsed=elapsed,
                           history=history, evaluations=evaluations, counts=counts))
        atomic_json(args.out/'results.json', summary(complete))

    def validate(full=False):
        score = evaluate(model, validation, context=args.context, limit=None if full else 8192)
        row = dict(updates=step, full=full, **score)
        if full:
            row['baselines'] = count_baselines(counts, validation)
        evaluations.append(row)
        print(json.dumps(dict(event='validation', **row)), flush=True)
        return row

    print(json.dumps(dict(event='start', parameters=model.parameter_cost(), protocol=protocol)), flush=True)
    if not evaluations:
        validate(); snapshot()
    while step < args.updates:
        if cursor == train.shape[1]-1:
            cursor = 0; epoch += 1; state = None
        x = train[:, cursor:cursor+args.context]
        y = train[:, cursor+1:cursor+args.context+1]
        future = train[:, cursor+1:cursor+args.context+16]
        for group in optimizer.param_groups:
            group['lr'] = args.lr*min(1., (step+1)/max(1, args.warmup_updates))
        state, row = train_update(model, optimizer, x, y, state, future)
        counts += torch.bincount((x*256+y).flatten(), minlength=65536).reshape(256, 256)
        cursor += args.context; step += 1; elapsed += row['training_seconds']
        row.update(updates=step, trained_targets=step*x.numel(), epoch=epoch, lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        with (args.out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if step % args.log_every == 0:
            print(json.dumps(dict(event='update', **row)), flush=True)
        complete = step == args.updates
        paused = args.max_updates is not None and step-first_step >= args.max_updates and not complete
        if step % args.validate_every == 0 or complete or paused:
            record = validate(full=step % 256 == 0 or complete or paused)
            # The first milestone must establish observation conditioning.
            # Fail closed at 256 instead of spending the remaining 768 updates.
            if step == 256:
                swapped = evaluate(model, validation, context=args.context, intervention='input')
                record['input_swap'] = swapped
                record['gate_passed'] = (record['nll'] < record['baselines']['unigram']['nll']-.05 and
                                         swapped['nll'] > record['nll']+.01)
                if not record['gate_passed']:
                    paused = not complete
                    record['stop_reason'] = 'Inspect the local observation path before extending the run.'
        if step % args.checkpoint_every == 0 or complete or paused:
            snapshot(complete)
        if paused:
            print(json.dumps(dict(event='paused', updates=step)), flush=True)
            return
    final = summary(True)
    final['baselines'] = count_baselines(counts, validation)
    final['input_swap'] = (evaluations[-1].get('input_swap') or
                           evaluate(model, validation, context=args.context, intervention='input'))
    if not model.cfg.local_only:
        final['upper_swap'] = evaluate(model, validation, context=args.context, intervention='upper')
        final['upper_swap_delta_nll'] = final['upper_swap']['nll']-evaluations[-1]['nll']
    if args.continuity:
        raw = torch.frombuffer(bytearray(args.continuity.read_bytes()), dtype=torch.uint8).long()
        if raw.numel() < 4050:
            raise ValueError('Continuity slice requires the original 50 streams of 81 bytes.')
        final['continuity_4k'] = evaluate(model, list(raw[:4050].reshape(50, 81)), context=args.context)
    final['samples'] = [sample(model, prompt) for prompt in ('The ', 'In the ')]
    final['routing'] = routing_diagnostic(model, validation)
    if source_digest() != digest:
        raise RuntimeError('Model/training source changed during the run.')
    final['wall_seconds_this_session'] = time.perf_counter()-started
    atomic_json(args.out/'results.json', final)
    print(json.dumps(dict(event='complete', updates=step, training_bytes_per_second=final['training_bytes_per_second'],
                         validation=evaluations[-1], input_swap=final['input_swap'],
                         upper_swap_delta_nll=final.get('upper_swap_delta_nll'))), flush=True)


if __name__ == '__main__':
    main()
