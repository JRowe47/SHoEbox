"""Read-only SHoEbox connectivity, credit-horizon and clipping audit.

Uses a supplied checkpoint; never steps its optimizer or rewrites training data.
Reports surrogate derivatives separately from finite changes in observations.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents
from experiments.text_regions.data import split_bytes
from experiments.text_regions.graph_mixed_train import intervention
from experiments.text_regions.state_credit_train import StateCreditRun
from experiments.text_regions.context_balance import parameter_groups
from experiments.text_regions.mixed_gradients import clip_mixed_gradients
from experiments.text_regions import gradient_balance
from scripts.evaluate_state_credit import counterfactual_pairs


def norm(tensors):
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in tensors if x is not None))


def stats(values):
    x = torch.tensor(values, dtype=torch.float64).flatten()
    return dict(count=x.numel(), minimum=float(x.min()), median=float(x.median()),
                mean=float(x.mean()), maximum=float(x.max()))


def load_model(saved, dtype=torch.float32):
    model = RegionLanguageModel(GraphConfig.from_dict(saved['config'])).to(dtype=dtype)
    model.load_state_dict(saved['weights'])
    model.eval()
    return model


def archived(root):
    output = {}
    for name, directory in [('balance_sum', root.parent/'context_balance/sum'),
                            ('balance_groups', root.parent/'context_balance/group_clip'),
                            ('state_credit', root/'state'), ('publication_credit', root/'publication')]:
        rows = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
        norms = [r['gradient_norm'] for r in rows]
        output[name] = dict(updates=len(rows), clipped_fraction=sum(x > 1 for x in norms)/len(rows),
            norm=stats(norms), median_scale=statistics.median(min(1., 1/(x+1e-6)) for x in norms),
            overflow_recoveries=sum(r['fp64_norm_recovered'] for r in rows))
    evaluation = json.loads((root/'evaluation.json').read_text())
    output['acquisition'] = {}
    for name in ('parent', 'state_best', 'publication_best'):
        rows = [json.loads(line) for line in (root/(evaluation[name]['signature']+'_test_traces.jsonl')).read_text().splitlines()]
        rows = [r for r in rows if r['kind'] == 'exact' and r['cycle'] == -1]
        output['acquisition'][name] = dict(count=len(rows), accuracy=sum(r['prediction']==r['answer'] for r in rows)/len(rows),
            nll=statistics.mean(r['nll'] for r in rows), predictions=dict(Counter(chr(r['prediction']) for r in rows)))
    return output


@torch.no_grad()
def forward_audit(saved, test):
    model = load_model(saved)
    docs = make_documents(TaskStreamConfig.from_dict(saved['data_config']), split='test',
        seed=saved['protocol']['data_seed'], count=8, distractor_bytes=test)
    # The normal generator, unchanged; focus on its first immediate query.
    native = []
    for doc in docs:
        stop = max(q.position for q in doc.queries if q.cycle == -1)+1
        r = model(torch.tensor([list(doc.data[:stop])]))
        for q in doc.queries:
            if q.cycle == -1:
                logits = r['logits'][0, q.position]
                native.append(dict(document=doc.index, delay=q.delay, answer=chr(q.answer),
                    prediction=chr(int(logits.argmax())), nll=float(F.cross_entropy(logits[None], torch.tensor([q.answer])))))

    # All four possible observed values, identical subsequent inputs per gap.
    gaps = (0, 8, 16, 32, 64, 128)
    probes = []
    for gap in gaps:
        values = [b'A x='+bytes([v])+b';'+test[:gap]+b'A x?' for v in b'0123']
        logits = model(torch.tensor([list(x) for x in values]))['logits'][:, -1]
        probs = logits.softmax(-1)
        target = torch.tensor(list(b'0123'))
        # Observation-conditioned margin versus each other observed value.
        contrast = []
        for a in range(4):
            for b in range(a+1, 4):
                contrast.append(float((logits[a, target[a]]-logits[a, target[b]])-
                                      (logits[b, target[a]]-logits[b, target[b]])))
        probes.append(dict(gap=gap, delay=gap+5, accuracy=float(logits.argmax(-1).eq(target).float().mean()),
            predictions=[chr(int(x)) for x in logits.argmax(-1)],
            nll=float(F.cross_entropy(logits, target)), mean_directional_contrast=statistics.mean(contrast),
            digit_probabilities=probs[:, target].tolist(),
            maximum_pair_logit_difference=float((logits[:, None]-logits[None]).abs().max())))

    return dict(native_acquisition=dict(count=len(native), accuracy=statistics.mean(r['prediction']==r['answer'] for r in native),
        nll=statistics.mean(r['nll'] for r in native), predictions=dict(Counter(r['prediction'] for r in native)), rows=native),
        balanced_probes=probes)


@torch.no_grad()
def telemetry_audit(saved, test):
    model = load_model(saved)
    # Do not hook recurrent modules: that changes stage fusion. Edge gate hooks
    # observe the normal execution path. Verify the observed run against an
    # unhooked run of exactly the same byte sequence before accepting telemetry.
    writes, recalls, edge_gates = defaultdict(list), defaultdict(list), defaultdict(list)
    handles = []
    for name, module in model.connections.edges.items():
        def collect_gate(module, args, result, key=name):
            edge_gates[key].extend(result.sigmoid().flatten().tolist())
        handles.append(module.gate.register_forward_hook(collect_gate))
    seq = [b'A x='+bytes([v])+b';'+test[:128]+b'A x?' for v in b'01']
    trajectory, state = [], None
    for position in range(len(seq[0])):
        result = model(torch.tensor([[x[position]] for x in seq]), state=state,
            return_write_activity=True, return_recall_activity=True)
        state = result['state']
        w = result['write_activity']
        for i, owner in enumerate(model.cfg.task_publication):
            writes[owner].append(dict(probability=w['write_probs'][:, :, i].flatten().tolist(),
                action=w['write_actions'][:, :, i].flatten().tolist(),
                demand=w['demand'][:, :, i].reshape(-1).tolist()))
        for owner, r in result['recall_activity'].items():
            recalls[owner].append(float(r['weights'][..., -1].mean()))
        lag = position-4
        if lag in (0, 1, 4, 8, 16, 32, 64, 128, 133):
            nodes = {name: norm(t[0]-t[1] for t in node.contextualizer+node.blocks) for name, node in state.nodes.items()}
            trajectory.append(dict(lag=lag, nodes=nodes,
                ingress=norm(t[0]-t[1] for t in state.ingress['text'].tensors() if t.is_floating_point()),
                ports={name: norm([port.payload[0]-port.payload[1]]) for name, port in state.ports.latest.items()},
                logits=norm([result['logits'][0]-result['logits'][1]])))
    observed_logits = result['logits'][:, -1].clone()
    for handle in handles:
        handle.remove()
    controls, base = {}, None
    x = torch.tensor([list(s) for s in seq])
    for control in ('intact', 'no_demand', 'no_feedback', 'blocked', 'always_state', 'always_write'):
        with intervention(model, control) as kwargs:
            r = model(x, **kwargs)
        logits = r['logits'][:, -1]
        base = logits if base is None else base
        controls[control] = dict(logit_delta=norm([logits-base]),
            nll=float(F.cross_entropy(logits, torch.tensor(list(b'01')))),
            predictions=[chr(int(v)) for v in logits.argmax(-1)])
    parity = float((observed_logits-base).abs().max())
    if parity != 0.:
        raise AssertionError(f'Telemetry altered forward values: {parity}')
    return dict(paired_trajectory=trajectory, interventions=controls, telemetry_maximum_difference=parity,
        write_gates={name: dict(probability=stats([p for r in rows for p in r['probability']]),
            sigmoid_slope=stats([p*(1-p) for r in rows for p in r['probability']]),
            write_fraction=statistics.mean(a for r in rows for a in r['action']),
            demand=stats([p for r in rows for p in r['demand']])) for name, rows in writes.items()},
        recall_null_fraction={name: statistics.mean(v) for name, v in recalls.items()},
        edge_gates={name: stats(v) for name, v in edge_gates.items()})


def gradient_audit(saved, checkpoint, raw):
    run = StateCreditRun(raw, GraphConfig.from_dict(saved['config']),
        TaskStreamConfig.from_dict(saved['data_config']), saved['protocol'])
    run.restore(checkpoint)
    named = list(run.model.named_parameters())
    report, snapshots = {}, {}
    original = run.apply_credit
    def capture(boundaries):
        before = {n: None if p.grad is None else p.grad.clone() for n, p in named}
        result = original(boundaries)
        changes = {}
        for n, p in named:
            if p.grad is not None:
                delta = p.grad if before[n] is None else p.grad-before[n]
                if bool(delta.ne(0).any()):
                    changes[n] = dict(parameters=p.numel(), norm=norm([delta]))
        report['last_delayed_additions'] = changes
        return result
    run.apply_credit = capture
    for source in ('language', 'memory'):
        # Skip at most four answer-free task windows, without optimizer updates.
        for attempt in range(4):
            gradients, row = run.source_gradient(source)
            if source != 'memory' or row['answers']:
                break
        by_group = {name: dict(norm=norm(gradients[i] for i in indices),
            parameters=sum(named[i][1].numel() for i in indices),
            tensors=len(indices), none=sum(gradients[i] is None for i in indices),
            nonzero=sum(gradients[i] is not None and bool(gradients[i].ne(0).any()) for i in indices))
            for name, indices in run.groups.items()}
        tensors = {n: dict(norm=norm([g]), parameters=p.numel(), missing=g is None)
                   for (n, p), g in zip(named, gradients)}
        report[source] = dict(**row, groups=by_group, delayed_additions=report['last_delayed_additions'],
            tensors=tensors, inspected_windows=attempt+1)
        snapshots[source] = gradients
    gradient_balance.combine(run.parameters, snapshots['language'], snapshots['memory'],
        run.groups, run.protocol['gradient_mode'], run.protocol['gradient_cap'])
    before = norm(p.grad for p in run.parameters)
    reported, overflow = clip_mixed_gradients(run.parameters, 1.)
    report['clipping'] = dict(before=before, reported=float(reported),
        after=norm(p.grad for p in run.parameters), recovered=overflow)
    del report['last_delayed_additions']
    return report


def answer_only_audit(saved, test):
    model = load_model(saved)
    values = [b'A x='+bytes([v])+b';'+test[:8]+b'A x?' for v in b'0123']
    result = model(torch.tensor([list(x) for x in values]))
    loss = F.cross_entropy(result['logits'][:, -1], torch.tensor(list(b'0123')))
    loss.backward()
    named = list(model.named_parameters())
    groups = parameter_groups(model)
    return dict(loss=float(loss.detach()), objective='Only four final answer cross-entropies; no syntax or auxiliary loss',
        groups={name: dict(norm=norm(named[i][1].grad for i in indices), tensors=len(indices),
            missing=[named[i][0] for i in indices if named[i][1].grad is None],
            zero=[named[i][0] for i in indices if named[i][1].grad is not None and not bool(named[i][1].grad.ne(0).any())])
            for name, indices in groups.items()})


def source_derivative(saved, test):
    """Differentiate to one occurrence, not to a reused embedding parameter.

    FP64 and frozen weights keep the diagnostic finite and avoid accumulating
    parameter gradients. Hard choices still use the model's ST derivative.
    """
    docs = make_documents(TaskStreamConfig.from_dict(saved['data_config']), split='long_delay',
        seed=saved['protocol']['data_seed'], count=8, distractor_bytes=test)
    pair = counterfactual_pairs(docs)[0]
    seq, q = pair['original'], pair['query']
    records = []
    for truncate in (0, 16):
        model = load_model(saved, torch.float64)
        model.requires_grad_(False)
        state, leaf, handle = None, None, None
        with torch.no_grad():
            if q['source_position']:
                state = model(torch.tensor([list(seq[:q['source_position']])]))['state']
        def inject(module, args, out):
            nonlocal leaf
            leaf = out.detach().clone().requires_grad_(True)
            return leaf
        handle = model.hat.byte_embedding.register_forward_hook(inject)
        for position in range(q['source_position'], len(seq)):
            if truncate and position % truncate == 0 and state is not None:
                state = state.detach()
            result = model(torch.tensor([[seq[position]]]), state=state)
            state = result['state']
            if handle is not None:
                handle.remove(); handle = None
        logits = result['logits'][0, -1]
        loss = F.cross_entropy(logits[None], torch.tensor([q['answer']]))
        g = torch.autograd.grad(loss, leaf, allow_unused=True)[0] if loss.requires_grad else None
        records.append(dict(truncate=truncate, source_position=q['source_position'],
            query_position=q['position'], delay=q['delay'], nll=float(loss.detach()),
            source_gradient_norm=norm([g]), source_gradient_connected=g is not None,
            logits=logits.detach().tolist(), finite_gradient=g is None or bool(g.isfinite().all())))
        print(json.dumps(dict(phase='derivative', truncate=truncate,
            gradient=records[-1]['source_gradient_norm'])), flush=True)
        del model, result, state, loss, logits, leaf
    return dict(records=records, forward_maximum_difference=max(abs(a-b) for a,b in
        zip(records[0]['logits'], records[1]['logits'])),
        note='Surrogate ST derivative; no historical credit is added to an observation occurrence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/state_credit/state/best.pt')
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/state_credit')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/headcheck')
    p.add_argument('--phase', choices=('archive', 'forward', 'telemetry', 'gradient', 'answer', 'derivative'), required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    raw = a.data.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == saved['raw_sha256'], 'Corpus mismatch'
    _, _, test = split_bytes(raw)
    start = time.perf_counter()
    if a.phase == 'archive':
        data = archived(a.root)
    elif a.phase == 'forward':
        data = forward_audit(saved, test)
    elif a.phase == 'telemetry':
        data = telemetry_audit(saved, test)
    elif a.phase == 'gradient':
        data = gradient_audit(saved, a.checkpoint, raw)
    elif a.phase == 'answer':
        data = answer_only_audit(saved, test)
    else:
        data = source_derivative(saved, test)
    output = dict(phase=a.phase, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, seconds=time.perf_counter()-start, data=data)
    a.out.mkdir(exist_ok=True, parents=True)
    (a.out/(a.phase+'.json')).write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(phase=a.phase, seconds=output['seconds'], saved=str(a.out/(a.phase+'.json')))), flush=True)


if __name__ == '__main__':
    main()
