"""Learned, record-encoded positive controls, not resource-matched SHoEbox rivals.

The syntax adapter exposes observed records and query symbols. It never solves
an instruction, matches a query to its record, or reads a scored answer byte.
GRU heads have explicit register addresses; attention learns query/key matching.
"""
import io
import time
import numpy as np
import torch
from torch import nn

from .tasks import COPY, KEY, MAP, SET, SWAP, VALUE, parse


class Lookup(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.compatibility = nn.Parameter(torch.randn(spec.key_pool, spec.key_pool)*.01)
        self.values = nn.Embedding(spec.vocab, 32)
        self.output = nn.Linear(32, spec.vocab)

    def forward(self, inputs, queries, lesion=None):
        keys, values = inputs
        scores = self.compatibility[queries.unsqueeze(-1), keys.unsqueeze(1)]
        if lesion == 'no_query':
            scores = scores*0
        probabilities = scores.softmax(-1)
        memory = self.values(values)
        if lesion == 'zero_state':
            memory = memory*0
        return self.output(probabilities @ memory)


class RegisterGRU(nn.Module):
    def __init__(self, spec, hidden=128):
        super().__init__()
        self.items, self.vocab = spec.items, spec.vocab
        self.recurrent = nn.GRU(4+spec.items*2+spec.vocab, hidden, batch_first=True)
        self.output = nn.Linear(hidden, spec.items*spec.vocab)

    def forward(self, inputs, queries, lesion=None):
        _, state = self.recurrent(inputs)
        if lesion == 'zero_state':
            state = state*0
        values = self.output(state[0]).reshape(-1, self.items, self.vocab)
        if lesion == 'no_query':
            queries = queries*0
        return values.gather(1, queries.unsqueeze(-1).expand(-1, -1, self.vocab))


class PositiveControl:
    def __init__(self, spec, seed=1, lr=None):
        self.spec = spec
        self.kind = 'attention_lookup' if spec.task in ('binding', 'one_hop') else 'state_gated_gru'
        torch.manual_seed(seed)
        self.net = Lookup(spec) if self.kind == 'attention_lookup' else RegisterGRU(spec)
        self.lr = (0.05 if self.kind == 'attention_lookup' else 0.01) if lr is None else lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def encode(self, cases, ignore_operation=False):
        rows = [parse(ep) for ep in cases]
        queries = torch.tensor([q for _, q in rows], dtype=torch.long)
        if self.kind == 'attention_lookup':
            keys = torch.tensor([[k for _, k, _ in records] for records, _ in rows], dtype=torch.long)
            base = KEY if self.spec.task == 'one_hop' else VALUE
            values = torch.tensor([[v-base for _, _, v in records] for records, _ in rows], dtype=torch.long)
            return (keys, values), queries
        dims = 4+2*self.spec.items+self.spec.vocab
        features = torch.zeros(len(rows), max(len(r) for r, _ in rows), dims)
        for i, (records, _) in enumerate(rows):
            if ignore_operation:
                records = records[:self.spec.items]
            for j, (op, key, arg) in enumerate(records):
                features[i, j, [MAP, SET, COPY, SWAP].index(op)] = 1
                features[i, j, 4+key] = 1
                if op in (COPY, SWAP):
                    features[i, j, 4+self.spec.items+arg-KEY] = 1
                else:
                    features[i, j, 4+2*self.spec.items+arg-VALUE] = 1
        return features, queries

    def predict(self, cases, lesion=None):
        self.net.eval()
        with torch.no_grad():
            x, q = self.encode(cases, ignore_operation=lesion == 'ignore_operation')
            return self.net(x, q, lesion).softmax(-1).numpy()

    def update(self, cases):
        self.net.train()
        x, q = self.encode(cases)
        labels = torch.tensor([ep.answer for ep in cases], dtype=torch.long)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.net(x, q)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
        self.optimizer.step()
        return float(loss.detach())

    def inventory(self):
        p = sum(x.numel() for x in self.net.parameters())
        if self.kind == 'attention_lookup':
            state = self.spec.items*(32*4+8)
        else:
            state = self.net.recurrent.hidden_size*4
        return {'parameters': p, 'persistent_state_bytes_per_episode': state,
                'state_scope': 'encoded record/value memory or GRU state; excludes observed input buffer',
                'input_adapter': 'record syntax and observed query symbols; no instruction execution or symbolic lookup',
                'optimizer': 'Adam', 'lr': self.lr,
                'not_a_matched_resource_comparison': True}

    def dumps(self):
        b = io.BytesIO()
        torch.save({'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}, b)
        return b.getvalue()

    def loads(self, payload):
        saved = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
        self.net.load_state_dict(saved['net'])
        self.optimizer.load_state_dict(saved['optimizer'])


def evaluate(control, cases, lesion=None):
    token_correct = exact = total = nll = 0
    changed_correct = changed_total = unchanged_correct = unchanged_total = 0
    for start in range(0, len(cases), 128):
        batch = cases[start:start+128]
        p = control.predict(batch, lesion)
        y = np.array([e.answer for e in batch])
        correct = p.argmax(-1) == y
        token_correct += int(correct.sum())
        exact += int(correct.all(1).sum())
        total += int(y.size)
        nll -= float(np.log(np.maximum(np.take_along_axis(p, y[..., None], -1), 1e-30)).sum())
        for ep, flags in zip(batch, correct):
            if ep.task in ('set', 'copy', 'swap'):
                records, queries = parse(ep)
                initial = {k: v-VALUE for _, k, v in records[:control.spec.items]}
                changed = np.array([initial[k] != value for k, value in zip(queries, ep.answer)])
                changed_correct += int(flags[changed].sum())
                changed_total += int(changed.sum())
                unchanged_correct += int(flags[~changed].sum())
                unchanged_total += int((~changed).sum())
    return {'cases': len(cases), 'token_accuracy': token_correct/total,
            'exact_accuracy': exact/len(cases), 'task_nll': nll/total,
            'changed_accuracy': changed_correct/changed_total if changed_total else None,
            'unchanged_accuracy': unchanged_correct/unchanged_total if unchanged_total else None,
            'native_byte_accuracy': None, 'output_scope': 'task alphabet from a record-encoded neural control'}
