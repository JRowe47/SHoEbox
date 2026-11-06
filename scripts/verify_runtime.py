"""Compare execution backends on a real checkpoint and a short continuation."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.joint_stability import load
from experiments.text_regions import byte_stream, gradient_balance
from experiments.text_regions.data import OrderedByteStreams
from experiments.text_regions.multifact import table_batches
from experiments.text_regions.multifact_data import make_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3)
    a = p.parse_args()
    if a.steps < 1:
        p.error('At least one continuation update is required.')
    torch.set_num_threads(1)
    model, _, _, saved = load(a.checkpoint)
    raw = (b'ROMEO: What light through yonder window breaks?\nJULIET: It is the east.\n')*2048
    chunk = OrderedByteStreams.from_bytes(raw, 8).next_chunk(16)
    snapshots = {}
    for mode in ('reference', 'optimized'):
        model.set_execution_backend(mode)
        model.zero_grad(set_to_none=True)
        r = byte_stream.forward(model, chunk.values, chunk.targets)
        r['loss'].backward()
        snapshots[mode] = dict(logits=r['logits'].detach().clone(), state=r['state'].detach(),
            loss=float(r['loss'].detach()), gradients={n: None if p.grad is None else p.grad.clone()
            for n, p in model.named_parameters()})
        del r
    old, new = snapshots['reference'], snapshots['optimized']
    torch.testing.assert_close(old['logits'], new['logits'], atol=0, rtol=0)
    assert old['loss'] == new['loss']
    assert len(old['state'].tensors()) == len(new['state'].tensors())
    for x, y in zip(old['state'].tensors(), new['state'].tensors()):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    error2 = norm2 = max_error = 0.
    for name, x in old['gradients'].items():
        y = new['gradients'][name]
        if x is None:
            assert y is None, name
            continue
        assert y is not None, name
        diff = x.double()-y.double()
        error2 += float(diff.square().sum()); norm2 += float(x.double().square().sum())
        max_error = max(max_error, float(diff.abs().max()))
    relative = (error2/max(norm2, 1e-30))**.5
    assert relative <= 5e-6, relative
    report = dict(checkpoint=str(a.checkpoint), exact_logits=True, exact_loss=True, exact_state=True,
                  gradient_relative_l2_error=relative, gradient_max_absolute_error=max_error,
                  continuation={})
    del snapshots, old, new, model
    end_weights = {}
    for mode in ('reference', 'optimized'):
        model, opt, _, saved = load(a.checkpoint)
        model.set_execution_backend(mode)
        stream = OrderedByteStreams.from_bytes(raw, 8)
        state = None
        rows = []
        for step in range(a.steps):
            kind = ('recall', 'shared_left', 'shared_right')[step % 3]
            batch = make_batch('train', step*16, 16, 20261107, (step % 3)*4, kind)
            state, row = gradient_balance.update(model, opt, stream.next_chunk(16), state,
                batch, table_batches(batch, 'train', step*16, 20261107), saved['reference'])
            rows.append(dict(text_nats=row['text_nats'], memory_loss=row['memory']['loss'],
                             gradient_norm=row['gradient_norm'], expert_executions=row['expert_executions']))
            print(json.dumps(dict(backend=mode, step=step+1, **rows[-1])), flush=True)
        report['continuation'][mode] = rows
        end_weights[mode] = {n: v.detach().clone() for n, v in model.state_dict().items()}
        del model, opt, state
    for old, new in zip(report['continuation']['reference'], report['continuation']['optimized']):
        assert old['expert_executions'] == new['expert_executions']
        for key in ('text_nats', 'memory_loss'):
            assert abs(old[key]-new[key]) <= 1e-4, (key, old[key], new[key])
    differences = []
    for name, x in end_weights['reference'].items():
        y = end_weights['optimized'][name]
        if x.is_floating_point():
            differences.append(float((x-y).abs().max()))
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4, msg=name)
        else:
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    report['continuation_max_weight_difference'] = max(differences)
    report['passed'] = True
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
