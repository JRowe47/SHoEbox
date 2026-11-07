"""Verify exact observation credit across the former 16-byte cutoff."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.task_stream import TaskStreamConfig, TaskChunk, make_documents
from experiments.text_regions.data import split_bytes
from experiments.text_regions.state_credit import make_state_graph_eligibility, state_window_forward
from experiments.text_regions.content_credit import content_window_forward
from scripts.evaluate_state_credit import counterfactual_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/state_credit/state/best.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/content_credit/long_probe.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    raw = a.data.read_bytes(); assert hashlib.sha256(raw).hexdigest() == saved['raw_sha256']
    _, _, test = split_bytes(raw)
    docs = make_documents(TaskStreamConfig.from_dict(saved['data_config']), split='long_delay',
        seed=saved['protocol']['data_seed'], count=8, distractor_bytes=test)
    pair = counterfactual_pairs(docs)[0]; seq, query = pair['original'], pair['query']
    # Pad after the queried position only, for the old trainer's divisor check.
    # All modes execute identical bytes; no padded position contributes a loss.
    seq += b' '*((-len(seq)) % 16)
    values = torch.tensor([list(seq)]); absent = torch.full_like(values, -100)
    reset = torch.zeros_like(values, dtype=torch.bool); reset[:, 0] = True
    chunk = TaskChunk(values, absent, absent, absent.ne(-100), reset, absent, absent)
    rows = []; started = time.perf_counter()
    for mode in ('full_reference','repaired','cut_16'):
        model = RegionLanguageModel(GraphConfig.from_dict(saved['config'])).double()
        model.load_state_dict(saved['weights']); model.eval(); model.requires_grad_(False)
        position, leaf = 0, None
        def inject(module, args, output):
            nonlocal position, leaf
            if position == query['source_position']:
                leaf = output.detach().clone().requires_grad_(True); output = leaf
            position += 1
            return output
        hook = model.hat.byte_embedding.register_forward_hook(inject)
        state, credit = model.initial_state(1), make_state_graph_eligibility(model, 1)
        if mode == 'cut_16':
            result, _, boundaries = state_window_forward(model, chunk, state, credit, 16)
        else:
            result, _, boundaries = content_window_forward(model, chunk, state, credit,
                len(seq) if mode == 'full_reference' else 16)
        logits = result['logits'][:, query['position']]
        loss = F.cross_entropy(logits, torch.tensor([query['answer']]))
        gradient = torch.autograd.grad(loss, leaf, allow_unused=True)[0]
        rows.append(dict(mode=mode, source_gradient_norm=float(gradient.norm()) if gradient is not None else 0.,
            source_gradient=gradient.flatten().tolist() if gradient is not None else None,
            logits=logits.detach().flatten().tolist(), nll=float(loss.detach()), boundary_receipts=len(boundaries)))
        hook.remove()
        print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('source_gradient','logits')}), flush=True)
        del model, state, credit, result, boundaries, leaf, logits, loss, gradient
        gc.collect()
    a0, b = rows[0], rows[1]
    error = max(abs(x-y) for x,y in zip(a0['source_gradient'], b['source_gradient']))
    forward = max(abs(x-y) for row in rows[1:] for x,y in zip(a0['logits'], row['logits']))
    assert rows[2]['source_gradient_norm'] == 0 and b['source_gradient_norm'] > 0
    assert error < 1e-10 and forward == 0.
    output = dict(delay=query['delay'], source_position=query['source_position'], query_position=query['position'],
        execution_segment=16, probe_horizon=len(seq), seconds=time.perf_counter()-started,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        gradient_maximum_difference=error, forward_maximum_difference=forward, rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    main()
