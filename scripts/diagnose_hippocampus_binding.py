"""Read-only causal audit of cues, selected memory and expert reinstatement.

Task metadata is used only after inference to summarize measurements. Hooks do
not change outputs, route candidates, or install diagnostic readouts in the model.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.data import split_bytes
from experiments.text_regions.paired_recall import paired_batch, paired_forward
from scripts.run_memory_selectivity import write_json


def rms(x):
    return float(x.square().mean().sqrt())


@torch.no_grad()
def audit(saved, raw, gap=0, count=2, seed=881, start=4000):
    cfg = GraphConfig.from_dict(saved['config'])
    data = TaskStreamConfig.from_dict(saved['data_config'])
    model = RegionLanguageModel(cfg)
    model.load_state_dict(saved['weights']); model.eval()
    h = model.hippocampus
    if h is None:
        raise ValueError('Audit requires enabled hippocampus.')
    _, validation, _ = split_bytes(raw)
    batch = paired_batch(data, split='validation', seed=seed, start=start,
                         count=count, gap=gap, distractor_bytes=validation)
    last = {}; candidate_ages = []; records_per_query = []
    query = h.query
    candidates = h._candidates
    def traced_candidates(row, request, code, cells, budget):
        result = candidates(row, request, code, cells, budget)
        candidate_ages.extend(request.watermark - x.created for x in result)
        records_per_query.append(len(result))
        return result
    h._candidates = traced_candidates
    def traced_query(state, keys, *args, **kwargs):
        candidate_ages.clear(); records_per_query.clear()
        value, receipts, work = query(state, keys, *args, **kwargs)
        last.update(keys=keys, residual=value, receipts=receipts, work=work,
                    candidate_ages=list(candidate_ages), counts=list(records_per_query), state=state)
        return value, receipts, work
    h.query = traced_query
    handles = []
    for name in model.memory_requesters.values():
        def hook(module, args, name=name):
            last.setdefault('payload', {})[name] = args[0]
        handles.append(model.nodes[name].register_forward_pre_hook(hook))
    def compare_hook(module, args, output):
        last['comparison'] = output.softmax(-1)
    handles.append(h.comparison.register_forward_hook(compare_hook))
    result = paired_forward(model, batch)
    scores = result['logits'][:, -1]
    predictions = scores.argmax(-1).reshape(-1, 2)
    correct = predictions.eq(batch['answers'].reshape(-1, 2))
    conflicting = batch['answers'].reshape(-1, 2).diff(dim=1).squeeze(1).ne(0)
    key = last['keys'].reshape(-1, 2, len(h.regions), cfg.hippocampus.address_width)
    delta = key[:, 0] - key[:, 1]
    residual = last['residual'].reshape(-1, 2, len(h.regions), cfg.width)
    selections = []
    selection_ages = []
    for row, receipts in zip(last['state'].rows, last['receipts']):
        ids = set()
        for receipt in receipts:
            for binding, handle in receipt.contributors:
                ids.add(handle)
                selection_ages.append(row.round - row.anchor(handle, row.round).ready_round)
        selections.append(ids)
    overlaps = [len(a & b) / max(1, len(a | b)) for a, b in zip(selections[::2], selections[1::2])]
    widths = {n: rms(x) for n, x in last['payload'].items()}
    codes, _ = h.separate(last['keys'].reshape(-1, cfg.hippocampus.address_width))
    code_overlap = (codes.reshape(-1, 2, len(h.regions), cfg.hippocampus.dg_cells).prod(1).sum(-1)
                    / cfg.hippocampus.dg_active)
    metrics = dict(gap=gap, templates=count, seed=seed, start=start,
        histories=len(predictions), conflicting_pairs=int(conflicting.sum()),
        both_correct=int(correct[conflicting].all(-1).sum()),
        query_switches=int(predictions[conflicting].diff(dim=1).ne(0).sum()),
        nll=float(F.cross_entropy(scores, batch['answers'])),
        cue_rms=rms(key), cue_query_delta_rms=rms(delta),
        cue_query_cosine=float((key[:,0]*key[:,1]).sum(-1).mean()),
        dg_query_overlap=float(code_overlap.mean()),
        residual_rms=rms(residual), residual_query_delta_rms=rms(residual[:,0]-residual[:,1]),
        payload_rms=widths, selected_jaccard=float(sum(overlaps)/len(overlaps)),
        candidate_age_mean=sum(last['candidate_ages'])/max(1,len(last['candidate_ages'])),
        candidate_age_max=max(last['candidate_ages'], default=0),
        selected_age_mean=sum(selection_ages)/max(1,len(selection_ages)),
        selected_age_max=max(selection_ages, default=0),
        final_round=last['state'].rows[0].round, source_byte_positions=batch['sources'][0],
        candidates_mean=sum(last['counts'])/len(last['counts']),
        ca1_mean=last['comparison'].mean(0).tolist(),
        memory_anchors=[sum(map(len,row.anchors.values())) for row in last['state'].rows[:2]])
    for hook in handles: hook.remove()
    h.query = query; h._candidates = candidates
    model.cfg = replace(cfg, hippocampus=replace(cfg.hippocampus, control='blocked'))
    h.config = model.cfg.hippocampus
    blocked = paired_forward(model, batch)['logits'][:, -1]
    metrics['blocked_logit_delta_rms'] = rms(scores-blocked)
    metrics['blocked_nll'] = float(F.cross_entropy(blocked, batch['answers']))
    metrics['blocked_prediction_changes'] = int(scores.argmax(-1).ne(blocked.argmax(-1)).sum())
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gaps', type=int, nargs='+', default=[0,32])
    p.add_argument('--count', type=int, default=2)
    a = p.parse_args(); torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    raw = (ROOT/'data/tiny_shakespeare.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']: raise ValueError('Corpus mismatch.')
    result = dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), measurements=[])
    for gap in a.gaps:
        row = audit(saved, raw, gap=gap, count=a.count)
        result['measurements'].append(row); a.out.parent.mkdir(parents=True,exist_ok=True)
        write_json(a.out,result); print(json.dumps(row), flush=True)

if __name__ == '__main__': main()
