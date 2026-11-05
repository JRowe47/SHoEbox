"""Validation-only bias calibration for an isolated, non-feedback producer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.publication_train import load, forward_batch, evaluate
from experiments.text_regions.publication_data import make_batch
from experiments.text_regions.composition_train import atomic_save

OFFSETS = (0., .5, 1., 1.5, 2., 2.5, 3.)
TASKS = (('recall','carried'), ('correction','carried'), ('correction','withheld'))


@torch.no_grad()
def calibrate(path, write=True):
    model, _, saved = load(path/'best.pt'); model.eval()
    assert saved['protocol']['policy'] == 'learned'
    assert tuple(model.nodes) == ('A',) and not model.cfg.edges and model.cfg.readout_mode == 'concat'
    assert model.cfg.rounds == 1 and not model.cfg.recurrent_feedback
    seed = saved['protocol']['seed']; count = saved['protocol']['validation_count']
    caches = []
    for kind, control in TASKS:
        batch = make_batch('validation', 0, count, seed, 16, kind, control)
        caches.append((batch, forward_batch(model, batch, 'learned')))
    choices = []
    for offset in OFFSETS:
        scores = []
        for batch, result in caches:
            payload = torch.zeros_like(result['candidate'][:, 0])
            valid = torch.zeros(count, dtype=torch.bool)
            predicted = result['logit'].ge(offset)
            for t in range(batch.inputs.shape[1]):
                payload = torch.where(batch.reset[:,t,None], 0., payload)
                valid &= ~batch.reset[:,t]
                payload = torch.where(predicted[:,t,None], result['candidate'][:,t], payload)
                valid |= predicted[:,t] & batch.available[:,t]
            logits = model.readout(payload*valid[:,None])
            decision = predicted & batch.available
            tp = int((decision & batch.writes).sum())
            fp = int((decision & ~batch.writes).sum())
            fn = int((~decision & batch.writes).sum())
            scores.append(dict(accuracy=float(logits.argmax(-1).eq(batch.answers).double().mean()),
                nats=float(F.cross_entropy(logits,batch.answers,reduction='sum'))/count,
                write_precision=tp/max(1,tp+fp), write_recall=tp/max(1,tp+fn),
                predictions=logits.argmax(-1).tolist(), targets=batch.answers.tolist()))
        minimum = min(s['accuracy'] for s in scores)
        decision_minimum = min(s[k] for s in scores for k in ('write_precision','write_recall'))
        key = [int(minimum >= .9 and decision_minimum >= .95), minimum, decision_minimum,
               sum(s['accuracy'] for s in scores)/3, -sum(s['nats'] for s in scores)/3, -offset]
        choices.append(dict(offset=offset, key=key, scores=scores))
    selected = max(choices, key=lambda x: x['key'])
    model.nodes['A'].publication_gate.bias.sub_(selected['offset'])
    checked = [evaluate(model, 'learned', 'validation', seed, count, 16, kind, control)
               for kind, control in TASKS]
    for cached, actual in zip(selected['scores'], checked):
        for field in ('accuracy','nats','write_precision','write_recall','predictions','targets'):
            assert cached[field] == actual[field], (field,cached[field],actual[field])
    result = dict(source_sha256=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256((ROOT/'docs/SHoEbox_Publication_Calibration_Protocol.md').read_bytes()).hexdigest(),
                  split='validation', count=count, seed=seed, selected=selected, candidates=choices,
                  actual_engine_checked=True)
    if write:
        (path/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
        atomic_save(path/'calibrated_inference.pt',dict(schema='shoebox-publication-inference-v1',
            config=model.cfg.to_dict(), weights=model.state_dict(), publication_policy='learned',
            gate_bias_offset=selected['offset'], selected_step=saved['step'], source_sha256=result['source_sha256'],
            parameters=model.parameter_summary(), parent_sha256=saved['protocol']['parent_sha256']))
    return result, model


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args = parser.parse_args(); torch.set_num_threads(1)
    result, _ = calibrate(args.run)
    print(json.dumps({'run':args.run.name,'offset':result['selected']['offset'],'validation_key':result['selected']['key']}))


if __name__ == '__main__': main()
