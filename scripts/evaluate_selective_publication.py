"""Evaluate preselected producer checkpoints; never use test data for selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.publication_train import load, evaluate
from experiments.text_regions.composition_train import atomic_save
from calibrate_publication import calibrate


def conditions():
    for gap in (16, 64):
        for kind, control in (('recall', 'carried'), ('correction', 'carried'), ('correction', 'withheld')):
            yield gap, kind, control
    for control in ('reset', 'no_writes', 'absent'):
        yield 16, 'correction', control


def evaluate_run(path, count=512):
    out = path/'evaluation.json'
    records = json.loads(out.read_text()) if out.exists() else {'groups': {}}
    def save():
        temp = out.with_suffix('.tmp'); temp.write_text(json.dumps(records, indent=2)+'\n'); temp.replace(out)
    for checkpoint in ('best', 'last'):
        model, _, saved = load(path/(checkpoint+'.pt')); model.eval()
        policy = saved['protocol']['policy']; seed = saved['protocol']['seed']
        metadata = dict(step=saved['step'], sha256=hashlib.sha256((path/(checkpoint+'.pt')).read_bytes()).hexdigest())
        if checkpoint in records and records[checkpoint] != metadata:
            raise ValueError('Checkpoint changed during evaluation.')
        records[checkpoint] = metadata
        runs = [(policy, *condition) for condition in conditions()] if checkpoint == 'best' else [
            (policy, 16, kind, control) for kind, control in
            (('recall','carried'), ('correction','carried'), ('correction','withheld'))]
        if checkpoint == 'best' and policy == 'learned':
            runs += [('always', 16, kind, control) for kind, control in
                     (('recall','carried'), ('correction','carried'), ('correction','withheld'))]
        for runtime_policy, gap, kind, control in runs:
            key = f'{checkpoint}/{runtime_policy}/{gap}/{kind}/{control}'
            if key in records['groups']:
                if records['groups'][key]['count'] != count: raise ValueError('Evaluation count changed.')
                continue
            score = evaluate(model, runtime_policy, 'test', seed, count, gap, kind, control)
            records['groups'][key] = score; save()
            print(json.dumps({'run': path.name, 'group': key, 'accuracy': score['accuracy'],
                              'precision': score['write_precision'], 'recall': score['write_recall']}), flush=True)
        if checkpoint == 'best':
            atomic_save(path/'inference.pt', {'schema': 'shoebox-publication-inference-v1',
                'config': model.cfg.to_dict(), 'weights': model.state_dict(), 'publication_policy': policy,
                'selected_step': saved['step'], 'parameters': model.parameter_summary(),
                'parent_sha256': saved['protocol']['parent_sha256']})
    if policy == 'learned':
        calibration, calibrated = calibrate(path)
        metadata = dict(offset=calibration['selected']['offset'],
                        source_sha256=calibration['source_sha256'],
                        sha256=hashlib.sha256((path/'calibrated_inference.pt').read_bytes()).hexdigest())
        if 'calibrated' in records and records['calibrated'] != metadata:
            raise ValueError('Calibration changed during evaluation.')
        records['calibrated'] = metadata
        for gap, kind, control in conditions():
            key = f'calibrated/learned/{gap}/{kind}/{control}'
            if key in records['groups']: continue
            score = evaluate(calibrated, 'learned', 'test', seed, count, gap, kind, control)
            records['groups'][key] = score; save()
            print(json.dumps({'run':path.name,'group':key,'accuracy':score['accuracy'],
                              'precision':score['write_precision'],'recall':score['write_recall']}),flush=True)
    records['complete'] = True; save()
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--count', type=int, default=512)
    args = p.parse_args(); torch.set_num_threads(1)
    evaluate_run(args.run, args.count)


if __name__ == '__main__': main()
