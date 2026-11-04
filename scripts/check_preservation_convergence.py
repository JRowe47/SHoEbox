"""Extend the worst training-gradient fit in each penalty/activation cell.

These four diagnostics are selected by training gradients only. They do not
replace any model in the fixed-budget main comparison.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.emission_adapter import sha
from experiments.text_regions.residual_adapter import (load_residuals, save_residuals,
    select_strength, score_residuals, write_json, write_bytes)
from experiments.text_regions.shared_preservation import fit_preserving
from experiments.text_regions.preservation_compare import audit_objective


def worker(source, out, case):
    torch.set_num_threads(2)
    family, name, run = case['family'], case['head'], case['run']
    bundle = load_residuals(source / run['name'] / (family + '.pt'))
    old = bundle['heads'][name]
    datasets = torch.load(ROOT / run['prior_run'] / 'features.pt', map_location='cpu', weights_only=True)['datasets']
    validation = torch.load(ROOT / run['residual_run'] / 'validation.pt', map_location='cpu', weights_only=True)
    fit = old.correction.fit
    head = fit_preserving(datasets['train'], bundle['base'], penalty=fit['preservation']['penalty'],
        hidden=fit['hidden'], seed=fit['seed'], activation=fit['activation'], l2=fit['l2'], max_iter=1600)
    head = select_strength(head, validation, fit_identities=bundle['fit_identities'])
    audit_objective(datasets['train'], head)
    bundle['heads'][name] = head
    out.mkdir(parents=True)
    save_residuals(out / 'extended.pt', bundle)
    restored = load_residuals(out / 'extended.pt')
    write_json(out / 'selection_lock.json', {'head': name, 'selection': head.selection})
    report = {'case': case, 'prior_fit': fit, 'extended_fit': head.correction.fit,
        'prior_strength': old.strength, 'extended_strength': head.strength,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT / 'experiments/text_regions/shared_preservation.py')},
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
            source / run['name'] / (family + '.pt'), ROOT / run['prior_run'] / 'features.pt',
            ROOT / run['residual_run'] / 'validation.pt')},
        'evaluations': {}}
    rows = []
    for split in ('test', 'independent'):
        scores, predictions = score_residuals(datasets[split], restored)
        report['evaluations'][split] = {'protocol': scores['protocol'], 'groups': [
            {**g, 'decoders': {n: m for n, m in g['decoders'].items() if n.startswith(name + '/')}}
            for g in scores['groups']]}
        rows.extend({**r, 'predictions': {n: p for n, p in r['predictions'].items() if n.startswith(name + '/')}}
            for r in predictions)
    write_bytes(out / 'predictions.jsonl', ''.join(json.dumps(r) + '\n' for r in rows).encode())
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out / 'results.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/shared_preservation_v1'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/preservation_convergence_v1'))
    parser.add_argument('--case', type=Path)
    args = parser.parse_args(); source = (ROOT / args.source).resolve(); out = (ROOT / args.out).resolve()
    if args.case:
        worker(source, out, json.loads(args.case.read_text()))
        return
    if out.exists():
        parser.error('Choose a new convergence output directory.')
    out.mkdir(parents=True)
    status = json.loads((source / 'status.json').read_text())
    reports = {r['name']: json.loads((source / r['name'] / 'results.json').read_text()) for r in status['runs']}
    cases = []
    for family in ('penalty_1', 'penalty_10'):
        for activation in ('identity', 'gelu'):
            options = [(r, n, f) for r in status['runs'] for n, f in reports[r['name']]['fits'][family].items()
                       if n.startswith(activation + '/')]
            r, name, fit = max(options, key=lambda item: item[2]['gradient_max'])
            case = {'family': family, 'activation': activation, 'head': name, 'run': r,
                    'training_gradient_max': fit['gradient_max'], 'max_iter': 1600}
            label = family + '_' + activation; path = out / (label + '.json'); write_json(path, case)
            cases.append((label, path))
    write_json(out / 'plan.json', {'selection': 'largest training gradient per penalty/activation cell',
        'main_study_unchanged': True, 'cases': [json.loads(p.read_text()) for _, p in cases]})

    def execute(pair):
        label, path = pair
        command = [sys.executable, str(Path(__file__)), '--source', str(source),
            '--out', str(out / label), '--case', str(path)]
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(label, result.returncode, flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(execute, cases))
    if any(codes):
        raise RuntimeError('Convergence diagnostic failed.')


if __name__ == '__main__':
    main()
