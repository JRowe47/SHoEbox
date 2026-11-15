"""Continue a saved model on fresh, disjoint data without restarting its rates."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shoe_multiscale import Model
from experiments.scale_100m.train import atomic, DocumentChunks, file_hash, frozen_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous-run', type=Path, required=True)
    p.add_argument('--previous-data', type=Path, nargs='+', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Checkpoint and stop after this many additional targets, for recovery checks.')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((a.previous_run/'results.json').read_text())
    origin = a.previous_run/'latest.shoe'
    assert prior.get('completed_utc'), 'The preceding phase must be complete'
    manifest = json.loads((a.data/'manifest.json').read_text())
    old_manifests = [json.loads((d/'manifest.json').read_text()) for d in a.previous_data]
    data_info, dev_info = manifest['files']['train'], manifest['files']['development']
    assert dev_info['sha256'] == prior['development_data_sha256']
    for info in (data_info, dev_info):
        assert file_hash(a.data/info['file']) == info['sha256']
        assert sum(d['targets'] for d in info['documents']) == info['targets']
    previous_docs = [d for m in old_manifests for split in m['files'].values() for d in split['documents']]
    docs, dev_docs = data_info['documents'], dev_info['documents']
    assert not {d['id'] for d in docs} & {d['id'] for d in previous_docs+dev_docs}
    assert not {d['text_sha256'] for d in docs} & {d['text_sha256'] for d in previous_docs+dev_docs}
    chunks = DocumentChunks(docs, a.seed)
    extra_budget = int(data_info['targets']); assert chunks.targets == extra_budget
    segment_hash = hashlib.sha256(chunks.segments.tobytes()).hexdigest()
    data = np.memmap(a.data/data_info['file'], mode='r', dtype=np.uint8)
    dev = np.memmap(a.data/dev_info['file'], mode='r', dtype=np.uint8)
    start_targets = int(prior['checkpoint_targets']); total_budget = start_targets+extra_budget
    started = time.perf_counter()
    source_hashes = {name:file_hash(ROOT/name) for name in prior['source_files']}
    assert source_hashes == prior['source_files'], 'This continuation requires the identical native runtime'

    if a.resume:
        r = json.loads((a.out/'results.json').read_text())
        model = Model.load(a.out/'latest.shoe')
        done = int(model.routing_stats()['levels'][0]['events'])-start_targets
        assert start_targets+done == r['checkpoint_targets']
        assert r['segment_sha256'] == segment_hash and r['seed'] == a.seed
        assert r['train_data_sha256'] == data_info['sha256']
        assert r['start_targets'] == start_targets and r['target_budget'] == total_budget
        assert model.stats()['learning_rates'] == r['fixed_learning_rates']
    else:
        assert not (a.out/'results.json').exists(), 'Use --resume for an existing run'
        assert file_hash(origin) == prior['checkpoint_sha256']
        model = Model.load(origin)
        assert int(model.routing_stats()['levels'][0]['events']) == start_targets
        assert model.parameter_count == prior['actual_parameters']
        done = 0
        r = dict(started_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=prior['source_commit'], source_files=source_hashes,
            source_has_local_scaling_patch=True, seed=a.seed,
            requested_parameters=prior['requested_parameters'], actual_parameters=model.parameter_count,
            experts=model.stats()['experts'], start_targets=start_targets,
            additional_target_budget=extra_budget, target_budget=total_budget,
            parent_checkpoint_sha256=prior['checkpoint_sha256'],
            parent_run_results_sha256=file_hash(a.previous_run/'results.json'),
            previous_training_data_sha256=[m['files']['train']['sha256'] for m in old_manifests],
            train_data_sha256=data_info['sha256'], development_data_sha256=dev_info['sha256'],
            train_documents=len(docs), previous_train_documents=sum(len(m['files']['train']['documents']) for m in old_manifests),
            development_documents=len(dev_docs), development_targets=dev_info['targets'],
            segment_sha256=segment_hash, chunk_targets=4096,
            schedule='Continue all saved rates unchanged; no rewarmup, rate restart, or optimizer reset.',
            fixed_learning_rates=model.stats()['learning_rates'],
            train_seconds=0., eval_seconds=0., checkpoint_seconds=0., wall_seconds=0.,
            previous_train_seconds=prior.get('cumulative_train_seconds',prior['train_seconds']),
            previous_wall_seconds=prior.get('cumulative_wall_seconds',prior['wall_seconds']),
            checkpoint_targets=start_targets, metrics=prior['metrics'].copy(),
            phase_start_metric_index=len(prior['metrics'])-1,
            environment=prior.get('environment', {}))
    previous_wall = r['wall_seconds']
    phase_limit = extra_budget if a.stop_after is None else min(extra_budget,a.stop_after)
    assert 0 <= done <= phase_limit <= extra_budget
    weighted_loss, window_n = 0., 0
    eval_points = set(range(2_000_000,extra_budget+1,2_000_000)) | {extra_budget}

    def record_point():
        nonlocal weighted_loss, window_n
        row = dict(targets=start_targets+done, additional_targets=done,
            train_seconds=r['previous_train_seconds']+r['train_seconds'],
            phase_train_seconds=r['train_seconds'], train_nll=weighted_loss/window_n if window_n else None,
            window_targets=window_n, development_nll=None, learning_rates=model.stats()['learning_rates'])
        if done in eval_points:
            t = time.perf_counter()
            row['development_nll'], per_doc = frozen_score(model,dev,dev_docs)
            r['eval_seconds'] += time.perf_counter()-t
            if done == extra_budget:
                atomic(a.out/'development_per_document.json',per_doc)
        row['routing'] = model.routing_stats()
        row['phase_wall_seconds'] = previous_wall+time.perf_counter()-started
        row['wall_seconds'] = r['previous_wall_seconds']+row['phase_wall_seconds']
        r['metrics'].append(row)
        r['wall_seconds'] = row['phase_wall_seconds']
        atomic(a.out/'progress.json',r)
        print(json.dumps({k:row[k] for k in ('targets','additional_targets','phase_train_seconds','train_nll','development_nll','phase_wall_seconds')}),flush=True)
        weighted_loss, window_n = 0., 0

    def checkpoint():
        t = time.perf_counter(); model.save(a.out/'latest.shoe')
        r['checkpoint_seconds'] += time.perf_counter()-t
        r['checkpoint_targets'] = start_targets+done
        r['wall_seconds'] = previous_wall+time.perf_counter()-started
        atomic(a.out/'results.json',r)

    try:
        if not a.resume:
            t = time.perf_counter(); loss,_ = frozen_score(model,dev,dev_docs)
            r['eval_seconds'] += time.perf_counter()-t
            assert loss == prior['metrics'][-1]['development_nll'], 'Parent evaluation must reproduce exactly'
            r['verified_initial_development_nll'] = loss
            r['parent_reload_exact'] = True
            os.link(origin,a.out/'latest.shoe')
            r['wall_seconds'] = time.perf_counter()-started
            atomic(a.out/'results.json',r)
            print(json.dumps(dict(parent_verified=True,targets=start_targets,development_nll=loss,
                rates=r['fixed_learning_rates'])),flush=True)
        while done < phase_limit:
            next_log = min(phase_limit,(done//250_000+1)*250_000)
            begin,end,reset = chunks.take(done,next_log-done); size=end-begin
            t = time.perf_counter()
            if reset:
                model.reset_context(clear_memory=True)
            loss = model.run(data[begin:end],data[begin+1:end+1])
            r['train_seconds'] += time.perf_counter()-t
            if not math.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at {start_targets+done}')
            weighted_loss += loss*size; window_n += size; done += size
            if done == next_log:
                assert int(model.routing_stats()['levels'][0]['events']) == start_targets+done
                assert model.stats()['learning_rates'] == r['fixed_learning_rates']
                record_point()
                if done%1_000_000 == 0 or done == phase_limit:
                    checkpoint()
        r['final_stats'] = model.stats(); r['final_routing'] = model.routing_stats()
        r['checkpoint_sha256'] = file_hash(a.out/'latest.shoe')
        r['checkpoint_bytes'] = (a.out/'latest.shoe').stat().st_size
        r['wall_seconds'] = previous_wall+time.perf_counter()-started
        r['cumulative_train_seconds'] = r['previous_train_seconds']+r['train_seconds']
        r['cumulative_wall_seconds'] = r['previous_wall_seconds']+r['wall_seconds']
        r['training_bytes_per_second'] = done/r['train_seconds'] if r['train_seconds'] else None
        r['additional_targets_completed'] = done
        if done == extra_budget:
            r['completed_utc'] = datetime.now(timezone.utc).isoformat()
        atomic(a.out/'results.json',r)
        print(json.dumps(dict(completed=done==extra_budget,targets=start_targets+done,
            added_targets=done,training_bytes_per_second=r['training_bytes_per_second'])),flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()
