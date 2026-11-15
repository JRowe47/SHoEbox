"""Train routing D on an exact document-aware FineWeb byte budget."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shoe_multiscale import Model


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class DocumentChunks:
    def __init__(self, documents, seed, size=4096):
        segments = [(start, min(start+size, d['stop']-1), i)
            for i, d in enumerate(documents) for start in range(d['start'], d['stop']-1, size)]
        order = np.random.default_rng(seed).permutation(len(segments))
        self.segments = np.array([segments[i] for i in order], dtype=np.int64)
        self.boundaries = np.concatenate(([0], np.cumsum(self.segments[:, 1]-self.segments[:, 0])))
        self.targets = int(self.boundaries[-1])

    def take(self, done, size):
        i = int(np.searchsorted(self.boundaries, done, side='right')-1)
        start, end, doc = self.segments[i]
        offset = done-int(self.boundaries[i])
        begin = int(start+offset)
        return begin, min(int(end), begin+size), offset == 0


def configure(model, position, total):
    # Same D peak rates; stretch warmup/cosine over the requested run.
    progress = position/total
    scale = max(.01, progress/.01) if progress < .01 else .1+.45*(1+math.cos(math.pi*min(1, (progress-.01)/.99)))
    model.configure(fast_rate=.008*scale, slow_rate=.08*scale, query_rate=.008*scale,
        head_rate=.16*scale, route_rate=.002, auxiliary_weight=.1,
        reconstruction_weight=.05, constituent_credit=1.)


def frozen_score(model, data, documents):
    total, n = 0., 0
    per_document = []
    with model.clone() as retired:
        retired.finish()
        with retired.inference_copy(cold=True, clear_memory=True) as frozen:
            for d in documents:
                frozen.reset_context(clear_memory=True)
                a, b = d['start'], d['stop']
                loss = frozen.run(data[a:b-1], data[a+1:b])
                total += loss*(b-a-1); n += b-a-1
                per_document.append(dict(row_idx=d['row_idx'], targets=b-a-1, nll=loss))
    return total/n, per_document


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parameters', type=int, default=100_000_000)
    p.add_argument('--targets', type=int, default=20_000_000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.data/'manifest.json').read_text())
    arrays = {}
    for name, entry in manifest['files'].items():
        assert file_hash(a.data/entry['file']) == entry['sha256']
        assert sum(d['stop']-d['start']-1 for d in entry['documents']) == entry['targets']
        arrays[name] = np.memmap(a.data/entry['file'], mode='r', dtype=np.uint8)
    docs = manifest['files']['train']['documents']
    dev_docs = manifest['files']['development']['documents']
    assert not {d['id'] for d in docs} & {d['id'] for d in dev_docs}
    assert not {d['text_sha256'] for d in docs} & {d['text_sha256'] for d in dev_docs}
    chunks = DocumentChunks(docs, a.seed)
    assert a.targets == chunks.targets
    segment_hash = hashlib.sha256(chunks.segments.tobytes()).hexdigest()
    data, dev = arrays['train'], arrays['development']
    started = time.perf_counter()
    if a.resume:
        result = json.loads((a.out/'results.json').read_text())
        model = Model.load(a.out/'latest.shoe')
        done = int(model.routing_stats()['levels'][0]['events'])
        assert done == result['checkpoint_targets']
        assert result['segment_sha256'] == segment_hash
        assert result['seed'] == a.seed and result['requested_parameters'] == a.parameters
    else:
        model = Model(parameters=a.parameters, seed=a.seed)
        done = 0
        try:
            source_commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, OSError):
            source_commit = (ROOT/'SOURCE_COMMIT.txt').read_text().strip() if (ROOT/'SOURCE_COMMIT.txt').exists() else 'unversioned; see exact source hashes'
        source = {str(f.relative_to(ROOT)): file_hash(f) for f in
                  [ROOT/'shoe_multiscale.py', *sorted((ROOT/'multiscale').glob('*.py')),
                   *sorted((ROOT/'multiscale').glob('*.hpp')), ROOT/'multiscale/native.cpp']}
        result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=source_commit,
            source_files=source, source_has_local_scaling_patch=True, seed=a.seed,
            requested_parameters=a.parameters, actual_parameters=model.parameter_count,
            experts=model.stats()['experts'], target_budget=a.targets,
            train_data_sha256=manifest['files']['train']['sha256'],
            development_data_sha256=manifest['files']['development']['sha256'],
            train_documents=len(docs), development_documents=len(dev_docs),
            development_targets=manifest['files']['development']['targets'],
            segment_sha256=segment_hash, chunk_targets=4096,
            schedule='200K-byte linear warmup; cosine decay across 20M to 10% of D peak rates; no 5M rate collapse.',
            peak_rates=dict(fast=.008, slow=.08, query=.008, head=.16),
            train_seconds=0., eval_seconds=0., checkpoint_seconds=0., wall_seconds=0.,
            checkpoint_targets=0, metrics=[], environment=dict(python=platform.python_version(),
                platform=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0))))
    previous_wall = result['wall_seconds']
    eval_points = {0, 250_000, 1_000_000, *range(2_000_000, a.targets+1, 2_000_000), a.targets}
    log_step = 250_000
    weighted_loss, window_n = 0., 0

    def record_point():
        nonlocal weighted_loss, window_n
        row = dict(targets=done, train_seconds=result['train_seconds'],
            train_nll=weighted_loss/window_n if window_n else None,
            window_targets=window_n, development_nll=None,
            learning_rates=model.stats()['learning_rates'])
        if done in eval_points:
            t = time.perf_counter()
            row['development_nll'], per_doc = frozen_score(model, dev, dev_docs)
            result['eval_seconds'] += time.perf_counter()-t
            if done == a.targets:
                atomic(a.out/'development_per_document.json', per_doc)
        row['routing'] = model.routing_stats()
        row['wall_seconds'] = previous_wall + time.perf_counter()-started
        result['metrics'].append(row)
        result['wall_seconds'] = row['wall_seconds']
        atomic(a.out/'progress.json', result)
        print(json.dumps({k: row[k] for k in ('targets', 'train_seconds', 'train_nll', 'development_nll', 'wall_seconds')}), flush=True)
        weighted_loss, window_n = 0., 0

    try:
        if not done:
            record_point()
        while done < a.targets:
            next_log = min(a.targets, (done//log_step+1)*log_step)
            start, end, reset = chunks.take(done, next_log-done)
            size = end-start
            t = time.perf_counter()
            if reset:
                model.reset_context(clear_memory=True)
            configure(model, done+size/2, a.targets)
            loss = model.run(data[start:end], data[start+1:end+1])
            result['train_seconds'] += time.perf_counter()-t
            if not math.isfinite(loss):
                raise RuntimeError(f'nonfinite training loss at {done}')
            weighted_loss += loss*size; window_n += size; done += size
            if done == next_log:
                # stats()['bytes'] is context-local and resets at each chunk.
                # Training-only routing observations persist across resets.
                assert int(model.routing_stats()['levels'][0]['events']) == done
                record_point()
                if done % 1_000_000 == 0 or done == a.targets:
                    t = time.perf_counter()
                    model.save(a.out/'latest.shoe')
                    result['checkpoint_seconds'] += time.perf_counter()-t
                    result['checkpoint_targets'] = done
                    result['wall_seconds'] = previous_wall+time.perf_counter()-started
                    atomic(a.out/'results.json', result)
        result['final_stats'] = model.stats()
        result['final_routing'] = model.routing_stats()
        result['checkpoint_sha256'] = file_hash(a.out/'latest.shoe')
        result['checkpoint_bytes'] = (a.out/'latest.shoe').stat().st_size
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        result['wall_seconds'] = previous_wall+time.perf_counter()-started
        result['training_bytes_per_second'] = a.targets/result['train_seconds']
        atomic(a.out/'results.json', result)
        print(json.dumps(dict(completed=True, targets=done, development_nll=result['metrics'][-1]['development_nll'],
            training_bytes_per_second=result['training_bytes_per_second'])), flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()
