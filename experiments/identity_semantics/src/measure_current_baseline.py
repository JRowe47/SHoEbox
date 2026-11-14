"""Action 1: isolated, reproducible comparison of the four production profiles.

Run from any directory. Each measurement uses a fresh child process. Native API
calls (including logits, worker quiescence and document drains) alone are timed;
warmup has its own clock. Evaluation and memory reads run between timed calls.
No timing hooks or counters are inserted into the native training loop.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PROFILES = ('reference', 'event', 'compact_serial', 'compact_threaded')
WORKLOADS = ('repeated', 'natural', 'documents')
TRAIN = b'The river runs through the valley. The king crossed the bridge. '
VALIDATION = b'The bird crossed the valley. The queen walked through the gate. '
PIN = '70ea7b0b4f203c94aabcaaae58a06e213f698031'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def source_manifest():
    files = {}
    for p in sorted((ROOT / 'runtime').rglob('*')):
        if p.suffix in ('.py', '.cpp', '.inc', '.json', '.npz'):
            files[str(p.relative_to(REPO))] = sha(p.read_bytes())
    return dict(revision=git('rev-parse', 'HEAD').decode().strip(),
                runtime_sha256=sha(json.dumps(files, sort_keys=True).encode()),
                files=files, driver_sha256=sha(Path(__file__).read_bytes()))


def fixtures(workload, targets, source_revision=PIN, validation_bytes=1024):
    """Return explicit documents: no target ever crosses a document boundary."""
    if workload == 'repeated':
        raw = (TRAIN * ((targets + 1) // len(TRAIN) + 1))[:targets + 1]
        docs = [raw]
        val = (VALIDATION * ((validation_bytes + 1) // len(VALIDATION) + 1))[:validation_bytes + 1]
        sources = ['embedded legacy repeated fixture', 'embedded distinct validation fixture']
    else:
        raw = b'\n\n'.join(git('show', f'{source_revision}:{name}')
                           for name in ('README.md', 'STATUS.md'))
        val = git('show', f'{source_revision}:docs/cortical_fineweb.md')[:validation_bytes + 1]
        sources = [f'{source_revision}:README.md', f'{source_revision}:STATUS.md',
                   f'{source_revision}:docs/cortical_fineweb.md (held out)']
        if workload == 'natural':
            docs = [raw[:targets + 1]]
        else:
            docs, pos, remaining = [], 0, targets
            while remaining:
                count = min(512, remaining)
                docs.append(raw[pos:pos + count + 1])
                pos += count + 1
                remaining -= count
    if sum(len(d) - 1 for d in docs) != targets or len(val) != validation_bytes + 1:
        raise ValueError('Insufficient fixed fixture data; do not silently repeat natural text')
    return docs, val, dict(sources=sources, documents=len(docs),
                          document_lengths=[len(d) for d in docs],
                          training_sha256=sha(b''.join(docs)), validation_sha256=sha(val))


def segments(docs, chunk, cuts=()):
    """(absolute target offset, x bytes, y bytes, document end)."""
    offset = 0
    for doc in docs:
        lo = 0
        while lo < len(doc) - 1:
            hi = min(lo + chunk, len(doc) - 1)
            for cut in cuts:
                if offset + lo < cut < offset + hi:
                    hi = cut - offset
            yield offset + lo, doc[lo:hi], doc[lo + 1:hi + 1], hi == len(doc) - 1
            lo = hi
        offset += len(doc) - 1


def rss_kib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    return None


def inventory(model):
    from memory_inspection import read_memory_index
    live = read_memory_index(model)
    for name, row in live.items():
        if row['invalid_retained_spans'] or row['missing_content_records']:
            raise AssertionError(f'Invalid live records in {name}: {row}')
    pipeline = model.inspect_pipeline()
    for key in ('banks', 'building_buffer', 'pending_buffer'):
        pipeline.pop(key, None)
    publications = model.publication_storage()
    publications = {k: v for k, v in publications.items() if not k.endswith(('_hits', '_misses'))}
    return dict(rss_kib=rss_kib(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                completion=model.completion_storage(), publications=publications,
                pipeline=pipeline, live=live)


def make_model(a):
    from runtime import fresh_model
    profile = 'pipeline' if a.profile.startswith('compact_') else a.profile
    m = fresh_model(seed=a.seed, profile=profile)
    if profile == 'pipeline':
        m.configure_pipeline_workers(2 if a.profile == 'compact_threaded' else 1,
                                     cpu=a.worker_cpu if a.profile == 'compact_threaded' else None)
    os.sched_setaffinity(0, {a.cpu})
    if m.capabilities()['inline_diagnostics']:
        raise RuntimeError('Action 1 measures the production build without inline diagnostics')
    return m


def worker(a):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'runtime'))
    from runtime import core
    from memory_inspection import capture_memory, memory_fingerprint
    docs, val, data = fixtures(a.workload, a.warmup + a.targets, a.data_revision, a.validation_bytes)
    validation = np.frombuffer(val, np.uint8)
    m = make_model(a)
    try:
        capabilities = m.capabilities()
        parameter_count = m.parameter_count
        lib = Path(m.lib._name)
        weights = dict(seed=a.seed, config=m.config, lace=m.lace, state_config=m.state_config,
                       probe=m.probe, utility=m.utility, event=m.event, reverse=m.reverse,
                       lr=a.lr, local_lr=a.lr / 3, route_lr=.002)
        def heldout():
            v = m.evaluation_copy(memory='cold_frozen')
            try:
                return float(v.run(validation[:-1], validation[1:], mode='eval'))
            finally:
                v.close()
        def timed(x, y, end):
            c0, t0 = time.process_time(), time.perf_counter()
            loss, logits = m.run(np.frombuffer(x, np.uint8), np.frombuffer(y, np.uint8),
                                 mode='periodic', lr=a.lr, lp=a.lr / 3, route=.002, logits=True)
            if a.workload == 'documents' and end:
                m.finish_document(mode='periodic', lr=a.lr, lp=a.lr / 3, route=.002, reset=True)
            elapsed, cpu = time.perf_counter() - t0, time.process_time() - c0
            return float(loss), logits, elapsed, cpu
        # Align the warmup boundary and quality/memory checkpoints exactly.
        checkpoints = sorted({min(n, a.targets) for n in (2048, 4096, 8192, 16384, 32768, a.targets)})
        cuts = [a.warmup] + [a.warmup + n for n in checkpoints]
        warm_wall = warm_cpu = seconds = cpu_seconds = weighted = 0.
        points, memory, chunks = [], [], []
        logit_hash = hashlib.sha256()
        start_phase = True
        for lo, x, y, end in segments(docs, a.chunk, cuts):
            if lo == a.warmup and start_phase:
                memory.append(dict(targets=0, inventory=inventory(m)))
                points.append(dict(targets=0, total_training_bytes=a.warmup,
                                   training_seconds=warm_wall, measured_seconds=0., heldout_nll=heldout()))
                start_phase = False
            loss, logits, dt, dc = timed(x, y, end)
            if lo < a.warmup:
                warm_wall += dt
                warm_cpu += dc
                continue
            seconds += dt
            cpu_seconds += dc
            weighted += loss * len(x)
            logit_hash.update(logits.tobytes())
            trained = lo + len(x) - a.warmup
            chunks.append(dict(targets=trained, seconds=dt, cpu_seconds=dc, train_nll=loss))
            if trained in checkpoints:
                # Read RSS before constructing the evaluation copy or full capture.
                memory.append(dict(targets=trained, inventory=inventory(m)))
                points.append(dict(targets=trained, total_training_bytes=a.warmup + trained,
                                   training_seconds=warm_wall + seconds, measured_seconds=seconds,
                                   heldout_nll=heldout()))
        if not chunks or chunks[-1]['targets'] != a.targets:
            raise AssertionError('No measured chunks')
        training_rss = memory[-1]['inventory']['peak_rss_kib']
        snapshot = capture_memory(m)
        row = dict(profile=a.profile, workload=a.workload, experts_per_region=core.E,
                   width=64, logical_regions=18, model_label='1M' if core.E == 12 else '5M',
                   fitted_parameter_count=parameter_count, targets=a.targets, warmup=a.warmup,
                   api_chunk_targets=a.chunk, logits=True, reset_at_document_end=a.workload == 'documents',
                   warmup_seconds=warm_wall, warmup_cpu_seconds=warm_cpu,
                   seconds=seconds, process_cpu_seconds=cpu_seconds,
                   bytes_per_second=a.targets / seconds, train_nll=weighted / a.targets,
                   heldout_nll=points[-1]['heldout_nll'], quality_curve=points,
                   memory_curve=memory, chunks=chunks, data=data, policy=weights,
                   capabilities=capabilities, caller_cpu=a.cpu,
                   worker_cpu=a.worker_cpu if a.profile == 'compact_threaded' else None,
                   training_peak_rss_kib=training_rss,
                   peak_rss_scope='process high-water; includes prior untimed validation copies',
                   logits_sha256=logit_hash.hexdigest(), continuation_sha256=memory_fingerprint(snapshot),
                   native_library=lib.name, native_library_sha256=sha(lib.read_bytes()),
                   state_section_bytes={k: v.nbytes for k, v in snapshot.arrays.items()},
                   integrity=True)
        atomic_json(a.output, row)
    finally:
        m.close()


def summarize(rows):
    result = []
    for workload, targets in sorted({(r['workload'], r['targets']) for r in rows}):
        for profile in PROFILES:
            group = [r for r in rows if (r['workload'], r['targets'], r['profile']) == (workload, targets, profile)]
            if not group:
                continue
            rates = [r['bytes_per_second'] for r in group]
            for field in ('logits_sha256', 'continuation_sha256'):
                if len({r[field] for r in group}) != 1:
                    raise AssertionError(f'Nondeterministic repeats: {workload}/{targets}/{profile}/{field}')
            result.append(dict(workload=workload, targets=targets, profile=profile, repeats=len(group),
                               median_bytes_per_second=statistics.median(rates), min_bytes_per_second=min(rates),
                               max_bytes_per_second=max(rates), cv_percent=100 * statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else None,
                               median_seconds=statistics.median(r['seconds'] for r in group),
                               median_process_cpu_seconds=statistics.median(r['process_cpu_seconds'] for r in group),
                               heldout_nll=group[0]['heldout_nll'], fitted_parameter_count=group[0]['fitted_parameter_count']))
        serial = [r for r in rows if (r['workload'], r['targets'], r['profile']) == (workload, targets, 'compact_serial')]
        threaded = [r for r in rows if (r['workload'], r['targets'], r['profile']) == (workload, targets, 'compact_threaded')]
        if serial and threaded:
            for field in ('logits_sha256', 'continuation_sha256'):
                if serial[0][field] != threaded[0][field]:
                    raise AssertionError(f'Serial/threaded mismatch: {workload}/{targets}/{field}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--profile', choices=PROFILES, default='reference')
    p.add_argument('--workload', choices=WORKLOADS, default='repeated')
    p.add_argument('--targets', type=int, default=16384)
    p.add_argument('--warmup', type=int, default=1024)
    p.add_argument('--chunk', type=int, default=512)
    p.add_argument('--validation-bytes', type=int, default=1024)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--lr', type=float, default=.04)
    p.add_argument('--cpu', type=int, default=0)
    p.add_argument('--worker-cpu', type=int, default=1)
    p.add_argument('--data-revision', default=PIN)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--suite', choices=('short', 'continuation', 'retention'), default='continuation')
    a = p.parse_args()
    if min(a.targets, a.warmup, a.chunk, a.validation_bytes, a.repeats) <= 0:
        p.error('Byte counts, chunk and repeats must be positive')
    if a.cpu == a.worker_cpu or not {a.cpu, a.worker_cpu} <= os.sched_getaffinity(0):
        p.error('Choose two distinct available CPUs')
    if a.worker:
        return worker(a)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'SHOEBOX_INLINE_DIAGNOSTICS': '0', 'SHOEBOX_EXPERTS': os.environ.get('SHOEBOX_EXPERTS', '12')}
    # Explicitly reject surprising build overrides; flags are recorded below.
    if env.get('SHOEBOX_CXX_FLAGS') or env.get('SHOEBOX_KERNEL', 'completion') != 'completion':
        p.error('Use the unmodified production completion build without CXX overrides')
    env['SHOEBOX_KERNEL'] = 'completion'
    targets = 2048 if a.suite == 'short' else 32768 if a.suite == 'retention' else a.targets
    workloads = ('natural',) if a.suite == 'retention' else WORKLOADS
    repeats = 1 if a.suite == 'retention' else a.repeats
    manifest = source_manifest()
    manifest.update(compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                    compiler_flags=['-O3', '-march=native', '-std=c++17', '-pthread',
                                    '-DSHOEBOX_EXPERTS=' + env['SHOEBOX_EXPERTS'], '-shared', '-fPIC'],
                    platform=platform.platform(), python=sys.version,
                    cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                    affinity=sorted(os.sched_getaffinity(0)), environment={k: env[k] for k in env if k.startswith(('SHOEBOX_', 'OMP_', 'OPENBLAS_', 'MKL_'))})
    result = dict(protocol=__doc__, suite=a.suite, manifest=manifest, runs=[], summary=[])
    with tempfile.TemporaryDirectory(prefix='shoebox-baseline-') as tmp:
        out = Path(tmp) / 'row.json'
        for workload in workloads:
            for repeat in range(repeats):
                order = PROFILES[repeat % 4:] + PROFILES[:repeat % 4]
                for position, profile in enumerate(order):
                    cmd = [sys.executable, __file__, '--worker', '--output', str(out), '--profile', profile,
                           '--workload', workload, '--targets', str(targets), '--warmup', str(a.warmup),
                           '--chunk', str(a.chunk), '--validation-bytes', str(a.validation_bytes),
                           '--seed', str(a.seed), '--lr', str(a.lr), '--cpu', str(a.cpu),
                           '--worker-cpu', str(a.worker_cpu), '--data-revision', a.data_revision]
                    subprocess.run(cmd, env=env, check=True)
                    row = json.loads(out.read_text())
                    row.update(repeat=repeat, order_position=position)
                    result['runs'].append(row)
                    result['summary'] = summarize(result['runs'])
                    atomic_json(a.output, result)
                    print(f'{a.suite} {workload} {repeat + 1}/{repeats} {profile}: '
                          f'{row["bytes_per_second"]:.1f} B/s, NLL {row["heldout_nll"]:.6f}', flush=True)
    result['serial_threaded_bitwise_equal'] = True
    result['completed'] = True
    atomic_json(a.output, result)


if __name__ == '__main__':
    main()
