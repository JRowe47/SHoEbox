"""External thread-directed instruction sampling of the unchanged production SO.

Linux x86-64 fallback for hosts without perf/ptrace. Only a standalone signal
attachment runs inside the sampled process, asynchronously; no model source,
per-byte timers, event counters or training hooks are changed. These diagnostic
runs are NEVER throughput evidence. Percentages are wall-time IP samples,
including waits, not CPU cost or inclusive stack attribution. Thread CPU time
is measured separately. Inlined work is attributed to its containing symbol.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import ctypes as ct
import json
import os
from pathlib import Path
import platform
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time

from measure_current_baseline import ROOT, PROFILES, PIN, fixtures, make_model, atomic_json, sha


def category(name):
    if any(s in name for s in ('collect', 'retention', 'ensure_index', 'archive_slot', 'prune')):
        return 'collection / retention'
    if any(s in name for s in ('derivative', 'adjoint', 'backward', 'union_flush', 'neural_credit', 'teach', 'backprop', 'event_boundary',
                              'constituent_update', 'expert_update', 'accumulate', 'local_prediction_update', 'context_update', 'learn_bottom')):
        return 'reverse / derivatives / credit'
    if any(s in name for s in ('bank_compute', 'pipeline_compute', 'content_bank', 'content_rows', 'content_materialize', 'content_expert_flush', 'pipeline_commit', 'pipeline_capture')):
        return 'bank update / versions'
    if any(s in name for s in ('forward', 'encode', 'publish', 'prepare', 'context_read', 'context_step', 'predict', 'finish_step', 'pattern_classify')):
        return 'forward / prediction / event'
    if any(s in name for s in ('futex', 'cond_wait', 'cond_clockwait', 'pthread_cond', 'syscall', 'nanosleep')):
        return 'wait / syscall'
    if any(s in name for s in ('malloc', 'free', 'memcpy', 'memmove', 'memset', 'operator new', 'operator delete')):
        return 'allocation / copying'
    return 'other / unresolved'


def symbol_table(path):
    result = subprocess.run(['nm', '-anSC', '--defined-only', str(path)], capture_output=True, text=True)
    if not result.stdout.strip():
        result = subprocess.run(['nm', '-DanSC', '--defined-only', str(path)], capture_output=True, text=True)
    entries = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and fields[2].lower() in ('t', 'w'):
            try:
                entries.append((int(fields[0], 16), int(fields[1], 16), fields[3]))
            except ValueError:
                pass
    entries.sort()
    return [v[0] for v in entries], entries


def task_ticks(pid):
    result = {}
    for path in Path(f'/proc/{pid}/task').glob('*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = int(fields[11]) + int(fields[12])
        except FileNotFoundError:
            pass
    return result


def child(a):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'runtime'))
    attachment = ct.CDLL(str(a.attachment))
    if attachment.sample_install(a.fd):
        raise OSError('Cannot attach sampler')
    a.seed = 7
    m = make_model(a)
    try:
        docs, _, data = fixtures('natural', a.targets + 1024, PIN)
        x = np.frombuffer(docs[0], np.uint8)
        m.run(x[:1024], x[1:1025], mode='periodic', lr=.04, lp=.04 / 3)
        print(json.dumps(dict(pid=os.getpid(), native_library=str(m.lib._name),
                              native_library_sha256=sha(Path(m.lib._name).read_bytes()), data=data)), flush=True)
        sys.stdin.readline()
        c0 = time.process_time()
        m.run(x[1024:-1], x[1025:], mode='periodic', lr=.04, lp=.04 / 3, logits=True)
        print(json.dumps(dict(done=True, process_cpu_seconds=time.process_time() - c0)), flush=True)
        # Keep worker threads alive while the parent records their CPU ticks.
        sys.stdin.readline()
    finally:
        m.close()


def sample_one(a, attachment):
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK)
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'SHOEBOX_INLINE_DIAGNOSTICS': '0', 'SHOEBOX_KERNEL': 'completion'}
    cmd = [sys.executable, __file__, '--child', '--profile', a.profile,
           '--targets', str(a.targets), '--cpu', str(a.cpu), '--worker-cpu', str(a.worker_cpu),
           '--attachment', str(attachment), '--fd', str(write_fd), '--output', str(a.output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                            pass_fds=(write_fd,), env=env)
    os.close(write_fd)
    try:
        ready = json.loads(proc.stdout.readline())
        pid = ready['pid']
        maps = []
        for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) < 6 or 'x' not in fields[1] or not fields[5].startswith('/'):
                continue
            start, end = (int(x, 16) for x in fields[0].split('-'))
            maps.append((start, end, start - int(fields[2], 16), fields[5]))
        # Symbols are loaded before sampling, so symbolization never delays it.
        tables = {path: symbol_table(path) for _, _, _, path in maps}
        ticks0 = task_ticks(pid)
        libc = ct.CDLL(None, use_errno=True)
        records, pending = [], b''
        proc.stdin.write('go\n'); proc.stdin.flush()
        attempted = 0
        next_signal = time.monotonic()
        while True:
            readable, _, _ = select.select([proc.stdout, read_fd], [], [], max(0., next_signal - time.monotonic()))
            if read_fd in readable:
                pending += os.read(read_fd, 65536)
                while len(pending) >= 24:
                    records.append(struct.unpack_from('QQQ', pending))
                    pending = pending[24:]
            if proc.stdout in readable:
                done = json.loads(proc.stdout.readline())
                break
            if time.monotonic() >= next_signal:
                for tid in ticks0:
                    if libc.syscall(234, pid, tid, signal.SIGUSR2) == 0:
                        attempted += 1
                next_signal = time.monotonic() + 1 / a.hz
        ticks1 = task_ticks(pid)
        proc.stdin.write('finish\n'); proc.stdin.flush()
        if proc.wait() != 0:
            raise RuntimeError('Sampling child failed')
        while True:
            data = os.read(read_fd, 65536)
            if not data:
                break
            pending += data
        while len(pending) >= 24:
            records.append(struct.unpack_from('QQQ', pending)); pending = pending[24:]
        if not records:
            raise RuntimeError('No instruction samples collected')
        counts = defaultdict(Counter)
        cpu_range = {}
        for tid, ip, cpu_ns in records:
            name = '<unmapped>'
            for start, end, base, path in maps:
                if start <= ip < end:
                    offsets, symbols = tables[path]
                    relative = ip - base
                    idx = bisect_right(offsets, relative) - 1
                    name = Path(path).name + ':<unresolved>'
                    if idx >= 0:
                        pos, size, sym = symbols[idx]
                        if relative < pos + size:
                            name = sym
                    break
            counts[tid][name] += 1
            if tid not in cpu_range:
                cpu_range[tid] = [cpu_ns, cpu_ns]
            cpu_range[tid][1] = cpu_ns
        threads = []
        for tid, counter in counts.items():
            groups = Counter()
            for name, n in counter.items():
                groups[category(name)] += n
            total = sum(counter.values())
            threads.append(dict(role='caller' if tid == pid else 'learning_worker', samples=total,
                                cpu_seconds=(ticks1.get(tid, ticks0[tid]) - ticks0[tid]) / os.sysconf('SC_CLK_TCK'),
                                categories={name: dict(samples=n, percent=100 * n / total) for name, n in groups.items()},
                                top_symbols=[dict(symbol=name, samples=n, percent=100 * n / total)
                                             for name, n in counter.most_common()]))
        return dict(profile=a.profile, targets=a.targets, warmup=1024, attempted_signals=attempted,
                    received_samples=len(records), threads=threads, process_cpu_seconds=done['process_cpu_seconds'],
                    native_library_sha256=ready['native_library_sha256'], data=ready['data'])
    finally:
        os.close(read_fd)
        if proc.poll() is None:
            proc.kill(); proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', type=int, default=8192)
    p.add_argument('--hz', type=float, default=100.)
    p.add_argument('--profile', choices=PROFILES, default='reference')
    p.add_argument('--cpu', type=int, default=0)
    p.add_argument('--worker-cpu', type=int, default=1)
    p.add_argument('--child', action='store_true')
    p.add_argument('--attachment', type=Path)
    p.add_argument('--fd', type=int)
    a = p.parse_args()
    if platform.machine() != 'x86_64' or sys.platform != 'linux':
        p.error('This sampler requires Linux x86-64')
    if os.environ.get('SHOEBOX_CXX_FLAGS'):
        p.error('Sampling must use the same unmodified production build')
    if a.child:
        return child(a)
    with tempfile.TemporaryDirectory(prefix='shoebox-sampler-') as tmp:
        attachment = Path(tmp) / 'sample_signal.so'
        subprocess.run(['gcc', '-O2', '-shared', '-fPIC', str(ROOT / 'src/native_sample_signal.c'), '-o', str(attachment)], check=True)
        result = dict(protocol=__doc__, hz_per_thread=a.hz, caller_cpu=a.cpu, worker_cpu=a.worker_cpu,
                      controller_sha256=sha(Path(__file__).read_bytes()),
                      attachment_source_sha256=sha((ROOT / 'src/native_sample_signal.c').read_bytes()),
                      attachment_compiler_flags=['-O2', '-shared', '-fPIC'], runs=[])
        for profile in PROFILES:
            a.profile = profile
            result['runs'].append(sample_one(a, attachment))
            atomic_json(a.output, result)
            print(profile, result['runs'][-1]['received_samples'], 'samples', flush=True)
    result['completed'] = True
    atomic_json(a.output, result)


if __name__ == '__main__':
    main()
