"""Exact pipeline continuation across irregular chunks, workers and disk restart."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

from measure_current_baseline import ROOT, PIN, atomic_json, fixtures, segments
sys.path.insert(0, str(ROOT / 'runtime'))
from runtime import fresh_model, Model
from memory_inspection import capture_memory, compare_memory, memory_fingerprint, read_memory_index


def check(workload, targets):
    docs, _, _ = fixtures(workload, targets, PIN)
    expected = reference = None
    rows = []
    for workers in (1, 2):
        m = fresh_model(seed=7, profile='pipeline', pipeline_workers=workers)
        logits = []
        restarted = False
        try:
            with tempfile.TemporaryDirectory(prefix='shoebox-baseline-resume-') as tmp:
                for offset, x, y, end in segments(docs, targets if workers == 1 else 97,
                                                  cuts=() if workers == 1 else (1, 2049, 8193)):
                    _, values = m.run(np.frombuffer(x, np.uint8), np.frombuffer(y, np.uint8),
                                      mode='periodic', lr=.04, lp=.04 / 3, logits=True)
                    logits.append(values)
                    if workload == 'documents' and end:
                        m.finish_document(mode='periodic', lr=.04, lp=.04 / 3, reset=True)
                    if workers == 2 and offset + len(x) == 2049:
                        before = capture_memory(m)
                        path = Path(tmp) / 'pending.npz'
                        m.save(path)
                        m.close()
                        m = Model.load(path)
                        # Exercise an execution-policy change without changing
                        # the saved delayed-learning schedule.
                        m.configure_pipeline_workers(1)
                        if not compare_memory(before, capture_memory(m))['equal']:
                            raise AssertionError('Checkpoint changed pending state')
                        restarted = True
                    if workers == 2 and offset + len(x) == 8193:
                        m.configure_pipeline_workers(2)
            actual = np.concatenate(logits)
            snapshot = capture_memory(m)
            live = read_memory_index(m)
            if any(v['missing_content_records'] or v['invalid_retained_spans'] for v in live.values()):
                raise AssertionError('Invalid retained records')
            if workers == 1:
                expected, reference = actual, snapshot
            else:
                np.testing.assert_array_equal(expected, actual)
                comparison = compare_memory(reference, snapshot)
                if not comparison['equal']:
                    raise AssertionError(comparison)
                if not restarted:
                    raise AssertionError('Restart boundary not visited')
            rows.append(dict(initial_workers=workers, restarted=restarted,
                             continuation_sha256=memory_fingerprint(snapshot), live=live))
        finally:
            m.close()
    return dict(workload=workload, targets=targets, bitwise_logits=True,
                bitwise_full_continuation=True, runs=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    rows = [check('natural', 16384), check('documents', 4096)]
    atomic_json(a.output, dict(completed=True, checks=rows))
    print('16,384-byte stream and 4,096-byte document restart/chunk/worker checks passed', flush=True)


if __name__ == '__main__':
    main()
