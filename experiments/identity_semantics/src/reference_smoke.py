"""Matched same-core reference smoke and throughput check; no corpus or models saved."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from runtime import fresh_model
from reference_acceptance import collect, check_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', nargs='+', choices=['composition', 'relays', 'completion'], default=['composition', 'relays', 'completion'])
    parser.add_argument('--targets', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reference_smoke.json')
    args = parser.parse_args()
    if args.targets < 1 or args.repeats < 1:
        parser.error('targets and repeats must be positive')
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    sentence = (b'The river runs through the valley. The king crossed the bridge. '
                b'A bird flew over the water. The queen opened the gate. ')
    data = np.frombuffer(sentence * ((args.targets + 2049) // len(sentence) + 1), np.uint8)
    models = {kernel: fresh_model(seed=7, kernel=kernel)
              for kernel in args.kernels}
    result = dict(cpu=cpu, seed=7, targets_per_round=args.targets, repeats=args.repeats,
                  workload='synthetic repeated sentences; a mechanism/timing smoke, not held-out language quality',
                  data_sha256=hashlib.sha256(data).hexdigest(), models={})
    try:
        for kernel, model in models.items():
            model.run(data[:2048], data[1:2049], mode='periodic', lr=.04, lp=.04 / 3)
            result['models'][kernel] = dict(parameters=model.parameter_count, rounds=[])
        kernels = list(models)
        for repeat in range(args.repeats):
            # Rotate order while keeping every run on the same pinned core.
            for kernel in kernels[repeat % len(kernels):] + kernels[:repeat % len(kernels)]:
                model = models[kernel]
                start = time.perf_counter()
                nll = model.run(data[:args.targets], data[1:args.targets + 1],
                                mode='periodic', lr=.04, lp=.04 / 3)
                seconds = time.perf_counter() - start
                assert np.isfinite(nll) and np.isfinite(model.flat).all()
                row = dict(nll=nll, seconds=seconds, bytes_per_second=args.targets / seconds)
                result['models'][kernel]['rounds'].append(row)
                print(kernel, repeat + 1, round(row['bytes_per_second']), flush=True)
        for kernel, model in models.items():
            model.finish_document(mode='periodic', lr=.04, lp=.04 / 3)
            row = result['models'][kernel]
            row['median_bytes_per_second'] = float(np.median([r['bytes_per_second'] for r in row['rounds']]))
            row['composition'] = model.composition_metrics()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            if kernel == 'completion':
                row.update(collect(model), clocks=model.completion_clocks())
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                row['acceptance_contracts'] = check_training(row)
                snapshot = model.snapshot_copy()
                try:
                    assert snapshot.fingerprint() == model.fingerprint()
                    np.testing.assert_array_equal(
                        model.run(data[:64], data[1:65], mode='periodic', logits=True)[1],
                        snapshot.run(data[:64], data[1:65], mode='periodic', logits=True)[1])
                    assert snapshot.fingerprint() == model.fingerprint()
                    row['grown_snapshot_continuation'] = True
                finally:
                    snapshot.close()
                a = model.evaluation_copy()
                b = model.evaluation_copy(diagnostic=True)
                try:
                    np.testing.assert_array_equal(a.run(data[:1024], logits=True)[1], b.run(data[:1024], logits=True)[1])
                    row['diagnostic_parity'] = True
                finally:
                    a.close()
                    b.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    finally:
        for model in models.values():
            model.close()


if __name__ == '__main__':
    main()
