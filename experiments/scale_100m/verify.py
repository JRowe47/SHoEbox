"""Compare optimized routing against a cap-only snapshot of the D runtime."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shoe_multiscale import Model


def digest(model):
    return hashlib.sha256(model.dumps()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('scalar_reference', a.reference / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data = np.frombuffer(a.data.read_bytes()[:90_001], np.uint8)
    evidence = []
    for budget, targets in [(12_000_000, 90_000), (100_000_000, 4_096)]:
        with module.Model(parameters=budget, seed=7) as old, Model(parameters=budget, seed=7) as new:
            assert digest(old) == digest(new)
            ts = time.perf_counter(); old_loss = old.run(data[:targets], data[1:targets+1]); old_time = time.perf_counter()-ts
            ts = time.perf_counter(); new_loss = new.run(data[:targets], data[1:targets+1]); new_time = time.perf_counter()-ts
            assert old_loss == new_loss
            assert digest(old) == digest(new)
            initial_digest = digest(new)
            assert old.evaluate(data[:514]) == new.evaluate(data[:514])
            assert digest(new) == initial_digest
            with Model.loads(old.dumps()) as restored:
                assert restored.run(data[:779], data[1:780]) == new.run(data[:779], data[1:780])
                assert digest(restored) == digest(new)
            evidence.append(dict(budget=budget, targets=targets, exact_loss=True, exact_state=True,
                frozen_evaluation_equal=True, evaluation_preserved_source=True, cross_runtime_continuation=True,
                scalar_seconds=old_time, optimized_seconds=new_time, speedup=old_time/new_time,
                optimized_bytes_per_second=targets/new_time, routing=new.routing_stats()))
            print(json.dumps(evidence[-1]), flush=True)
    a.out.write_text(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
