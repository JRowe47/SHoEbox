"""Check a fresh checkout without private data or trained checkpoints."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from runtime import Model, SnapshotPipeline, core, engine, fresh_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        engine.KERNEL = 'baseline'
        reference = Model.load(args.checkpoint)
        engine.KERNEL = 'native'
        candidate = Model.load(args.checkpoint)
    else:
        reference = fresh_model(seed=871, kernel='baseline')
        candidate = fresh_model(seed=871)
    data = np.frombuffer((b'SHoEbox CPU runtime: source mixing, regional routing and memory.\n' * 100), dtype=np.uint8)
    rates = dict(mode='periodic', lr=.008, lp=.008/3, route=.002)
    assert reference.fingerprint() == candidate.fingerprint()
    for lo, hi in [(0, 537), (537, 4096)]:
        assert reference.run(data[lo:hi], data[lo+1:hi+1], **rates) == candidate.run(data[lo:hi], data[lo+1:hi+1], **rates)
        assert reference.fingerprint() == candidate.fingerprint()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'resume.npz'
        candidate.save(checkpoint)
        resumed = Model.load(checkpoint)
        assert candidate.fingerprint() == resumed.fingerprint()
        for m in [candidate, resumed]:
            m.run(data[4096:4352], data[4097:4353], **rates)
        assert candidate.fingerprint() == resumed.fingerprint()
        resumed.close()
        submitted = candidate.fingerprint()
        with SnapshotPipeline(data[:1025], directory) as pipeline:
            pipeline.submit(candidate, 4352)
            candidate.run(data[4352:4608], data[4353:4609], **rates)
        assert pipeline.results[0]['snapshot_fingerprint'] == submitted
        saved = Model.load(Path(directory) / 'at_4352.npz')
        assert saved.fingerprint() == submitted
        saved.close()
    result = dict(experts=core.E, parameters=candidate.parameter_count,
                  loaded_checkpoint=bool(args.checkpoint), native_reference_exact=True,
                  save_resume_exact=True, snapshot_pipeline_exact=True)
    candidate.close()
    reference.close()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
