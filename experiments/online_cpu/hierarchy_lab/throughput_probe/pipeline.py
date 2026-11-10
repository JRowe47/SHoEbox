"""Bounded validation/checkpoint overlap for an unchanged online trainer.

The producer must finish its native call before snapshot_copy(). Every worker
owns an independent snapshot. Validation never changes producer parameters,
state or learning-rate policy. Flush close() before consuming final results.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,time
import model

def process_snapshot(snapshot,targets,validation,path,cpu):
    if cpu is not None:os.sched_setaffinity(0,{cpu})
    try:
        before=snapshot.fingerprint();start=time.perf_counter()
        snapshot.save(path);save_seconds=time.perf_counter()-start
        start=time.perf_counter();score=model.evaluate(snapshot,validation)
        evaluation_seconds=time.perf_counter()-start
        assert before==snapshot.fingerprint()
        return dict(targets=targets,validation_nll=score['nll'],validation_targets=score['targets'],checkpoint=str(path),snapshot_fingerprint=before,save_seconds=save_seconds,evaluation_seconds=evaluation_seconds)
    finally:snapshot.close()

class SnapshotPipeline:
    def __init__(self,validation,directory,worker_cpu=None):
        self.validation=validation;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.worker_cpu=worker_cpu;self.executor=ThreadPoolExecutor(max_workers=1);self.pending=None;self.results=[];self.snapshot_seconds=0.;self.wait_seconds=0.
    def collect(self):
        if self.pending is not None:
            start=time.perf_counter();self.results.append(self.pending.result());self.wait_seconds+=time.perf_counter()-start;self.pending=None
    def submit(self,trainer,targets):
        self.collect()  # At most one pending snapshot: bounded memory and lag.
        start=time.perf_counter();snapshot=trainer.snapshot_copy();self.snapshot_seconds+=time.perf_counter()-start
        path=self.directory/f'at_{targets}.npz'
        self.pending=self.executor.submit(process_snapshot,snapshot,targets,self.validation,path,self.worker_cpu)
    def close(self):
        try:self.collect()
        finally:self.executor.shutdown(wait=True)
        return self.results
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
