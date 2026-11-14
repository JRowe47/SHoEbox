"""Canonical CPU reference entry point, independent of the older PyTorch package.

    from shoebox_reference import fresh_model
    model = fresh_model()
"""
from pathlib import Path
import sys

_RUNTIME = Path(__file__).resolve().parent / 'experiments/identity_semantics/runtime'
sys.path.insert(0, str(_RUNTIME))
from runtime import Model, SnapshotPipeline, core, evaluate, fresh_model

from memory_inspection import capture_memory, compare_memory, read_memory_index, read_indexed_storage

__all__ = ['capture_memory', 'compare_memory', 'read_memory_index', 'read_indexed_storage', 'Model', 'SnapshotPipeline', 'core', 'evaluate', 'fresh_model']
