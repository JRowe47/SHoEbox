"""Read and compare model memory outside the training loop.

Call only after run() returns, on the model-owning thread. Captures own their
arrays. No observer, clock, hook or comparison is installed in the trainer.
Historic diagnostics are excluded by explicit native byte ranges, not by
ignoring whole state sections. Operational execution/usage counts stay intact.
"""
import ctypes as ct
from dataclasses import dataclass
import hashlib
import numpy as np

SECTIONS=('state','lace_state','learned_state','composition_state','neural_state')

@dataclass
class MemorySnapshot:
    arrays: dict
    pipeline: dict
    index: dict


def read_memory_index(model):
    """Current live inventory; this is not a cumulative event history."""
    from runtime import core
    rows=np.zeros((18,12),np.uint64)
    model.lib.inspect_live_memory.argtypes=[ct.c_void_p,ct.c_void_p]
    model.lib.inspect_live_memory(model.handle,rows.ctypes.data)
    fields=('executions','publication_serial','archived_neural','pending_neural',
            'pending_input','event_ordinal','pattern_classes','ring_records',
            'retained_publications','invalid_retained_spans','constituents','missing_content_records')
    return {name:dict(zip(fields,map(int,row))) for name,row in zip(core.NAMES,rows)}


def diagnostic_ranges(model):
    model.lib.diagnostic_memory_ranges.argtypes=[ct.c_void_p]
    model.lib.diagnostic_memory_ranges.restype=ct.c_size_t
    n=model.lib.diagnostic_memory_ranges(None);rows=np.empty((n,3),np.uint64)
    model.lib.diagnostic_memory_ranges(rows.ctypes.data)
    return [(SECTIONS[int(section)],int(offset),int(size)) for section,offset,size in rows]


def normalize_observations(arrays,ranges):
    """Zero only historical diagnostic fields in detached copies."""
    out={k:np.ascontiguousarray(v).copy() for k,v in arrays.items()}
    for name,offset,size in ranges:
        if name in out:
            raw=out[name].view(np.uint8).reshape(-1)
            if offset+size>raw.size:raise ValueError(f'{name}: incompatible state layout')
            raw[offset:offset+size]=0
    return out


def decode_pending(state,flat,experts):
    """Expand either v1 or v2 pending state to the same indexed bank results."""
    state=np.ascontiguousarray(state,dtype=np.uint8)
    if not state.size:return dict(header=(),banks={})
    h=state[:96].view(np.uint64);pos=96;banks={};bank_count=18*experts
    w=flat[:bank_count*4096].reshape(bank_count,4096)
    for _ in range(int(h[9])):
        bank=int(state[pos:pos+8].view(np.uint64)[0]);pos+=8
        if h[0]==1:
            weights=state[pos:pos+16384].view(np.float32).copy();pos+=16384
        elif h[0]==2:
            flags=int(state[pos:pos+8].view(np.uint64)[0]);pos+=8;weights=w[bank].copy()
        else:raise ValueError('Unknown pipeline extension')
        bias=state[pos:pos+256].view(np.float32).copy();pos+=256
        keys=state[pos:pos+256].view(np.float32).copy();pos+=256
        if h[0]==2 and flags&1:weights=state[pos:pos+16384].view(np.float32).copy();pos+=16384
        banks[bank]=np.concatenate((weights,bias,keys))
    if pos!=state.size:raise ValueError('Trailing pipeline state')
    return dict(header=tuple(map(int,h[1:9])),banks=banks)


def capture_memory(model,*,include_stream=True):
    """Copy fitted parameters and optionally all continuation state."""
    from runtime import core,engine
    # Export parks completed pending work; it never advances the version.
    pending=model.pipeline_state()
    arrays={'flat':model.flat.copy(),'extra':model.extra_flat.copy(),
            'composition_fitted':model.composition_state(fitted=True),
            'event_fitted':model.event_state(fitted=True),'pattern_fitted':model.pattern_state(fitted=True)}
    if include_stream:
        arrays['state']=engine.parent.legacy.base.state(model)
        for key in ('lace_state','learned_state','probe_state','utility_state','neural_state',
                    'content_expert_state','pattern_state','paired_memory_state','reverse_state','event_state',
                    'composition_state','precision_state','traffic_state','gate_bias_state','activation_state'):
            arrays[key]=getattr(model,key)()
        if model.retention_policy():arrays['retention_state']=model.retention_state()
    arrays=normalize_observations(arrays,diagnostic_ranges(model))
    # The fitted composition prefix includes the legacy symbol-update counter.
    for name,offset,size in diagnostic_ranges(model):
        if name=='composition_state' and offset+size<=arrays['composition_fitted'].size:
            arrays['composition_fitted'].view(np.uint8).reshape(-1)[offset:offset+size]=0
    return MemorySnapshot(arrays,decode_pending(pending,arrays['flat'],core.E),read_memory_index(model))


def compare_memory(before,after,*,atol=0.,rtol=0.):
    """Compare detached data only; never call back into either model."""
    changes={}
    left=dict(before.arrays);right=dict(after.arrays)
    for bank,a in before.pipeline['banks'].items():left[f'pending_bank_{bank}']=a
    for bank,a in after.pipeline['banks'].items():right[f'pending_bank_{bank}']=a
    for name in sorted(left.keys()|right.keys()):
        if name not in left or name not in right:changes[name]={'missing':True};continue
        a,b=left[name],right[name]
        if a.shape!=b.shape or a.dtype!=b.dtype:changes[name]={'shape_or_type':True};continue
        if np.array_equal(a.view(np.uint8),b.view(np.uint8)):continue
        if np.issubdtype(a.dtype,np.floating):
            delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
            if np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False):continue
            changes[name]={'different':int(np.count_nonzero(a!=b)),'max_abs':float(delta.max(initial=0))}
        else:changes[name]={'different_bytes':int(np.count_nonzero(a.view(np.uint8)!=b.view(np.uint8)))}
    if before.pipeline['header']!=after.pipeline['header']:changes['pipeline_header']={'before':before.pipeline['header'],'after':after.pipeline['header']}
    return dict(equal=not changes,changes=changes)


def memory_fingerprint(snapshot):
    digest=hashlib.sha256()
    for name,value in sorted(snapshot.arrays.items()):digest.update(name.encode());digest.update(value.tobytes())
    digest.update(repr(snapshot.pipeline['header']).encode())
    for bank,value in sorted(snapshot.pipeline['banks'].items()):digest.update(str(bank).encode());digest.update(value.tobytes())
    return digest.hexdigest()
