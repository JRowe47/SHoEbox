"""Public E8 lifecycle and explicit research checkpoint conversion."""
import ctypes as C
import hashlib
import struct

import numpy as np

from ..public import Model
from ..model import _bytes
from .backend import library

MAGIC = 0x3850454f4853
RESEARCH_MAGIC = 0x38455041514853
HEADER = b"SHOEME01"


class E8Model(Model):
    def __init__(self, *, parameters=1_000_000, seed=7, policy=None,
                 levels=4, core=None, **options):
        if levels != 4 or core not in (None, "minimal") or options:
            raise ValueError("E8 requires four minimal scales; use a historical constructor for other architectures")
        if isinstance(parameters, bool) or not isinstance(parameters, (int, np.integer)) or not 0 < parameters <= 2**31-1:
            raise ValueError("parameters must be a positive int32")
        # Validate the nominal budget before the qualified readout deduction.
        # Preserve the historical public cap and rejection type.
        if parameters > 100_000_000:
            raise RuntimeError("parameters must not exceed 100,000,000")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32")
        self._lib = library()
        self._attach(self._lib.qr_new(parameters, seed, 1, 4, 1))
        try:
            self._check(self._lib.qr_configure(self._adapter, .16, 1., .1))
        except Exception:
            self.close()
            raise

    def _attach(self, ptr):
        self._adapter = ptr
        if not ptr:
            raise RuntimeError(self._lib.ms_error().decode())
        self._profile = "addressed"
        self._handle = self._lib.pe_core(ptr)

    @classmethod
    def _from_pointer(cls, ptr):
        obj = object.__new__(cls)
        obj._lib = library()
        obj._attach(ptr)
        return obj

    def close(self):
        if getattr(self, "_adapter", None):
            self._lib.qr_free(self._adapter)
            self._adapter = None
            self._handle = None

    def run(self, inputs, targets, *, mask=None, local_labels=None):
        """Mean next-byte loss. Optional labels describe observed windows only.

        Six integers per input: query byte (-1 absent), four packed key/value
        windows (-1 absent, otherwise 256*key+value), observed request (0/1).
        No labels are inferred from input bytes. Generic calls freeze local heads.
        A zero mask suppresses that next-byte objective, not its local labels.
        """
        result = self.run_objectives(inputs, targets, mask=mask, local_labels=local_labels)
        return result["loss_sum"] / result["targets"] if result["targets"] else 0.

    def run_objectives(self, inputs, targets, *, mask=None, local_labels=None):
        """Same updates as run(), returning exact loss sum and charged counts."""
        x, y = _bytes(inputs), _bytes(targets)
        if x.size != y.size or not x.size:
            raise ValueError("inputs and targets must have the same nonzero length")
        weights = None
        if mask is not None:
            weights = _bytes(mask)
            if len(weights) != len(x) or np.any(weights > 1):
                raise ValueError("mask must contain one 0/1 byte per input")
        labels = None
        if local_labels is not None:
            a = np.asarray(local_labels)
            bounds = np.array([255, 65535, 65535, 65535, 65535, 1])
            if a.shape != (len(x), 6) or a.dtype.kind not in "iu" or np.any(a < -1) or np.any(a > bounds):
                raise ValueError("local_labels require six in-range integers per input")
            labels = np.ascontiguousarray(a, dtype=np.int32)
        self._alive()
        loss, count = C.c_double(), C.c_size_t()
        self._check(self._lib.pe_run(self._adapter, x.ctypes.data, y.ctypes.data,
                    weights.ctypes.data if weights is not None else None,
                    labels.ctypes.data if labels is not None else None,
                    len(x), C.byref(loss), C.byref(count)))
        return dict(loss_sum=loss.value, targets=count.value, input_bytes=len(x))

    def logits(self):
        self._alive()
        a = (C.c_float * 256)()
        self._check(self._lib.qr_probs(self._adapter, a))
        return np.array(a)

    def observe(self, byte):
        if not isinstance(byte, (int, np.integer)) or not 0 <= byte <= 255:
            raise ValueError("byte must be an integer in 0..255")
        self._alive()
        a = (C.c_float * 256)()
        self._check(self._lib.pe_observe(self._adapter, byte, a))
        return np.array(a)

    def clone(self):
        self._alive()
        return self._from_pointer(self._lib.pe_clone(self._adapter, 0))

    def inference_copy(self, *, cold=True, clear_memory=False):
        if clear_memory and not cold:
            raise ValueError("clear_memory requires a cold context")
        self._alive()
        m = self._from_pointer(self._lib.pe_clone(self._adapter, 1))
        if cold:
            m.reset_context(clear_memory=clear_memory)
        return m

    def readout_stats(self):
        self._alive()
        a = (C.c_double * 24)()
        self._check(self._lib.qr_info(self._adapter, a))
        return dict(policy="e8", core_parameters=int(a[0]), readout_parameters=int(a[1]),
                    head_targets=int(a[3]), prediction_scratch_bytes=int(a[8]),
                    allocated_head_bytes=int(a[13]), allocated_adapter_bytes=int(a[14]),
                    local_training_windows=int(a[16]), optimizer_state_bytes=int(a[17]),
                    existing_stream_and_retained_bytes=int(a[19]),
                    gate_targets=int(a[21]), local_projection_steps=int(a[22]),
                    local_presence_steps=int(a[23]), added_persistent_sequence_state_bytes=0)

    def stats(self):
        result = super().stats()
        result.update(self.readout_stats())
        result.update(parameters=result['core_parameters'] + result['readout_parameters'],
                      core_checkpoint_version=19, checkpoint_version="e8-public-1")
        return result

    def reconstruction_features(self):
        """Final completed L3 state for a separately budgeted external decoder."""
        self._alive()
        out = (C.c_float * 64)()
        self._check(self._lib.qr_features(self._adapter, out))
        return np.array(out)

    def apply_reconstruction_credit(self, gradient):
        """Apply a separately supplied final-L3 gradient, then finish its epoch."""
        x = np.asarray(gradient)
        if x.shape != (64,) or not np.all(np.isfinite(x)) or np.any(np.abs(x) > np.finfo(np.float32).max):
            raise ValueError("reconstruction gradient must contain 64 finite float32 values")
        self._alive()
        self._check(self._lib.qr_credit(self._adapter, (C.c_float * 64)(*x)))

    def benchmark(self, data, count):
        """Native full-256-byte inference checksum; frozen models only."""
        a = _bytes(data)
        if not len(a) or type(count) is not int or count < 0:
            raise ValueError("nonempty data and a nonnegative integer count required")
        self._alive()
        out = C.c_double()
        self._check(self._lib.qr_benchmark(self._adapter, a.tobytes(), len(a), count, C.byref(out)))
        return out.value

    def dumps(self):
        self._alive()
        n = C.c_size_t()
        self._check(self._lib.qr_save(self._adapter, None, 0, C.byref(n)))
        buf = C.create_string_buffer(n.value)
        self._check(self._lib.qr_save(self._adapter, buf, n.value, C.byref(n)))
        return HEADER + hashlib.sha256(buf.raw).digest() + buf.raw

    @classmethod
    def loads(cls, data):
        if len(data) < 60 or data[:8] != HEADER or hashlib.sha256(data[40:]).digest() != data[8:40]:
            raise ValueError("invalid or corrupted public E8 checkpoint")
        return cls._from_pointer(library().pe_load(data[40:], len(data)-40))

    @classmethod
    def import_research(cls, data):
        if len(data) < 52 or hashlib.sha256(data[32:]).digest() != data[:32]:
            raise ValueError("invalid E8 research checksum")
        if struct.unpack_from("=Q", data, 32)[0] != RESEARCH_MAGIC:
            raise ValueError("explicit importer accepts only E8 research checkpoints")
        raw = struct.pack("=Q", MAGIC) + data[40:]
        model = cls.loads(HEADER + hashlib.sha256(raw).digest() + raw)
        receipt = dict(source_format="EXP01-E8-research-1", destination_format="e8-public-1",
                       source_sha256=hashlib.sha256(data).hexdigest(),
                       destination_sha256=hashlib.sha256(model.dumps()).hexdigest(),
                       payload_after_magic_exact=model.dumps()[48:] == data[40:])
        return model, receipt
