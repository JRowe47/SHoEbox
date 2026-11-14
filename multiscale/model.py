"""NumPy/ctypes API; native loops own all per-byte computation and learning."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
_LIB = None


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = (HERE / "native.cpp").read_bytes() + (HERE / "native.hpp").read_bytes()
    flags = ["-std=c++17", "-O3", "-march=native", "-ffp-contract=off", "-fPIC", "-shared"]
    compiler = os.environ.get("CXX", "g++")
    signature = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256(source + str(flags).encode() + signature).hexdigest()[:20]
    cache = Path(os.environ.get("SHOE_MULTISCALE_CACHE", str(Path(tempfile.gettempdir()) / "shoe_multiscale")))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"native_{key}.so"
    if not path.exists():
        temporary = cache / f"native_{key}.{os.getpid()}.so"
        subprocess.run([compiler, *flags, str(HERE / "native.cpp"), "-o", str(temporary)], check=True)
        temporary.replace(path)
    lib = C.CDLL(str(path))
    p, i, n, f = C.c_void_p, C.c_int, C.c_size_t, C.c_float
    specs = {
        "ms_error": ([], C.c_char_p), "ms_create": ([i, i, C.c_uint32, i, i], p),
        "ms_destroy": ([p], None), "ms_clone": ([p, i, i], p),
        "ms_run": ([p, p, p, n, C.POINTER(C.c_double)], i),
        "ms_observe": ([p, C.c_uint8, p], i), "ms_logits": ([p, p], i),
        "ms_reset": ([p, i], i), "ms_finish": ([p], i),
        "ms_configure": ([p, f, f, f, f], i), "ms_intervene": ([p, i, i], i),
        "ms_stats": ([p, p], i), "ms_save_size": ([p], n),
        "ms_save": ([p, p, n], i), "ms_load": ([p, n], p),
        "ms_bind": ([p, p, n, p, n, C.POINTER(C.c_uint64)], i),
        "ms_lookup": ([p, p, n, C.POINTER(C.c_uint64)], i),
        "ms_payload": ([p, C.c_uint64, p, C.POINTER(n)], i),
    }
    for name, (args, result) in specs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    _LIB = lib
    return lib


def _bytes(x):
    if isinstance(x, str):
        x = x.encode("utf-8")
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(x, dtype=np.uint8)
    a = np.asarray(x)
    if a.ndim != 1 or a.dtype != np.uint8:
        raise TypeError("inputs must be bytes or a one-dimensional uint8 array")
    return np.ascontiguousarray(a)


class Model:
    def __init__(self, *, levels=4, parameters=1_000_000, seed=7, router="indexed", associations=False):
        if router not in ("indexed", "scan"):
            raise ValueError("router must be 'indexed' or 'scan'")
        self._lib = _library()
        self._handle = self._lib.ms_create(levels, parameters, seed, router == "indexed", associations)
        self._check_pointer()

    def _check_pointer(self):
        if not self._handle:
            raise RuntimeError(self._lib.ms_error().decode())

    def _alive(self):
        if not self._handle:
            raise RuntimeError("model has been closed")
        return self._handle

    def _check(self, code):
        if code:
            raise RuntimeError(self._lib.ms_error().decode())

    @classmethod
    def _wrap(cls, handle):
        obj = cls.__new__(cls)
        obj._lib, obj._handle = _library(), handle
        obj._check_pointer()
        return obj

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.ms_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self._alive()
        return self

    def __exit__(self, *exc):
        self.close()

    def configure(self, *, learning_rate=.08, head_rate=.16, auxiliary_weight=.1, route_rate=.002):
        self._check(self._lib.ms_configure(self._alive(), learning_rate, head_rate, auxiliary_weight, route_rate))

    def run(self, inputs, targets):
        x, y = _bytes(inputs), _bytes(targets)
        if x.size != y.size or not x.size:
            raise ValueError("inputs and targets must have the same nonzero length")
        loss = C.c_double()
        self._check(self._lib.ms_run(self._alive(), x.ctypes.data, y.ctypes.data, x.size, C.byref(loss)))
        return loss.value

    def train(self, data):
        """Train n-1 next-byte targets. For streaming, pass x/y to run directly."""
        if not self.stats()["training"]:
            raise RuntimeError("this is a frozen inference model")
        a = _bytes(data)
        return self.run(a[:-1], a[1:])

    def logits(self):
        """Normalized log probabilities over all 256 bytes."""
        a = np.empty(256, np.float32)
        self._check(self._lib.ms_logits(self._alive(), a.ctypes.data))
        return a

    def observe(self, byte):
        if not isinstance(byte, (int, np.integer)) or not 0 <= byte <= 255:
            raise ValueError("byte must be an integer in 0..255")
        a = np.empty(256, np.float32)
        self._check(self._lib.ms_observe(self._alive(), byte, a.ctypes.data))
        return a

    def clone(self):
        return self._wrap(self._lib.ms_clone(self._alive(), 0, 0))

    def inference_copy(self, *, cold=True):
        """Freeze a weight snapshot; preserve source pending updates unchanged."""
        return self._wrap(self._lib.ms_clone(self._alive(), 1, cold))

    def evaluate(self, data, *, mask=-1, shuffle=-1):
        a = _bytes(data)
        with self.inference_copy() as m:
            m.intervene(mask=mask, shuffle=shuffle)
            return m.run(a[:-1], a[1:])

    def intervene(self, *, mask=-1, shuffle=-1):
        """Mask or coordinate-permute one slow level during frozen evaluation."""
        self._check(self._lib.ms_intervene(self._alive(), mask, shuffle))

    def finish(self):
        """Apply accumulated local gradients at an explicit truncation boundary.

        Held outputs become constants until replaced by new events. No targets
        or unobserved future units are fabricated for unfinished forecasts.
        """
        self._check(self._lib.ms_finish(self._alive()))

    def reset_context(self, *, clear_memory=False):
        self._check(self._lib.ms_reset(self._alive(), clear_memory))

    def stats(self):
        a = np.zeros(48, np.float64)
        self._check(self._lib.ms_stats(self._alive(), a.ctypes.data))
        levels = int(a[7])
        return dict(parameters=int(a[0]), bytes=int(a[1]), training=bool(a[2]),
                    route_comparisons=int(a[3]), memory_queries=int(a[4]),
                    memory_candidates=int(a[5]), memory_writes=int(a[6]),
                    levels=levels, experts=a[8:8+levels].astype(int).tolist(),
                    events=a[12:12+levels].astype(int).tolist(),
                    updates=a[16:16+levels].astype(int).tolist(),
                    pending=a[20:20+levels].astype(int).tolist(),
                    native_targets=a[24:24+levels].astype(int).tolist(),
                    native_loss_sum=a[28:28+levels].tolist(),
                    state_energy=a[32:32+levels].tolist())

    @property
    def parameter_count(self):
        return self.stats()["parameters"]

    def dumps(self):
        size = self._lib.ms_save_size(self._alive())
        if not size:
            raise RuntimeError(self._lib.ms_error().decode())
        buf = C.create_string_buffer(size)
        self._check(self._lib.ms_save(self._alive(), buf, size))
        raw = buf.raw
        # Header-level checksum also rejects accidental counter/payload changes.
        return b"SHOEMS01" + hashlib.sha256(raw).digest() + raw

    @classmethod
    def loads(cls, data):
        if len(data) < 40 or data[:8] != b"SHOEMS01" or hashlib.sha256(data[40:]).digest() != data[8:40]:
            raise ValueError("invalid or corrupted multiscale checkpoint")
        raw = data[40:]
        return cls._wrap(_library().ms_load(raw, len(raw)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(self.dumps())
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls.loads(Path(path).read_bytes())

    def bind(self, key, value):
        """Admit an already observed exact key (1..64 bytes) and payload (<=256)."""
        k, v, handle = _bytes(key), _bytes(value), C.c_uint64()
        self._check(self._lib.ms_bind(self._alive(), k.ctypes.data, k.size, v.ctypes.data, v.size, C.byref(handle)))
        return handle.value

    def lookup(self, key):
        k, handle = _bytes(key), C.c_uint64()
        self._check(self._lib.ms_lookup(self._alive(), k.ctypes.data, k.size, C.byref(handle)))
        return handle.value or None

    def payload(self, handle):
        a, size = np.empty(256, np.uint8), C.c_size_t()
        self._check(self._lib.ms_payload(self._alive(), handle, a.ctypes.data, C.byref(size)))
        return a[:size.value].tobytes()

    def generate(self, prompt, *, length=256, temperature=.8, seed=7):
        a = _bytes(prompt)
        if not a.size or length < 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("a nonempty prompt, nonnegative length and positive temperature are required")
        rng = np.random.default_rng(seed)
        result = bytearray(a.tobytes())
        with self.inference_copy() as m:
            for byte in a:
                logits = m.observe(int(byte))
            for _ in range(length):
                z = logits.astype(np.float64) / temperature
                p = np.exp(z - z.max())
                byte = int(rng.choice(256, p=p / p.sum()))
                result.append(byte)
                logits = m.observe(byte)
        return bytes(result)


def fresh_model(**kwargs):
    return Model(**kwargs)
