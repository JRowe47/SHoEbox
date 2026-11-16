"""Current public SHoEbox plus isolated compression and frozen-probe helpers."""
import ctypes as C
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np
from multiscale.model import Model
from .tasks import VALUE

ROOT = Path(__file__).resolve().parents[2]
_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    compiler = os.environ.get("CXX", "g++")
    flags = ["-std=c++17", "-O3", "-march=native", "-ffp-contract=off", "-shared", "-fPIC"]
    h = hashlib.sha256(str(flags).encode()+subprocess.check_output([compiler, "--version"]))
    for p in [ROOT/"multiscale/native.cpp", *sorted((ROOT/"multiscale").glob("*.hpp")), Path(__file__).with_name("native.cpp")]:
        h.update(p.read_bytes())
    cache = Path(tempfile.gettempdir())/"shoebox-mad"
    cache.mkdir(exist_ok=True)
    dest = cache/(h.hexdigest()[:20]+".so")
    if not dest.exists():
        temporary = dest.with_suffix(f".{os.getpid()}.so")
        subprocess.run([compiler, *flags, "-I"+str(ROOT/"multiscale"),
                        str(Path(__file__).with_name("native.cpp")), "-o", str(temporary)], check=True)
        temporary.replace(dest)
    lib = C.CDLL(str(dest))
    p, i, f = C.c_void_p, C.c_int, C.POINTER(C.c_float)
    specs = {"ms_error": ([], C.c_char_p), "mad_observe": ([p, C.c_uint8], i),
             "mad_train": ([p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t, C.POINTER(C.c_double), C.POINTER(C.c_size_t)], i),
             "mad_features": ([p, f], i), "mad_credit": ([p, f], i),
             "mad_score": ([p, C.c_char_p, C.c_char_p, C.c_size_t, i, i, i, C.POINTER(C.c_double)], i),
             "mad_permute": ([p, i, C.c_uint32, C.POINTER(i)], i),
             "mad_inventory": ([p, C.POINTER(C.c_double)], i)}
    for name, (args, result) in specs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    _LIB = lib
    return lib


class CompressionHead:
    """A position-wise MLP decoding one 64-float aggregation state."""
    def __init__(self, length, vocab, seed):
        rng = np.random.default_rng(seed ^ 0x434F4D50)
        self.w1 = rng.normal(0, .05, (128, 64)).astype(np.float32)
        self.b1 = np.zeros(128, np.float32)
        self.w2 = rng.normal(0, .05, (length, vocab, 128)).astype(np.float32)
        self.b2 = np.zeros((length, vocab), np.float32)

    @property
    def arrays(self):
        return {k: getattr(self, k) for k in ("w1", "b1", "w2", "b2")}

    @property
    def parameters(self):
        return sum(x.size for x in self.arrays.values())

    def predict(self, x):
        hidden = np.tanh(self.w1 @ x + self.b1)
        z = self.w2 @ hidden + self.b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p/p.sum(1, keepdims=True)

    def gradients(self, x, y):
        hidden = np.tanh(self.w1 @ x + self.b1)
        p = self.predict(x)
        loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()
        dz = p.copy()
        dz[np.arange(len(y)), y] -= 1
        dz /= len(y)
        dh = np.einsum("lv,lvh->h", dz, self.w2)*(1-hidden*hidden)
        dx = self.w1.T @ dh
        gradients = dict(w1=np.outer(dh, x), b1=dh, w2=dz[..., None]*hidden, b2=dz)
        return float(loss), dx, gradients

    def train(self, x, y, lr):
        loss, dx, gradients = self.gradients(x, y)
        for k, g in gradients.items():
            getattr(self, k)[:] -= lr*g
        return loss, dx


class BenchModel:
    def __init__(self, spec, parameters=1_000_000, seed=7, steps=0, lr_scale=1., auxiliary=.1):
        self.spec = spec
        self.head = CompressionHead(spec.length, spec.vocab, seed) if spec.task == "compression" else None
        self.head_rate = .2*lr_scale
        extra = self.head.parameters if self.head else 0
        self.model = Model(parameters=parameters-extra, seed=seed,
                           loops={"steps": [0, 0, steps, steps], "tolerance": 0.} if steps else None)
        self.model.configure(fast_rate=.008*lr_scale, slow_rate=.08*lr_scale,
                             query_rate=.008*lr_scale, head_rate=.16*lr_scale,
                             route_rate=.002*lr_scale, auxiliary_weight=auxiliary)
        self.lib = library()

    def call(self, name, *args):
        if getattr(self.lib, name)(self.model._alive(), *args):
            raise RuntimeError(self.lib.ms_error().decode())

    def close(self):
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def features(self):
        x = (C.c_float*64)()
        self.call("mad_features", x)
        return np.array(x)

    def train_episode(self, ep, remaining):
        self.model.reset_context(clear_memory=True)
        cost = len(ep.data) if self.head else len(ep.data)-1
        used = min(cost, remaining)
        n = min(used, len(ep.data)-1)
        loss, targets = C.c_double(), C.c_size_t()
        if n:
            self.call("mad_train", ep.data[:n], ep.data[1:n+1], ep.mask[1:n+1], n, C.byref(loss), C.byref(targets))
        task_loss = None
        if self.head and used == cost:
            self.call("mad_observe", ep.data[-1])
            task_loss, dx = self.head.train(self.features(), ep.answer, self.head_rate)
            self.call("mad_credit", (C.c_float*64)(*dx))
        return used, targets.value, loss.value, task_loss

    def clone(self, steps=None, intervention=None):
        obj = self.__class__.__new__(self.__class__)
        obj.spec, obj.lib, obj.head_rate = self.spec, self.lib, self.head_rate
        obj.model = self.model.inference_copy(clear_memory=True)
        obj.head = None
        if self.head:
            obj.head = CompressionHead(self.spec.length, self.spec.vocab, 0)
            for k, v in self.head.arrays.items():
                getattr(obj.head, k)[:] = v
        if steps is not None:
            obj.model.configure_loops(steps=[0, 0, steps, steps], tolerance=0.)
        obj.intervention_count = None
        if intervention == "no_memory":
            obj.model.intervene_memory("no_read")
            obj.model.intervene_recall("no_read")
        elif intervention in ("permute_l2", "permute_l3"):
            n = C.c_int()
            obj.call("mad_permute", int(intervention[-1]), 47, C.byref(n))
            obj.intervention_count = n.value
        elif intervention not in (None, "zero_bottleneck"):
            obj.close()
            raise ValueError("unknown intervention")
        obj.zero_bottleneck = intervention == "zero_bottleneck"
        return obj

    def score(self, ep, rollout=0):
        self.model.reset_context(clear_memory=True)
        if self.head:
            for byte in ep.data:
                self.call("mad_observe", byte)
            x = self.features()
            if getattr(self, "zero_bottleneck", False):
                x[:] = 0
            p = self.head.predict(x)
            nll = -float(np.log(np.maximum(p[np.arange(len(ep.answer)), ep.answer], 1e-30)).sum())
            correct = p.argmax(1) == ep.answer
            return [len(ep.answer), nll, nll, int(correct.sum()), int(correct.all()), int(correct.all()), int(correct.sum())]
        out = (C.c_double*7)()
        self.call("mad_score", ep.data, ep.mask, len(ep.data), VALUE, self.spec.vocab, rollout, out)
        return list(out)

    def inventory(self):
        out = (C.c_double*10)()
        self.call("mad_inventory", out)
        names = ("core_parameters", "levels", "state_width", "recurrent_vector_bytes",
                 "minkron_matrix_bytes", "address_matrix_bytes", "retained_record_bytes",
                 "child_buffer_bytes", "allocated_loop_training_tape_bytes", "retained_slots_per_level")
        result = dict(zip(names, map(int, out)))
        result["readout_parameters"] = self.head.parameters if self.head else 0
        result["parameters"] = result["core_parameters"]+result["readout_parameters"]
        result["experts_per_level"] = self.model.stats()["experts"]
        result["compression_bottleneck_bytes"] = 256 if self.head else None
        return result

    def dumps(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("core.shoe", self.model.dumps())
            z.writestr("policy.json", json.dumps({"version": 1, "spec": self.spec.__dict__, "head_rate": self.head_rate}))
            if self.head:
                arrays = io.BytesIO()
                np.savez(arrays, **self.head.arrays)
                z.writestr("head.npz", arrays.getvalue())
        payload = buf.getvalue()
        return hashlib.sha256(payload).digest()+payload

    @classmethod
    def loads(cls, raw):
        from .tasks import TaskSpec
        if len(raw) < 32 or hashlib.sha256(raw[32:]).digest() != raw[:32]:
            raise ValueError("MAD checkpoint checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(raw[32:])) as z:
            policy = json.loads(z.read("policy.json"))
            if policy["version"] != 1:
                raise ValueError("unsupported MAD checkpoint")
            obj = cls.__new__(cls)
            obj.spec, obj.head_rate = TaskSpec(**policy["spec"]), float(policy["head_rate"])
            if not np.isfinite(obj.head_rate) or obj.head_rate < 0:
                raise ValueError("invalid head rate")
            obj.lib, obj.head = library(), None
            obj.model = Model.loads(z.read("core.shoe"))
            try:
                if obj.spec.task == "compression":
                    obj.head = CompressionHead(obj.spec.length, obj.spec.vocab, 0)
                    with np.load(io.BytesIO(z.read("head.npz")), allow_pickle=False) as arrays:
                        for k, v in obj.head.arrays.items():
                            if arrays[k].shape != v.shape or not np.isfinite(arrays[k]).all():
                                raise ValueError("invalid compression head")
                            v[:] = arrays[k]
                return obj
            except Exception:
                obj.close()
                raise
