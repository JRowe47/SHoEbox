"""NumPy/ctypes API; native loops own all per-byte computation and learning."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import struct
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
_LIBS = {}


def _library(profile="addressed"):
    if profile not in ("addressed", "addressed_v4", "addressed_v3", "addressed_v2", "legacy"):
        raise ValueError("memory_profile must be 'addressed', 'addressed_v4', 'addressed_v3', 'addressed_v2' or 'legacy'")
    if profile in _LIBS:
        return _LIBS[profile]
    directory = HERE if profile == "addressed" else HERE / profile
    source = (directory / "native.cpp").read_bytes() + (directory / "native.hpp").read_bytes()
    if profile != "legacy":
        source += (directory / "address_memory.hpp").read_bytes()
    if profile in ("addressed", "addressed_v4"):
        source += (directory / "recall.hpp").read_bytes()
    if profile == "addressed":
        source += (directory / "recall_credit.hpp").read_bytes()
    if profile == "addressed":
        source += (directory / "routing_geometry.hpp").read_bytes() + (directory / "routing_geometry_impl.hpp").read_bytes()
        source += (directory / "semantic_laces.hpp").read_bytes() + (directory / "semantic_laces_impl.hpp").read_bytes()
    flags = ["-std=c++17", "-O3", "-march=native", "-ffp-contract=off", "-fPIC", "-shared"]
    compiler = os.environ.get("CXX", "g++")
    signature = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256(source + str(flags).encode() + signature).hexdigest()[:20]
    cache = Path(os.environ.get("SHOE_MULTISCALE_CACHE", str(Path(tempfile.gettempdir()) / "shoe_multiscale")))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"native_{key}.so"
    if not path.exists():
        temporary = cache / f"native_{key}.{os.getpid()}.so"
        subprocess.run([compiler, *flags, str(directory / "native.cpp"), "-o", str(temporary)], check=True)
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
    if profile != "legacy":
        specs.update(ms_reconstruction=([p, f], i), ms_memory_intervene=([p, i], i),
                     ms_address_stats=([p, p], i))
    if profile in ("addressed", "addressed_v4"):
        specs.update(ms_recall_intervene=([p, i], i), ms_recall_stats=([p, p], i))
    if profile == "addressed":
        specs.update(ms_learning=([p, f, f, f, f], i), ms_learning_stats=([p, p], i),
                     ms_laces=([p,i],i), ms_lace_stats=([p,p],i),
                     ms_lace_intervene=([p,i],i), ms_lace_records=([p,p],i),
                     plateau_probe=([p,p,n,p,p],i),ms_routing=([p,i],i),
                     ms_swap_experts=([p,i],i),ms_routing_stats=([p,p],i),ms_balance=([p,i],i),
                     ms_language_credit=([p,p,n],i),ms_credit_stats=([p,p],i),ms_recall_credit=([p,p,n],i))
    for name, (args, result) in specs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    _LIBS[profile] = lib
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
    def __init__(self, *, levels=4, parameters=1_000_000, seed=7, router=None, associations=False,
                 memory_profile="addressed", semantic_laces=None, routing_arm=None, balance_policy=None,
                 language_credit=None, recall_credit=None):
        """Create corrected C with coverage warmup and no mature usage penalty.

        Explicit C/D arms retain legacy balancing unless balance_policy="warmup".
        Loading preserves the saved policy. Explicit routing_arm="A" retains the
        historical indexed/lace defaults.
        semantic_laces=True also explicitly selects that historical arm.
        Historical memory profiles retain their original indexed routing.
        """
        implicit_arm = routing_arm is None
        if routing_arm is None:
            routing_arm = "C" if memory_profile == "addressed" and semantic_laces is not True else "A"
        if routing_arm not in ("A", "B", "C", "D"):
            raise ValueError("routing_arm must be 'A', 'B', 'C' or 'D'")
        if memory_profile != "addressed" and routing_arm != "A":
            raise ValueError("routing arms B/C/D require memory_profile='addressed'")
        if semantic_laces is None:
            semantic_laces = memory_profile == "addressed" and routing_arm == "A"
        if semantic_laces and (routing_arm != "A" or memory_profile != "addressed"):
            raise ValueError("semantic laces require routing arm A and memory_profile='addressed'")
        if router is None:
            router = "indexed" if routing_arm == "A" else "scan"
        if router not in ("indexed", "scan"):
            raise ValueError("router must be 'indexed' or 'scan'")
        if routing_arm != "A" and router != "scan":
            raise ValueError("routing arms B/C/D use exact search; set router='scan'")
        if balance_policy not in (None, "legacy", "warmup"):
            raise ValueError("balance_policy must be 'legacy' or 'warmup'")
        if balance_policy is not None and routing_arm not in ("C", "D"):
            raise ValueError("balance_policy requires routing C or D")
        if language_credit is not None:
            if memory_profile != "addressed":
                raise ValueError("language credit requires the addressed runtime")
            self._validate_credit(language_credit, levels)
        if recall_credit is not None:
            if memory_profile != "addressed":raise ValueError("recall credit requires the addressed runtime")
            self._validate_credit(recall_credit,levels)
        self._profile = memory_profile
        self._lib = _library(memory_profile)
        self._handle = self._lib.ms_create(levels, parameters, seed, router == "indexed", associations)
        self._check_pointer()
        if memory_profile == "addressed":
            self.set_semantic_laces(semantic_laces)
            self._check(self._lib.ms_routing(self._alive(), {"A":0,"B":1,"C":2,"D":3}[routing_arm]))
            if routing_arm in ("C", "D"):
                self.set_balance_policy(balance_policy or ("warmup" if implicit_arm else "legacy"))
            if language_credit is not None:
                self.set_language_credit(language_credit)
            if recall_credit is not None:
                self.set_recall_credit(recall_credit)

    def set_recall_credit(self, gains):
        """Credit retained values to live producers within the current epoch.

        Change this policy at a fresh or explicitly finished training boundary.
        Retained values whose producers have retired remain detached.
        """
        if self._profile!="addressed":raise ValueError("recall credit requires the addressed runtime")
        values=self._validate_credit(gains,self.stats()['levels'])
        self._check(self._lib.ms_recall_credit(self._alive(),values.ctypes.data,len(values)))

    @staticmethod
    def _validate_credit(gains, levels):
        values=np.asarray(gains,dtype=np.float64)
        if values.shape!=(levels,) or not np.all(np.isfinite(values)) or np.any(values<0) or np.any(values>np.finfo(np.float32).max):
            raise ValueError("language_credit requires one finite nonnegative gain per level")
        return np.ascontiguousarray(values,dtype=np.float32)

    def set_language_credit(self, gains):
        """Set language gains at native-state entry, including pending blocks.

        Readout-head and auxiliary rates remain unchanged. Saved gains survive
        resume; legacy checkpoints retain unit gains until explicitly changed.
        """
        if self._profile!="addressed":
            raise ValueError("language credit requires the addressed runtime")
        values=self._validate_credit(gains,self.stats()['levels'])
        self._check(self._lib.ms_language_credit(self._alive(),values.ctypes.data,len(values)))

    def set_balance_policy(self, policy):
        """Explicitly change future balance updates; retain current bias/state.

        Old checkpoints keep their saved policy unless this method is called.
        """
        if self._profile != "addressed" or policy not in ("legacy", "warmup"):
            raise ValueError("balance_policy requires 'legacy' or 'warmup' on an addressed model")
        self._check(self._lib.ms_balance(self._alive(), {"legacy":1,"warmup":2}[policy]))

    def swap_experts(self, level=-1):
        self._check(self._lib.ms_swap_experts(self._alive(),level))

    def routing_stats(self):
        a=np.zeros(64,np.float64);self._check(self._lib.ms_routing_stats(self._alive(),a.ctypes.data))
        names=("events","seeded","minimum_selections","maximum_selections","predict_targets","predict_loss_sum","minimum_bias","maximum_bias","seed_cues","window_count")
        return dict(arm="ABCD"[int(a[0])],balance_policy=("pilot_uniform","legacy","warmup")[int(a[1])] if a[0]>=2 else "none",levels=[dict(zip(names,a[4+i*15:14+i*15])) for i in range(self.stats()["levels"])])

    def set_semantic_laces(self, enabled=True):
        self._check(self._lib.ms_laces(self._alive(),int(enabled)))

    def intervene_laces(self, mode=1):
        self._check(self._lib.ms_lace_intervene(self._alive(),mode))

    def lace_records(self):
        a=np.zeros((4,128,16),np.float64)
        self._check(self._lib.ms_lace_records(self._alive(),a.ctypes.data))
        return a

    def lace_stats(self):
        a=np.zeros(64,np.float64)
        self._check(self._lib.ms_lace_stats(self._alive(),a.ctypes.data))
        names=("events","matches","misses","allocations","promotions","rejections","trial_routes","retained_routes","no_spare","records","trials","retained","reserved","dormant","state_bytes")
        return dict(enabled=bool(a[0]),training_targets=int(a[1]),levels=[dict(zip(names,map(int,a[4+i*15:19+i*15]))) for i in range(self.stats()["levels"])])

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
    def _wrap(cls, handle, profile):
        obj = cls.__new__(cls)
        obj._profile = profile
        obj._lib, obj._handle = _library(profile), handle
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

    def configure(self, *, learning_rate=None, fast_rate=None, slow_rate=None, query_rate=None,
                  head_rate=None, auxiliary_weight=None, route_rate=None,
                  reconstruction_weight=None, constituent_credit=None, language_credit=None):
        """Update rates independently. Omitted fields preserve the v5 policy.

        learning_rate is a backward-compatible alias for fast_rate. Historical
        backends preserve their original shared-rate configuration semantics.
        """
        values = (learning_rate, fast_rate, slow_rate, query_rate, head_rate,
                  auxiliary_weight, route_rate, reconstruction_weight, constituent_credit)
        if any(v is not None and (not np.isfinite(v) or v < 0 or v > np.finfo(np.float32).max) for v in values):
            raise ValueError("rates and weights must be finite and nonnegative")
        if learning_rate is not None and fast_rate is not None:
            raise ValueError("use learning_rate or fast_rate, not both")
        if language_credit is not None:
            if self._profile!="addressed":
                raise ValueError("language credit requires the addressed runtime")
            self._validate_credit(language_credit,self.stats()['levels'])
        fast = fast_rate if fast_rate is not None else learning_rate
        if self._profile == "addressed":
            policy = self.stats()["learning_rates"]
            fast = policy["fast"] if fast is None else fast
            slow = policy["slow"] if slow_rate is None else slow_rate
            query = policy["query"] if query_rate is None else query_rate
            head = policy["head"] if head_rate is None else head_rate
            aux = policy["auxiliary_weight"] if auxiliary_weight is None else auxiliary_weight
            route = policy["route"] if route_rate is None else route_rate
            credit = policy["constituent_credit"] if constituent_credit is None else constituent_credit
        else:
            if any(v is not None for v in (fast_rate, slow_rate, query_rate, constituent_credit)):
                raise ValueError("separate rates and constituent credit require a v5 model")
            fast = .08 if fast is None else fast
            head = .16 if head_rate is None else head_rate
            aux = .1 if auxiliary_weight is None else auxiliary_weight
            route = .002 if route_rate is None else route_rate
        if self._profile == "legacy" and reconstruction_weight:
            raise ValueError("legacy checkpoints retain their original learning objectives")
        self._check(self._lib.ms_configure(self._alive(), fast, head, aux, route))
        if self._profile == "addressed":
            self._check(self._lib.ms_learning(self._alive(), fast, slow, query, credit))
        if reconstruction_weight is not None and self._profile != "legacy":
            self._check(self._lib.ms_reconstruction(self._alive(), reconstruction_weight))
        if language_credit is not None:
            self.set_language_credit(language_credit)

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
        return self._wrap(self._lib.ms_clone(self._alive(), 0, 0), self._profile)

    def inference_copy(self, *, cold=True, clear_memory=False):
        """Freeze weights; cold context preserves stored associations by default.

        Use clear_memory=True for an explicitly empty episodic store. The
        source, including pending updates and bindings, is never modified.
        """
        if clear_memory and not cold:
            raise ValueError("clear_memory requires a cold context")
        # Separate recurrent context from the persistent episode store, also
        # for older checkpoint backends whose native cold clone cleared both.
        m = self._wrap(self._lib.ms_clone(self._alive(), 1, 0), self._profile)
        if cold:
            m.reset_context(clear_memory=clear_memory)
        return m

    def evaluate(self, data, *, mask=-1, shuffle=-1, memory="full", clear_memory=False):
        a = _bytes(data)
        with self.inference_copy(clear_memory=clear_memory) as m:
            m.intervene(mask=mask, shuffle=shuffle)
            if memory != "full":
                m.intervene_memory(memory)
            return m.run(a[:-1], a[1:])

    def intervene(self, *, mask=-1, shuffle=-1):
        """Mask or coordinate-permute one slow level during frozen evaluation."""
        self._check(self._lib.ms_intervene(self._alive(), mask, shuffle))

    def intervene_memory(self, mode="full"):
        """Frozen controls: full, no_read, no_time or no_content.

        Set before consuming the evaluation prefix to remove a write channel
        throughout that prefix. This never changes a live training policy.
        """
        modes = {"full": 0, "no_read": 1, "no_time": 2, "no_content": 3}
        if mode not in modes:
            raise ValueError("unknown memory intervention")
        if self._profile == "legacy":
            raise ValueError("legacy runtime has no addressed-memory channel")
        self._check(self._lib.ms_memory_intervene(self._alive(), modes[mode]))

    def intervene_recall(self, mode="full"):
        """Frozen retained-unit controls; set before consuming a cold prefix.

        Modes: full, no_read, recency_only, no_fixed_values, no_learned_values.
        This controls the v4 retained units, separately from the compact matrix.
        """
        modes = {"full": 0, "no_read": 1, "recency_only": 2,
                 "no_fixed_values": 3, "no_learned_values": 4}
        if mode not in modes:
            raise ValueError("unknown recall intervention")
        if self._profile not in ("addressed", "addressed_v4"):
            raise ValueError("retained-unit recall requires a version-4 model")
        self._check(self._lib.ms_recall_intervene(self._alive(), modes[mode]))

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
        result = dict(parameters=int(a[0]), bytes=int(a[1]), training=bool(a[2]),
                    route_comparisons=int(a[3]), memory_queries=int(a[4]),
                    memory_candidates=int(a[5]), memory_writes=int(a[6]),
                    levels=levels, experts=a[8:8+levels].astype(int).tolist(),
                    events=a[12:12+levels].astype(int).tolist(),
                    updates=a[16:16+levels].astype(int).tolist(),
                    pending=a[20:20+levels].astype(int).tolist(),
                    native_targets=a[24:24+levels].astype(int).tolist(),
                    native_loss_sum=a[28:28+levels].tolist(),
                    state_energy=a[32:32+levels].tolist(), memory_profile=self._profile,
                    checkpoint_version={"legacy": 1, "addressed_v2": 2, "addressed_v3": 3, "addressed_v4": 4, "addressed": 7}[self._profile],
                    reconstruction_targets=a[40:40+levels].astype(int).tolist(),
                    reconstruction_loss_sum=a[44:44+levels].tolist())
        if self._profile != "legacy":
            b = np.zeros(24, np.float64)
            self._check(self._lib.ms_address_stats(self._alive(), b.ctypes.data))
            result.update(clock_ticks=b[:levels].astype(int).tolist(),
                          address_rows=b[4:4+levels].astype(int).tolist(),
                          address_values=b[8:8+levels].astype(int).tolist(),
                          address_energy=b[12:12+levels].tolist(),
                          query_content_energy=b[16:16+levels].tolist(),
                          reconstruction_weight=float(b[20]), memory_intervention=int(b[21]))
        result["retained_recall"] = self._profile in ("addressed", "addressed_v4")
        if self._profile in ("addressed", "addressed_v4"):
            c = np.zeros(16, np.float64)
            self._check(self._lib.ms_recall_stats(self._alive(), c.ctypes.data))
            result.update(retained_units=c[:levels].astype(int).tolist(),
                          retained_occurrences=c[4:4+levels].astype(int).tolist(),
                          recall_slots_per_level=int(c[8]), recall_intervention=int(c[9]),
                          retained_stream_bytes=int(c[10]), recall_parameters=int(c[11]),
                          recall_query_period_bytes=1)
        if self._profile == "addressed":
            c = np.zeros(32, np.float64)
            self._check(self._lib.ms_learning_stats(self._alive(), c.ctypes.data))
            result.update(learning_rates=dict(fast=c[0], slow=c[1], query=c[2], head=c[3],
                                             constituent_credit=c[4], auxiliary_weight=c[7], route=c[8]),
                          constituent_epoch_bytes=int(c[5]), constituent_epoch_limit=int(c[6]),
                          constituent_pending=c[12:12+levels].astype(int).tolist(),
                          constituent_credit_edges=c[16:16+levels].astype(int).tolist(),
                          task_gradient_energy=c[20:20+levels].tolist(),
                          constituent_gradient_energy=c[24:24+levels].tolist(),
                          content_updates=c[28:28+levels].astype(int).tolist(),
                          fast_bank_role="context_dependent_transition")
        if self._profile == "addressed":
            gains=np.zeros(8,np.float64)
            self._check(self._lib.ms_credit_stats(self._alive(),gains.ctypes.data))
            result['language_credit']=gains[:levels].tolist()
            result['recall_credit']=gains[4:4+levels].tolist()
            lace=np.zeros(64,np.float64);self._check(self._lib.ms_lace_stats(self._alive(),lace.ctypes.data))
            result["checkpoint_version"]=int(lace[2])
            routing = np.zeros(64, np.float64)
            self._check(self._lib.ms_routing_stats(self._alive(), routing.ctypes.data))
            result.update(routing_arm="ABCD"[int(routing[0])], semantic_laces=bool(lace[0]),
                          balance_policy=("pilot_uniform","legacy","warmup")[int(routing[1])] if routing[0]>=2 else "none")
        return result

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
        if len(raw) < 16:
            raise ValueError("truncated native checkpoint")
        version = struct.unpack_from("=I", raw, 8)[0]
        profiles = {1: "legacy", 2: "addressed_v2", 3: "addressed_v3", 4: "addressed_v4", 5: "addressed", 6: "addressed", 7: "addressed", 8: "addressed", 9: "addressed", 10: "addressed", 11: "addressed", 12: "addressed"}
        if version not in profiles:
            raise ValueError("unsupported multiscale checkpoint version")
        profile = profiles[version]
        return cls._wrap(_library(profile).ms_load(raw, len(raw)), profile)

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

    def generate(self, prompt, *, length=256, temperature=.8, seed=7, clear_memory=False):
        a = _bytes(prompt)
        if not a.size or length < 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("a nonempty prompt, nonnegative length and positive temperature are required")
        rng = np.random.default_rng(seed)
        result = bytearray(a.tobytes())
        with self.inference_copy(clear_memory=clear_memory) as m:
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
