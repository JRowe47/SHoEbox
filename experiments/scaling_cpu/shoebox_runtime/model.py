"""Width-aware CPU SHoEbox. No import-time architecture globals.

Checkpoints contain all learned parameters, policies, fixed transforms and native
stream state. Widths 64/128/256 and independent regional banks can coexist.
"""
from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import tempfile

import numpy as np
from .topology import NAMES, RECIPES, CADENCE

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FP = ct.POINTER(ct.c_float)
IP = ct.POINTER(ct.c_int)
DP = ct.POINTER(ct.c_double)
U8P = ct.POINTER(ct.c_ubyte)
MODES = {'eval': 0, 'dfa': 2, 'periodic': 6}
STATE_ARRAYS = {
    'state': ('state_size', 'export_state', 'import_state'),
    'lace_state': ('lace_state_size', 'export_laces', 'import_laces'),
    'learned_state': ('learned_state_size', 'export_learned', 'import_learned'),
    'probe_state': ('probe_state_size', 'probe_export', 'probe_import'),
    'utility_state': ('utility_state_size', 'utility_export', 'utility_import'),
}
A1_FIELDS = ('prototype_scale', 'follow_cooldown', 'reference_lr', 'decoder_scale')
A1_DEFAULT = np.array([1., 0., 4e-5, 1.], np.float32)


@dataclass(frozen=True)
class Architecture:
    width: int = 64
    experts: tuple = (64,) * 18

    def __post_init__(self):
        if isinstance(self.width, bool) or not isinstance(self.width,(int,np.integer)) or self.width not in (64, 128, 256):
            raise ValueError('width must be 64, 128 or 256')
        object.__setattr__(self, 'width', int(self.width))
        counts = (self.experts,) * 18 if isinstance(self.experts, int) else tuple(self.experts)
        if len(counts) != 18 or any(isinstance(e, bool) or not isinstance(e, (int, np.integer)) or not 2 <= e <= 256 for e in counts):
            raise ValueError('Provide 18 integer expert counts in [2, 256]')
        object.__setattr__(self, 'experts', tuple(int(e) for e in counts))

    @property
    def offsets(self):
        return np.r_[0, np.cumsum(self.experts)].astype(np.int32)

    @property
    def shapes(self):
        d, total = self.width, sum(self.experts)
        return dict(W=(total, d, d), bias=(total, d), keys=(total, d),
                    P=(18, d, d), pbias=(18, d), readout=(256, 2*d), rbias=(256,))

    @property
    def extra_shapes(self):
        d, bottom = self.width, sum(self.experts[:2])
        return dict(state_gates=(2, 3, d), prototypes=(bottom, d), task_offsets=(bottom, d))

    @property
    def parameter_count(self):
        return sum(int(np.prod(s)) for s in [*self.shapes.values(), *self.extra_shapes.values()])

    def as_dict(self):
        return dict(width=self.width, experts=list(self.experts))


def views(flat, shapes):
    out = {}; pos = 0
    if flat.size != sum(int(np.prod(s)) for s in shapes.values()):
        raise ValueError('Parameter array does not match architecture')
    for key, shape in shapes.items():
        n = int(np.prod(shape)); out[key] = flat[pos:pos+n].reshape(shape); pos += n
    return out


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.pending.{os.getpid()}')
    try:
        with temporary.open('w') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(arch):
    compiler = os.environ.get('CXX', 'g++')
    flags = ['-O3', '-march=native', '-std=c++17', '-shared', '-fPIC']
    if os.environ.get('SHOEBOX_SANITIZE') == '1':
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=undefined']
    header = (f'#define SHOEBOX_WIDTH {arch.width}\n#define SHOEBOX_EXPERTS {max(arch.experts)}\n'
              f'constexpr int EXPERT_COUNTS[18]{{{",".join(map(str,arch.experts))}}};\n'
              f'constexpr int BANK_OFFSETS[19]{{{",".join(map(str,arch.offsets))}}};\n'
              f'constexpr int TOTAL_EXPERTS={sum(arch.experts)},BOTTOM_EXPERTS={sum(arch.experts[:2])};\n')
    sources = sorted((ROOT/'native').glob('*'))
    signature = subprocess.check_output([compiler, '--version']) + platform.machine().encode()
    # -march=native caches must not be reused across machines with different ISA.
    cpu = Path('/proc/cpuinfo')
    if cpu.exists():
        signature += next((line for line in cpu.read_bytes().splitlines() if line.startswith(b'flags')), b'')
    digest = hashlib.sha256(header.encode() + signature + json.dumps(flags).encode() + b''.join(p.read_bytes() for p in sources if p.is_file())).hexdigest()
    cache = Path(os.environ.get('SHOEBOX_BUILD_CACHE', str(PROJECT/'.build'))) / digest[:20]
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache/'runtime.so'
    with (cache/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not lib.exists():
            (cache/'architecture.inc').write_text(header)
            temporary = cache/f'runtime.{os.getpid()}.tmp.so'
            try:
                subprocess.run([compiler, *flags, '-I', str(cache), str(ROOT/'native/kernel.cpp'), '-o', str(temporary)], check=True)
                os.replace(temporary, lib)
            finally:
                temporary.unlink(missing_ok=True)
    return lib, digest


def defaults(width):
    settings = json.loads((PROJECT/'configs/legacy_policy.json').read_text())
    settings['lace'].update(sdr_k=width//4, credit_features=width)
    settings['state_config'].update(temporal_features=width//4, route_features=width)
    return settings


def initialize(arch, seed):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    d = arch.width
    flat = np.zeros(sum(int(np.prod(s)) for s in arch.shapes.values()), np.float32)
    p = views(flat, arch.shapes)
    # Independent streams keep decoder and common expert prefixes unchanged
    # when only the number of experts changes.
    for r, e in enumerate(arch.experts):
        lo, hi = arch.offsets[r:r+2]
        p['W'][lo:hi] = np.random.default_rng(np.random.SeedSequence([seed, 10, r])).normal(0, .25/np.sqrt(d), (e,d,d))
        keys = np.random.default_rng(np.random.SeedSequence([seed, 20, r])).normal(0, 1, (e,d)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        p['keys'][lo:hi] = keys
    p['P'][:] = np.eye(d, dtype=np.float32)[None] * .25
    p['readout'][:] = np.random.default_rng(np.random.SeedSequence([seed, 30])).normal(0, .02*np.sqrt(64/d), p['readout'].shape)
    return flat


def fixed_transports(width):
    # Use exactly the established calibration/transports at width 64. Wider
    # transforms use independent random streams and train-only calibration.
    if width == 64:
        with np.load(PROJECT/'configs/legacy_fixed_64.npz', allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    rng = np.random.default_rng(1729); d = width
    embedding = rng.choice([-1., 1.], (256,d)).astype(np.float32)*.5
    feedback = rng.normal(0, 1/np.sqrt(d), (256,d)).astype(np.float32)
    slots = []; starts = [0]
    for recipe in RECIPES:
        for src, weight, count, delay in recipe:
            for age in range(count-1, -1, -1):
                slots.append((NAMES.index(src), age, delay, float(np.sqrt(weight/4))))
        starts.append(len(slots))
    slots = np.asarray(slots, np.float32)
    return dict(embedding=embedding, feedback=feedback, slots=slots,
                starts=np.asarray(starts,np.int32),
                perm=np.stack([rng.permutation(d) for _ in slots]).astype(np.int32),
                signs=rng.choice([-1.,1.],(len(slots),d)).astype(np.float32),
                fperm=np.stack([rng.permutation(d) for _ in range(18)]).astype(np.int32),
                fsign=rng.choice([-1.,1.],(18,d)).astype(np.float32),
                cadence=CADENCE.copy(), head=np.array([1],np.int32),
                address_center=np.zeros((2,d),np.float32))


class Model:
    def __init__(self, architecture=None, seed=1, flat=None, extra=None, fixed=None,
                 config=None, lace=None, state_config=None, probe=None, utility=None,
                 teaching=None, regional_gains=None, frozen_regions=None,
                 task_route_lr_scale=1., a1_learning=None):
        self.handle = None
        self.arch = architecture if isinstance(architecture, Architecture) else Architecture(**(architecture or {}))
        self.width = self.arch.width; self.seed = seed
        settings = defaults(self.width)
        for name, supplied in [('config',config),('lace',lace),('state_config',state_config),('probe',probe),('utility',utility)]:
            base = settings[name]
            if supplied:
                if set(supplied)-set(base): raise ValueError(f'Unknown {name} settings: {set(supplied)-set(base)}')
                base.update(supplied)
            setattr(self, name, base)
        self._validate_policy()
        self.flat = initialize(self.arch, seed) if flat is None else np.array(flat,dtype=np.float32,copy=True,order='C').reshape(-1)
        self.params = views(self.flat, self.arch.shapes)
        n = sum(int(np.prod(s)) for s in self.arch.extra_shapes.values())
        self.extra_flat = np.zeros(n,np.float32) if extra is None else np.array(extra,dtype=np.float32,copy=True,order='C').reshape(-1)
        self.extra = views(self.extra_flat, self.arch.extra_shapes)
        if extra is None:
            c = self.state_config
            for r, key in enumerate(('write_a','write_b')):
                self.extra['state_gates'][r,0] = np.arctanh(c['initial_read']/c['read_limit'])
                self.extra['state_gates'][r,2] = np.log(c[key]/(1-c[key]))
            self.extra['prototypes'][:] = self.params['keys'][:sum(self.arch.experts[:2])]
        if not np.isfinite(self.flat).all() or not np.isfinite(self.extra_flat).all(): raise ValueError('Nonfinite parameters')
        self.fixed = {k:np.array(v,copy=True,order='C') for k,v in (fixed_transports(self.width) if fixed is None else fixed).items()}
        self._validate_fixed()
        self.fixed['cadence'][15:17] = [self.config['ec_c'],self.config['ec_x']]
        for k in range(int(self.fixed['starts'][17]),int(self.fixed['starts'][18])):
            if int(self.fixed['slots'][k,0]) == 15:self.fixed['slots'][k,3] = np.sqrt(.5)*self.config['ec_y_gain']
        library, self.build_digest = build(self.arch)
        self.lib = ct.CDLL(str(library)); self._bind()
        f = self.fixed
        self.handle = self.lib.create(self.flat.ctypes.data_as(FP), f['embedding'].ctypes.data_as(FP),f['feedback'].ctypes.data_as(FP),f['slots'].ctypes.data_as(FP),f['starts'].ctypes.data_as(IP),f['perm'].ctypes.data_as(IP),f['signs'].ctypes.data_as(FP),f['fperm'].ctypes.data_as(IP),f['fsign'].ctypes.data_as(FP),f['cadence'].ctypes.data_as(IP),int(f['head'][0]))
        if not self.handle:raise MemoryError('Native model allocation failed')
        self._push('configure',list(self.config)[:24],self.config)
        self.lib.set_reference(self.handle,f['address_center'].ctypes.data_as(FP))
        self._push('configure_laces',list(self.lace),self.lace)
        a=np.array(list(self.state_config.values()),np.float32)
        self.lib.configure_state(self.handle,self.extra_flat.ctypes.data_as(FP),a.ctypes.data_as(FP))
        self._push('probe_configure',list(self.probe),self.probe)
        self._push('utility_configure',list(self.utility),self.utility)
        self.configure_teaching(np.ones((18,2),np.float32) if teaching is None else teaching)
        self.configure_regional_gains(np.ones(18,np.float32) if regional_gains is None else regional_gains)
        self.configure_training_policy(np.zeros(18,np.int32) if frozen_regions is None else frozen_regions,task_route_lr_scale)
        self.configure_a1_learning(a1_learning)

    def _validate_policy(self):
        c,l,s=self.config,self.lace,self.state_config
        for group in (c,l,s,self.probe,self.utility):
            if any(not np.isfinite(v) for v in group.values() if not isinstance(v,str)):raise ValueError('Nonfinite policy')
        b,w=int(c['buckets']),int(c['ways'])
        if c['buckets']!=b or c['ways']!=w or b<4 or b&(b-1) or w<1 or b*w>8192:raise ValueError('Invalid memory buckets/ways')
        for key in ('ec_c','ec_x','stride','read_stride'):
            if int(c[key])!=c[key] or c[key]<1:raise ValueError('Invalid cadence')
        if c['mode'] not in ('dfa','periodic') or (c['wake_threshold']>0 and c['credit']==2):raise ValueError('Invalid EC policy')
        for k in ('sdr_k','credit_features'):
            if int(l[k])!=l[k] or not 1<=l[k]<=self.width:raise ValueError('Invalid lace feature budget')
        if int(l['depth'])!=l['depth'] or not 0<=l['depth']<=3 or l['credit_features'] not in (self.width,l['sdr_k']):raise ValueError('Invalid lace depth/credit width')
        for k in ('credit_stride','deep_stride'):
            if int(l[k])!=l[k] or l[k]<1:raise ValueError('Invalid credit cadence')
        if l['temperature']<=0:raise ValueError('Invalid routing temperature')
        for k in ('temporal_features','route_features'):
            if int(s[k])!=s[k] or not 1<=s[k]<=self.width:raise ValueError('Invalid state feature budget')
        if int(s['credit_stride'])!=s['credit_stride'] or s['credit_stride']<1:raise ValueError('Invalid temporal credit cadence')
        if s['read_limit']<=0 or abs(s['initial_read'])>=s['read_limit']:raise ValueError('Invalid state read range')
        if not 0<s['write_a']<1 or not 0<s['write_b']<1:raise ValueError('Invalid write probability')
        if min(s[k] for k in ('ema_mass_start','ema_mass_end','ema_anneal_events','temperature','task_offset_limit'))<=0:raise ValueError('Invalid state scales')
        if not 0<self.utility['mean_rate']<=1 or not 0<=self.probe['center_rate']<=1:raise ValueError('Invalid centering rate')

    def _validate_fixed(self):
        d=self.width;f=self.fixed
        shapes=dict(embedding=(256,d),feedback=(256,d),starts=(19,),fperm=(18,d),fsign=(18,d),cadence=(18,),head=(1,),address_center=(2,d))
        for k,shape in shapes.items():
            if k not in f or f[k].shape!=shape:raise ValueError(f'Invalid fixed array: {k}')
        if f['slots'].ndim!=2 or f['slots'].shape[1]!=4:raise ValueError('Invalid slots')
        n=len(f['slots'])
        if f['perm'].shape!=(n,d) or f['signs'].shape!=(n,d):raise ValueError('Invalid transport shape')
        ints=('starts','perm','fperm','cadence','head')
        for k,a in f.items():
            dtype=np.int32 if k in ints else np.float32
            if a.dtype!=dtype:raise ValueError(f'Wrong dtype for {k}')
            if not np.isfinite(a).all():raise ValueError('Nonfinite fixed values')
        if f['starts'][0]!=0 or f['starts'][-1]!=n or np.any(np.diff(f['starts'])<0):raise ValueError('Invalid slot offsets')
        if np.any(f['cadence']<1) or f['head'][0] not in (0,1):raise ValueError('Invalid schedule/head')
        if np.any(f['perm']<0) or np.any(f['perm']>=d) or np.any(f['fperm']<0) or np.any(f['fperm']>=d):raise ValueError('Invalid coordinate index')
        slots=f['slots']
        if np.any(slots[:,:3]!=slots[:,:3].astype(np.int32)) or np.any(slots[:,0]<0) or np.any(slots[:,0]>=18) or np.any(slots[:,1]<0) or np.any(slots[:,1]>1) or np.any((slots[:,2]!=0)&(slots[:,2]!=1)):raise ValueError('Invalid transport recipe')
        if max(np.diff(f['starts']))>8:raise ValueError('Transport cue capacity exceeded')

    def _bind(self):
        l=self.lib;v=ct.c_void_p
        bindings={
            'create':([FP,FP,FP,FP,IP,IP,FP,IP,FP,IP,ct.c_int],v),
            'run':([v,U8P,U8P,ct.c_int,ct.c_int,ct.c_float,ct.c_float,ct.c_float,ct.c_int,FP],ct.c_double),
            'configure_state':([v,FP,FP],None),
            'training_policy_configure':([v,IP,ct.c_float],ct.c_int),
            'force_revival':([v,ct.c_int,ct.c_int,ct.c_int],ct.c_int),
            'check_masks':([FP,ct.POINTER(ct.c_uint64),ct.c_int,ct.c_int,ct.c_int],None),
            'check_sparse_update':([FP,FP,ct.c_float,FP,ct.c_int],None),
            'observation_count':([v,ct.POINTER(ct.c_int64)],None),
            'draw':([v,ct.c_float,FP],ct.c_int),
            'block_before':([v,ct.c_int64],None),
            'verification_expert_update':([v,ct.c_int,FP,ct.c_float,ct.c_int],None),
        }
        for name in ('destroy','reset','clear_memory','reset_context'):
            bindings[name]=([v],None)
        for name in ('configure','configure_laces','configure_state_policy','set_reference','probe_configure','utility_configure','probe_get_means','probe_set_means','utility_mean','utility_set_mean','query_vectors','publications','learned_snapshot','info'):
            bindings[name]=([v,FP],None)
        for name in ('teaching_configure','regional_gain_configure','a1_learning_configure'):
            bindings[name]=([v,FP],ct.c_int)
        for name in ('diagnostic_enable','utility_ablate','utility_observe','set_ablation','freeze_writes','expert_update_observe'):
            bindings[name]=([v,ct.c_int],None)
        for name in ('diagnostic_export','persistent_metrics','lace_metrics','learned_metrics','expert_update_metrics','regional_training_totals'):
            bindings[name]=([v,DP],None)
        bindings['expert_update_metrics_import']=([v,DP],None)
        for size,export,restore in STATE_ARRAYS.values():
            bindings[size]=([],ct.c_size_t);bindings[export]=([v,v],None);bindings[restore]=([v,v],None)
        for name,(args,result) in bindings.items():
            fn=getattr(l,name);fn.argtypes=args;fn.restype=result
        l.lace_codes.argtypes=[v,ct.POINTER(ct.c_uint64)];l.lace_codes.restype=None

    def _push(self,method,fields,settings):
        a=np.array([settings[k] for k in fields],np.float32)
        getattr(self.lib,method)(self.handle,a.ctypes.data_as(FP))

    def configure_adaptation(self, *, prototype_scale=None, center_rate=None, mean_rate=None):
        if prototype_scale is not None:
            if not np.isfinite(prototype_scale) or not 0<prototype_scale<=1:raise ValueError('Prototype scale must be in (0,1]')
            self.state_config['reserved']=0. if prototype_scale==1 else float(prototype_scale)
            self._push('configure_state_policy',list(self.state_config),self.state_config)
        if center_rate is not None:
            if not np.isfinite(center_rate) or not 0<=center_rate<=1:raise ValueError('Invalid center rate')
            self.probe['center_rate']=float(center_rate);self._push('probe_configure',list(self.probe),self.probe)
        if mean_rate is not None:
            if not np.isfinite(mean_rate) or not 0<mean_rate<=1:raise ValueError('Invalid mean rate')
            self.utility['mean_rate']=float(mean_rate);self._push('utility_configure',list(self.utility),self.utility)

    def configure_teaching(self,scales):
        a=np.array(scales,np.float32,copy=True,order='C')
        if a.shape!=(18,2) or not self.lib.teaching_configure(self.handle,a.ctypes.data_as(FP)):raise ValueError('Invalid teaching scales')
        self.teaching=a

    def configure_regional_gains(self,gains):
        a=np.array(gains,np.float32,copy=True,order='C')
        if a.shape!=(18,) or not self.lib.regional_gain_configure(self.handle,a.ctypes.data_as(FP)):raise ValueError('Invalid regional gains')
        self.regional_gains=a

    def configure_training_policy(self,frozen_regions,task_route_lr_scale=1.):
        raw=np.asarray(frozen_regions)
        if raw.shape!=(18,) or not np.isin(raw,[0,1]).all():raise ValueError('Invalid frozen regions')
        a=np.array(raw,np.int32,copy=True,order='C')
        if not self.lib.training_policy_configure(self.handle,a.ctypes.data_as(IP),float(task_route_lr_scale)):raise ValueError('Invalid training policy')
        self.frozen_regions=a;self.task_route_lr_scale=float(np.float32(task_route_lr_scale))

    def configure_a1_learning(self,policy=None):
        if policy is None:a=A1_DEFAULT.copy()
        elif isinstance(policy,dict):
            if set(policy)-set(A1_FIELDS):raise ValueError('Unknown A1 setting')
            values=dict(zip(A1_FIELDS,A1_DEFAULT));values.update(policy);a=np.array([values[k] for k in A1_FIELDS],np.float32)
        else:a=np.array(policy,np.float32,copy=True,order='C')
        if a.shape!=(4,) or not self.lib.a1_learning_configure(self.handle,a.ctypes.data_as(FP)):raise ValueError('Invalid A1 policy')
        self.a1_learning=a

    @property
    def parameter_count(self):return int(self.flat.size+self.extra_flat.size)
    @property
    def observations(self):
        n=ct.c_int64();self.lib.observation_count(self.handle,ct.byref(n));return n.value

    def run(self,x,y=None,mode='eval',lr=.008,lp=.008/3,route=.002,logits=False):
        if not self.handle:raise RuntimeError('Model is closed')
        if mode not in MODES:raise ValueError('Use eval, dfa or periodic')
        if any(not np.isfinite(v) or v<0 for v in (lr,lp,route)):raise ValueError('Rates must be finite and nonnegative')
        for a in (x,y):
            if a is not None:
                a=np.asarray(a)
                if a.ndim!=1 or a.dtype!=np.uint8:raise ValueError('Inputs must be one-dimensional uint8 arrays')
        x=np.ascontiguousarray(x);y=x.copy() if y is None else np.ascontiguousarray(y)
        if len(x)!=len(y) or len(x)>2**31-1:raise ValueError('Invalid input/target lengths')
        out=np.zeros((len(x),256) if logits else 1,np.float32)
        value=self.lib.run(self.handle,x.ctypes.data_as(U8P),y.ctypes.data_as(U8P),len(x),MODES[mode],lr,lp,route,int(logits),out.ctypes.data_as(FP))
        if not np.isfinite(value):raise FloatingPointError('Nonfinite native loss')
        return (value,out) if logits else value

    def state_arrays(self):
        arrays={}
        for key,(size,export,_) in STATE_ARRAYS.items():
            a=np.zeros(getattr(self.lib,size)(),np.uint8);getattr(self.lib,export)(self.handle,a.ctypes.data);arrays[key]=a
        return arrays

    def _restore(self,arrays):
        for key,(size,_,restore) in STATE_ARRAYS.items():
            a=np.ascontiguousarray(arrays[key],dtype=np.uint8)
            if a.ndim!=1 or len(a)!=getattr(self.lib,size)():raise ValueError(f'Incompatible native {key} layout')
            getattr(self.lib,restore)(self.handle,a.ctypes.data)

    def settings(self):
        return {k:dict(getattr(self,k)) for k in ('config','lace','state_config','probe','utility')}

    def _copy_base(self):
        return Model(self.arch,self.seed,self.flat,self.extra_flat,self.fixed,**self.settings(),
                     teaching=self.teaching,regional_gains=self.regional_gains,frozen_regions=self.frozen_regions,
                     task_route_lr_scale=self.task_route_lr_scale,a1_learning=self.a1_learning)

    def snapshot_copy(self):
        m=self._copy_base();m._restore(self.state_arrays());return m

    def means(self):
        a=np.zeros((18,self.width),np.float32);self.lib.probe_get_means(self.handle,a.ctypes.data_as(FP));return a
    def ymean(self):
        a=np.zeros(self.width,np.float32);self.lib.utility_mean(self.handle,a.ctypes.data_as(FP));return a
    def publications(self):
        a=np.zeros((18,self.width),np.float32);self.lib.publications(self.handle,a.ctypes.data_as(FP));return a
    def queries(self):
        a=np.zeros((2,self.width),np.float32);self.lib.query_vectors(self.handle,a.ctypes.data_as(FP));return a

    def frozen_copy(self):
        m=self._copy_base()
        a=self.means();m.lib.probe_set_means(m.handle,a.ctypes.data_as(FP))
        a=self.ymean();m.lib.utility_set_mean(m.handle,a.ctypes.data_as(FP));return m

    def reset(self):
        # Preserve fitted centering and policies while clearing stream state.
        q=self.means();y=self.ymean()
        self.lib.reset(self.handle)
        for key in ('probe_state','utility_state'):
            size,_,restore=STATE_ARRAYS[key];a=np.zeros(getattr(self.lib,size)(),np.uint8);getattr(self.lib,restore)(self.handle,a.ctypes.data)
        self._push('probe_configure',list(self.probe),self.probe);self._push('utility_configure',list(self.utility),self.utility)
        self.lib.probe_set_means(self.handle,q.ctypes.data_as(FP));self.lib.utility_set_mean(self.handle,y.ctypes.data_as(FP))

    def stats(self):
        e=max(self.arch.experts);a=np.zeros(18*(e+2)+5,np.float32);self.lib.info(self.handle,a.ctypes.data_as(FP))
        raw=a[36:36+18*e].reshape(18,e)
        counts=[raw[r,:n].tolist() for r,n in enumerate(self.arch.experts)]
        return dict(executions=a[:18].tolist(),local_mse=a[18:36].tolist(),expert_counts=counts,hc_reads=float(a[-5]),hc_writes=float(a[-4]),correct=float(a[-3]),observations=self.observations,state_bytes=int(a[-1]))

    def regional_totals(self):
        e=max(self.arch.experts);a=np.zeros(18*(e+2),np.float64);self.lib.regional_training_totals(self.handle,a.ctypes.data_as(DP))
        counts=a[36:].reshape(18,e)
        return dict(executions=a[:18].tolist(),local_mse_sum=a[18:36].tolist(),expert_counts=[counts[r,:n].tolist() for r,n in enumerate(self.arch.experts)])

    def update_totals(self):
        a=np.zeros((18,2,6),np.float64);self.lib.expert_update_metrics(self.handle,a.ctypes.data_as(DP));return a

    def observe_updates(self,previous=None):
        self.lib.expert_update_observe(self.handle,1)
        if previous is not None:
            a=np.ascontiguousarray(previous,dtype=np.float64)
            if a.shape!=(18,2,6) or not np.isfinite(a).all():raise ValueError('Invalid update observations')
            self.lib.expert_update_metrics_import(self.handle,a.ctypes.data_as(DP))

    def credit_totals(self):
        lace=np.zeros((18,12),np.float64);bottom=np.zeros((2,12),np.float64)
        self.lib.lace_metrics(self.handle,lace.ctypes.data_as(DP));self.lib.learned_metrics(self.handle,bottom.ctypes.data_as(DP))
        return dict(lace=lace.tolist(),bottom_temporal=bottom.tolist())

    def diagnostics(self):
        d=self.width;ncoord=18*4*d;nprod=18*4*d*d
        a=np.zeros(18+ncoord+nprod+18*8,np.float64);self.lib.diagnostic_export(self.handle,a.ctypes.data_as(DP))
        n=a[:18];s=a[18:18+ncoord].reshape(18,4,d);prod=a[18+ncoord:18+ncoord+nprod].reshape(18,4,d,d);ex=a[-18*8:].reshape(18,8)
        out={}
        for r,name in enumerate(NAMES):
            row=dict(samples=int(n[r]),margin=float(ex[r,0]/max(1,n[r])),saturated_h_fraction=float(ex[r,1]/max(1,n[r]*d)))
            for k,label in enumerate(('bottom','context','input','output')):
                mean=s[r,k]/max(1,n[r]);moment=prod[r,k]/max(1,n[r]);ev=np.maximum(0,np.linalg.eigvalsh(moment-np.outer(mean,mean)));p=ev/max(ev.sum(),1e-30)
                row[label]=dict(effective_rank=float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-30))))) if ev.sum()>1e-14 else 0.,variance=float(ev.sum()))
            out[name]=row
        return out

    def fingerprint(self):
        h=hashlib.sha256(json.dumps(self.arch.as_dict(),sort_keys=True).encode())
        for a in (self.flat,self.extra_flat,*self.state_arrays().values(),self.teaching,self.regional_gains,self.frozen_regions,self.a1_learning):h.update(a.tobytes())
        h.update(json.dumps(self.settings(),sort_keys=True).encode());h.update(np.float32(self.task_route_lr_scale).tobytes())
        for k,v in sorted(self.fixed.items()):h.update(k.encode());h.update(v.tobytes())
        return h.hexdigest()

    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(flat=self.flat,extra=self.extra_flat,**self.state_arrays(),version=np.array([20],np.int32),
                    architecture_json=np.array(json.dumps(self.arch.as_dict())),settings_json=np.array(json.dumps(self.settings())),
                    seed=np.array([self.seed],np.int64),teaching_scales=self.teaching,regional_gains=self.regional_gains,
                    frozen_regions=self.frozen_regions,task_route_lr_scale=np.array(self.task_route_lr_scale,np.float32),a1_learning_scales=self.a1_learning,
                    build_digest=np.array(self.build_digest),metadata_json=np.array(json.dumps(metadata or {},sort_keys=True)),
                    **{'fixed_'+k:v for k,v in self.fixed.items()})
        temporary=path.with_name(path.name+f'.pending.{os.getpid()}')
        try:
            with temporary.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            version=int(a['version'][0])
            if version==20:
                arch=Architecture(**json.loads(str(a['architecture_json'])));settings=json.loads(str(a['settings_json']));seed=int(a['seed'][0])
            elif version in (16,17,18,19):
                arch=Architecture(64,int(a['experts'][0]));settings={k:json.loads(str(a[k+'_json'])) for k in ('config','lace','state_config','probe','utility')};seed=1
            else:raise ValueError('Supported checkpoint versions: 16-20')
            m=cls(arch,seed,flat=a['flat'],extra=a['extra'],fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},**settings,
                  teaching=a['teaching_scales'] if 'teaching_scales' in a else None,
                  regional_gains=a['regional_gains'] if 'regional_gains' in a else None,
                  frozen_regions=a['frozen_regions'] if 'frozen_regions' in a else None,
                  task_route_lr_scale=float(a['task_route_lr_scale']) if 'task_route_lr_scale' in a else 1.,
                  a1_learning=a['a1_learning_scales'] if 'a1_learning_scales' in a else None)
            try:m._restore(a)
            except Exception:m.close();raise
            return m

    def close(self):
        if getattr(self,'handle',None):self.lib.destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):self.close()


def evaluate(model,data,chunk=8192,diagnostics=False,mean_y=False,block_recall=False,route_trace_stride=None):
    if len(data)<2:raise ValueError('Evaluation needs at least one target')
    if chunk<1 or (route_trace_stride is not None and route_trace_stride<1):raise ValueError('Invalid evaluation chunk')
    if route_trace_stride is not None:chunk=route_trace_stride
    with model.frozen_copy() as m:
        m.lib.diagnostic_enable(m.handle,int(diagnostics));m.lib.utility_ablate(m.handle,int(mean_y))
        if block_recall:m.lib.set_ablation(m.handle,3)
        total=0.;n=0;trace=[];codes=np.zeros((18,2*(m.width//64)+1),np.uint64)
        for start in range(0,len(data)-1,chunk):
            stop=min(len(data)-1,start+chunk);k=stop-start
            total+=k*m.run(data[start:stop],data[start+1:stop+1]);n+=k
            if route_trace_stride is not None:
                m.lib.lace_codes(m.handle,codes.ctypes.data_as(ct.POINTER(ct.c_uint64)));trace.append(codes[:,-1].tolist())
        stats=m.regional_totals();effective={}
        for name,counts in zip(NAMES,stats['expert_counts']):
            p=np.asarray(counts,np.float64)/max(1,sum(counts));effective[name]=float(np.exp(-sum(p*np.log(np.maximum(p,1e-30)))))
        out=dict(nll=total/n,targets=n,effective_experts=effective)
        if diagnostics:out['diagnostics']=m.diagnostics()
        if route_trace_stride is not None:out['routing_trace']=trace;out['trace_stride']=route_trace_stride
        return out
