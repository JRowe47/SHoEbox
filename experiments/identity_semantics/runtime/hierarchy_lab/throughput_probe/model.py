from pathlib import Path
import ctypes as ct, fcntl, hashlib, importlib.util, json, os, subprocess, threading
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('utility_parent_model',ROOT.parent/'scale_probe/model.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
core=parent.core
from reference_policy import REFERENCE_KERNEL, REFERENCE_PROFILE, settings_for
DEFAULT=dict(center_y=0.,split_norm=0.,residual=0.,upper_local_gain=1.,innovation=0.,mean_rate=.001,reserved1=0.,reserved2=0.)
FIELDS=list(DEFAULT)
A1_FIELDS=['prototype_scale','follow_cooldown','reference_lr','decoder_scale']
A1_DEFAULT=np.array([1.,0.,4e-5,1.],np.float32)
KERNEL=os.environ.get('SHOEBOX_KERNEL',REFERENCE_KERNEL)
FLAGS=os.environ.get('SHOEBOX_CXX_FLAGS','').split()
COMPLETION_KERNELS=('completion','completion_checks')
MESSAGE_KERNELS=('messages','message_checks',*COMPLETION_KERNELS)
DIAGNOSTIC_KERNELS=('contributions','message_checks','completion_checks')
RELAY_KERNELS=('relays','contributions',*MESSAGE_KERNELS)
MESSAGE_DEFAULT=dict(payload_gain=1.,feedback_activation=.1,feedback_routing=.1,semantic_lr=1.)
CONTRIBUTION_KINDS=('none','neural_output','content_descriptor','relay_context',
                    'retained_input','semantic_residual','whole_source','expert_transform','semantic_payload','feedback_channel','content_expert')
_CONSTRUCTION_LOCK=threading.RLock()
def build():
    source=ROOT/(KERNEL+'.cpp');digest=hashlib.sha256(source.read_bytes()+((ROOT/'composition.cpp').read_bytes() if KERNEL in ('binding','semantics','semantics_fixed',*RELAY_KERNELS) else b'')+((ROOT/'relays.cpp').read_bytes() if KERNEL in ('contributions',*MESSAGE_KERNELS) else b'')+((ROOT/'messages.cpp').read_bytes() if KERNEL in ('message_checks',*COMPLETION_KERNELS) else b'')+((ROOT/'completion.cpp').read_bytes() if KERNEL=='completion_checks' else b'')+b''.join(p.read_bytes() for p in sorted(ROOT.glob('*.inc')))+((ROOT/'e14.cpp').read_bytes() if KERNEL=='e14_fixture' else b'')+str(FLAGS).encode()).hexdigest()[:16]
    lib=ROOT/f'_{KERNEL}_e{core.E}_{digest}.so'
    with open(ROOT/f'.build_e{core.E}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not lib.exists():
            tmp=lib.with_suffix(f'.{os.getpid()}.tmp')
            subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread',f'-DSHOEBOX_EXPERTS={core.E}',*FLAGS,'-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
            os.replace(tmp,lib)
    return lib
class Model(parent.Model):
    def __init__(self,*args,utility=None,kernel=None,teaching=None,regional_gains=None,frozen_regions=None,task_route_lr_scale=1.,a1_learning=None,f3_task_route_gain=1.,exploration=None,traffic=None,composition=None,messages=None,local_credit=None,content_experts=None,pattern_classes=None,paired_memory=None,**kwargs):
        global KERNEL
        with _CONSTRUCTION_LOCK:
            previous=KERNEL
            try:
                if kernel is not None:KERNEL=kernel
                if KERNEL in COMPLETION_KERNELS:
                    settings=settings_for(core.E,KERNEL)
                    args=list(args)
                    for name,index in (('config',0),('lace',4),('state_config',5)):
                        if len(args)>index:args[index]={**settings[name],**(args[index] or {})}
                        else:kwargs[name]={**settings[name],**(kwargs.get(name) or {})}
                    kwargs['probe']={**settings['probe'],**(kwargs.get('probe') or {})}
                    utility={**settings['utility'],**(utility or {})}
                    def argument(name,index,default=None):
                        return args[index] if len(args)>index else kwargs.get(name,default)
                    def supply(name,index,value):
                        if len(args)>index:args[index]=value
                        else:kwargs[name]=value
                    if argument('params',2) is None:
                        supply('params',2,core.initialize(argument('seed',1,1))[0])
                    if argument('fixed',3) is None:
                        size='1m' if core.E==12 else '5m'
                        with np.load(ROOT.parents[1]/'configs'/(size+'_fixed.npz'),allow_pickle=False) as arrays:
                            supply('fixed',3,{key:arrays[key].copy() for key in arrays.files})
                    if (composition or {}).get('enabled',True) and any(kwargs['probe'][k] for k in ('port_mode','port_gain','balance')):
                        raise ValueError('Completion has no port_mode/port_gain/balance implementation; use neutral zeros')
                self._initialize(*args,utility=utility,**kwargs)
                self.configure_teaching(np.ones((18,2),np.float32) if teaching is None else teaching)
                self.configure_regional_gains(np.ones(18,np.float32) if regional_gains is None else regional_gains)
                self.configure_training_policy(np.zeros(18,np.int32) if frozen_regions is None else frozen_regions,task_route_lr_scale)
                self.configure_a1_learning(a1_learning)
                self.configure_f3_task_route(f3_task_route_gain)
                self.configure_path_prototypes()
                self.configure_precision()
                self.configure_boundaries()
                self.configure_exploration(exploration)
                self.configure_traffic(traffic)
                self.configure_composition(composition)
                self.configure_messages(messages)
                self.configure_local_credit(self.runtime_kernel in COMPLETION_KERNELS if local_credit is None else local_credit)
                self.configure_content_experts(self.runtime_kernel in COMPLETION_KERNELS if content_experts is None else content_experts)
                self.configure_pattern_classes(self.runtime_kernel in COMPLETION_KERNELS if pattern_classes is None else pattern_classes)
                self.configure_paired_memory(self.runtime_kernel in COMPLETION_KERNELS if paired_memory is None else paired_memory)
            finally:KERNEL=previous
    def _initialize(self,*args,utility=None,**kwargs):
        self.runtime_kernel=KERNEL
        self.utility={**DEFAULT,**(utility or {})}
        if not 0<self.utility['mean_rate']<=1 or self.utility['upper_local_gain']<0:raise ValueError('invalid utility policy')
        previous=parent.build
        try:parent.build=build;super().__init__(*args,**kwargs)
        finally:parent.build=previous
        fp=ct.POINTER(ct.c_float)
        self.lib.utility_configure.argtypes=[ct.c_void_p,fp]
        self.lib.utility_state_size.restype=ct.c_size_t
        for n in ['utility_export','utility_import']:getattr(self.lib,n).argtypes=[ct.c_void_p,ct.c_void_p]
        for n in ['utility_mean','utility_set_mean']:getattr(self.lib,n).argtypes=[ct.c_void_p,fp]
        for n in ['utility_ablate','utility_observe']:getattr(self.lib,n).argtypes=[ct.c_void_p,ct.c_int]
        self.lib.utility_metrics.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        cfg=np.array([self.utility[k] for k in FIELDS],np.float32)
        self.lib.utility_configure(self.handle,cfg.ctypes.data_as(fp))
        self._utility_ready=True
    def configure_composition(self,policy=None):
        cfg=dict(enabled=self.runtime_kernel in ('composition','binding','semantics','semantics_fixed',*RELAY_KERNELS),threshold=.35,min_count=3,max_children=8,context_gain=.2)
        if policy is not None:
            if set(policy)-set(cfg):raise ValueError('Unknown composition policy field')
            cfg.update(policy)
        if not isinstance(cfg['enabled'],(bool,np.bool_)):raise ValueError('enabled must be boolean')
        for key in ('min_count','max_children'):
            if isinstance(cfg[key],bool) or int(cfg[key])!=cfg[key]:raise ValueError(key+' must be an integer')
        if not np.isfinite([cfg['threshold'],cfg['context_gain']]).all():raise ValueError('Nonfinite composition policy')
        if hasattr(self.lib,'composition_configure'):
            self.lib.composition_configure.argtypes=[ct.c_void_p,ct.c_int,ct.c_float,ct.c_int,ct.c_int,ct.c_float]
            self.lib.composition_configure.restype=ct.c_int
            if not self.lib.composition_configure(self.handle,int(cfg['enabled']),cfg['threshold'],int(cfg['min_count']),int(cfg['max_children']),cfg['context_gain']):raise ValueError('Invalid composition policy')
        elif cfg['enabled']:raise ValueError('Composition kernel required')
        self.composition=cfg
    def configure_messages(self,policy=None):
        cfg=dict(MESSAGE_DEFAULT if self.runtime_kernel in MESSAGE_KERNELS else dict.fromkeys(MESSAGE_DEFAULT,0.))
        if policy is not None:
            if set(policy)-set(cfg):raise ValueError('Unknown learned-message policy field')
            cfg.update(policy)
        values=np.array(list(cfg.values()),np.float32)
        if not np.isfinite(values).all() or np.any(values<0) or np.any(values[:3]>1) or values[3]>8:
            raise ValueError('Message gains must be in [0,1], semantic_lr in [0,8]')
        if hasattr(self.lib,'message_configure'):
            self.lib.message_configure.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.message_configure.restype=ct.c_int
            if not self.lib.message_configure(self.handle,values.ctypes.data):raise ValueError('Message policy requires a fresh stream')
        elif np.any(values):raise ValueError('Learned-message kernel required')
        self.messages=cfg
    def message_metrics(self):
        if not hasattr(self.lib,'message_metrics'):return {}
        a=np.zeros((18,4),np.float64);self.lib.message_metrics.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.message_metrics(self.handle,a.ctypes.data)
        fields=('semantic_deliveries','semantic_gradient_norm','feedback_deliveries','missing_feedback_producers')
        return {name:dict(zip(fields,row.tolist())) for name,row in zip(core.NAMES,a)}
    def completion_metrics(self):
        if not hasattr(self.lib,'completion_metrics'):return {}
        a=np.zeros((18,5),np.uint64);self.lib.completion_metrics.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.completion_metrics(self.handle,a.ctypes.data)
        fields=('semantic_nodes','expired_semantic_handles','semantic_budget_omissions','completed_sequences','semantic_capacity_prunes')
        return {name:dict(zip(fields,map(int,row))) for name,row in zip(core.NAMES,a)}
    def completion_gates(self):
        if not hasattr(self.lib,'completion_gates'):return np.zeros(0,np.float32)
        a=np.zeros(16,np.float32);self.lib.completion_gates.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.completion_gates(self.handle,a.ctypes.data);return a
    def completion_clocks(self):
        if not hasattr(self.lib,'completion_clocks'):return {}
        a=np.zeros((18,2),np.int64);self.lib.completion_clocks.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.completion_clocks(self.handle,a.ctypes.data)
        return {name:tuple(map(int,row)) for name,row in zip(core.NAMES,a)}
    def completion_capacity(self):
        if not hasattr(self.lib,'completion_capacity'):return 0
        self.lib.completion_capacity.argtypes=[ct.c_void_p];self.lib.completion_capacity.restype=ct.c_size_t
        return int(self.lib.completion_capacity(self.handle))
    def acceptance_metrics(self):
        """Publication distributions and objective calls since reset/import.

        These observations do not enter checkpoints or fingerprints. Span bin 0
        is zero; bin b>0 is [2**(b-1), 2**b-1]. Temporal lengths count local
        frames in L2-G5, excluding relays and EC/Y source-role arity.
        """
        if not hasattr(self.lib,'acceptance_status'):return {}
        arrays=[np.zeros((18,n),np.uint64) for n in (13,65,65,9,9,2)]
        self.lib.acceptance_status.argtypes=[ct.c_void_p]*7
        self.lib.acceptance_status(self.handle,*(a.ctypes.data for a in arrays))
        names=('publications','primary_span_sum','primary_span_max','coverage_span_sum',
               'coverage_span_max','formed','recognized','provisional','relays',
               'invalid_primary_spans','invalid_coverage_spans','primary_outside_coverage','unclocked_fusions')
        return {name:{**dict(zip(names,map(int,arrays[0][r]))),
                      'primary_span_histogram':arrays[1][r].tolist(),
                      'coverage_span_histogram':arrays[2][r].tolist(),
                      'sequence_length_histogram':arrays[3][r].tolist(),
                      'recognized_sequence_length_histogram':arrays[4][r].tolist(),
                      'language_dfa_update_calls':int(arrays[5][r,0]),
                      'local_update_calls':int(arrays[5][r,1])}
                for r,name in enumerate(core.NAMES)}
    def reset_acceptance_metrics(self):
        if not hasattr(self.lib,'acceptance_reset'):raise ValueError('Completion kernel required')
        self.lib.acceptance_reset.argtypes=[ct.c_void_p]
        self.lib.acceptance_reset(self.handle)
    def completion_storage(self):
        """Live record payload bytes, excluding allocator and scratch overhead."""
        if not hasattr(self.lib,'completion_storage_status'):return {}
        self.lib.completion_storage_status.argtypes=[ct.c_void_p,ct.c_void_p]
        out=np.zeros(10,np.uint64)
        self.lib.completion_storage_status(self.handle,out.ctypes.data)
        names=('input_bytes','legacy_input_bytes','upper_record_bytes','legacy_record_bytes',
               'composition_bytes','legacy_composition_bytes','archived_records',
               'temporal_records','archive_payload_bytes','legacy_archive_payload_bytes')
        return dict(zip(names,map(int,out)))

    def publication_storage(self):
        """Shared publication payloads, sparse gradients and role-cache reuse."""
        if not hasattr(self.lib,'publication_storage_status'):return {}
        self.lib.publication_storage_status.argtypes=[ct.c_void_p,ct.c_void_p]
        out=np.zeros(16,np.uint64)
        self.lib.publication_storage_status(self.handle,out.ctypes.data)
        names=('event_handle_bytes','legacy_event_bytes','content_payload_bytes','context_payload_bytes',
               'content_references','context_references','unique_content_payloads','unique_context_payloads',
               'neural_gradient_blocks','input_gradient_blocks','gradient_payload_bytes','role_cache_bytes',
               'content_cache_hits','content_cache_misses','context_cache_hits','context_cache_misses')
        return dict(zip(names,map(int,out)))

    def neural_state(self):
        if not hasattr(self.lib,'neural_state_size'):return np.zeros(0,np.uint8)
        self.lib.neural_state_size.argtypes=[ct.c_void_p];self.lib.neural_state_size.restype=ct.c_size_t
        self.lib.neural_export.argtypes=[ct.c_void_p,ct.c_void_p]
        out=np.zeros(self.lib.neural_state_size(self.handle),np.uint8)
        self.lib.neural_export(self.handle,out.ctypes.data);return out
    def import_neural(self,data):
        data=np.ascontiguousarray(data,dtype=np.uint8)
        if not hasattr(self.lib,'neural_import'):
            if data.size:raise ValueError('Neural ownership requires the completion kernel')
            return
        self.lib.neural_import.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];self.lib.neural_import.restype=ct.c_int
        if not self.lib.neural_import(self.handle,data.ctypes.data,data.size):raise ValueError('Invalid neural ownership snapshot')
    def neural_metrics(self):
        if not hasattr(self.lib,'neural_metrics'):return {}
        out=np.zeros((18,5),np.uint64);self.lib.neural_metrics.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.neural_metrics(self.handle,out.ctypes.data)
        fields=('archived','reclaimed','peak_archived','missing_live_dependencies','live_archived')
        return {name:dict(zip(fields,map(int,row))) for name,row in zip(core.NAMES,out)}
    def configure_local_credit(self,enabled=True):
        if not isinstance(enabled,(bool,np.bool_)):raise ValueError('local_credit must be boolean')
        if hasattr(self.lib,'local_credit_configure'):
            self.lib.local_credit_configure.argtypes=[ct.c_void_p,ct.c_int]
            self.lib.local_credit_configure(self.handle,int(enabled))
        elif enabled:raise ValueError('Local neural prediction credit requires the completion kernel')
        self.local_credit=bool(enabled)
    def local_credit_metrics(self):
        """Per-objective diagnostics; reset on stream reset or snapshot load."""
        if not hasattr(self.lib,'local_credit_status'):return {}
        out=np.zeros((18,8),np.uint64);self.lib.local_credit_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.local_credit_status(self.handle,out.ctypes.data)
        fields=('objectives','neural_deliveries','feedback_deliveries','input_deliveries',
                'neural_updates','input_updates','relay_hops','missing_producers')
        return {name:dict(zip(fields,map(int,row))) for name,row in zip(core.NAMES,out)}
    def configure_content_workers(self,count=1,cpus=None):
        """Transient CPU execution setting; checkpoints keep the learning policy."""
        if isinstance(count,(bool,np.bool_)) or int(count)!=count or not 1<=count<=32:
            raise ValueError('count must be an integer from 1 to 32')
        if not hasattr(self.lib,'content_workers_configure'):
            if count!=1:raise ValueError('Content workers require the completion runtime')
            return
        values=None
        if cpus is not None:
            cpus=list(cpus)
            if len(cpus)!=count or any(isinstance(x,(bool,np.bool_)) or int(x)!=x for x in cpus):
                raise ValueError('cpus must contain one integer CPU ID per worker')
            if len(set(cpus))!=count or not set(cpus)<=set(os.sched_getaffinity(0)):
                raise ValueError('cpus must be distinct and available to this process')
            values=np.array(cpus,dtype=np.int32)
        self.lib.content_workers_configure.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
        self.lib.content_workers_configure.restype=ct.c_int
        if not self.lib.content_workers_configure(self.handle,int(count),None if values is None else values.ctypes.data):
            raise ValueError('Invalid content worker configuration')

    def configure_content_experts(self,enabled=True):
        if not isinstance(enabled,(bool,np.bool_)):raise ValueError('content_experts must be boolean')
        if hasattr(self.lib,'content_expert_configure'):
            self.lib.content_expert_configure.argtypes=[ct.c_void_p,ct.c_int]
            self.lib.content_expert_configure(self.handle,int(enabled))
        elif enabled:raise ValueError('Content experts require the completion kernel')
        self.content_experts=bool(enabled)
    def content_expert_state(self):
        if not hasattr(self.lib,'content_expert_state_size'):return np.zeros(0,np.uint8)
        self.lib.content_expert_state_size.argtypes=[ct.c_void_p];self.lib.content_expert_state_size.restype=ct.c_size_t
        self.lib.content_expert_export.argtypes=[ct.c_void_p,ct.c_void_p]
        out=np.zeros(self.lib.content_expert_state_size(self.handle),np.uint8)
        self.lib.content_expert_export(self.handle,out.ctypes.data);return out
    def import_content_experts(self,data):
        data=np.ascontiguousarray(data,dtype=np.uint8)
        if not hasattr(self.lib,'content_expert_import'):
            if data.size:raise ValueError('Content eligibility requires the completion kernel')
            return
        self.lib.content_expert_import.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];self.lib.content_expert_import.restype=ct.c_int
        if not self.lib.content_expert_import(self.handle,data.ctypes.data,data.size):raise ValueError('Invalid content-expert eligibility snapshot')
    def content_expert_metrics(self):
        if not hasattr(self.lib,'content_expert_status'):return {}
        out=np.zeros((18,6),np.uint64);self.lib.content_expert_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.content_expert_status(self.handle,out.ctypes.data)
        fields=('encodes','updates','local_updates','task_updates','router_updates','missing_producers')
        return {name:dict(zip(fields,map(int,row))) for name,row in zip(core.NAMES,out)}
    def configure_pattern_classes(self,enabled=True):
        """Change recognition policy only at a clean stream boundary."""
        if not isinstance(enabled,(bool,np.bool_)):raise ValueError('pattern_classes must be boolean')
        if hasattr(self.lib,'pattern_configure'):
            self.lib.pattern_configure.argtypes=[ct.c_void_p,ct.c_int]
            self.lib.pattern_configure.restype=ct.c_int
            if not self.lib.pattern_configure(self.handle,int(enabled)):
                raise ValueError('Pattern-class policy requires a fresh stream boundary')
        elif enabled:raise ValueError('Pattern classes require the completion kernel')
        self.pattern_classes=bool(enabled)
    def pattern_state(self,fitted=False):
        if not hasattr(self.lib,'pattern_state_size'):return np.zeros(0,np.uint8)
        self.lib.pattern_state_size.argtypes=[ct.c_void_p,ct.c_int];self.lib.pattern_state_size.restype=ct.c_size_t
        self.lib.pattern_export.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]
        out=np.zeros(self.lib.pattern_state_size(self.handle,int(fitted)),np.uint8)
        self.lib.pattern_export(self.handle,out.ctypes.data,int(fitted));return out
    def import_pattern_classes(self,data):
        data=np.ascontiguousarray(data,np.uint8)
        if not hasattr(self.lib,'pattern_import'):
            if data.size:raise ValueError('Pattern state requires the completion kernel')
            return
        self.lib.pattern_import.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];self.lib.pattern_import.restype=ct.c_int
        if not self.lib.pattern_import(self.handle,data.ctypes.data,data.size):raise ValueError('Invalid pattern-class state')
    def pattern_metrics(self):
        if not hasattr(self.lib,'pattern_status'):return {}
        out=np.zeros((18,18),np.uint64);self.lib.pattern_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.pattern_status(self.handle,out.ctypes.data)
        fields=('observed','created','reused','promoted','recognized','provisional','evicted',
                'unknown_eval','transitions','class_hits','forecasts','forecast_hits',
                'sequence_publications','primary_span_sum','sequence_frames','table_evictions',
                'dictionary_size','dictionary_observations')
        return {name:dict(zip(fields,map(int,row))) for name,row in zip(core.NAMES,out)}
    def pattern_events(self):
        if not hasattr(self.lib,'pattern_events'):return {}
        out=np.zeros((18,2),np.uint64);self.lib.pattern_events.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.pattern_events(self.handle,out.ctypes.data)
        return {name:tuple(map(int,row)) for name,row in zip(core.NAMES,out)}
    def configure_paired_memory(self,enabled=True):
        """Select the episodic contract at a fresh stream with empty memory."""
        if not isinstance(enabled,(bool,np.bool_)):raise ValueError('paired_memory must be boolean')
        if hasattr(self.lib,'paired_memory_configure'):
            self.lib.paired_memory_configure.argtypes=[ct.c_void_p,ct.c_int]
            self.lib.paired_memory_configure.restype=ct.c_int
            if not self.lib.paired_memory_configure(self.handle,int(enabled)):
                raise ValueError('Paired-memory policy requires a fresh stream and empty memory')
        elif enabled:raise ValueError('Paired memory requires the completion kernel')
        self.paired_memory=bool(enabled)
    def paired_memory_state(self):
        if not hasattr(self.lib,'paired_memory_state_size'):return np.zeros(0,np.uint8)
        self.lib.paired_memory_state_size.argtypes=[ct.c_void_p];self.lib.paired_memory_state_size.restype=ct.c_size_t
        self.lib.paired_memory_export.argtypes=[ct.c_void_p,ct.c_void_p]
        out=np.zeros(self.lib.paired_memory_state_size(self.handle),np.uint8)
        self.lib.paired_memory_export(self.handle,out.ctypes.data);return out
    def import_paired_memory(self,data):
        data=np.ascontiguousarray(data,np.uint8)
        if not hasattr(self.lib,'paired_memory_import'):
            if data.size:raise ValueError('Paired state requires the completion kernel')
            return
        self.lib.paired_memory_import.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];self.lib.paired_memory_import.restype=ct.c_int
        if not self.lib.paired_memory_import(self.handle,data.ctypes.data,data.size):raise ValueError('Invalid paired-memory state')
    def paired_memory_metrics(self):
        if not hasattr(self.lib,'paired_memory_status'):return {}
        out=np.zeros(27,np.uint64);self.lib.paired_memory_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.paired_memory_status(self.handle,out.ctypes.data)
        fields=('content_publications','context_publications','candidate_pairs','admissions','replacements','reinforcements',
                'novelty_rejected','priority_rejected','joint_reads','comparisons','hits','utility_credit',
                'retired_utility_handles','incomplete_reads','cached_reads','incomplete_writes',
                'occupied','serial','stream_epoch','returned_episode','read_tick',
                'content_stream','content_seq','content_tick','context_stream','context_seq','context_tick')
        return dict(zip(fields,map(int,out)))
    def capabilities(self):
        """Report effective mechanisms, including controls absent from this kernel."""
        completion=self.runtime_kernel in COMPLETION_KERNELS and self.composition['enabled']
        reference=completion and all(all(getattr(self,group).get(k)==v for k,v in policy.items())
            for group,policy in settings_for(core.E).items()) and self.messages==MESSAGE_DEFAULT and self.local_credit and self.content_experts and self.pattern_classes and self.paired_memory
        return dict(kernel=self.runtime_kernel,reference_profile=REFERENCE_PROFILE if reference else 'custom' if completion else None,
            checkpoint_version=39 if self.runtime_kernel in COMPLETION_KERNELS else None,
            composition_abi=10 if self.runtime_kernel in COMPLETION_KERNELS else None,
            features=dict(multi_source_binding=self.composition['enabled'] and self.runtime_kernel in ('binding','semantics','semantics_fixed',*RELAY_KERNELS),
                relay_provenance=self.runtime_kernel in RELAY_KERNELS and self.composition['enabled'],
                learned_completion=completion,primary_occurrence_clock=completion,
                retained_neural_producers=completion,
                acceptance_observations=completion,
                local_neural_prediction_credit=completion and self.local_credit and self.lace['depth']>0,
                expert_constituent_content=completion and self.content_experts,
                reusable_pattern_classes=completion and self.pattern_classes,
                class_predictive_completion=completion and self.pattern_classes,
                paired_episodic_memory=completion and self.paired_memory,
                recursive_constituent_content=completion and self.messages['payload_gain']>0,
                semantic_credit=self.composition['enabled'] and self.messages['semantic_lr']>0,
                feedback_activation=self.composition['enabled'] and self.messages['feedback_activation']>0,
                feedback_routing=self.composition['enabled'] and self.messages['feedback_routing']>0,
                lace_credit=self.lace['depth']>0,retained_input_credit=self.composition['enabled'] and self.lace['depth']>0,
                learned_bottom_state=bool(self.state_config['enabled']),
                bottom_task_routing=self.state_config['task_gain']>0,
                prototype_ema=bool(self.state_config['ema']),prototype_revival=self.state_config['revive_every']>0,
                ec_content_recall=bool(self.config['port'] and self.config['gain'] and not self.config['no_returns']),
                ec_context_recall=bool(self.config['port'] and self.config['gain'] and not self.config['no_returns']),
                memory_key_trace=self.config['key_mix']>0,direct_feedback_alignment=self.lace['dfa_gain']>0,
                residual_a1_objective=self.utility['residual']>0,local_innovation_objective=self.utility['innovation']>0,
                full_temporal_credit=self.state_config['credit_stride']==1 and self.state_config['temporal_features']==64,
                full_deep_credit=self.lace['deep_stride']==1 and self.lace['credit_features']==64),
            unavailable_experiment_controls=['exploration','traffic_bias','ec_activation','gate_bias','path_prototypes','precision','boundary_policy','f3_task_multiplier'] if completion else [],
            inactive_legacy_controls=['port_mode','port_gain','balance','route_mode','center_rate','nominal_cadence'] if completion else [],
            semantic_credit_capacity=self.completion_capacity(),
            memory_contract=('paired EC content/context episodes; joint retrieval and asynchronous latest-completed pairing'
                             if completion and self.paired_memory else 'two independent EC associative transition channels'))
    def composition_state(self,fitted=False):
        if not hasattr(self.lib,'composition_state_size'):return np.zeros(0,np.uint8)
        self.lib.composition_state_size.restype=ct.c_size_t
        self.lib.composition_state_size.argtypes=[ct.c_void_p]
        self.lib.composition_fitted_size.restype=ct.c_size_t
        self.lib.composition_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.composition_state_size(self.handle),np.uint8);self.lib.composition_export(self.handle,a.ctypes.data)
        return a[:self.lib.composition_fitted_size()].copy() if fitted else a
    def import_composition(self,state,fitted=False):
        a=np.ascontiguousarray(state,np.uint8)
        if not a.size:return
        if not hasattr(self.lib,'composition_import'):raise ValueError('Composition state requires composition kernel')
        self.lib.composition_import.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
        self.lib.composition_import.restype=ct.c_int
        if not self.lib.composition_import(self.handle,a.ctypes.data,a.size,int(fitted)):raise ValueError('Composition state ABI or policy mismatch')
    def composition_metrics(self):
        a=np.zeros((18,20),np.float64)
        if hasattr(self.lib,'composition_metrics'):
            self.lib.composition_metrics.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.composition_metrics(self.handle,a.ctypes.data)
        names=('consumed','published','compounds','recognized','provisional','held','table_evictions','span_sum','input_credit_received','input_credit_forwarded','expired_input_edges','overwritten_input_credit','identity_credit_relays','singleton_relays','bound_frames','held_roles','fresh_roles','startup_holds')
        return {name:dict(zip(names,a[r,:len(names)].tolist())) for r,name in enumerate(core.NAMES)}
    def content_events(self):
        if not hasattr(self.lib,'composition_events'):raise ValueError('Composition kernel required')
        a=np.zeros((18,84),np.uint64);codes=np.zeros((18,64),np.float32)
        self.lib.composition_events.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.lib.composition_events(self.handle,a.ctypes.data,codes.ctypes.data)
        result={}
        for r,name in enumerate(core.NAMES):
            v=a[r];children=v[12:].reshape(8,9)[:int(v[6])]
            result[name]=dict(seq=int(v[0]),id=(int(v[1]),int(v[2])),span=(int(v[3]),int(v[4])),tick=int(v[5]),count=int(v[6]),recognized=bool(v[7]),kind=int(v[8]),pending=int(v[9]),seen=int(v[10]),bytes=int(v[11]),children=children.copy(),code=codes[r].copy())
        return result
    def semantic_parameters(self):
        if not hasattr(self.lib,'semantic_parameters'):return np.zeros((0,),np.float32)
        a=np.zeros((16,64,8),np.float32)
        self.lib.semantic_parameters.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.semantic_parameters(self.handle,a.ctypes.data);return a
    def relay_teaching_status(self):
        if not hasattr(self.lib,'relay_teaching_status'):return {}
        a=np.zeros((18,8),np.float64)
        self.lib.relay_teaching_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.relay_teaching_status(self.handle,a.ctypes.data)
        names=('scheduled','neural_targets','relay_targets','nonlearning_targets','expired_producers','duplicate_schedules','pending_tick','pending_index')
        return {name:dict(zip(names,map(int,a[r]))) for r,name in enumerate(core.NAMES)}
    def relay_event_status(self):
        if not hasattr(self.lib,'relay_event_status'):return {}
        a=np.zeros((18,8),np.uint64);confidence=np.zeros(18,np.float32)
        self.lib.relay_event_status.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.lib.relay_event_status(self.handle,a.ctypes.data,confidence.ctypes.data)
        def signed(v):return int(v) if v<2**63 else int(v)-2**64
        return {name:dict(local_arity=int(a[r,0]),relayed=bool(a[r,1]),confidence=float(confidence[r]),
                         origin=dict(region=signed(a[r,2]),tick=int(a[r,3]),index=signed(a[r,4]),
                                     seq=int(a[r,5]),id=(int(a[r,6]),int(a[r,7])))) for r,name in enumerate(core.NAMES)}
    def semantic_updates(self):
        if not hasattr(self.lib,'semantic_updates'):return {}
        a=np.zeros(16,np.uint64);self.lib.semantic_updates.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.semantic_updates(self.handle,a.ctypes.data)
        return dict(zip(core.NAMES[2:],map(int,a)))
    def binding_role_status(self):
        if not hasattr(self.lib,'binding_role_status'):return {}
        a=np.zeros((18,3),np.uint64)
        self.lib.binding_role_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.binding_role_status(self.handle,a.ctypes.data)
        return {name:dict(present=int(a[r,0]),fresh=int(a[r,1]),kind=int(a[r,2])) for r,name in enumerate(core.NAMES)}
    def finish_document(self,mode='eval',lr=.12,lp=.04,route=.002,reset=False):
        if self.runtime_kernel in DIAGNOSTIC_KERNELS and mode!='eval':raise ValueError('Contribution checks cannot train at document boundaries')
        if self.composition['enabled']:
            if mode not in core.MODES or mode=='spatial':raise ValueError('Unsupported composition training mode')
            self.lib.composition_finish.argtypes=[ct.c_void_p,ct.c_int,ct.c_float,ct.c_float,ct.c_float]
            self.lib.composition_finish.restype=ct.c_int
            if not self.lib.composition_finish(self.handle,core.MODES[mode],lr,lp,route):raise RuntimeError('Pending composition events did not drain')
        if reset:self.lib.reset_context(self.handle)
    def ablate(self,value=0):
        if self.runtime_kernel in DIAGNOSTIC_KERNELS and value!=0:raise ValueError('Use configure_contribution on a fresh diagnostic copy')
        return super().ablate(value)
    def run(self,x,y=None,mode='eval',lr=.12,lp=.04,route=.002,logits=False):
        if self.runtime_kernel in DIAGNOSTIC_KERNELS and (mode!='eval' or not self.composition['enabled']):
            raise ValueError('Contribution checks require evaluation with composition enabled')
        if self.composition['enabled'] and mode=='spatial':raise ValueError('Legacy spatial mode has no event/state credit; use periodic or hybrid')
        return super().run(x,y,mode=mode,lr=lr,lp=lp,route=route,logits=logits)
    def configure_activation(self,width,scale,ramp=131072):
        if not hasattr(self.lib,'e13_configure'):raise ValueError('EC-C activation needs e13 kernel')
        self.lib.e13_configure.argtypes=[ct.c_void_p,ct.c_float,ct.c_float,ct.c_int64];self.lib.e13_configure.restype=ct.c_int
        if not self.lib.e13_configure(self.handle,float(width),float(scale),int(ramp)):raise ValueError('Invalid EC-C activation')
    def activation_policy(self):
        if not hasattr(self.lib,'e13_policy'):return dict(enabled=False)
        self.lib.e13_policy.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        a=np.zeros(6,np.float64);self.lib.e13_policy(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        return dict(enabled=bool(a[0]),width=float(a[1]),scale=float(a[2]),mix=float(a[3]),trained=int(a[4]),ramp=int(a[5])) if a[0] else dict(enabled=False)
    def activation_state(self):
        if not hasattr(self.lib,'e13_state_size'):return np.zeros(0,np.uint8)
        self.lib.e13_state_size.restype=ct.c_size_t;self.lib.e13_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.e13_state_size(),np.uint8);self.lib.e13_export(self.handle,a.ctypes.data);return a
    def import_activation(self,state,fresh=False):
        a=np.ascontiguousarray(state,np.uint8)
        if not a.size:return
        self.lib.e13_state_size.restype=ct.c_size_t
        if a.size!=self.lib.e13_state_size():raise ValueError('EC-C activation state ABI mismatch')
        self.lib.e13_import.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.e13_import(self.handle,a.ctypes.data)
        if fresh:
            self.lib.e13_clear_cache.argtypes=[ct.c_void_p];self.lib.e13_clear_cache(self.handle)
    @property
    def parameter_count(self):return super().parameter_count+(16 if self.runtime_kernel in COMPLETION_KERNELS else 0)+(8192 if self.runtime_kernel.startswith("semantics") or self.runtime_kernel in RELAY_KERNELS else 0)+((core.E if self.runtime_kernel.startswith('r02') else 16*core.E) if self.gate_bias_policy()['enabled'] else 0)
    def configure_gate_bias(self,gain=250.,limit=2.,step_limit=.01):
        if not hasattr(self.lib,'e14_configure'):raise ValueError('Gate bias requires e14 kernel')
        self.lib.e14_configure.argtypes=[ct.c_void_p,ct.c_float,ct.c_float,ct.c_float];self.lib.e14_configure.restype=ct.c_int
        if not self.lib.e14_configure(self.handle,gain,limit,step_limit):raise ValueError('Invalid gate bias policy')
    def gate_bias_policy(self):
        if not hasattr(self.lib,'e14_policy'):return dict(enabled=False)
        self.lib.e14_policy.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];a=np.zeros(4,np.float32);self.lib.e14_policy(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)))
        return dict(enabled=True,gain=float(a[1]),limit=float(a[2]),step_limit=float(a[3])) if a[0] else dict(enabled=False)
    def gate_bias_state(self):
        if not hasattr(self.lib,'e14_state_size'):return np.zeros(0,np.uint8)
        self.lib.e14_state_size.restype=ct.c_size_t;self.lib.e14_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.e14_state_size(),np.uint8);self.lib.e14_export(self.handle,a.ctypes.data);return a
    def import_gate_bias(self,state):
        a=np.ascontiguousarray(state,np.uint8)
        if not a.size:return
        self.lib.e14_state_size.restype=ct.c_size_t
        if a.size!=self.lib.e14_state_size():raise ValueError('Gate bias state ABI mismatch')
        self.lib.e14_import.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.e14_import(self.handle,a.ctypes.data)
    def gate_biases(self):
        a=np.zeros((16,core.E),np.float32)
        if not hasattr(self.lib,'e14_biases'):return a
        self.lib.e14_biases.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];self.lib.e14_biases(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return a
    def set_gate_biases(self,values):
        a=np.ascontiguousarray(values,np.float32)
        if a.shape!=(16,core.E) or not np.isfinite(a).all():raise ValueError('Expected finite upper-expert biases')
        self.lib.e14_set_biases.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];self.lib.e14_set_biases(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)))
    def configure_path_prototypes(self,scales=None):
        a=np.ones(18,np.float32) if scales is None else np.ascontiguousarray(scales,np.float32)
        allowed=np.ones(18,bool);allowed[[5,9,11]]=False
        if a.shape!=(18,) or not np.isfinite(a).all() or np.any(a<=0) or np.any(a>4) or np.any(a[allowed]!=1):raise ValueError('Only F2/F3/F4 prototype multipliers may vary')
        if hasattr(self.lib,'r03_configure'):
            self.lib.r03_configure.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];self.lib.r03_configure.restype=ct.c_int
            if not self.lib.r03_configure(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float))):raise ValueError('Invalid regional prototype policy')
        elif np.any(a!=1):raise ValueError('Regional prototypes require R03 kernel')
        self.path_prototypes=a.copy()
    def configure_precision(self,policy=None):
        a=np.array([0,1,0,1] if policy is None else policy,np.float32)
        if a.shape!=(4,) or not np.isfinite(a).all() or a[0] not in (0,1) or a[1] not in (1,8,32) or not 0<=a[2]<1 or not 0<a[3]<=16:raise ValueError('Invalid R04 policy')
        if hasattr(self.lib,'r04_configure'):
            self.lib.r04_configure.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.r04_configure.restype=ct.c_int
            assert self.lib.r04_configure(self.handle,a.ctypes.data)
        elif not np.array_equal(a,np.array([0,1,0,1],np.float32)):raise ValueError('R04 kernel required')
        self.precision_policy=a.copy()
    def precision_state(self):
        if not hasattr(self.lib,'r04_state_size'):return np.zeros(0,np.uint8)
        self.lib.r04_state_size.argtypes=[ct.c_void_p];self.lib.r04_state_size.restype=ct.c_size_t
        a=np.zeros(self.lib.r04_state_size(self.handle),np.uint8)
        self.lib.r04_export.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.r04_export(self.handle,a.ctypes.data);return a
    def import_precision(self,a):
        a=np.ascontiguousarray(a,np.uint8)
        assert len(a)==len(self.precision_state())
        if len(a):
            self.lib.r04_import.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.r04_import(self.handle,a.ctypes.data)
    def configure_boundaries(self,policy=None):
        a=np.array([0,1,0,1] if policy is None else policy,np.float32)
        if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Invalid R05 policy')
        if hasattr(self.lib,'r05_configure'):
            self.lib.r05_configure.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.r05_configure.restype=ct.c_int
            if not self.lib.r05_configure(self.handle,a.ctypes.data):raise ValueError('Invalid R05 policy')
        elif not np.array_equal(a,[0,1,0,1]):raise ValueError('R05 kernel required')
        self.boundary_policy=a.copy()
    def boundary_metrics(self):
        a=np.zeros(96,np.float64)
        if hasattr(self.lib,'r05_metrics'):
            self.lib.r05_metrics.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.r05_metrics(self.handle,a.ctypes.data)
        return a
    def import_boundary_metrics(self,a):
        a=np.ascontiguousarray(a,np.float64);assert a.shape==(96,)
        if hasattr(self.lib,'r05_metrics_import'):
            self.lib.r05_metrics_import.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.r05_metrics_import(self.handle,a.ctypes.data)
    def boundary(self,lr=0.,train=False):
        self.lib.r05_boundary.argtypes=[ct.c_void_p,ct.c_float,ct.c_int];self.lib.r05_boundary.restype=ct.c_int
        assert self.lib.r05_boundary(self.handle,float(lr),int(train)), 'undrained boundary credit'
    def configure_f3_task_route(self,gain=1.):
        gain=float(np.float32(gain))
        if not np.isfinite(gain) or not 0<gain<=16:raise ValueError('Invalid F3 task routing multiplier')
        if hasattr(self.lib,'e05_configure'):
            self.lib.e05_configure.argtypes=[ct.c_void_p,ct.c_float];self.lib.e05_configure.restype=ct.c_int
            if not self.lib.e05_configure(self.handle,gain):raise ValueError('Native F3 gain rejected')
        elif gain!=1.:raise ValueError('F3 gain requires the E05 kernel')
        self.f3_task_route_gain=gain
    def configure_exploration(self,policy=None):
        cfg=dict(enabled=False,threshold=float(np.float32(.01755600795149803)),seed=7107,one_in=100)
        if policy is not None:
            if set(policy)-set(cfg):raise ValueError('Unknown exploration field')
            cfg.update(policy)
        cfg['enabled']=bool(cfg['enabled']);cfg['threshold']=float(np.float32(cfg['threshold']));cfg['seed']=int(cfg['seed']);cfg['one_in']=int(cfg['one_in'])
        if not np.isfinite(cfg['threshold']) or cfg['threshold']<0 or not 0<=cfg['seed']<2**64 or not 1<=cfg['one_in']<2**31:raise ValueError('Invalid exploration policy')
        if hasattr(self.lib,'e07_configure'):
            self.lib.e07_configure.argtypes=[ct.c_void_p,ct.c_int,ct.c_float,ct.c_uint64,ct.c_int];self.lib.e07_configure.restype=ct.c_int
            if not self.lib.e07_configure(self.handle,int(cfg['enabled']),cfg['threshold'],cfg['seed'],cfg['one_in']):raise ValueError('Native exploration rejected')
        elif cfg['enabled']:raise ValueError('Exploration requires E07 kernel')
        self.exploration=cfg
    def configure_traffic(self,policy=None):
        cfg=dict(enabled=False,strength=float(np.float32(.01755600795149803*.1)),alpha=1/4096)
        if policy is not None:
            if set(policy)-set(cfg):raise ValueError('Unknown traffic field')
            cfg.update(policy)
        cfg['enabled']=bool(cfg['enabled']);cfg['strength']=float(np.float32(cfg['strength']));cfg['alpha']=float(np.float32(cfg['alpha']))
        if not np.isfinite(cfg['strength']) or not 0<=cfg['strength']<=.1 or not np.isfinite(cfg['alpha']) or not 0<cfg['alpha']<1:raise ValueError('Invalid traffic policy')
        if hasattr(self.lib,'e08_configure'):
            self.lib.e08_configure.argtypes=[ct.c_void_p,ct.c_int,ct.c_float,ct.c_float];self.lib.e08_configure.restype=ct.c_int
            if not self.lib.e08_configure(self.handle,int(cfg['enabled']),cfg['strength'],cfg['alpha']):raise ValueError('Native traffic policy rejected')
        elif cfg['enabled']:raise ValueError('Traffic bias requires E08 kernel')
        self.traffic=cfg
    def init_traffic(self,reference):
        a=np.ascontiguousarray(reference,np.float32)
        if a.shape!=(core.E,):raise ValueError('Expected one reference per expert')
        self.lib.e08_init.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];self.lib.e08_init.restype=ct.c_int
        if not self.lib.e08_init(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float))):raise ValueError('Invalid reference shares')
    def traffic_state(self):
        if not hasattr(self.lib,'e08_state_size'):return np.zeros(0,np.uint8)
        self.lib.e08_state_size.restype=ct.c_size_t;self.lib.e08_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.e08_state_size(),np.uint8);self.lib.e08_export(self.handle,a.ctypes.data);return a
    def import_traffic(self,state):
        a=np.ascontiguousarray(state,np.uint8)
        if len(a)==0:return
        assert len(a)==len(self.traffic_state())
        self.lib.e08_import.argtypes=[ct.c_void_p,ct.c_void_p];self.lib.e08_import(self.handle,a.ctypes.data)
    def configure_retention(self,policy):
        if policy not in (0,1,2):raise ValueError('Retention policy must be 0, 1, or 2')
        if not hasattr(self.lib,'retention_configure'):
            if policy:raise ValueError('Use the credit kernel for B1 retention')
            return
        self.lib.retention_configure.argtypes=[ct.c_void_p,ct.c_int]
        self.lib.retention_configure(self.handle,policy)
    def retention_policy(self):
        if not hasattr(self.lib,'retention_policy'):return 0
        self.lib.retention_policy.argtypes=[ct.c_void_p];self.lib.retention_policy.restype=ct.c_int
        return self.lib.retention_policy(self.handle)
    def retention_state(self):
        self.lib.retention_state_size.restype=ct.c_size_t
        self.lib.retention_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.retention_state_size(),np.uint8);self.lib.retention_export(self.handle,a.ctypes.data);return a
    def import_retention(self,a):
        a=np.ascontiguousarray(a,np.uint8)
        assert len(a)==len(self.retention_state())
        self.lib.retention_import.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.retention_import(self.handle,a.ctypes.data)
    def retention_metrics(self):
        self.lib.retention_metrics.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        a=np.zeros(12,np.float64);self.lib.retention_metrics(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        fields=['archived','recovered_edges','recovered_norm_sum','flushed','flush_norm_sum','overwritten_pending','invalid_generation','update_samples','actual_update_norm_sum','changed_coordinates','max_credit_norm','reserved']
        return dict(zip(fields,a.tolist()))
    def utility_state(self):
        a=np.zeros(self.lib.utility_state_size(),np.uint8);self.lib.utility_export(self.handle,a.ctypes.data);return a
    def configure_teaching(self,scales):
        a=np.ascontiguousarray(scales,np.float32)
        if a.shape!=(18,2) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>64):
            raise ValueError('Teaching scales must be finite 18 x 2 values in [0,64]')
        if hasattr(self.lib,'teaching_configure'):
            self.lib.teaching_configure.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)]
            self.lib.teaching_configure.restype=ct.c_int
            if not self.lib.teaching_configure(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float))):
                raise ValueError('Native teaching configuration rejected')
        elif np.any(a!=1):raise ValueError('This kernel does not support teaching scales')
        self.teaching=a.copy()
    def ymean(self):
        a=np.zeros(64,np.float32);self.lib.utility_mean(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return a
    def configure_regional_gains(self,gains):
        a=np.ascontiguousarray(gains,np.float32)
        allowed=np.ones(18,bool);allowed[[10,12]]=False
        if a.shape!=(18,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>4) or np.any(a[allowed]!=1):
            raise ValueError('Only E4/V4 outgoing gains may differ from unity, in [0,4]')
        if hasattr(self.lib,'regional_gain_configure'):
            self.lib.regional_gain_configure.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)]
            self.lib.regional_gain_configure.restype=ct.c_int
            if not self.lib.regional_gain_configure(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float))):raise ValueError('Native gain policy rejected')
        elif np.any(a!=1):raise ValueError('Kernel does not support regional gains')
        self.regional_gains=a.copy()
    def set_ymean(self,a):
        a=np.ascontiguousarray(a,dtype=np.float32);assert a.shape==(64,)
        self.lib.utility_set_mean(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)))
    def configure_training_policy(self,frozen_regions,task_route_lr_scale=1.):
        a=np.ascontiguousarray(frozen_regions,np.int32)
        scale=float(np.float32(task_route_lr_scale))
        if a.shape!=(18,) or np.any((a!=0)&(a!=1)) or np.any(a[6:]) or not 0<scale<=32:
            raise ValueError('Freeze policy supports orders 1-2 only, with finite route scale in (0,32]')
        if hasattr(self.lib,'training_policy_configure'):
            self.lib.training_policy_configure.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int),ct.c_float]
            self.lib.training_policy_configure.restype=ct.c_int
            if not self.lib.training_policy_configure(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_int)),scale):raise ValueError('Native freeze policy rejected')
        elif np.any(a) or scale!=1:raise ValueError('Kernel does not support freezing')
        self.frozen_regions=a.copy();self.task_route_lr_scale=scale
    def bottom_fitted_state(self):
        self.lib.bottom_fitted_state_size.restype=ct.c_size_t
        self.lib.bottom_fitted_state.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.bottom_fitted_state_size(),np.uint8)
        self.lib.bottom_fitted_state(self.handle,a.ctypes.data);return a
    def configure_a1_learning(self,policy=None):
        """Scale A1 prototypes and decoder weights without freezing experts.

        Effective A1 prototype scale is its inherited scale times
        prototype_scale times clip(main_lr / reference_lr, 0, 1) when
        follow_cooldown is enabled. B1, Y decoder weights, decoder bias,
        and loss-driven routing keep their own existing policies.
        """
        if policy is None:a=A1_DEFAULT.copy()
        elif isinstance(policy,dict):
            if set(policy)-set(A1_FIELDS):raise ValueError('Unknown A1 learning policy field')
            values=dict(zip(A1_FIELDS,A1_DEFAULT));values.update(policy)
            a=np.array([values[k] for k in A1_FIELDS],np.float32)
        else:a=np.ascontiguousarray(policy,np.float32)
        if a.shape!=(4,) or not np.isfinite(a).all() or not 0<=a[0]<=1 or a[1] not in (0,1) or a[2]<=0 or not 0<=a[3]<=1:
            raise ValueError('A1 multipliers must be in [0,1], follow_cooldown boolean, reference_lr positive')
        if hasattr(self.lib,'a1_learning_configure'):
            self.lib.a1_learning_configure.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)]
            self.lib.a1_learning_configure.restype=ct.c_int
            if not self.lib.a1_learning_configure(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float))):raise ValueError('Native A1 policy rejected')
        elif not np.array_equal(a,A1_DEFAULT):raise ValueError('Kernel does not support A1 learning policy')
        self.a1_learning_array=a.copy()
        self.a1_learning={k:(bool(v) if k=='follow_cooldown' else float(v)) for k,v in zip(A1_FIELDS,a)}
    def reset(self):
        super().reset()
        if getattr(self,'_utility_ready',False):
            saved=self.utility_state();clean=np.zeros_like(saved);stop=(8+64)*4;clean[:stop]=saved[:stop]
            self.lib.utility_import(self.handle,clean.ctypes.data);self.lib.utility_observe(self.handle,0);self.lib.utility_ablate(self.handle,0)
    def frozen_copy(self,*,kernel=None):
        if self.runtime_kernel in DIAGNOSTIC_KERNELS:raise ValueError('Create independent contribution checks from the original model')
        if kernel is not None and (self.runtime_kernel,kernel) not in (('relays','contributions'),('messages','message_checks'),('completion','completion_checks')):
            raise ValueError('Use the diagnostic kernel matching the production kernel')
        m=Model(config=self.config,params=self.params,fixed=self.fixed,lace=self.lace,state_config=self.state_config,extra=self.extra_flat,probe=self.probe,utility=self.utility,kernel=kernel or self.runtime_kernel,teaching=self.teaching,regional_gains=self.regional_gains,frozen_regions=self.frozen_regions,task_route_lr_scale=self.task_route_lr_scale,a1_learning=self.a1_learning,f3_task_route_gain=self.f3_task_route_gain,exploration=self.exploration,traffic=self.traffic,composition=self.composition,messages=self.messages,local_credit=self.local_credit,content_experts=self.content_experts,pattern_classes=self.pattern_classes,paired_memory=self.paired_memory)
        m.configure_path_prototypes(self.path_prototypes)
        m.configure_boundaries(self.boundary_policy)
        m.configure_precision(self.precision_policy)
        m.import_precision(self.precision_state())
        m.import_traffic(self.traffic_state())
        m.import_gate_bias(self.gate_bias_state())
        m.import_activation(self.activation_state(),fresh=True)
        m.import_composition(self.composition_state(fitted=True),fitted=True)
        m.import_pattern_classes(self.pattern_state(fitted=True))
        a=self.means();m.lib.probe_set_means(m.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));m.set_ymean(self.ymean());return m
    def evaluation_copy(self,*,memory='cold_adaptive',diagnostic=False):
        """Fresh stream, frozen neural/dictionary evaluation, explicit memory.

        cold: empty store; warm: checkpoint store, coordinate centers and clock.
        adaptive: causal admission from the observed prefix; frozen: no writes.
        Query normalization and recurrent inference state still evolve in eval.
        Run this copy with mode='eval'. The historical frozen_copy() is cold_adaptive.
        """
        if self.runtime_kernel not in ('relays','messages','completion'):raise ValueError('Explicit memory evaluation requires a relay or message kernel')
        if memory not in ('cold_adaptive','cold_frozen','warm_frozen','warm_adaptive'):
            raise ValueError('Unknown evaluation memory policy')
        m=self.frozen_copy(kernel=('completion_checks' if self.runtime_kernel=='completion' else 'message_checks' if self.runtime_kernel=='messages' else 'contributions') if diagnostic else None)
        if memory.startswith('warm'):
            m.lib.evaluation_copy_memory.argtypes=[ct.c_void_p,ct.c_void_p]
            m.lib.evaluation_copy_memory(m.handle,self.handle)
        m.freeze_writes(memory.endswith('frozen'))
        return m
    def evaluation_memory_info(self):
        if not hasattr(self.lib,'evaluation_memory_info'):raise ValueError('Relay memory diagnostics required')
        a=np.zeros(5,np.uint64)
        self.lib.evaluation_memory_info.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.evaluation_memory_info(self.handle,a.ctypes.data)
        return dict(occupied=tuple(map(int,a[:2])),store_digest=f'{int(a[2]):016x}',
                    frozen=bool(a[3]),clock=int(a[4]))
    def configure_contribution(self,kind='none',region=None):
        """Select one intervention before evaluating a fresh diagnostic copy."""
        if self.runtime_kernel not in DIAGNOSTIC_KERNELS:raise ValueError('Use evaluation_copy(diagnostic=True)')
        if kind not in CONTRIBUTION_KINDS:raise ValueError('Unknown contribution intervention')
        if isinstance(region,str):
            if region not in core.NAMES:raise ValueError('Unknown region')
            region=core.NAMES.index(region)
        if kind=='none':
            if region is not None:raise ValueError('The disabled control has no region')
            r=-1
        else:
            if isinstance(region,(bool,np.bool_)) or not isinstance(region,(int,np.integer)) or not 0<=region<18:
                raise ValueError('Expected a region name or integer in [0, 17]')
            r=int(region)
        self.lib.contribution_configure.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
        self.lib.contribution_configure.restype=ct.c_int
        if not self.lib.contribution_configure(self.handle,CONTRIBUTION_KINDS.index(kind),r):
            raise ValueError('Intervention requires a fresh, unablated stream; semantic residuals exist only at L2 and above')
    def contribution_metrics(self):
        if self.runtime_kernel not in DIAGNOSTIC_KERNELS:raise ValueError('Diagnostic kernel required')
        a=np.zeros((18,5),np.uint64);h=np.zeros(18,np.uint64)
        self.lib.contribution_metrics.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.lib.contribution_metrics(self.handle,a.ctypes.data,h.ctypes.data)
        names=('publications','neural_executions','span_sum','max_span','intervention_applications')
        return {name:{**dict(zip(names,map(int,a[r]))),'event_hash':f'{int(h[r]):016x}'} for r,name in enumerate(core.NAMES)}
    def snapshot_copy(self):
        if self.runtime_kernel in DIAGNOSTIC_KERNELS:raise ValueError('Diagnostic copies cannot be resumed; start from the original checkpoint')
        # Own every mutable array before dispatching an immutable snapshot.
        m=Model(config=self.config,params=core.unflatten(self.flat.copy()),fixed={k:v.copy() for k,v in self.fixed.items()},lace=self.lace,state_config=self.state_config,extra=self.extra_flat.copy(),probe=self.probe,utility=self.utility,kernel=self.runtime_kernel,teaching=self.teaching,regional_gains=self.regional_gains,frozen_regions=self.frozen_regions,task_route_lr_scale=self.task_route_lr_scale,a1_learning=self.a1_learning,f3_task_route_gain=self.f3_task_route_gain,exploration=self.exploration,traffic=self.traffic,composition=self.composition,messages=self.messages,local_credit=self.local_credit,content_experts=self.content_experts,pattern_classes=self.pattern_classes,paired_memory=self.paired_memory)
        for arr,method in [(parent.legacy.base.state(self),'import_state'),(self.lace_state(),'import_laces'),(self.learned_state(),'import_learned'),(self.probe_state(),'probe_import'),(self.utility_state(),'utility_import')]:
            getattr(m.lib,method)(m.handle,arr.ctypes.data)
        m.configure_path_prototypes(self.path_prototypes)
        m.configure_boundaries(self.boundary_policy)
        m.configure_precision(self.precision_policy)
        m.import_precision(self.precision_state())
        m.import_traffic(self.traffic_state())
        if self.retention_policy():m.import_retention(self.retention_state())
        m.import_gate_bias(self.gate_bias_state())
        m.import_activation(self.activation_state())
        m.import_boundary_metrics(self.boundary_metrics())
        m.import_composition(self.composition_state())
        m.import_neural(self.neural_state())
        m.import_content_experts(self.content_expert_state())
        m.import_pattern_classes(self.pattern_state())
        m.import_paired_memory(self.paired_memory_state())
        return m
    def base_fingerprint(self):
        digest=hashlib.sha256(super().fingerprint().encode()+self.probe_state().tobytes()+self.utility_state().tobytes()).hexdigest()
        if np.any(self.teaching!=1):digest=hashlib.sha256(digest.encode()+self.teaching.tobytes()).hexdigest()
        if np.any(self.regional_gains!=1):digest=hashlib.sha256(digest.encode()+self.regional_gains.tobytes()).hexdigest()
        if np.any(self.frozen_regions) or self.task_route_lr_scale!=1:digest=hashlib.sha256(digest.encode()+self.frozen_regions.tobytes()+np.float32(self.task_route_lr_scale).tobytes()).hexdigest()
        if not np.array_equal(self.a1_learning_array,A1_DEFAULT):digest=hashlib.sha256(digest.encode()+self.a1_learning_array.tobytes()).hexdigest()
        return digest
    def fingerprint(self):
        digest=self.base_fingerprint()
        if self.neural_state().size:digest=hashlib.sha256(digest.encode()+self.neural_state().tobytes()).hexdigest()
        if self.runtime_kernel in COMPLETION_KERNELS:digest=hashlib.sha256(digest.encode()+bytes([self.local_credit,self.content_experts,self.pattern_classes,self.paired_memory])+self.content_expert_state().tobytes()+self.pattern_state().tobytes()+self.paired_memory_state().tobytes()).hexdigest()
        if self.runtime_kernel in MESSAGE_KERNELS:digest=hashlib.sha256(digest.encode()+json.dumps(self.messages,sort_keys=True).encode()).hexdigest()
        if self.composition_state().size:digest=hashlib.sha256(digest.encode()+self.composition_state().tobytes()).hexdigest()
        if not np.array_equal(self.boundary_policy,[0,1,0,1]):digest=hashlib.sha256(digest.encode()+self.boundary_policy.tobytes()+self.boundary_metrics().tobytes()).hexdigest()
        if self.precision_policy[0] or self.precision_policy[3]!=1: digest=hashlib.sha256(digest.encode()+self.precision_policy.tobytes()+self.precision_state().tobytes()).hexdigest()
        if np.any(self.path_prototypes!=1):digest=hashlib.sha256(digest.encode()+self.path_prototypes.tobytes()).hexdigest()
        if self.activation_policy()["enabled"]:digest=hashlib.sha256(digest.encode()+self.activation_state().tobytes()).hexdigest()
        if self.gate_bias_policy()["enabled"]:digest=hashlib.sha256(digest.encode()+self.gate_bias_state().tobytes()).hexdigest()
        if self.traffic['enabled']:digest=hashlib.sha256(digest.encode()+json.dumps(self.traffic,sort_keys=True).encode()+self.traffic_state().tobytes()).hexdigest()
        if self.exploration['enabled']:digest=hashlib.sha256(digest.encode()+json.dumps(self.exploration,sort_keys=True).encode()).hexdigest()
        if self.f3_task_route_gain!=1.:digest=hashlib.sha256(digest.encode()+np.float32(self.f3_task_route_gain).tobytes()).hexdigest()
        if self.retention_policy():digest=hashlib.sha256(digest.encode()+self.retention_state().tobytes()).hexdigest()
        return digest
    def save(self,path):
        if self.runtime_kernel in DIAGNOSTIC_KERNELS:raise ValueError('Evaluation interventions cannot be saved as training checkpoints')
        parent.legacy.base.atomic_checkpoint(path,flat=self.flat,extra=self.extra_flat,state=parent.legacy.base.state(self),lace_state=self.lace_state(),learned_state=self.learned_state(),probe_state=self.probe_state(),utility_state=self.utility_state(),
            version=np.array([39 if self.runtime_kernel=="completion" else 32 if self.runtime_kernel=="messages" else 31 if self.runtime_kernel=="relays" else 30 if self.runtime_kernel.startswith("semantics") else 29 if self.runtime_kernel=="binding" else 28 if self.runtime_kernel=="composition" else 27 if self.runtime_kernel.startswith("r05") else 26 if self.runtime_kernel.startswith("r04") else 25 if self.runtime_kernel.startswith("r03") else 24 if self.runtime_kernel.startswith("r02") and (self.activation_policy()["enabled"] or self.gate_bias_policy()["enabled"]) else 23 if self.gate_bias_policy()["enabled"] else 21 if self.traffic['enabled'] else (20 if self.retention_policy() else 19)],np.int32),experts=np.array([core.E],np.int32),config_json=np.array(json.dumps(self.config)),lace_json=np.array(json.dumps(self.lace)),state_config_json=np.array(json.dumps(self.state_config)),probe_json=np.array(json.dumps(self.probe)),utility_json=np.array(json.dumps(self.utility)),teaching_scales=self.teaching,regional_gains=self.regional_gains,frozen_regions=self.frozen_regions,task_route_lr_scale=np.array(self.task_route_lr_scale,np.float32),a1_learning_scales=self.a1_learning_array,f3_task_route_gain=np.array(self.f3_task_route_gain,np.float32),exploration_json=np.array(json.dumps(self.exploration)),traffic_json=np.array(json.dumps(self.traffic)),traffic_state=self.traffic_state(),
            **({'gate_bias_state':self.gate_bias_state(),'gate_bias_kernel':np.array('r02' if self.runtime_kernel.startswith('r02') else 'e14_calibrated' if self.gate_bias_policy()['gain']>10000 else 'e14'),'gate_bias_json':np.array(json.dumps(self.gate_bias_policy()))} if self.gate_bias_policy()['enabled'] else {}),
            **({'activation_state':self.activation_state(),'activation_json':np.array(json.dumps(self.activation_policy())),'activation_kernel':np.array('r02' if self.runtime_kernel.startswith('r02') else 'e13')} if self.activation_policy()['enabled'] else {}),
            **({"path_prototypes":self.path_prototypes} if self.runtime_kernel.startswith("r03") else {}),
            **({"precision_policy":self.precision_policy,"precision_state":self.precision_state()} if self.runtime_kernel.startswith(("r04","r05")) else {}),
            **({"boundary_policy":self.boundary_policy,"boundary_metrics":self.boundary_metrics()} if self.runtime_kernel.startswith("r05") else {}),
            **({'r08_kernel':np.array('r04_credit')} if self.runtime_kernel.startswith('r04_credit') else {}),
            **({'neural_state':self.neural_state(),'local_credit_policy':np.array(self.local_credit),'content_expert_policy':np.array(self.content_experts),'content_expert_state':self.content_expert_state(),'pattern_class_policy':np.array(self.pattern_classes),'pattern_class_state':self.pattern_state(),'paired_memory_policy':np.array(self.paired_memory),'paired_memory_state':self.paired_memory_state()} if self.runtime_kernel in COMPLETION_KERNELS else {}),
            **({'messages_json':np.array(json.dumps(self.messages))} if self.runtime_kernel in MESSAGE_KERNELS else {}),
            composition_state=self.composition_state(),composition_json=np.array(json.dumps(self.composition)),experiment_kernel=np.array(self.runtime_kernel),
            kernel_sha256=np.array(hashlib.sha256((ROOT/(self.runtime_kernel+'.cpp')).read_bytes()).hexdigest()),**({'retention_state':self.retention_state(),'retention_version':np.array([1],np.int32)} if self.retention_policy() else {}),**{'fixed_'+k:v for k,v in self.fixed.items()})
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            if int(a['version'][0])==33:
                raise ValueError('Version-33 completion uses coverage clocks; convert_to_completion.py starts a fresh stream with primary clocks')
            if int(a['version'][0])==34:
                raise ValueError('Version-34 completion has expiring neural producers; convert_to_completion.py starts a fresh stream with neural ownership')
            if int(a['version'][0]) in (35,36,37,38,39) and ('neural_state' not in a):
                raise ValueError('Completion checkpoints require saved neural ownership')
            if int(a['version'][0]) in (35,36,37,38,39) and ('messages_json' not in a or str(a['experiment_kernel'])!='completion'):
                raise ValueError('Completion checkpoints require their saved policy and production kernel')
            if int(a['version'][0])==32 and ('messages_json' not in a or str(a['experiment_kernel'])!='messages'):
                raise ValueError('Message checkpoints require their saved policy and production kernel')
            assert int(a['version'][0]) in (16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,35,36,37,38,39) and int(a['experts'][0])==core.E
            m=cls(config=json.loads(str(a['config_json'])),params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=json.loads(str(a['lace_json'])),state_config=json.loads(str(a['state_config_json'])),extra=a['extra'],probe=json.loads(str(a['probe_json'])),utility=json.loads(str(a['utility_json'])),composition=json.loads(str(a['composition_json'])) if 'composition_json' in a else None,kernel=str(a['experiment_kernel']) if 'experiment_kernel' in a else str(a['r08_kernel']) if 'r08_kernel' in a else 'r05' if int(a['version'][0])==27 else 'r04' if int(a['version'][0])==26 else 'r03' if int(a['version'][0])==25 else 'r02' if int(a['version'][0])==24 else ('e14_calibrated' if 'gate_bias_kernel' in a and str(a['gate_bias_kernel'])=='e14_calibrated' else 'e14') if 'gate_bias_state' in a else 'e13' if 'activation_state' in a else ('credit' if 'retention_state' in a else 'native'))
            if int(a['version'][0]) in (36,37,38,39):
                if 'local_credit_policy' not in a or a['local_credit_policy'].shape!=() or a['local_credit_policy'].dtype!=np.dtype(bool):
                    m.close();raise ValueError('Checkpoint requires its saved local-credit policy')
                m.configure_local_credit(bool(a['local_credit_policy']))
            elif m.runtime_kernel in COMPLETION_KERNELS:m.configure_local_credit(False)
            if int(a['version'][0]) in (37,38,39):
                if ('content_expert_state' not in a or 'content_expert_policy' not in a or
                    a['content_expert_policy'].shape!=() or a['content_expert_policy'].dtype!=np.dtype(bool)):
                    m.close();raise ValueError('Version-37 requires saved content-expert policy and eligibility')
                m.configure_content_experts(bool(a['content_expert_policy']))
            elif m.runtime_kernel in COMPLETION_KERNELS:m.configure_content_experts(False)
            if int(a['version'][0]) in (38,39):
                if ('pattern_class_state' not in a or 'pattern_class_policy' not in a or
                    a['pattern_class_policy'].shape!=() or a['pattern_class_policy'].dtype!=np.dtype(bool)):
                    m.close();raise ValueError('Version-38 requires saved pattern-class policy and state')
                m.configure_pattern_classes(bool(a['pattern_class_policy']))
            elif m.runtime_kernel in COMPLETION_KERNELS:m.configure_pattern_classes(False)
            if int(a['version'][0])==39:
                if ('paired_memory_state' not in a or 'paired_memory_policy' not in a or
                    a['paired_memory_policy'].shape!=() or a['paired_memory_policy'].dtype!=np.dtype(bool)):
                    m.close();raise ValueError('Version-39 requires saved paired-memory policy and state')
                m.configure_paired_memory(bool(a['paired_memory_policy']))
            elif m.runtime_kernel in COMPLETION_KERNELS:m.configure_paired_memory(False)
            if 'messages_json' in a:m.configure_messages(json.loads(str(a['messages_json'])))
            if 'gate_bias_state' in a:m.import_gate_bias(a['gate_bias_state'])
            if 'activation_state' in a:m.import_activation(a['activation_state'])
            if 'traffic_json' in a:
                m.configure_traffic(json.loads(str(a['traffic_json'])))
                m.import_traffic(a['traffic_state'])
            if 'exploration_json' in a:m.configure_exploration(json.loads(str(a['exploration_json'])))
            if 'boundary_policy' in a:m.configure_boundaries(a['boundary_policy']);m.import_boundary_metrics(a['boundary_metrics'])
            if 'precision_policy' in a:
                m.configure_precision(a['precision_policy']);m.import_precision(a['precision_state'])
            if 'path_prototypes' in a:m.configure_path_prototypes(a['path_prototypes'])
            if 'f3_task_route_gain' in a:m.configure_f3_task_route(float(a['f3_task_route_gain']))
            if 'teaching_scales' in a:m.configure_teaching(a['teaching_scales'])
            if int(a['version'][0])>=17:m.configure_regional_gains(a['regional_gains'])
            if int(a['version'][0])>=18:m.configure_training_policy(a['frozen_regions'],float(a['task_route_lr_scale']))
            if int(a['version'][0])>=19:m.configure_a1_learning(a['a1_learning_scales'])
            for key,method,size in [('state','import_state',m.lib.state_size()),('lace_state','import_laces',m.lib.lace_state_size()),('learned_state','import_learned',m.lib.learned_state_size()),('probe_state','probe_import',m.lib.probe_state_size()),('utility_state','utility_import',m.lib.utility_state_size())]:
                arr=np.ascontiguousarray(a[key],dtype=np.uint8);assert arr.size==size,(key,arr.size,size)
                getattr(m.lib,method)(m.handle,arr.ctypes.data)
            if 'composition_json' in a:m.configure_composition(json.loads(str(a['composition_json'])))
            if int(a['version'][0]) in (35,36):
                data=np.ascontiguousarray(a['composition_state'],np.uint8)
                m.lib.composition_import_v9.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t]
                m.lib.composition_import_v9.restype=ct.c_int
                if not m.lib.composition_import_v9(m.handle,data.ctypes.data,data.size):
                    m.close();raise ValueError('Invalid version-9 composition snapshot')
            elif 'composition_state' in a:m.import_composition(a['composition_state'])
            if 'neural_state' in a:m.import_neural(a['neural_state'])
            if int(a['version'][0]) in (37,38,39):m.import_content_experts(a['content_expert_state'])
            if int(a['version'][0]) in (38,39):m.import_pattern_classes(a['pattern_class_state'])
            if int(a['version'][0])==39:m.import_paired_memory(a['paired_memory_state'])
            if 'retention_state' in a:
                assert int(a['retention_version'][0])==1
                m.import_retention(a['retention_state'])
        return m
    def utility_metrics(self):
        a=np.zeros((18,8),np.float64);self.lib.utility_metrics(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        return {n:dict(total_norm=v[0],total_calls=v[1],local_norm=v[2],local_calls=v[3]) for n,v in zip(core.NAMES,a)}

def evaluate(model,data,mean_y=False,documents=None):
    m=model.frozen_copy();m.lib.utility_ablate(m.handle,int(mean_y));rows=[]
    if documents is None:
        spans=[(i,min(i+2048,len(data)-1),str(i)) for i in range(0,len(data)-1,2048)]
    else:
        spans=[(max(0,d['start']-1),min(len(data)-1,d['start']+d['bytes_used']-1),d['id']) for d in documents]
    for lo,hi,ident in spans:
        if hi>lo:rows.append(dict(id=ident,targets=hi-lo,nll=m.run(data[lo:hi],data[lo+1:hi+1])))
    counts=np.array(m.stats()['expert_counts']);prob=counts/np.maximum(1,counts.sum(1,keepdims=True));eff=np.exp(-(prob*np.log(np.maximum(prob,1e-30))).sum(1))
    out=dict(nll=float(np.average([v['nll'] for v in rows],weights=[v['targets'] for v in rows])),targets=sum(v['targets'] for v in rows),effective_experts=dict(zip(core.NAMES,eff.tolist())),documents=rows)
    m.close();return out
