"""Forward-only policy, compact persistence and explicit return to training."""
import ctypes as ct
import hashlib
import json
import numpy as np

POLICY = dict(runtime='forward-only-v1', entry='drain-materialized-v1', resume='reset-context-v1')

class InferenceMixin:
    def finish_training(self):
        """Commit materialized slow-Y, reverse, replay and bank work.

        Does not invent targets for pending next-event teachers or flush a
        document. Entering inference retires that remaining eligibility.
        """
        if self.runtime_kernel not in ('completion', 'completion_checks'):
            raise ValueError('Forward-only runtime requires learned completion')
        self.lib.inference_finish_boundary.argtypes=[ct.c_void_p]
        if not self.lib.inference_finish_boundary(self.handle):
            raise ValueError('Transition requires a quiescent composition stream')
        return self

    def enter_inference(self):
        """Drain materialized work, retain forward values and release eligibility."""
        if self.forward_only:return self
        self.finish_training()
        self.lib.inference_enter.argtypes=[ct.c_void_p]
        if not self.lib.inference_enter(self.handle):raise ValueError('Cannot enter inference at this boundary')
        self.forward_only=True
        return self

    def resume_training(self, *, reset_context=False):
        """Start fresh eligibility, preserving fitted weights and episodic store."""
        if not self.forward_only:return self
        if reset_context is not True:
            raise ValueError('Forward-only inference has no eligibility; use resume_training(reset_context=True)')
        self.lib.inference_resume.argtypes=[ct.c_void_p,ct.c_int]
        if not self.lib.inference_resume(self.handle,1):raise ValueError('Unable to reset inference context')
        self.forward_only=False
        return self

    def inference_copy(self, *, memory='cold_adaptive'):
        """Independent frozen stream; continue preserves the current prefix.

        cold/warm selects empty/copied episodes; adaptive/frozen selects writes.
        The original training stream and its pending versions stay untouched.
        continue drains materialized work only in the copied stream.
        """
        m=self.snapshot_copy() if memory=='continue' else self.evaluation_copy(memory=memory)
        try:return m.enter_inference()
        except Exception:m.close();raise

    def inspect_inference(self):
        a=np.zeros(16,np.uint64)
        self.lib.inference_inspect.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.inference_inspect(self.handle,a.ctypes.data)
        names=('forward_only','fixed_model_bytes','derivative_slab_bytes','derivative_vector_bytes',
               'neural_archives','context_archives','neural_records','context_records','forward_publications',
               'temporal_archives','reverse_roots','pending_bank_job','pending_bank_scopes','slow_y_samples',
               'forward_state_bytes','recycled_archive_bytes')
        return dict(zip(names,map(int,a)))

    def inference_state(self):
        if not self.forward_only:raise ValueError('Inference state requires enter_inference()')
        self.lib.inference_state_size.restype=ct.c_size_t
        self.lib.inference_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.inference_state_size(),np.uint8)
        if not self.lib.inference_export(self.handle,a.ctypes.data):raise ValueError('Inference export failed')
        return a

    def import_inference(self,state):
        a=np.ascontiguousarray(state,np.uint8)
        self.lib.inference_import.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t]
        if not self.lib.inference_import(self.handle,a.ctypes.data,a.size):
            raise ValueError('Invalid forward-only state or policy')

    def _inference_arrays(self):
        config={key:getattr(self,key) for key in ('config','lace','state_config','probe','utility','messages',
            'composition','reverse','event','prediction','frames','bank_pipeline','bank_epochs','local_credit',
            'content_experts','pattern_classes','paired_memory','task_route_lr_scale','f3_task_route_gain',
            'a1_learning','teaching','regional_gains','frozen_regions')}
        config['kernel']=self.runtime_kernel
        def default(v):
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,np.generic):return v.item()
            raise TypeError(type(v).__name__)
        return dict(version=np.array([47],np.int32),experts=np.array([self.params['W'].shape[1]],np.int32),
            inference_policy=np.array(json.dumps(POLICY,sort_keys=True)),
            inference_config_json=np.array(json.dumps(config,default=default,sort_keys=True)),
            flat=self.flat.copy(),extra=self.extra_flat.copy(),inference_state=self.inference_state(),
            composition_fitted=self.composition_state(fitted=True),pattern_fitted=self.pattern_state(fitted=True),
            event_fitted=self.event_state(fitted=True),prediction_fitted=self.prediction_state(fitted=True),
            paired_memory_state=self.paired_memory_state(),**{'fixed_'+k:v.copy() for k,v in self.fixed.items()})

    def _inference_fingerprint(self):
        digest=hashlib.sha256()
        for key,value in sorted(self._inference_arrays().items()):
            digest.update(key.encode());digest.update(value.tobytes())
        return digest.hexdigest()

    @classmethod
    def _load_inference_arrays(cls,a):
        from model import core
        required=('version','experts','inference_policy','inference_config_json','flat','extra','inference_state',
                  'composition_fitted','pattern_fitted','event_fitted','prediction_fitted','paired_memory_state')
        if any(k not in a for k in required) or json.loads(str(a['inference_policy']))!=POLICY:
            raise ValueError('Version-47 requires its forward-only policy and complete state')
        if a['version'].shape!=(1,) or int(a['version'][0])!=47 or a['experts'].shape!=(1,) or int(a['experts'][0])!=core.E:
            raise ValueError('Inference checkpoint model size/version mismatch')
        config=json.loads(str(a['inference_config_json']))
        if config.get('kernel')!='completion' or not config.get('composition',{}).get('enabled',True):
            raise ValueError('Inference checkpoint requires production completion')
        m=cls(params=core.unflatten(a['flat'].copy()),extra=a['extra'].copy(),
              fixed={k[6:]:a[k].copy() for k in a if k.startswith('fixed_')},**config)
        try:
            m.import_composition(a['composition_fitted'],fitted=True)
            m.import_pattern_classes(a['pattern_fitted'])
            m.import_event(a['event_fitted'],fitted=True)
            m.import_prediction(a['prediction_fitted'],fitted=True)
            m.enter_inference()
            m.import_inference(a['inference_state'])
            m.import_paired_memory(a['paired_memory_state'])
            return m
        except Exception:m.close();raise
