from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runtime/hierarchy_lab/throughput_probe'
# Keep a portable base copy for reproducible packaging.
baseline=root/'model_legacy.py'
if not baseline.exists():baseline.write_text((root/'model.py').read_text())
s=baseline.read_text()
def change(old,new,count=1):
    global s
    if s.count(old)!=count:raise RuntimeError((old[:80],s.count(old),count))
    s=s.replace(old,new)
change('exploration=None,traffic=None,**kwargs):','exploration=None,traffic=None,composition=None,**kwargs):')
change('                self.configure_traffic(traffic)','                self.configure_traffic(traffic)\n                self.configure_composition(composition)')
methods='''    def configure_composition(self,policy=None):
        cfg=dict(enabled=self.runtime_kernel=='composition',threshold=.35,min_count=3,max_children=8,context_gain=.2)
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
    def composition_state(self,fitted=False):
        if not hasattr(self.lib,'composition_state_size'):return np.zeros(0,np.uint8)
        self.lib.composition_state_size.restype=ct.c_size_t
        self.lib.composition_fitted_size.restype=ct.c_size_t
        self.lib.composition_export.argtypes=[ct.c_void_p,ct.c_void_p]
        a=np.zeros(self.lib.composition_state_size(),np.uint8);self.lib.composition_export(self.handle,a.ctypes.data)
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
        names=('consumed','published','compounds','recognized','provisional','held','table_evictions','span_sum','input_credit_received','input_credit_forwarded','expired_input_edges','overwritten_input_credit','identity_credit_relays','singleton_relays')
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
    def finish_document(self,mode='eval',lr=.12,lp=.04,route=.002,reset=False):
        if self.composition['enabled']:
            if mode not in core.MODES or mode=='spatial':raise ValueError('Unsupported composition training mode')
            self.lib.composition_finish.argtypes=[ct.c_void_p,ct.c_int,ct.c_float,ct.c_float,ct.c_float]
            self.lib.composition_finish.restype=ct.c_int
            if not self.lib.composition_finish(self.handle,core.MODES[mode],lr,lp,route):raise RuntimeError('Pending composition events did not drain')
        if reset:self.lib.reset_context(self.handle)
    def run(self,x,y=None,mode='eval',lr=.12,lp=.04,route=.002,logits=False):
        if self.composition['enabled'] and mode=='spatial':raise ValueError('Legacy spatial mode has no event/state credit; use periodic or hybrid')
        return super().run(x,y,mode=mode,lr=lr,lp=lp,route=route,logits=logits)
'''
change('    def configure_activation(self,width,scale,ramp=131072):',methods+'    def configure_activation(self,width,scale,ramp=131072):')
change('exploration=self.exploration,traffic=self.traffic)', 'exploration=self.exploration,traffic=self.traffic,composition=self.composition)',2)
change('        a=self.means();m.lib.probe_set_means', '        m.import_composition(self.composition_state(fitted=True),fitted=True)\n        a=self.means();m.lib.probe_set_means')
change('        m.import_boundary_metrics(self.boundary_metrics())\n        return m', '        m.import_boundary_metrics(self.boundary_metrics())\n        m.import_composition(self.composition_state())\n        return m')
change('        digest=self.base_fingerprint()', '        digest=self.base_fingerprint()\n        if self.composition_state().size:digest=hashlib.sha256(digest.encode()+self.composition_state().tobytes()).hexdigest()')
change('version=np.array([27 if', 'version=np.array([28 if self.runtime_kernel=="composition" else 27 if')
change('            kernel_sha256=np.array(', '            composition_state=self.composition_state(),composition_json=np.array(json.dumps(self.composition)),experiment_kernel=np.array(self.runtime_kernel),\n            kernel_sha256=np.array(')
change("kernel=str(a['r08_kernel']) if", "kernel=str(a['experiment_kernel']) if 'experiment_kernel' in a else str(a['r08_kernel']) if")
change('in (16,17,18,19,20,21,22,23,24,25,26,27)', 'in (16,17,18,19,20,21,22,23,24,25,26,27,28)')
change("            if 'retention_state' in a:\n                assert", "            if 'composition_json' in a:m.configure_composition(json.loads(str(a['composition_json'])))\n            if 'composition_state' in a:m.import_composition(a['composition_state'])\n            if 'retention_state' in a:\n                assert")
(root/'model.py').write_text(s)
p=root.parents[1]/'runtime.py'
t=p.read_text().replace("def fresh_model(seed=1, kernel='native'):","def fresh_model(seed=1, kernel='composition'):")
p.write_text(t)
