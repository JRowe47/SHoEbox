"""Add binding selection, ABI/version handling and role diagnostics."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runtime/hierarchy_lab/throughput_probe'
p=root/'model.py';s=p.read_text()
changes=[
 ("source.read_bytes()+b''.join", "source.read_bytes()+((ROOT/'composition.cpp').read_bytes() if KERNEL=='binding' else b'')+b''.join"),
 ("enabled=self.runtime_kernel=='composition'","enabled=self.runtime_kernel in ('composition','binding')"),
 ('version=np.array([28 if self.runtime_kernel=="composition"','version=np.array([29 if self.runtime_kernel=="binding" else 28 if self.runtime_kernel=="composition"'),
 ('(16,17,18,19,20,21,22,23,24,25,26,27,28)','(16,17,18,19,20,21,22,23,24,25,26,27,28,29)'),
 ("'identity_credit_relays','singleton_relays')","'identity_credit_relays','singleton_relays','bound_frames','held_roles','fresh_roles','startup_holds')")]
for old,new in changes:
 if new not in s:
  assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new,1)
method='''    def binding_role_status(self):
        if not hasattr(self.lib,'binding_role_status'):return {}
        a=np.zeros((18,3),np.uint64)
        self.lib.binding_role_status.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.binding_role_status(self.handle,a.ctypes.data)
        return {name:dict(present=int(a[r,0]),fresh=int(a[r,1]),kind=int(a[r,2])) for r,name in enumerate(core.NAMES)}
'''
if method not in s:s=s.replace('    def finish_document(',method+'    def finish_document(',1)
p.write_text(s)
