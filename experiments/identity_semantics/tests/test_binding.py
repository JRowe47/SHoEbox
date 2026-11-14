import ctypes as ct
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core

class BindingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.path=ROOT/'results/binding_fixture.so'
  subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O2','-march=native','-std=c++17','-shared','-fPIC',str(ROOT/'tests/binding_fixture.cpp'),'-o',str(cls.path)],check=True)
  cls.fx=ct.CDLL(str(cls.path));f=cls.fx
  f.fixture_stage.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int64,ct.c_int64,ct.c_int64,ct.c_float]
  f.fixture_prepare.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p];f.fixture_prepare.restype=ct.c_int
  f.fixture_inputs.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
  f.fixture_bundle_gradient.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
  f.fixture_context_delta.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_float]
  f.fixture_current_cues.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
 def make(self,kernel='binding'):
  m=fresh_model(kernel=kernel);self.addCleanup(m.close);return m
 def stage(self,m,r,offset=0,change=None,context=0):
  slots=[]
  for k in range(int(m.fixed['starts'][r]),int(m.fixed['starts'][r+1])):
   src,age,delay,w=m.fixed['slots'][k];src,age=int(src),int(age)
   if delay:continue
   value=(src*13+age*17+41)%256;begin=8-2*age;end=10-2*age;seq=2-age
   if change and change[0]==k:
    value+=change[1];begin+=change[2];end+=change[2];seq+=change[3]
   self.fx.fixture_stage(m.handle,src,age,value,seq,begin+offset,end+offset,context);slots.append((k,src,age))
  return slots
 def prepare(self,m,r,learn=0):
  meta=np.zeros(29,np.uint64);code=np.zeros(64,np.float32)
  yes=self.fx.fixture_prepare(m.handle,r,learn,meta.ctypes.data,code.ctypes.data)
  return yes,meta,code
 def test_01_every_declared_immediate_role_is_a_constituent(self):
  for r in range(3,18):
   m=self.make();slots=self.stage(m,r);yes,meta,code=self.prepare(m,r)
   self.assertTrue(yes,core.NAMES[r]);self.assertEqual(meta[2],len(slots));self.assertGreater(np.linalg.norm(code),0)
   self.assertEqual([(int(meta[5+3*i]),int(meta[6+3*i]),int(meta[7+3*i])) for i in range(len(slots))],[(src,2-age,k) for k,src,age in slots])
 def test_02_each_secondary_source_changes_identity_and_neural_code(self):
  for r in range(3,18):
   base=self.make();slots=self.stage(base,r);_,ids,code=self.prepare(base,r)
   for k,src,age in slots[1:]:
    m=self.make();self.stage(m,r,change=(k,7,0,0));_,b,c=self.prepare(m,r)
    self.assertFalse(np.array_equal(ids[:2],b[:2]),(r,k));self.assertGreater(np.linalg.norm(code-c),.1,(r,k));m.close()
   base.close()
 def test_03_no_fresh_roles_no_duplicate_publication(self):
  m=self.make();self.stage(m,10);self.assertTrue(self.prepare(m,10,1)[0]);before=m.composition_state(fitted=True).copy()
  self.assertFalse(self.prepare(m,10,1)[0]);np.testing.assert_array_equal(before,m.composition_state(fitted=True))
 def test_04_secondary_update_wakes_region_and_marks_held_roles(self):
  m=self.make();slots=self.stage(m,10);self.prepare(m,10);k=slots[-1][0]
  self.stage(m,10,change=(k,7,0,1));yes,meta,_=self.prepare(m,10)
  self.assertTrue(yes);self.assertEqual(meta[4],1<<(k-int(m.fixed['starts'][10])));self.assertGreater(int(meta[3]).bit_count(),1)
 def test_05_alignment_invariance_and_relative_geometry_sensitivity(self):
  a=self.make();b=self.make();c=self.make();slots=self.stage(a,10);self.stage(b,10,offset=123);self.stage(c,10,change=(slots[-1][0],0,3,0))
  _,ia,va=self.prepare(a,10);_,ib,vb=self.prepare(b,10);_,ic,vc=self.prepare(c,10)
  np.testing.assert_array_equal(ia[:2],ib[:2]);np.testing.assert_array_equal(va,vb)
  self.assertFalse(np.array_equal(ia[:2],ic[:2]));self.assertGreater(np.linalg.norm(va-vc),.01)
 def test_06_context_does_not_relabel_content(self):
  a=self.make();b=self.make();self.stage(a,10,context=0);self.stage(b,10,context=10)
  _,ia,va=self.prepare(a,10);_,ib,vb=self.prepare(b,10);np.testing.assert_array_equal(ia[:2],ib[:2]);np.testing.assert_array_equal(va,vb)
 def test_07_source_removal_preserves_surviving_roles(self):
  m=self.make();slots=self.stage(m,10);m.ablate(10+8);yes,meta,_=self.prepare(m,10)
  self.assertTrue(yes);actual=[int(meta[7+3*i]) for i in range(int(meta[2]))];self.assertEqual(actual,[k for k,src,age in slots if src!=8])
 def test_08_context_jacobian_and_distinct_occurrences(self):
  m=self.make();slots=self.stage(m,10);g=np.linspace(-.3,.6,64,dtype=np.float32);grad=np.zeros((18,2,64),np.float32)
  self.fx.fixture_bundle_gradient(m.handle,10,g.ctypes.data,grad.ctypes.data)
  for k,src,age in slots:
   for j in (3,31):
    vals=[]
    for d in (-.001,.001):
     self.fx.fixture_context_delta(m.handle,src,age,j,d);q=np.zeros(64,np.float32);c=q.copy();self.fx.fixture_inputs(m.handle,10,q.ctypes.data,c.ctypes.data);vals.append(float(g@c));self.fx.fixture_context_delta(m.handle,src,age,j,-d)
    self.assertAlmostEqual((vals[1]-vals[0])/.002,float(grad[src,age,j]),delta=4e-4)
 def test_09_actual_router_cue_distinguishes_parallel_paths(self):
  m=self.make();m.run(np.arange(60,dtype=np.uint8));codes=m.content_events()
  for name in ('S2','R2','F2'):self.assertGreater(np.linalg.norm(codes['L2']['code']-codes[name]['code']),.1)
  for r in (3,4,5):
   cue=np.zeros(64,np.float32);self.fx.fixture_current_cues(m.handle,r,cue.ctypes.data);np.testing.assert_array_equal(cue,codes[core.NAMES[r]]['code'])
 def test_10_slot_gain_controls_neural_input(self):
  a=self.make();b=self.make();x=np.frombuffer(b'A meaningful sequence of symbols. '*8,np.uint8)
  for k in range(int(b.fixed['starts'][17]),int(b.fixed['starts'][18])):
   if int(b.fixed['slots'][k,0])==15:b.fixed['slots'][k,3]=0
  la=a.run(x,logits=True)[1];lb=b.run(x,logits=True)[1];self.assertGreater(np.max(np.abs(la-lb)),1e-5)
 def test_11_exact_resume_and_chunk_independence(self):
  a=self.make();x=np.arange(200,dtype=np.uint8);a.run(x,mode='periodic',lr=.01,lp=.003)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'m.npz';a.save(p);b=Model.load(p);self.addCleanup(b.close);self.assertEqual(a.fingerprint(),b.fingerprint())
   aa=a.run(x,mode='periodic',lr=.01,lp=.003,logits=True)[1];bb=np.concatenate([b.run(z,mode='periodic',lr=.01,lp=.003,logits=True)[1] for z in np.array_split(x,5)])
   np.testing.assert_array_equal(aa,bb);self.assertEqual(a.fingerprint(),b.fingerprint())
 def test_12_eval_freezes_dictionaries_and_targets_do_not_leak(self):
  a=self.make();b=self.make();x=np.arange(200,dtype=np.uint8);before=a.composition_state(fitted=True)
  aa=a.run(x,np.zeros_like(x),logits=True)[1];bb=b.run(x,np.ones_like(x),logits=True)[1]
  np.testing.assert_array_equal(aa,bb);np.testing.assert_array_equal(before,a.composition_state(fitted=True))
 def test_13_finish_has_no_synthetic_bytes_and_reset_clears_roles(self):
  m=self.make();m.run(np.arange(21,dtype=np.uint8),mode='periodic',lr=.01,lp=.003);before=m.content_events()['A1']['bytes'];m.finish_document(mode='periodic',lr=.01,lp=.003)
  self.assertEqual(before,m.content_events()['A1']['bytes']);m.finish_document(reset=True);self.assertTrue(all(e['seq']==0 for e in m.content_events().values()))
 def test_14_legacy_composition_abi_is_rejected(self):
  a=self.make();b=self.make('composition')
  with self.assertRaises(ValueError):a.import_composition(b.composition_state())

if __name__=='__main__':unittest.main(verbosity=2)
