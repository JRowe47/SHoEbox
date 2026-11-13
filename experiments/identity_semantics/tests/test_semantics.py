import ctypes as ct
import subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core
from test_binding import BindingTests
class SemanticsBindingTests(BindingTests):
 @classmethod
 def setUpClass(cls):
  cls.path=ROOT/'results/semantics_fixture.so'
  subprocess.run(['g++','-O2','-march=native','-std=c++17','-shared','-fPIC',str(ROOT/'tests/semantics_fixture.cpp'),'-o',str(cls.path)],check=True)
  cls.fx=ct.CDLL(str(cls.path));f=cls.fx
  f.fixture_stage.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int64,ct.c_int64,ct.c_int64,ct.c_float]
  f.fixture_prepare.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p];f.fixture_prepare.restype=ct.c_int
  f.fixture_inputs.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
  f.fixture_bundle_gradient.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
  f.fixture_context_delta.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_float]
  f.fixture_current_cues.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
  f.semantic_fixture_structure.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
  f.semantic_fixture_encode.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
  f.semantic_fixture_change.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int64]
  f.semantic_fixture_weight.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_float]
  f.semantic_fixture_gradient.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_float]
  f.semantic_fixture_saved_gradient.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_float]
  f.semantic_fixture_clear_dictionary.argtypes=[ct.c_void_p]
  f.semantic_fixture_q.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
 def make(self,kernel='semantics'):
  m=fresh_model(kernel=kernel);self.addCleanup(m.close);return m
 def stage(self,m,r,offset=0,change=None,context=0):
  slots=super().stage(m,r,offset,change,context)
  for _,src,age in slots:self.fx.semantic_fixture_structure(m.handle,src,age)
  return slots
 def sequence(self,m,offset=0):
  for age,value in [(1,65),(0,66)]:
   self.fx.fixture_stage(m.handle,0,age,value,2-age,offset+1-age,offset+2-age,0)
   self.fx.semantic_fixture_structure(m.handle,0,age)
 def encoded(self,m,r=2,kind=1):
  ids=np.zeros(2,np.uint64);v=np.zeros(64,np.float32)
  self.fx.semantic_fixture_encode(m.handle,r,kind,ids.ctypes.data,v.ctypes.data);return ids,v
 def test_15_temporal_geometry_duration_namespace_and_kind(self):
  a=self.make();self.sequence(a);ia,va=self.encoded(a)
  for field,delta in [(0,3),(1,1)]:
   b=self.make();self.sequence(b);self.fx.semantic_fixture_change(b.handle,0,0,field,delta);ib,vb=self.encoded(b)
   self.assertFalse(np.array_equal(ia,ib));self.assertGreater(np.linalg.norm(va-vb),.1)
  for r,kind in [(15,1),(16,1),(2,2)]:
   ib,vb=self.encoded(a,r,kind);self.assertFalse(np.array_equal(ia,ib));self.assertGreater(np.linalg.norm(va-vb),.1)
 def test_16_alignment_and_occurrence_do_not_change_symbol(self):
  a=self.make();b=self.make();self.sequence(a);self.sequence(b,10000)
  self.fx.semantic_fixture_change(b.handle,0,0,3,1000)
  for aa,bb in zip(self.encoded(a),self.encoded(b)):np.testing.assert_array_equal(aa,bb)
 def test_17_stale_child_embedding_cannot_relabel_parent(self):
  a=self.make();self.sequence(a);before=self.encoded(a)
  self.fx.semantic_fixture_change(a.handle,0,0,2,100)
  for aa,bb in zip(before,self.encoded(a)):np.testing.assert_array_equal(aa,bb)
 def test_18_shared_adapter_gradient_matches_finite_difference(self):
  m=self.make();self.sequence(m);g=np.linspace(-.3,.3,64,dtype=np.float32)
  for j,k in [(2,0),(17,3),(61,7)]:
   self.fx.semantic_fixture_weight(m.handle,2,j,k,.7)
   scores=[]
   for delta in [-.01,.01]:
    self.fx.semantic_fixture_weight(m.handle,2,j,k,delta);scores.append(float(g.astype(np.float64)@self.encoded(m)[1].astype(np.float64)));self.fx.semantic_fixture_weight(m.handle,2,j,k,-delta)
   before=m.semantic_parameters();self.fx.semantic_fixture_gradient(m.handle,2,g.ctypes.data,.1);after=m.semantic_parameters()
   actual=float((before[0,j,k]-after[0,j,k])/.1)
   self.assertAlmostEqual(actual,(scores[1]-scores[0])/.02,delta=1e-5)
 def test_19_learning_changes_vector_without_changing_id(self):
  m=self.make();self.sequence(m);ids,v=self.encoded(m);g=np.linspace(-.2,.4,64,dtype=np.float32)
  for _ in range(20):self.fx.semantic_fixture_gradient(m.handle,2,g.ctypes.data,.1)
  ids2,v2=self.encoded(m);np.testing.assert_array_equal(ids,ids2);self.assertGreater(np.linalg.norm(v-v2),.001)
  self.assertLess(float(g@v2),float(g@v))
 def test_20_saved_features_survive_new_occurrence(self):
  a=self.make();b=self.make();g=np.linspace(-.2,.4,64,dtype=np.float32)
  for m in [a,b]:self.sequence(m);self.fx.semantic_fixture_gradient(m.handle,2,g.ctypes.data,0)
  self.sequence(b,1000);self.fx.fixture_stage(b.handle,0,0,90,900,900,901,100)
  for m in [a,b]:self.fx.semantic_fixture_saved_gradient(m.handle,2,g.ctypes.data,.01)
  np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
 def test_21_eval_and_frozen_regions_freeze_semantic_weights(self):
  a=self.make();a.configure_training_policy(np.array([1]*6+[0]*12,np.int32));before=a.semantic_parameters()
  x=np.arange(300,dtype=np.uint8);a.run(x,mode='periodic');np.testing.assert_array_equal(before[:4],a.semantic_parameters()[:4])
  b=self.make();b.run(x,mode='periodic');self.assertGreater(np.linalg.norm(b.semantic_parameters()),0)
  before=b.semantic_parameters();b.run(x);np.testing.assert_array_equal(before,b.semantic_parameters())
 def test_22_teacher_content_input_and_router_cue_are_same_vector(self):
  m=self.make();m.utility['innovation']=1.;m.lib.utility_configure(m.handle,np.array([m.utility[k] for k in __import__('model').FIELDS],np.float32).ctypes.data_as(ct.POINTER(ct.c_float)))
  m.run(np.arange(60,dtype=np.uint8),mode='periodic',lr=0,lp=0)
  for r in [3,4,6,10,14,15,16]:
   cue=np.zeros(64,np.float32);q=cue.copy();self.fx.fixture_current_cues(m.handle,r,cue.ctypes.data);self.fx.semantic_fixture_q(m.handle,r,q.ctypes.data)
   np.testing.assert_array_equal(cue,q);np.testing.assert_array_equal(q,m.content_events()[core.NAMES[r]]['code'])
 def test_23_dictionary_eviction_does_not_remap_vector(self):
  m=self.make();self.sequence(m);a=self.encoded(m);self.fx.semantic_fixture_clear_dictionary(m.handle)
  for aa,bb in zip(a,self.encoded(m)):np.testing.assert_array_equal(aa,bb)
 def test_24_binding_abi_rejected_and_fixed_control_no_updates(self):
  m=self.make();b=self.make('binding')
  with self.assertRaises(ValueError):m.import_composition(b.composition_state())
  f=self.make('semantics_fixed');f.run(np.arange(300,dtype=np.uint8),mode='periodic');self.assertEqual(np.linalg.norm(f.semantic_parameters()),0)
 def test_25_disabled_composition_matches_binding(self):
  a=self.make();b=self.make('binding')
  for m in [a,b]:m.configure_composition({'enabled':False})
  x=np.arange(200,dtype=np.uint8)
  np.testing.assert_array_equal(a.run(x,mode='periodic',logits=True)[1],b.run(x,mode='periodic',logits=True)[1]);np.testing.assert_array_equal(a.flat,b.flat)
 def test_26_fitted_copy_preserves_semantics_and_resets_stream(self):
  m=self.make();m.run(np.arange(100,dtype=np.uint8),mode='periodic');c=m.frozen_copy();self.addCleanup(c.close)
  np.testing.assert_array_equal(m.semantic_parameters(),c.semantic_parameters());self.assertTrue(all(v['seq']==0 for v in c.content_events().values()))
del BindingTests
if __name__=='__main__':unittest.main(verbosity=2)
