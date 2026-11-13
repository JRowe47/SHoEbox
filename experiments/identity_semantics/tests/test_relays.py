"""Saved-producer teaching, transparent recognition, and exact continuation."""
import ctypes as ct
import tempfile
from pathlib import Path
import unittest
import numpy as np
from test_semantics import SemanticsBindingTests
from runtime import Model,core


class RelayTests(SemanticsBindingTests):
 kernel='relays'
 fixture_name='relays_fixture.so'
 fixture_source='relays_fixture.cpp'

 @classmethod
 def setUpClass(cls):
  super().setUpClass()
  for name,args in {
   'relay_fixture_schedule':[ct.c_void_p,ct.c_int],
   'relay_fixture_teach':[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_float,ct.c_int],
   'relay_fixture_public_output':[ct.c_void_p,ct.c_int,ct.c_float],
   'relay_fixture_expire':[ct.c_void_p,ct.c_int],
   'relay_fixture_forward':[ct.c_void_p,ct.c_int,ct.c_int],
   'relay_fixture_recognition':[ct.c_void_p,ct.c_int,ct.c_int],
   'relay_fixture_snapshot':[ct.c_void_p,ct.c_int,ct.c_void_p],
   'relay_fixture_familiar_pair':[ct.c_void_p,ct.c_int,ct.c_void_p],
  }.items():getattr(cls.fx,name).argtypes=args

 def scheduled(self,r=3):
  m=self.make();m.run(np.arange(40,dtype=np.uint8),mode='periodic',lr=.01,lp=.003)
  self.fx.relay_fixture_schedule(m.handle,r);return m

 def teach(self,m,r=3,relay=1,mode=6,rate=.03):
  target=np.linspace(-.2,.4,64,dtype=np.float32)
  self.fx.relay_fixture_teach(m.handle,r,target.ctypes.data,mode,rate,relay)
  return target

 def test_27_teacher_uses_saved_producer_not_public_state(self):
  a=self.scheduled();b=a.snapshot_copy();self.addCleanup(b.close)
  before=a.flat.copy();semantic_before=a.semantic_parameters()
  self.fx.relay_fixture_public_output(b.handle,3,.97)
  self.teach(a);self.teach(b)
  np.testing.assert_array_equal(a.flat,b.flat)
  np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
  self.assertGreater(np.linalg.norm(a.flat-before),0)
  self.assertGreater(np.linalg.norm(a.semantic_parameters()-semantic_before),0)

 def test_28_predictor_gradient_uses_saved_activation(self):
  m=self.scheduled();saved=np.zeros(257,np.float32)
  self.fx.relay_fixture_snapshot(m.handle,3,saved.ctypes.data)
  before={k:v.copy() for k,v in m.params.items()}
  q=self.teach(m);err=np.clip(saved[:64]-q,-1,1)
  h=saved[128:192];scale=.03/(1+float(h.astype(np.float64)@h.astype(np.float64)))
  np.testing.assert_allclose(m.params['P'][3]-before['P'][3],-scale*np.outer(err,h),atol=2e-8)
  np.testing.assert_allclose(m.params['pbias'][3]-before['pbias'][3],-scale*err,atol=2e-8)

 def test_29_relay_consumes_once_and_does_not_schedule(self):
  m=self.scheduled();self.teach(m);before=m.flat.copy();metrics=m.relay_teaching_status()['S2'].copy()
  self.teach(m);np.testing.assert_array_equal(before,m.flat)
  self.assertEqual(metrics,m.relay_teaching_status()['S2']);self.assertEqual(metrics['pending_tick'],0)
  self.assertGreater(metrics['relay_targets'],0)

 def test_30_recognition_and_intrinsic_structure_survive_two_relays(self):
  for known in (0,1):
   m=self.make();self.stage(m,6);self.prepare(m,6)
   self.fx.relay_fixture_recognition(m.handle,6,known)
   child=m.content_events()['L3'];origin=m.relay_event_status()['L3']['origin']
   before=m.composition_state(fitted=True).copy()
   for source,receiver in [(6,10),(10,14)]:self.fx.relay_fixture_forward(m.handle,source,receiver)
   result=m.content_events()['G5'];status=m.relay_event_status()['G5']
   for key in ('id','span','count','kind','recognized'):self.assertEqual(child[key],result[key])
   for key in ('children','code'):np.testing.assert_array_equal(child[key],result[key])
   self.assertEqual(origin,status['origin']);self.assertEqual(status['local_arity'],1)
   self.assertTrue(status['relayed']);self.assertEqual(status['confidence'],.875 if known else .25)
   self.assertEqual(m.composition_metrics()['G5']['compounds'],0)
   np.testing.assert_array_equal(before,m.composition_state(fitted=True))

 def test_31_familiar_units_still_form_provisional_new_relation(self):
  m=self.make();self.stage(m,6);self.prepare(m,6);self.fx.relay_fixture_recognition(m.handle,6,1)
  self.fx.relay_fixture_forward(m.handle,6,0)
  result=np.zeros(5,np.int32);self.fx.relay_fixture_familiar_pair(m.handle,2,result.ctypes.data)
  np.testing.assert_array_equal(result,[0,2,1,2,0])

 def test_32_nonlearning_target_expires_without_weights_changing(self):
  for mode in (0,2,4):
   m=self.scheduled();before=m.flat.copy();semantic=m.semantic_parameters()
   self.teach(m,mode=mode);np.testing.assert_array_equal(before,m.flat)
   np.testing.assert_array_equal(semantic,m.semantic_parameters())
   self.assertEqual(m.relay_teaching_status()['S2']['pending_tick'],0)

 def test_33_frozen_zero_scale_and_expired_provenance(self):
  for policy in ('frozen','zero','expired'):
   m=self.scheduled();before=m.flat.copy();semantic=m.semantic_parameters()
   if policy=='frozen':
    frozen=np.zeros(18,np.int32);frozen[3]=1;m.configure_training_policy(frozen)
   elif policy=='zero':
    scales=np.ones((18,2),np.float32);scales[3,1]=0;m.configure_teaching(scales)
   else:self.fx.relay_fixture_expire(m.handle,3)
   self.teach(m);np.testing.assert_array_equal(before,m.flat)
   np.testing.assert_array_equal(semantic,m.semantic_parameters())
   if policy=='expired':self.assertEqual(m.relay_teaching_status()['S2']['expired_producers'],1)

 def test_34_checkpoint_keeps_pending_forecast_and_exact_update(self):
  m=self.scheduled();self.fx.relay_fixture_public_output(m.handle,3,.9)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'relay.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
   self.assertEqual(n.runtime_kernel,'relays');self.assertEqual(m.fingerprint(),n.fingerprint())
   self.teach(m);self.teach(n);self.assertEqual(m.fingerprint(),n.fingerprint())

 def test_35_reset_and_old_abi(self):
  m=self.scheduled();old=self.make('semantics')
  with self.assertRaises(ValueError):m.import_composition(old.composition_state())
  with self.assertRaises(ValueError):m.import_composition(old.composition_state(fitted=True),fitted=True)
  m.finish_document(reset=True)
  self.assertTrue(all(v['pending_tick']==0 for v in m.relay_teaching_status().values()))

 def test_36_live_stream_accounts_for_every_forecast(self):
  m=self.make();data=np.frombuffer(b'one two three. one two four! '*300,np.uint8)
  m.run(data,mode='periodic',lr=.01,lp=.003)
  metrics=m.relay_teaching_status();self.assertGreater(metrics['L2']['relay_targets'],0)
  for name,row in metrics.items():
   self.assertEqual(row['expired_producers'],0,name);self.assertEqual(row['duplicate_schedules'],0,name)
   consumed=sum(row[k] for k in ('neural_targets','relay_targets','nonlearning_targets'))
   self.assertEqual(row['scheduled'],consumed+int(row['pending_tick']>0),name)

 def test_37_transport_preserves_pending_producer(self):
  a=self.scheduled();b=a.snapshot_copy();self.addCleanup(b.close)
  pending=b.relay_teaching_status()['S2'].copy()
  self.fx.relay_fixture_forward(b.handle,0,3)
  self.assertEqual(pending,b.relay_teaching_status()['S2'])
  self.teach(a);self.teach(b)
  np.testing.assert_array_equal(a.flat,b.flat)
  np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())

 def test_38_fitted_conversion_preserves_parameters_and_starts_clean(self):
  import sys
  root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
  from convert_to_relays import convert
  old=self.make('semantics');old.run(np.arange(100,dtype=np.uint8),mode='periodic')
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'old.npz';dest=Path(tmp)/'new.npz';old.save(source)
   summary=convert(source,dest);new=Model.load(dest);self.addCleanup(new.close)
   self.assertFalse(summary['exact_continuation']);self.assertEqual(new.parameter_count,1031936)
   np.testing.assert_array_equal(old.flat,new.flat)
   np.testing.assert_array_equal(old.extra_flat,new.extra_flat)
   np.testing.assert_array_equal(old.semantic_parameters(),new.semantic_parameters())
   np.testing.assert_array_equal(old.composition_state(fitted=True)[4:],new.composition_state(fitted=True)[4:])
   self.assertTrue(all(v['seq']==0 for v in new.content_events().values()))
   self.assertTrue(all(v['pending_tick']==0 for v in new.relay_teaching_status().values()))
   with self.assertRaises(ValueError):convert(source,dest)


del SemanticsBindingTests
if __name__=='__main__':unittest.main(verbosity=2)
