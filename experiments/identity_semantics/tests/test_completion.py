"""Completed role frames, recursive code Jacobians and version-34 lifecycle."""
import ctypes as ct
import subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core

class CompletionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  (ROOT/'results').mkdir(exist_ok=True)
  path=ROOT/'results'/f'completion_fixture_e{core.E}.so'
  subprocess.run(['g++','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(ROOT/'tests/completion_fixture.cpp'),'-o',str(path)],check=True)
  cls.fx=ct.CDLL(str(path));p=ct.c_void_p;i=ct.c_int;f=ct.c_float;l=ct.c_int64
  signatures=dict(completion_fixture_chain=[p,i,f,i,p,p],completion_fixture_update=[p,p,f],
   completion_fixture_grow=[p,p],completion_fixture_update_grown=[p],
   completion_fixture_gate=[p,i,f],completion_fixture_source=[p,i,i,l,l],completion_fixture_seed_frame=[p,i],
   completion_fixture_prepare=[p,i,i,i,p],completion_fixture_erase_patterns=[p],message_fixture_weights=[p],
   semantic_fixture_weight=[p,i,i,i,f],relay_fixture_forward=[p,i,i],relay_fixture_schedule=[p,i],
   completion_fixture_clock_case=[p,i,p],relay_fixture_teach=[p,i,p,i,f,i],relay_fixture_public_output=[p,i,f])
  for name,args in signatures.items():getattr(cls.fx,name).argtypes=args
  cls.text=np.frombuffer(b'The king walked across the street. '*120,np.uint8)
 def make(self,**policy):
  m=fresh_model(kernel='completion');self.addCleanup(m.close)
  if policy:m.configure_messages(policy)
  return m
 def chain(self,m,j=-1,pulse=0,relay=0):
  a=np.zeros((2,64),np.float32);ids=np.zeros(2,np.uint64)
  self.fx.completion_fixture_chain(m.handle,j,pulse,relay,a.ctypes.data,ids.ctypes.data);return a,ids
 def source(self,m,src,ident,begin,end):self.fx.completion_fixture_source(m.handle,src,ident,begin,end)
 def prepare(self,m,r=6,drain=0,learn=0):
  a=np.zeros(32,np.int64);self.fx.completion_fixture_prepare(m.handle,r,drain,learn,a.ctypes.data);return a
 def test_two_bindings_export_changed_child_content_with_fixed_identity(self):
  m=self.make();self.fx.message_fixture_weights(m.handle)
  a,ids=self.chain(m);b,other=self.chain(m,17,.2)
  self.assertGreater(np.linalg.norm(a[0]-b[0]),1e-4)
  np.testing.assert_array_equal(ids,other);np.testing.assert_array_equal(a[1],b[1])
 def test_two_hop_saved_jacobian_and_repeated_role_accumulation(self):
  m=self.make();self.fx.message_fixture_weights(m.handle);g=np.linspace(-.2,.3,64,dtype=np.float32)
  for r,j,k in ((3,17,3),(7,23,4),(10,41,5)):
   scores=[]
   for delta in (-.004,.004):
    self.fx.semantic_fixture_weight(m.handle,r,j,k,delta);a,_=self.chain(m);scores.append(float(a[0].astype(float)@g))
    self.fx.semantic_fixture_weight(m.handle,r,j,k,-delta)
   self.chain(m);before=m.semantic_parameters();self.fx.completion_fixture_update(m.handle,g.ctypes.data,.05)
   actual=(before[r-2,j,k]-m.semantic_parameters()[r-2,j,k])/.05
   self.assertAlmostEqual(actual,(scores[1]-scores[0])/.008,delta=2e-5)
 def test_learned_payload_gate_jacobian(self):
  m=self.make();self.fx.message_fixture_weights(m.handle);g=np.linspace(-.2,.3,64,dtype=np.float32);scores=[]
  for delta in (-.005,.005):
   self.fx.completion_fixture_gate(m.handle,7,delta);a,_=self.chain(m);scores.append(float(a[0].astype(float)@g));self.fx.completion_fixture_gate(m.handle,7,-delta)
  self.chain(m);before=m.completion_gates();self.fx.completion_fixture_update(m.handle,g.ctypes.data,.1)
  self.assertAlmostEqual(float((before[5]-m.completion_gates()[5])/.1),(scores[1]-scores[0])/.01,delta=2e-5)
 def test_saved_ancestry_survives_source_replacement_and_relay(self):
  a=self.make();b=self.make();g=np.linspace(-.2,.3,64,dtype=np.float32)
  for m in (a,b):self.fx.message_fixture_weights(m.handle);self.chain(m,relay=1)
  for i in range(180):self.source(b,3,i,2*i,2*i+2)
  for m in (a,b):self.fx.completion_fixture_update(m.handle,g.ctypes.data,.1)
  np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
  self.assertGreater(np.linalg.norm(a.semantic_parameters()[1]-a.semantic_parameters()[0]),0)
 def test_primary_occurrences_are_disjoint_and_side_updates_do_not_commit(self):
  m=self.make();self.source(m,3,70,0,2)
  self.source(m,2,1,0,2);self.assertEqual(self.prepare(m)[0],0)
  self.source(m,2,2,2,4);first=self.prepare(m);self.assertEqual(first[1],1)
  self.assertEqual(tuple(first[6:8]),(0,4));self.assertEqual(first[5],3)
  roles=first[8:17].reshape(3,3);self.assertEqual(list(roles[:,0]),[2,2,3]);self.assertEqual(list(roles[:2,1]),[1,2])
  self.source(m,3,71,4,6);self.assertEqual(self.prepare(m)[0],0)
  self.source(m,2,3,4,6);self.assertEqual(self.prepare(m)[0],0)
  self.source(m,2,4,6,8);emitted=self.prepare(m);self.assertEqual(emitted[0],1)
  tail=self.prepare(m,drain=1);self.assertEqual(tuple(tail[6:8]),(4,8));self.assertEqual(tail[4],4)
 def test_recognition_changes_completion_of_unseen_frame_transition(self):
  outcomes=[]
  for familiar in (False,True):
   m=self.make();self.source(m,3,70,0,2)
   for i in range(4):
    self.source(m,2,i+1,2*i,2*i+2)
    if i%2:
     if familiar:self.fx.completion_fixture_seed_frame(m.handle,6)
     result=self.prepare(m)
   outcomes.append(tuple(result[:2]))
  self.assertEqual(outcomes,[(1,1),(0,2)])
 def test_drain_preserves_odd_primary_tail_without_reusing_previous_occurrence(self):
  m=self.make();self.source(m,3,70,0,2)
  for i in range(3):self.source(m,2,i+1,2*i,2*i+2);self.prepare(m)
  self.prepare(m,drain=1);last=self.prepare(m,drain=1)
  self.assertEqual(last[0],1);self.assertEqual(tuple(last[6:8]),(4,6));self.assertEqual(last[5],2)
  self.assertEqual(self.prepare(m,drain=1)[0],0)
 def test_hierarchy_spans_and_publication_rates(self):
  m=self.make();m.run(self.text,mode='periodic');m.finish_document(mode='periodic')
  metrics=m.composition_metrics();means={r:v['span_sum']/max(v['published'],1) for r,v in metrics.items()}
  self.assertLess(means['L2'],means['L3']);self.assertLess(means['L3'],means['F4']);self.assertLess(means['F4'],means['G5'])
  self.assertGreater(metrics['L2']['published'],metrics['L3']['published']);self.assertGreater(metrics['L3']['published'],metrics['F4']['published'])
  for v in m.completion_metrics().values():self.assertEqual(v['expired_semantic_handles'],0);self.assertEqual(v['semantic_capacity_prunes'],0);self.assertEqual(v['semantic_budget_omissions'],0)
  for v in m.relay_teaching_status().values():self.assertEqual(v['expired_producers'],0);self.assertEqual(v['duplicate_schedules'],0)
  before={k:v['published'] for k,v in metrics.items()};m.finish_document(mode='periodic')
  self.assertEqual(before,{k:v['published'] for k,v in m.composition_metrics().items()})
 def test_checkpoint_and_chunked_continuation_include_pending_frames_and_credit(self):
  m=self.make();m.run(self.text[:1177],mode='periodic')
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'m.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
   self.assertEqual(m.fingerprint(),n.fingerprint())
   a=m.run(self.text[1177:2200],mode='periodic',logits=True)[1]
   b=np.concatenate([n.run(x,mode='periodic',logits=True)[1] for x in np.array_split(self.text[1177:2200],7)])
   np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
 def test_diagnostic_parity_and_content_preservation(self):
  m=self.make();m.run(self.text[:500],mode='periodic');a=m.evaluation_copy();b=m.evaluation_copy(diagnostic=True)
  self.addCleanup(a.close);self.addCleanup(b.close)
  np.testing.assert_array_equal(a.run(self.text[:800],logits=True)[1],b.run(self.text[:800],logits=True)[1]);base=b.contribution_metrics()
  for kind in ('expert_transform','semantic_payload','semantic_residual','content_expert'):
   c=m.evaluation_copy(diagnostic=True)
   try:
    c.configure_contribution(kind,'S3');c.run(self.text[:800]);self.assertEqual([v['event_hash'] for v in base.values()],[v['event_hash'] for v in c.contribution_metrics().values()])
   finally:c.close()
 def test_source_freeze_and_semantic_learning_disable(self):
  m=self.make(semantic_lr=0);before=m.semantic_parameters();m.run(self.text[:600],mode='periodic')
  np.testing.assert_array_equal(before,m.semantic_parameters());np.testing.assert_array_equal(m.completion_gates(),0)
  m=self.make();self.fx.message_fixture_weights(m.handle);frozen=np.zeros(18,np.int32);frozen[3]=1;m.configure_training_policy(frozen)
  self.chain(m);before=m.semantic_parameters();g=np.ones(64,np.float32)*.1;self.fx.completion_fixture_update(m.handle,g.ctypes.data,.1)
  np.testing.assert_array_equal(before[1],m.semantic_parameters()[1]);self.assertGreater(np.linalg.norm(before[5]-m.semantic_parameters()[5]),0)
 def test_foreign_abi_rejected_and_parameter_count(self):
  m=self.make();old=fresh_model(kernel='messages')
  try:
   self.assertEqual(m.parameter_count,old.parameter_count+16)
   with self.assertRaises(ValueError):m.import_composition(old.composition_state())
  finally:old.close()
 def test_duplicate_primary_occurrence_role_is_rejected(self):
  m=self.make();first=int(m.fixed['starts'][6]);m.fixed['slots'][first+1,1]=m.fixed['slots'][first,1]
  with self.assertRaises(ValueError):m.configure_composition()
 def test_explicit_conversion_preserves_fitted_weights_and_resets_events(self):
  sys.path.insert(0,str(ROOT/'src'));from convert_to_completion import convert
  for kernel in ('relays','messages'):
   old=fresh_model(kernel=kernel)
   try:
    old.run(self.text[:400],mode='periodic')
    with tempfile.TemporaryDirectory() as tmp:
     source=Path(tmp)/'old.npz';target=Path(tmp)/'new.npz';old.save(source);convert(source,target);m=Model.load(target)
     try:
      np.testing.assert_array_equal(old.flat,m.flat);np.testing.assert_array_equal(old.semantic_parameters(),m.semantic_parameters())
      self.assertTrue(all(v['seq']==0 for v in m.content_events().values()));np.testing.assert_array_equal(m.completion_gates(),0)
      self.assertEqual(m.runtime_kernel,'completion')
     finally:m.close()
     with self.assertRaises(ValueError):convert(source,target)
   finally:old.close()
 def test_relay_teaching_uses_saved_recursive_code_once(self):
  a=self.make();a.run(self.text[:600],mode='periodic');self.fx.relay_fixture_schedule(a.handle,11)
  b=a.snapshot_copy();self.addCleanup(b.close);self.fx.relay_fixture_public_output(b.handle,11,.9);self.fx.relay_fixture_forward(b.handle,14,11)
  target=np.linspace(-.2,.3,64,dtype=np.float32)
  for m in (a,b):self.fx.relay_fixture_teach(m.handle,11,target.ctypes.data,6,.03,1)
  np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
  before=b.semantic_parameters();self.fx.relay_fixture_teach(b.handle,11,target.ctypes.data,6,.03,1);np.testing.assert_array_equal(before,b.semantic_parameters())

 def test_primary_clock_ignores_side_coverage_but_respects_real_gaps(self):
  for overlap in (0,1,2,3):
   with self.subTest(overlap=overlap):
    m=self.make();result=np.zeros(7,np.float64)
    self.fx.completion_fixture_clock_case(m.handle,overlap,result.ctypes.data)
    self.assertEqual(result[0],1.0)
    self.assertEqual(result[1],16)
    self.assertEqual(result[4],int(overlap>=2))
    self.assertEqual(result[6],1 if overlap>=2 else 2)
    if overlap==1:
     self.assertEqual(result[2],0)
     self.assertEqual(result[3],16)
 def test_primary_clocks_propagate_from_atoms_through_the_whole_graph(self):
  m=self.make();m.run(self.text,mode='periodic');m.finish_document(mode='periodic')
  events=m.content_events();clocks=m.completion_clocks()
  for region,event in events.items():
   with self.subTest(region=region):
    self.assertGreater(event['seq'],0)
    begin,end=clocks[region];covered_begin,covered_end=event['span']
    self.assertLess(begin,end)
    self.assertLessEqual(covered_begin,begin)
    self.assertLessEqual(end,covered_end)
  self.assertEqual(clocks['A1'],events['A1']['span'])
  self.assertEqual(clocks['B1'],events['B1']['span'])
  self.assertEqual(clocks['L2'],events['L2']['span'])
 def test_live_semantic_pool_grows_and_resumes_without_pruning_or_work_cutoff(self):
  m=self.make();result=np.zeros(4,np.uint64)
  self.fx.completion_fixture_grow(m.handle,result.ctypes.data)
  self.assertEqual(tuple(result),(32768,1,16416,0))
  self.assertEqual(m.completion_capacity(),32768)
  for metrics in m.completion_metrics().values():
   self.assertEqual(metrics['expired_semantic_handles'],0)
   self.assertEqual(metrics['semantic_budget_omissions'],0)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'grown.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
   self.assertEqual(n.completion_capacity(),32768)
   self.assertEqual(m.fingerprint(),n.fingerprint())
   for model in (m,n):self.fx.completion_fixture_update_grown(model.handle)
   np.testing.assert_array_equal(m.semantic_parameters(),n.semantic_parameters())
   self.assertEqual(m.fingerprint(),n.fingerprint())
   malformed=m.composition_state()[:-1]
   with self.assertRaises(ValueError):n.import_composition(malformed)
   n.reset();self.assertEqual(n.completion_capacity(),16384)
   self.assertTrue(all(event['seq']==0 for event in n.content_events().values()))

if __name__=='__main__':unittest.main(verbosity=2)
