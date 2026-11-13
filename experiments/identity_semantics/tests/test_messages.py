"""New message contracts, exact zero-control parity, Jacobians and lifecycle."""
import ctypes as ct
import subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core

class MessageTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  (ROOT/'results').mkdir(exist_ok=True)
  path=ROOT/'results'/f'messages_fixture_e{core.E}.so'
  subprocess.run(['g++','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(ROOT/'tests/messages_fixture.cpp'),'-o',str(path)],check=True)
  cls.fx=ct.CDLL(str(path));p=ct.c_void_p;i=ct.c_int;f=ct.c_float
  signatures=dict(message_fixture_weights=[p],message_fixture_payload=[p,i,i,i,f,i,p,p],message_fixture_semantic_credit=[p,i,p,f],
   message_fixture_pulse=[p,i,i,f],message_fixture_feedback_isolate=[p,i],message_fixture_state=[p,i,p,p],message_fixture_credit=[p,i,i,p,p],message_fixture_route_keys=[p,i,p],
   message_fixture_gate_gradient=[p,i,p,p],message_fixture_gate_objective=[p,i,i,f,p,p],semantic_fixture_weight=[p,i,i,i,f],
   semantic_fixture_change=[p,i,i,i,ct.c_int64],relay_fixture_forward=[p,i,i],
   relay_fixture_schedule=[p,i],relay_fixture_teach=[p,i,p,i,f,i],relay_fixture_public_output=[p,i,f])
  for name,args in signatures.items():getattr(cls.fx,name).argtypes=args
  cls.text=np.frombuffer((b'The king and queen walked through the city.\n'*30),np.uint8)
 def make(self,kernel='messages',warm=True,**policy):
  m=fresh_model(kernel=kernel);self.addCleanup(m.close)
  if policy:m.configure_messages(policy)
  if warm:
   if kernel=='messages':self.fx.message_fixture_weights(m.handle)
   m.run(self.text[:100])
  return m
 def payload(self,m,r,src=-1,j=0,delta=0,reencode=0):
  a=np.zeros((3,64),np.float32);ids=np.zeros(2,np.uint64)
  self.fx.message_fixture_payload(m.handle,r,src,j,delta,reencode,a.ctypes.data,ids.ctypes.data);return a,ids
 def state(self,m,r):
  a=np.zeros((3,64),np.float32);ids=np.zeros(2,np.int32);self.fx.message_fixture_state(m.handle,r,a.ctypes.data,ids.ctypes.data);return a,ids
 def test_01_child_payload_changes_input_preserves_identity_structure_context(self):
  for seed in (1,2):
   for src,r in ((10,14),(10,16),(12,14),(6,17),(11,17)):
    m=fresh_model(seed,kernel='messages');self.addCleanup(m.close);m.run(self.text[:100])
    a,ids=self.payload(m,r);b,newids=self.payload(m,r,src,17,.2)
    self.assertGreater(np.linalg.norm(a[0]-b[0]),.001)
    np.testing.assert_array_equal(ids,newids);np.testing.assert_array_equal(a[1:],b[1:])
 def test_02_semantic_edge_jacobians_include_reused_occurrences(self):
  g=np.linspace(-.2,.25,64,dtype=np.float32)
  for src,r in ((10,14),(10,16),(12,14),(6,17),(11,17)):
   m=self.make()
   for j,k in ((3,0),(17,3),(51,7)):
    scores=[]
    for delta in (-.003,.003):
     self.fx.semantic_fixture_weight(m.handle,src,j,k,delta)
     q,_=self.payload(m,r,reencode=1);scores.append(float(q[0].astype(float)@g))
     self.fx.semantic_fixture_weight(m.handle,src,j,k,-delta)
    self.payload(m,r,reencode=1);before=m.semantic_parameters();weights=m.flat.copy()
    self.fx.message_fixture_semantic_credit(m.handle,r,g.ctypes.data,.05)
    actual=(before[src-2,j,k]-m.semantic_parameters()[src-2,j,k])/.05
    self.assertAlmostEqual(actual,(scores[1]-scores[0])/.006,delta=2e-5)
    np.testing.assert_array_equal(weights,m.flat)
 def test_03_saved_semantic_occurrence_survives_replacement_and_relay(self):
  g=np.linspace(-.2,.3,64,dtype=np.float32);a=self.make();b=self.make()
  for m in (a,b):self.payload(m,14)
  # Alter the held source after the consumer has saved its real derivatives.
  self.fx.semantic_fixture_change(b.handle,10,0,2,100)
  self.fx.semantic_fixture_change(b.handle,10,0,3,10000)
  for m in (a,b):self.fx.message_fixture_semantic_credit(m.handle,14,g.ctypes.data,.03)
  np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
  c=self.make();self.fx.relay_fixture_forward(c.handle,10,12)
  before=c.semantic_parameters();self.payload(c,14);self.fx.message_fixture_semantic_credit(c.handle,14,g.ctypes.data,.03)
  self.assertGreater(np.linalg.norm(c.semantic_parameters()[8]-before[8]),0)
  np.testing.assert_array_equal(c.semantic_parameters()[10],before[10])
 def test_04_all_gains_zero_reproduces_relay_training_exactly(self):
  a=self.make('relays',warm=False);b=self.make(warm=False,payload_gain=0,feedback_activation=0,feedback_routing=0)
  for x in np.array_split(self.text,4):
   np.testing.assert_array_equal(a.run(x,mode='periodic',logits=True)[1],b.run(x,mode='periodic',logits=True)[1])
  np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
  self.assertEqual(a.relay_teaching_status(),b.relay_teaching_status())
 def test_05_feedback_changes_actual_routing_cues_for_all_g5_consumers(self):
  for r in (9,10,11,12):
   a=self.make();b=self.make();self.fx.message_fixture_pulse(b.handle,14,23,1.)
   for m in (a,b):m.run(self.text[100:101])
   self.assertGreater(np.linalg.norm(self.state(a,r)[0][1]-self.state(b,r)[0][1]),.001)
   self.assertEqual(a.content_events()[core.NAMES[r]]['id'],b.content_events()[core.NAMES[r]]['id'])
 def test_06_feedback_zero_preserves_content_only_routing(self):
  a=self.make(feedback_activation=0,feedback_routing=0);b=self.make(feedback_activation=0,feedback_routing=0)
  self.fx.message_fixture_pulse(b.handle,14,23,1.)
  for m in (a,b):m.run(self.text[100:101])
  for r in (9,10,11,12):np.testing.assert_array_equal(self.state(a,r)[0][1],self.state(b,r)[0][1])
 def test_07_real_feedback_edge_jacobian(self):
  g=np.linspace(-.1,.15,64,dtype=np.float32);src=14
  for r in (9,10,11,12):
   for j in (3,23,51):
    a=self.make();self.fx.message_fixture_feedback_isolate(a.handle,r);a.run(self.text[100:101]);gradient=np.zeros(64,np.float32)
    self.fx.message_fixture_credit(a.handle,r,src,g.ctypes.data,gradient.ctypes.data)
    scores=[];choices=[]
    for delta in (-.002,.002):
     m=self.make();self.fx.message_fixture_feedback_isolate(m.handle,r)
     self.fx.message_fixture_pulse(m.handle,src,j,delta);m.run(self.text[100:101])
     state,choice=self.state(m,r);scores.append(float(state[0].astype(float)@g));choices.append(choice)
    np.testing.assert_array_equal(*choices)
    self.assertAlmostEqual(float(gradient[j]),(scores[1]-scores[0])/.004,delta=3e-5)
 def test_08_saved_smooth_route_gate_jacobian(self):
  m=self.make();g=np.linspace(-.1,.2,64,dtype=np.float32);r=11;gradient=np.zeros(64,np.float32)
  self.fx.message_fixture_gate_gradient(m.handle,r,g.ctypes.data,gradient.ctypes.data)
  for j in np.argsort(abs(gradient))[-3:]:
   values=[]
   for delta in (-.003,.003):
    v=np.zeros(1,np.float32);self.fx.message_fixture_gate_objective(m.handle,r,int(j),delta,g.ctypes.data,v.ctypes.data);values.append(v[0])
   self.assertAlmostEqual(float(gradient[j]),float((values[1]-values[0])/.006),delta=2e-5)
 def test_09_feedback_can_switch_chosen_expert_near_boundary(self):
  # Use the production sparse mask. Pulse an admitted coordinate, then place
  # equal-norm keys on either side of a separating plane in the common mask.
  a=self.make();a.run(self.text[100:101]);qa=self.state(a,11)[0][1]
  k=next(k for k in range(a.fixed['starts'][11],a.fixed['starts'][12]) if a.fixed['slots'][k,0]==14 and a.fixed['slots'][k,2])
  coordinate=int(a.fixed['perm'][k,np.argmax(abs(qa))])
  b=self.make();self.fx.message_fixture_pulse(b.handle,14,coordinate,1.);b.run(self.text[100:101]);qb=self.state(b,11)[0][1]
  common=np.zeros(64,np.float32);common[np.intersect1d(np.argsort(abs(qa))[-16:],np.argsort(abs(qb))[-16:])]=1
  mid=(qa+qb)/2*common;d=(qb-qa)*common;d-=mid*(d@mid)/(mid@mid);d/=np.linalg.norm(d)
  a=self.make();b=self.make();self.fx.message_fixture_pulse(b.handle,14,coordinate,1.)
  for m in (a,b):self.fx.message_fixture_route_keys(m.handle,11,d.ctypes.data);m.run(self.text[100:101])
  self.assertNotEqual(self.state(a,11)[1][0],self.state(b,11)[1][0])
 def test_10_resume_chunking_and_policy_roundtrip(self):
  m=self.make(warm=False,payload_gain=.5,feedback_activation=.2,feedback_routing=.3)
  m.run(self.text[:300],mode='periodic')
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'m.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
   self.assertEqual(m.fingerprint(),n.fingerprint());self.assertEqual(m.messages,n.messages)
   a=m.run(self.text[300:600],mode='periodic',logits=True)[1]
   b=np.concatenate([n.run(x,mode='periodic',logits=True)[1] for x in np.array_split(self.text[300:600],3)])
   np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
 def test_11_diagnostic_parity_freezing_and_channel_interventions(self):
  m=self.make();m.run(self.text[:300],mode='periodic');a=m.evaluation_copy();b=m.evaluation_copy(diagnostic=True)
  self.addCleanup(a.close);self.addCleanup(b.close)
  np.testing.assert_array_equal(a.run(self.text,logits=True)[1],b.run(self.text,logits=True)[1])
  base=b.contribution_metrics()
  for kind in ('neural_output','expert_transform','semantic_payload','semantic_residual','feedback_channel','relay_context','whole_source'):
   c=m.evaluation_copy(diagnostic=True);self.addCleanup(c.close);c.configure_contribution(kind,'E4')
   before=c.semantic_parameters();c.run(self.text);np.testing.assert_array_equal(before,c.semantic_parameters())
   if kind!='whole_source':
    self.assertEqual([v['event_hash'] for v in base.values()],[v['event_hash'] for v in c.contribution_metrics().values()])
   with self.assertRaises(ValueError):c.run(self.text,mode='periodic')
 def test_12_no_missing_credit_or_teaching_after_training_and_drain(self):
  m=self.make(warm=False);m.run(self.text,mode='periodic');m.finish_document(mode='periodic')
  for v in m.message_metrics().values():self.assertEqual(v['missing_feedback_producers'],0)
  for v in m.relay_teaching_status().values():self.assertEqual(v['expired_producers'],0);self.assertEqual(v['duplicate_schedules'],0)
  for v in m.composition_metrics().values():self.assertEqual(v['expired_input_edges'],0)
 def test_13_validation_rejects_bad_policy_and_foreign_abi(self):
  m=self.make(warm=False);old=self.make('relays',warm=False)
  with self.assertRaises(ValueError):m.import_composition(old.composition_state())
  for policy in ({'payload_gain':-1},{'feedback_routing':2},{'semantic_lr':float('nan')},{'nope':1}):
   with self.assertRaises(ValueError):m.configure_messages(policy)
  m.run(self.text[:2])
  with self.assertRaises(ValueError):m.configure_messages({'payload_gain':.1})
 def test_14_explicit_old_checkpoint_conversion(self):
  sys.path.insert(0,str(ROOT/'src'));from convert_to_messages import convert
  m=self.make('relays',warm=False);m.run(self.text[:300],mode='periodic')
  with tempfile.TemporaryDirectory() as tmp:
   old=Path(tmp)/'old.npz';new=Path(tmp)/'new.npz';m.save(old);convert(old,new)
   n=Model.load(new);self.addCleanup(n.close);np.testing.assert_array_equal(m.flat,n.flat)
   np.testing.assert_array_equal(m.semantic_parameters(),n.semantic_parameters())
   self.assertTrue(all(e['seq']==0 for e in n.content_events().values()))
   with self.assertRaises(ValueError):convert(old,new)

 def test_15_semantic_learning_zero_freezes_all_adapters(self):
  for policy in ({'semantic_lr':0},{'semantic_lr':0,'payload_gain':0,'feedback_activation':0,'feedback_routing':0}):
   m=self.make(warm=False,**policy);before=m.semantic_parameters();flat=m.flat.copy()
   m.run(self.text,mode='periodic');np.testing.assert_array_equal(before,m.semantic_parameters())
   self.assertGreater(np.linalg.norm(m.flat-flat),0);self.assertTrue(all(v==0 for v in m.semantic_updates().values()))
 def test_16_relay_target_uses_saved_message_eligibility_exactly_once(self):
  a=self.make(warm=False);a.run(self.text[:100],mode='periodic');self.fx.relay_fixture_schedule(a.handle,11)
  b=a.snapshot_copy();self.addCleanup(b.close);self.fx.relay_fixture_public_output(b.handle,11,.97)
  self.fx.relay_fixture_forward(b.handle,14,11)
  target=np.linspace(-.2,.3,64,dtype=np.float32)
  for m in (a,b):self.fx.relay_fixture_teach(m.handle,11,target.ctypes.data,6,.03,1)
  np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
  before=b.semantic_parameters();flat=b.flat.copy();self.fx.relay_fixture_teach(b.handle,11,target.ctypes.data,6,.03,1)
  np.testing.assert_array_equal(before,b.semantic_parameters());np.testing.assert_array_equal(flat,b.flat)
 def test_17_source_freeze_applies_to_transferred_semantic_credit(self):
  for policy in ('freeze','zero_task_rate'):
   m=self.make()
   if policy=='freeze':
    frozen=np.zeros(18,np.int32);frozen[3]=1;m.configure_training_policy(frozen)
   else:
    scales=np.ones((18,2),np.float32);scales[3,0]=0;m.configure_teaching(scales)
   g=np.linspace(-.2,.3,64,dtype=np.float32);before=m.semantic_parameters();self.payload(m,7)
   self.fx.message_fixture_semantic_credit(m.handle,7,g.ctypes.data,.03)
   np.testing.assert_array_equal(before[1],m.semantic_parameters()[1])
   self.assertGreater(np.linalg.norm(before[5]-m.semantic_parameters()[5]),0)

if __name__=='__main__':unittest.main()
