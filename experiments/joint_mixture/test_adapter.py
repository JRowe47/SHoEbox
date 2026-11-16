"""EXP01-E numerical, causality and continuation contracts."""
import unittest,hashlib
import numpy as np
from .adapter import Adapter,prefix_labels
from experiments.joint_readout.stream import new_cursor,advance
from .tasks import dataset,shifted
from experiments.constituent_readout.adapter import Adapter as D3

class MixtureTests(unittest.TestCase):
 def test_historical_pair_numerical_parity(self):
  ep=shifted(dataset(4,'train')[0],29)
  for joint in (False,True):
   for kind in ('native','paired','paired_latest'):
    with Adapter(kind=kind,joint=joint) as a,D3(kind=kind,joint=joint) as b:
     self.assertEqual(a.train_episode(ep,999),b.train_episode(ep,999))
     np.testing.assert_array_equal(a.log_probs(),b.log_probs())
     a.finish();b.finish()
     with a.clone() as x,b.clone() as y:self.assertEqual(x.score(ep),y.score(ep))
 def test_checkpoint_all_partial_phases(self):
  data=bytes(np.random.default_rng(83).integers(0,256,193,dtype=np.uint8))
  for at in range(1,66):
   with Adapter() as a:
    a.run(data[:at],data[1:at+1])
    with Adapter.loads(a.dumps()) as b:
     self.assertEqual(a.run(data[at:-1],data[at+1:]),b.run(data[at:-1],data[at+1:]))
     self.assertEqual(a.dumps(),b.dumps())
 def test_cursor_arbitrary_interruption(self):
  with Adapter() as a:
   cursor=new_cursor(all_bytes=True);cursor,_=advance(a,cursor,83)
   with Adapter.loads(a.dumps()) as b:
    whole,_=advance(a,cursor,1000)
    partial,_=advance(b,cursor,371)
    with Adapter.loads(b.dumps()) as c:
     resumed,_=advance(c,partial,629)
     self.assertEqual(a.dumps(),c.dumps());self.assertEqual(whole,resumed)
 def test_causal_readonly_and_generic_counts(self):
  with Adapter() as a:
   a.run(b'abcdef',b'bcdefg');inv=a.inventory()
   self.assertEqual(inv['gate_targets'],6);self.assertEqual(inv['local_training_windows'],0)
   self.assertEqual(inv['added_persistent_sequence_state_bytes'],0)
   self.assertLessEqual(inv['parameters'],500000)
   with a.clone() as f:
    before=f.dumps();p=f.log_probs();f.gate();self.assertEqual(before,f.dumps())
    with Adapter.loads(before) as g:np.testing.assert_array_equal(p,g.log_probs())
   ep=dataset(4,'dev')[0]
   from dataclasses import replace
   for at in (9,32,63,64,80):
    changed=replace(ep,data=ep.data[:at]+bytes([202])*(len(ep.data)-at))
    np.testing.assert_array_equal(prefix_labels(ep)[:at],prefix_labels(changed)[:at])
 def test_core_actually_updates(self):
  ep=shifted(dataset(4,'train')[0],11)
  with Adapter() as a:
   old=a.core_dumps();a.train_episode(ep,999);a.finish();self.assertNotEqual(old,a.core_dumps())
   with Adapter.loads(a.dumps()) as b:self.assertEqual(a.core_dumps(),b.core_dumps())
 def test_mixture_normalization_and_loss_parity(self):
  import ctypes as C
  with Adapter() as a:
   a.run(b'abcde'*30,b'bcdea'*30)
   with a.clone() as f:
    for byte in b'abcd':f.observe(byte)
    p=f.log_probs();self.assertAlmostEqual(float(np.exp(p).sum()),1.,places=5)
    before=f.dumps()
    for target in (0,32,97,255):
     with Adapter.loads(before) as b:
      loss=C.c_double();b.call('qr_stream_score',b'X'+bytes([target]),2,C.byref(loss))
      with Adapter.loads(before) as c:
       c.observe(88);self.assertAlmostEqual(loss.value,-float(c.log_probs()[target]),places=5)
 def test_budget_and_policy_validation(self):
  for budget in (500000,1000000,5000000):
   with Adapter(parameters=budget) as a:
    self.assertLessEqual(a.inventory()['parameters'],budget)
    self.assertEqual(a.inventory()['readout_parameters'],9965)
    with a.clone(intervention='gate_off') as f:self.assertEqual(f.gate(),0.)
    with a.clone(intervention='gate_on') as f:self.assertEqual(f.gate(),1.)
    with a.clone() as f:
     with self.assertRaises(RuntimeError):f.call('qr_query_policy',0)
if __name__=='__main__':unittest.main()
