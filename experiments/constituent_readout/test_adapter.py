import unittest
from dataclasses import replace
import numpy as np
from .adapter import Adapter,PARAMETERS,prefix_labels
from .tasks import dataset,balanced_offsets,shifted,fold
from experiments.query_access.adapter import Adapter as Native

class PairTests(unittest.TestCase):
 def test_capacity_and_frozen_state(self):
  with Adapter() as a:
   self.assertEqual(a.inventory()['readout_parameters'],PARAMETERS)
   self.assertLessEqual(a.inventory()['parameters'],500000)
   self.assertEqual(a.inventory()['added_persistent_sequence_state_bytes'],0)
   a.train_episode(dataset(4,'train')[0],129);before=a.dumps()
   with a.clone() as f:
    original=f.inventory()['normalization_samples']
    for t in range(131):f.observe(t%256);f.log_probs()
    self.assertEqual(original,f.inventory()['normalization_samples'])
    self.assertLessEqual(f.inventory()['source_count'],64)
    self.assertAlmostEqual(float(np.exp(f.log_probs()).sum()),1,places=6)
   self.assertEqual(before,a.dumps())
 def test_native_parity(self):
  with Adapter(kind='native',joint=True) as a,Native(kind='native',joint=True) as b:
   for ep in balanced_offsets(dataset(8,'train')):
    self.assertEqual(a.train_episode(ep,999)[:3],b.train_episode(ep,999)[:3])
    np.testing.assert_array_equal(a.log_probs(),b.log_probs())
 def test_continuation_boundaries(self):
  ep=shifted(dataset(4,'train')[0],7);labs=prefix_labels(ep)
  for joint in (False,True):
   for at in (1,3,4,7,8,15,16,63,64,65,127,128):
    with Adapter(joint=joint) as a:
     a.run(ep.data[:at],ep.data[1:at+1],ep.mask[1:at+1],labs[:at])
     with Adapter.loads(a.dumps()) as b:
      args=(ep.data[at:-1],ep.data[at+1:],ep.mask[at+1:],labs[at:])
      self.assertEqual(a.run(*args),b.run(*args));self.assertEqual(a.dumps(),b.dumps())
 def test_causal_local_labels(self):
  ep=shifted(dataset(4,'train')[0],33);labs=prefix_labels(ep)
  for at in range(len(ep.data)-1):
   changed=replace(ep,data=ep.data[:at+1]+bytes([255])*(len(ep.data)-at-1))
   np.testing.assert_array_equal(labs[:at+1],prefix_labels(changed)[:at+1])
  for off in range(64):
   x=shifted(ep,off);l=prefix_labels(x)
   self.assertEqual(int((l[:,0]>=0).sum()),2)
   self.assertEqual(int((l[:,1:]>=0).sum()),2)
 def test_inference_has_no_label_inputs(self):
  with Adapter() as a:
   with a.clone() as b,a.clone() as c:
    for byte in b'observed prefix':b.observe(byte);c.observe(byte)
    np.testing.assert_array_equal(b.log_probs(),c.log_probs())
    with self.assertRaises(RuntimeError):b.run(b'x',b'y',b'\x00',np.zeros((1,5),np.int32))
  with Adapter() as a,Adapter() as b:
   a.run(b'prefix',b'aaaaaa',bytes(6));b.run(b'prefix',b'zzzzzz',bytes(6))
   self.assertEqual(a.dumps(),b.dumps())
 def test_split_and_metrics(self):
  from experiments.query_access.run import swapped_queries
  groups=[{e.split_key for e in dataset(256,s)} for s in ('train','dev','test')]
  self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
  ep=shifted(dataset(4,'dev')[0],31)
  with Adapter() as a,a.clone() as f:
   scored=f.score(ep,2,2);f.reset();correct=[]
   for i,byte in enumerate(ep.data):
    if ep.mask[i]:p=int(f.log_probs().argmax());correct.append(int(p==byte));byte=p
    if i+1<len(ep.data):f.observe(byte)
   self.assertEqual(scored[7:9],correct);self.assertEqual(scored[5],int(all(correct)))
 def test_checkpoint_validation(self):
  with Adapter() as a:
   with self.assertRaises(ValueError):Adapter.loads(a.dumps()[:-1])
   with a.clone(intervention='unpair') as f,Adapter.loads(f.dumps()) as b:
    np.testing.assert_array_equal(f.log_probs(),b.log_probs())
if __name__=='__main__':unittest.main()
