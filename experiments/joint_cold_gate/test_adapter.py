"""E8 calibrated-byte probability, causal updates and checkpoint contracts."""
import unittest,struct,ctypes as C,hashlib
import numpy as np
from .adapter import Adapter,prefix_labels
from .tasks import dataset,balanced_offsets
from .stream import new_cursor,advance
from experiments.joint_readout.adapter import Adapter as E1
class CalibratedTests(unittest.TestCase):
 def test_cached_probability_parity(self):
  from experiments.joint_calibrated.adapter import Adapter as E4
  with E4() as a:
   for ep in balanced_offsets(dataset(4,'train')):a.train_episode(ep,999)
   raw=bytearray(a.dumps()[32:]);struct.pack_into('<Q',raw,0,0x38455041514853)
   with Adapter.loads(hashlib.sha256(raw).digest()+raw) as b:
    for byte in bytes(range(80)):
     a.observe(byte);b.observe(byte);np.testing.assert_array_equal(a.log_probs(),b.log_probs())
 def test_local_gate_frozen_without_labels(self):
  def gate(raw):
   raw=raw[32:];n=struct.unpack_from('<Q',raw,12)[0];at=20+n+7605*4+4+2+8+16+4
   return raw[at:at+65*4]
  with Adapter() as a:
   for ep in dataset(8,'train'):a.train_episode(ep,999)
   raw=a.dumps();a.reset();a.run(b'abcde'*30,b'bcdea'*30)
   self.assertEqual(gate(raw),gate(a.dumps()));self.assertNotEqual(raw,a.dumps())
 def test_byte_distribution_and_loss(self):
  with Adapter() as a:
   for ep in balanced_offsets(dataset(4,'train')):a.train_episode(ep,999)
   with a.clone() as f:
    for b in b'abcdef':f.observe(b)
    p=f.log_probs();self.assertAlmostEqual(float(np.exp(p).sum()),1.,places=5)
    raw=f.dumps()
    for target in (0,32,97,255):
     with Adapter.loads(raw) as b,Adapter.loads(raw) as c:
      v=C.c_double();b.call('qr_stream_score',b'X'+bytes([target]),2,C.byref(v));c.observe(88)
      self.assertAlmostEqual(v.value,-float(c.log_probs()[target]),places=4)
    self.assertEqual(raw,f.dumps())
 def test_generic_targets_do_not_rewrite_local_codes(self):
  def local(raw):
   raw=raw[32:];n=struct.unpack_from('<Q',raw,12)[0];return raw[20+n:20+n+7605*4]
  with Adapter() as a:
   a.train_episode(dataset(4,'train')[0],999);old=a.dumps();old_gate=a.inventory()["gate_targets"]
   a.reset();a.run(b'abcde'*40,b'bcdea'*40)
   self.assertEqual(local(old),local(a.dumps()));self.assertNotEqual(old,a.dumps())
   self.assertEqual(a.inventory()['readout_parameters'],7670);self.assertEqual(old_gate,a.inventory()['gate_targets'])
 def test_partial_phases_and_outer_resume(self):
  data=bytes(np.random.default_rng(91).integers(0,256,150,dtype=np.uint8))
  for at in range(1,66):
   with Adapter() as a:
    a.run(data[:at],data[1:at+1])
    with Adapter.loads(a.dumps()) as b:
     self.assertEqual(a.run(data[at:-1],data[at+1:]),b.run(data[at:-1],data[at+1:]));self.assertEqual(a.dumps(),b.dumps())
  with Adapter() as a:
   cursor=new_cursor(all_bytes=True);cursor,_=advance(a,cursor,83,cases=dataset(2048,'train'))
   with Adapter.loads(a.dumps()) as b:
    whole,_=advance(a,cursor,1000,cases=dataset(2048,'train'));partial,_=advance(b,cursor,371,cases=dataset(2048,'train'))
    with Adapter.loads(b.dumps()) as c:
     resumed,_=advance(c,partial,629,cases=dataset(2048,'train'));self.assertEqual(a.dumps(),c.dumps());self.assertEqual(whole,resumed)
 def test_native_and_checkpoint_policy_separation(self):
  with Adapter(kind='native') as a,E1(kind='native') as b:
   data=b'abcde'*20;self.assertEqual(a.run(data[:-1],data[1:]),b.run(data[:-1],data[1:]));np.testing.assert_array_equal(a.log_probs(),b.log_probs())
   with self.assertRaises(RuntimeError):Adapter.loads(b.dumps())
if __name__=='__main__':unittest.main()
