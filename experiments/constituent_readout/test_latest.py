"""D2 persistence, streaming, policy and frozen inference contracts."""
import unittest
import numpy as np
from .adapter import Adapter,prefix_labels
from .tasks import dataset,shifted
class LatestTests(unittest.TestCase):
 def test_latest_continuation_and_freezing(self):
  ep=shifted(dataset(4,'train')[0],29);labels=prefix_labels(ep)
  for joint in (False,True):
   for at in (1,4,8,16,63,64,65,129):
    with Adapter(kind='paired_latest',joint=joint) as a:
     a.run(ep.data[:at],ep.data[1:at+1],ep.mask[1:at+1],labels[:at])
     with Adapter.loads(a.dumps()) as b:
      self.assertEqual(b.inventory()['kind'],'paired_latest')
      args=(ep.data[at:-1],ep.data[at+1:],ep.mask[at+1:],labels[at:])
      self.assertEqual(a.run(*args),b.run(*args));self.assertEqual(a.dumps(),b.dumps())
 def test_policy_change_is_frozen_only(self):
  with Adapter(kind='paired') as a:
   with self.assertRaises(RuntimeError):a.call('qr_query_policy',1)
   before=a.dumps()
   for bad in ([256,-1,-1,-1,-1],[1,65536,-1,-1,-1],[-2,-1,-1,-1,-1]):
    with self.assertRaises(RuntimeError):a.run(b'x',b'y',bytes(1),np.array([bad],dtype=np.int32))
    self.assertEqual(before,a.dumps())
   with a.clone() as f:
    old=f.dumps();f.call('qr_query_policy',1);self.assertEqual(f.inventory()['kind'],'paired_latest')
    f.call('qr_query_policy',0);self.assertEqual(old,f.dumps())
    with self.assertRaises(RuntimeError):f.call('qr_query_policy',2)
 def test_frozen_predictions_and_target_independence(self):
  ep=shifted(dataset(4,'dev')[0],43)
  with Adapter(kind='paired_latest') as a:
   a.train_episode(ep,999)
   with a.clone() as f:
    for b in ep.data[:61]:f.observe(b)
    before=f.dumps();p=f.log_probs();self.assertEqual(before,f.dumps())
    with Adapter.loads(before) as g:np.testing.assert_array_equal(p,g.log_probs())
if __name__=='__main__':unittest.main()
