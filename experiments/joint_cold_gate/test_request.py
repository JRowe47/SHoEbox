import unittest
from dataclasses import replace
import numpy as np
from .adapter import Adapter,prefix_labels
from .tasks import dataset
from experiments.joint_local_gate.adapter import Adapter as E6,prefix_labels as labels6
from experiments.mad.tasks import ANSWER

class RequestTests(unittest.TestCase):
 def test_label_is_observed_request_only(self):
  ep=dataset(4,'train')[0];labels=prefix_labels(ep)
  self.assertEqual(labels[:,5].tolist(),[int(b==ANSWER) for b in ep.data[:-1]])
  for at in (1,8,20,60):
   if at>=len(ep.data)-1:continue
   changed=replace(ep,data=ep.data[:at]+bytes([255])*(len(ep.data)-at),mask=bytes(len(ep.mask)))
   np.testing.assert_array_equal(labels[:at],prefix_labels(changed)[:at])
 def test_generic_control_trajectory_matches_e6(self):
  with Adapter(kind='paired_gated') as a,E6(kind='paired_gated') as b:
   for ep in dataset(8,'train'):
    a.reset();b.reset()
    self.assertEqual(a.run(ep.data[:-1],ep.data[1:],labels=prefix_labels(ep)),b.run(ep.data[:-1],ep.data[1:],labels=labels6(ep)))
    self.assertEqual(a.dumps()[40:],b.dumps()[40:])
if __name__=='__main__':unittest.main()
