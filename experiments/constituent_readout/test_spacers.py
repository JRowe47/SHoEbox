import unittest
from .tasks_v3 import dataset,spaced,shifted,fold
from .tasks import dataset as old_dataset
from .adapter import prefix_labels
from experiments.mad_next.tasks import Spec,solve
class SpacerTests(unittest.TestCase):
 def test_new_audit_excludes_prior_audit(self):
  sets=[{e.split_key for e in dataset(1024 if s=='test' else 256,s)} for s in ('train','dev','test')]
  self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
  self.assertFalse(sets[2]&{e.split_key for e in old_dataset(1024,'test')})
 def test_spacing_keeps_answers_labels_and_balanced_phases(self):
  ep=dataset(4,'train')[0]
  for phase in range(64):
   for gaps in ([0,0],[0,7],[7,0],[7,7],[3,5]):
    x=shifted(spaced(ep,gaps),(phase-sum(gaps))%64)
    self.assertEqual(x.mask.index(1)%64,phase)
    self.assertEqual(solve(x,Spec('binding',vocab=2)),ep.answer)
    self.assertEqual(x.split_key,ep.split_key)
    labels=prefix_labels(x);self.assertEqual(int((labels[:,0]>=0).sum()),2);self.assertEqual(int((labels[:,1:]>=0).sum()),2)
 def test_length_validation(self):
  ep=dataset(4,'train')[0]
  for gaps in ([8,0],[-1,0],[0],[]):
   with self.assertRaises(ValueError):spaced(ep,gaps)
if __name__=='__main__':unittest.main()
