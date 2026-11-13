import sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from runtime import fresh_model,Model
from convert_to_semantics import convert
class SemanticsDeliveryTests(unittest.TestCase):
 def test_conversion_preserves_weights_resets_new_semantic_state(self):
  a=fresh_model(kernel='binding');self.addCleanup(a.close);a.run(np.arange(60,dtype=np.uint8),mode='periodic')
  with tempfile.TemporaryDirectory() as tmp:
   old=Path(tmp)/'old.npz';new=Path(tmp)/'new.npz';a.save(old);result=convert(old,new);b=Model.load(new);self.addCleanup(b.close)
   self.assertEqual(result['version'],30);self.assertFalse(result['exact_continuation']);self.assertEqual(b.runtime_kernel,'semantics')
   np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.extra_flat,b.extra_flat);self.assertEqual(np.linalg.norm(b.semantic_parameters()),0)
   self.assertTrue(all(e['seq']==0 for e in b.content_events().values()))
   with self.assertRaises(ValueError):convert(old,new)
