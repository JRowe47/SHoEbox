import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from runtime import Model,fresh_model
from convert_to_binding import convert

class BindingDeliveryTests(unittest.TestCase):
 def test_weights_convert_without_reinterpreting_old_state(self):
  a=fresh_model(kernel='composition');a.run(np.arange(32,dtype=np.uint8),mode='periodic',lr=.01,lp=.003)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'old.npz';q=Path(t)/'new.npz';a.save(p);convert(p,q);b=Model.load(q)
   np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.extra_flat,b.extra_flat)
   self.assertTrue(all(e['seq']==0 for e in b.content_events().values()))
   self.assertEqual(b.persistent_metrics()['occupied'],0)
   with self.assertRaises(ValueError):convert(p,q)
   b.close()
  a.close()
 def test_disabled_binding_matches_legacy_numerics(self):
  a=fresh_model(kernel='native');b=fresh_model(kernel='binding');b.configure_composition({'enabled':False});x=np.arange(128,dtype=np.uint8)
  va=a.run(x,mode='periodic',lr=.01,lp=.003,logits=True);vb=b.run(x,mode='periodic',lr=.01,lp=.003,logits=True)
  self.assertEqual(va[0],vb[0]);np.testing.assert_array_equal(va[1],vb[1]);np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.extra_flat,b.extra_flat);a.close();b.close()

if __name__=='__main__':unittest.main(verbosity=2)
