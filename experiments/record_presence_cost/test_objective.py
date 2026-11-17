"""Independent Python logistic/hybrid checks and exact same-weight trajectories."""
import ctypes as C,hashlib,math,unittest
import numpy as np
from .backend import Model,library,header,NEGATIVE_OLD,NEGATIVE_NEW
from .tasks import cases
from experiments.discriminative_keys.backend import Model as A5,header as a5_header
from experiments.discriminative_keys.test_objective import reference,code
from experiments.joint_cold_gate.adapter import prefix_labels

def logistic(z,y):return 4*(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)
def as_a5(raw):
 payload=(0x3541353050584553).to_bytes(8,'little')+raw[40:]
 return hashlib.sha256(payload).digest()+payload
class ObjectiveTests(unittest.TestCase):
 def test_exact_source_isolation(self):
  self.assertEqual(header('a6').replace('0x3641353050584553ULL','0x3541353050584553ULL').replace(NEGATIVE_NEW,NEGATIVE_OLD),a5_header('a5'))
 def test_independent_logistic_gradient_and_finite_differences(self):
  lib=library('a6')
  for z in np.r_[np.linspace(-40,40,321),-100.,100.,-1e6,1e6]:
   for y in (0,1):
    out=(C.c_float*2)();lib.a6_presence_probe(float(z),y,out);p,e=out
    self.assertTrue(math.isfinite(e) and math.isfinite(logistic(z,y)))
    self.assertEqual(e,float(np.float32(np.float32(p-y)*np.float32(4))))
    expected=(-4/(1+math.exp(z)) if y else 4/(1+math.exp(-z))) if abs(z)<700 else (4*(z>0)-4*y)
    self.assertAlmostEqual(e,expected,delta=5e-7)
    if abs(z)<=40:
     h=1e-3;self.assertAlmostEqual((logistic(z+h,y)-logistic(z-h,y))/(2*h),expected,delta=8e-8)
 def test_all_256_hybrid_reference_and_codebook_unchanged(self):
  lib=library('a6');raw=(C.c_float*4096)();lib.a5_code_probe(raw)
  np.testing.assert_array_equal(np.array(raw).reshape(256,16),[[code(b,r) for r in range(16)] for b in range(256)])
  rng=np.random.default_rng(615)
  for label in range(256):
   z=(np.ones(16) if label==0 else -np.ones(16) if label==255 else rng.uniform(-1,1,16)).astype(np.float32)
   out=(C.c_double*290)();self.assertEqual(lib.a5_objective_probe(z.ctypes.data_as(C.POINTER(C.c_float)),label,out),0)
   got=np.array(out);ref=reference(z,label)
   np.testing.assert_allclose(got[:274],ref[:274],atol=2e-14,rtol=2e-14);np.testing.assert_array_equal(got[274:],ref[274:])
 def test_same_weight_inference_admission_generic_credit(self):
  with Model('a6',seed=7) as a:
   for c in cases('train',2,condition='overwrite16'):
    a.reset();a.run(c.ep.data[:-1],c.ep.data[1:],labels=prefix_labels(c.ep))
   with A5.loads(as_a5(a.dumps()),'a5') as b:
    self.assertEqual(a.dumps()[40:],b.dumps()[40:]);data=bytes(range(256))*2
    self.assertEqual(a.run(data[:-1],data[1:]),b.run(data[:-1],data[1:]));self.assertEqual(a.dumps()[40:],b.dumps()[40:])
    for byte in data:
     a.observe(byte);b.observe(byte);np.testing.assert_array_equal(a.log_probs(),b.log_probs())
    self.assertEqual(a.dumps()[40:],b.dumps()[40:])
if __name__=='__main__':unittest.main()
