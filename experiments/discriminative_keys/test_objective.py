"""Float64 independent Python reference and bit-exact policy isolation."""
import ctypes as C,hashlib,math,unittest
import numpy as np
from .backend import Model,library,header
from experiments.relative_chronological_retention.backend import Model as A4,header as a4_header
from .tasks import cases
from experiments.joint_cold_gate.adapter import prefix_labels

def code(b,r):
 x=((b+1)*0x9e3779b9+(r+1)*0x85ebca6b)&0xffffffff
 x^=x>>16;x=(x*0x7feb352d)&0xffffffff;x^=x>>15;x=(x*0x846ca68b)&0xffffffff;x^=x>>16
 return float(np.float32(.7 if x&1 else -.7))

def reference(z,label):
 codes=[[code(b,r) for r in range(16)] for b in range(256)]
 logits=[-4*sum((float(z[r])-codes[b][r])**2 for r in range(16)) for b in range(256)]
 largest=max(logits);weights=[math.exp(s-largest) for s in logits];denom=sum(weights);p=[w/denom for w in weights]
 mu=[sum(p[b]*codes[b][r] for b in range(256)) for r in range(16)]
 grad=[float(np.float32(.5*(float(z[r])+mu[r]-2*codes[label][r])*(1-float(z[r])**2))) for r in range(16)]
 loss=[.25*sum((float(z[r])-codes[label][r])**2 for r in range(16)),-math.log(p[label])/16]
 return np.array(loss+p+mu+grad)

def as_a4_for_test(raw):
 # Test-only same-weight control. Production loaders reject the A5 envelope.
 payload=(0x3441353050584553).to_bytes(8,'little')+raw[40:]
 return hashlib.sha256(payload).digest()+payload

class ObjectiveTests(unittest.TestCase):
 def test_all_codes_and_independent_objective_gradient(self):
  lib=library('a5');raw=(C.c_float*4096)();lib.a5_code_probe(raw)
  native=np.array(raw).reshape(256,16);expected=np.array([[code(b,r) for r in range(16)] for b in range(256)])
  np.testing.assert_array_equal(native,expected)
  collisions=[(b,a) for a in range(256) for b in range(a) if np.array_equal(native[a],native[b])]
  print('Fixed 256-byte codebook collision pairs:',collisions)
  rng=np.random.default_rng(515)
  for label in range(256):
   z=(np.ones(16) if label==0 else -np.ones(16) if label==255 else rng.uniform(-1,1,16)).astype(np.float32)
   out=(C.c_double*290)();self.assertEqual(lib.a5_objective_probe(z.ctypes.data_as(C.POINTER(C.c_float)),label,out),0)
   got=np.array(out);ref=reference(z,label)
   np.testing.assert_allclose(got[:274],ref[:274],atol=2e-14,rtol=2e-14)
   np.testing.assert_array_equal(got[274:],ref[274:])
 def test_unchanged_inference_and_generic_native_updates_at_identical_weights(self):
  with Model('a5',seed=7) as a:
   for c in cases('train',2,condition='overwrite16'):
    a.reset();a.run(c.ep.data[:-1],c.ep.data[1:],labels=prefix_labels(c.ep))
   with A4.loads(as_a4_for_test(a.dumps()),'a4') as b:
    self.assertEqual(a.dumps()[40:],b.dumps()[40:])
    data=bytes(range(256))*2
    self.assertEqual(a.run(data[:-1],data[1:]),b.run(data[:-1],data[1:]))
    self.assertEqual(a.dumps()[40:],b.dumps()[40:])
    for byte in data:
     a.observe(byte);b.observe(byte);np.testing.assert_array_equal(a.log_probs(),b.log_probs())
    self.assertEqual(a.dumps()[40:],b.dumps()[40:])
 def test_exact_source_isolation(self):
  new=header('a5');old=a4_header('a4')
  from pathlib import Path
  new=new.replace((Path(__file__).parent/'objective.hpp').read_text()+'\n','').replace('0x3541353050584553ULL','0x3441353050584553ULL')
  new=new.replace('a5_reconstruct(head.key.data()+4*QRANK,x,labels[0],lr*4);','for(int r=0;r<QRANK;++r)reconstruct(head.key[4*QRANK+r],x,byte_code(labels[0],r),lr*4);')
  new=new.replace('a5_reconstruct(head.key.data()+phase*QRANK,x,key,lr*4);','for(int r=0;r<QRANK;++r)reconstruct(head.key[phase*QRANK+r],x,byte_code(key,r),lr*4);')
  self.assertEqual(new,old)
if __name__=='__main__':unittest.main()
