import tempfile
import unittest
from pathlib import Path
import ctypes as ct
import numpy as np
from shoebox_runtime import Model, Architecture, evaluate
from shoebox_runtime.model import FP, initialize, views


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=np.frombuffer((b'The quick brown fox crosses the network. Another byte arrives.\n'*30),dtype=np.uint8)

    def test_parameter_budgets(self):
        self.assertEqual(Architecture(64,64).parameter_count,4_990_720)
        self.assertEqual(Architecture(64,256).parameter_count,19_638_016)
        self.assertEqual(Architecture(128,64).parameter_count,19_565_824)
        self.assertEqual(Architecture(256,[128,128,96,96,96,96,80,80,80,80,64,64,64,64,48,64,64,96]).parameter_count,99_727_616)

    def test_bank_growth_preserves_common_initialization(self):
        a,b=Architecture(64,12),Architecture(64,64)
        p,q=views(initialize(a,7),a.shapes),views(initialize(b,7),b.shapes)
        np.testing.assert_array_equal(p['readout'],q['readout'])
        for r in range(18):
            for k in ('W','keys'):
                np.testing.assert_array_equal(p[k][a.offsets[r]:a.offsets[r+1]],q[k][b.offsets[r]:b.offsets[r]+12])

    def test_masks_and_sparse_update_cross_word_boundaries(self):
        rng=np.random.default_rng(52)
        for d in (64,128,256):
            with self.subTest(width=d), Model(Architecture(d,2)) as m:
                inputs=np.ascontiguousarray(np.r_[rng.normal(size=(3,d)),np.zeros((1,d))],dtype=np.float32)
                # Force the important boundaries into the largest coordinates.
                for j in (0,63,64,127,128,191,192,255):
                    if j<d:inputs[0,j]=1000+j
                for k in (1,d//4,d-1,d):
                    out=np.zeros((len(inputs),d//64),np.uint64)
                    m.lib.check_masks(inputs.ctypes.data_as(FP),out.ctypes.data_as(ct.POINTER(ct.c_uint64)),len(inputs),k,0)
                    for row,mask in zip(inputs,out):
                        expected=set(np.lexsort((np.arange(d),-np.abs(row)))[:k])
                        actual={j for j in range(d) if int(mask[j//64])&(1<<(j%64))}
                        self.assertEqual(expected,actual)
                    w=rng.normal(size=d).astype(np.float32);before=w.copy();u=rng.normal(size=d).astype(np.float32)
                    m.lib.check_sparse_update(w.ctypes.data_as(FP),u.ctypes.data_as(FP),.125,inputs[0].ctypes.data_as(FP),k)
                    ids=np.lexsort((np.arange(d),-np.abs(inputs[0])))[:k]
                    expected=before.copy();expected[ids]-=np.float32(.125)*u[ids]
                    np.testing.assert_allclose(w,expected,rtol=2e-7,atol=1e-7)
                    inactive=np.setdiff1d(np.arange(d),ids);np.testing.assert_array_equal(w[inactive],before[inactive])

    def test_heterogeneous_banks_resume_and_snapshots(self):
        for d in (64,128,256):
            counts=[2+r%5 for r in range(18)]
            with self.subTest(width=d),Model(Architecture(d,counts),seed=9) as m:
                x=self.data[:-1];y=self.data[1:];cut=333
                m.run(x[:cut],y[:cut],mode='periodic')
                before=m.fingerprint()
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'resume.npz';m.save(path,{'offset':cut})
                    with Model.load(path) as resumed,m.snapshot_copy() as snapshot:
                        self.assertEqual(before,resumed.fingerprint());self.assertEqual(before,snapshot.fingerprint())
                        loss=m.run(x[cut:],y[cut:],mode='periodic')
                        self.assertEqual(loss,resumed.run(x[cut:],y[cut:],mode='periodic'))
                        self.assertEqual(m.fingerprint(),resumed.fingerprint())
                        self.assertEqual(before,snapshot.fingerprint())
                totals=m.regional_totals()
                for n,count,execs in zip(counts,totals['expert_counts'],totals['executions']):
                    self.assertEqual(len(count),n);self.assertEqual(sum(count),execs)
                fp=m.fingerprint();ev=evaluate(m,self.data[:256]);self.assertEqual(fp,m.fingerprint());self.assertTrue(np.isfinite(ev['nll']))
                self.assertEqual(m.observations,len(x))
                # Learned signal must reach the widened portion of selected banks.
                if d>64:
                    initial=views(initialize(m.arch,9),m.arch.shapes)['W']
                    self.assertTrue(np.any(m.params['W'][:,64:,64:]!=initial[:,64:,64:]))
                m.lib.diagnostic_enable(m.handle,1);m.run(x[:64],y[:64]);self.assertEqual(len(m.diagnostics()),18)

    def test_policy_and_checkpoint_validation(self):
        for bad in (0,32,65,512):
            with self.assertRaises(ValueError):Architecture(bad,64)
        for bad in ([1]*18,[64]*17,[2.5]*18):
            with self.assertRaises(ValueError):Architecture(64,bad)
        with Model(Architecture(64,2)) as m:
            with self.assertRaises(ValueError):m.configure_a1_learning({'prototype_scale':float('nan')})
            with self.assertRaises(ValueError):m.configure_teaching(np.full((18,2),float('nan')))
            with self.assertRaises(ValueError):m.run(np.array([256]),np.array([1]))
            m.run(self.data[:64],self.data[1:65],mode='periodic')
            clock=m.observations
            m.configure_adaptation(prototype_scale=.1,center_rate=.0001,mean_rate=.0001)
            self.assertEqual(clock,m.observations)
            m.configure_a1_learning({'prototype_scale':.1,'follow_cooldown':1,'reference_lr':.008,'decoder_scale':.1})
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'policy.npz';m.save(path)
                with Model.load(path) as r:self.assertEqual(m.fingerprint(),r.fingerprint())


if __name__=='__main__':unittest.main()
