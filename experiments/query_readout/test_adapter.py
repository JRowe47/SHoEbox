import unittest
import numpy as np
from .adapter import Adapter,PARAMETERS
from experiments.mad_next.tasks import Spec,dataset


class QueryTests(unittest.TestCase):
    def test_parameter_budget_and_same_capacity(self):
        for core in ('minimal','full'):
            with Adapter(core=core,kind='additive') as a,Adapter(core=core,kind='bilinear') as b:
                self.assertEqual(a.inventory()['parameters'],b.inventory()['parameters'])
                self.assertLessEqual(a.inventory()['parameters'],500000)
                self.assertEqual(a.inventory()['readout_parameters'],PARAMETERS)

    def test_exact_partial_continuation(self):
        for core in ('minimal','full'):
            for kind in ('native','additive','bilinear'):
                with Adapter(core=core,kind=kind) as a:
                    x=bytes(range(256))*3;a.run(x[:131],x[1:132])
                    with Adapter.loads(a.dumps()) as b:
                        self.assertEqual(a.run(x[131:509],x[132:510]),b.run(x[131:509],x[132:510]))
                        self.assertEqual(a.dumps(),b.dumps())

    def test_frozen_clone_does_not_mutate_source(self):
        with Adapter() as a:
            a.run(b'hello world',b'ello world!');before=a.dumps()
            with a.clone() as b:
                for i in range(80):b.observe(i)
                self.assertAlmostEqual(float(np.exp(b.log_probs()).sum()),1.,places=6)
                with self.assertRaises(RuntimeError):b.run(b'a',b'b')
            self.assertEqual(before,a.dumps())

    def test_adopt_and_zero_residual(self):
        with Adapter(parameters=500000-PARAMETERS,kind='native') as native,Adapter(kind='bilinear',joint=False) as a:
            native.run(b'observed',b'bserved!');native.finish();a.adopt(native)
            with native.clone() as n:
                np.testing.assert_array_equal(n.log_probs(),a.log_probs())
                for byte in b'more bytes':a.observe(byte);n.observe(byte)
                np.testing.assert_array_equal(n.log_probs(),a.log_probs())

    def test_feature_inspection_read_only(self):
        with Adapter() as a:
            a.run(bytes(range(80)),bytes(range(1,81)));before=a.dumps()
            x=a.features();self.assertEqual(x.shape,(432,));self.assertTrue(np.isfinite(x).all())
            self.assertEqual(a.dumps(),before)

    def test_target_and_mask_causality(self):
        with Adapter() as a,Adapter() as b:
            a.run(b'prefix',b'aaaaaa',bytes(6));b.run(b'prefix',b'zzzzzz',bytes(6))
            self.assertEqual(a.dumps(),b.dumps())
            np.testing.assert_array_equal(a.log_probs(),b.log_probs())

    def test_corrupt_and_closed(self):
        with Adapter() as a:
            raw=a.dumps()
            with self.assertRaises(ValueError):Adapter.loads(raw[:-1])
            with self.assertRaises(ValueError):a.run(b'a',b'bb')
        with self.assertRaises(RuntimeError):a.log_probs()

    def test_partial_cost(self):
        ep=dataset(Spec('binding',vocab=2),4,'train')[0]
        with Adapter() as a:
            used,targets,_,_=a.train_episode(ep,13)
            self.assertEqual((used,targets),(13,0));self.assertEqual(a.inventory()['bytes'],13)


if __name__=='__main__':unittest.main()
