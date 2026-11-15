import unittest
from shoe_multiscale import Model

class LanguageCreditTests(unittest.TestCase):
    def test_gains_roundtrip_pending_and_frozen_state(self):
        for levels in (1,2,3,4):
            gains=[1,1,4,16][:levels]
            with Model(levels=levels,language_credit=gains) as m:
                m.train(b'A pending stream of observed words. '*19)
                self.assertEqual(m.stats()['language_credit'],gains)
                with Model.loads(m.dumps()) as q:
                    self.assertEqual(q.stats()['language_credit'],gains)
                    self.assertEqual(m.train(b'Extra observations.'*37),q.train(b'Extra observations.'*37))
                    self.assertEqual(m.dumps(),q.dumps())
                before=m.dumps();m.evaluate(b'An independent evaluation.'*25)
                self.assertEqual(m.dumps(),before)
                with m.inference_copy() as f:
                    with self.assertRaises(RuntimeError):f.set_language_credit(gains)

    def test_validation_and_unit_gain_legacy_serialization(self):
        with Model(routing_arm='C',balance_policy='warmup') as m:
            m.train(b'Historical checkpoints keep their original policy.'*13)
            old=m.dumps();m.set_language_credit([1]*4);self.assertEqual(m.dumps(),old)
            m.configure(language_credit=[1,1,4,16]);self.assertEqual(m.stats()['checkpoint_version'],11)
            m.configure(language_credit=[1]*4);self.assertEqual(m.dumps(),old)
            for gains in ([1], [1,1,-1,1], [1,1,float('nan'),1], [1,1,float('inf'),1], [[1]*4]):
                with self.assertRaises(ValueError):m.configure(language_credit=gains)
                self.assertEqual(m.dumps(),old)
        with self.assertRaises(ValueError):Model(memory_profile='legacy',language_credit=[1]*4)

    def test_recall_pending_eviction_and_boundary_continuation(self):
        import numpy as np
        data=np.random.default_rng(127).integers(0,256,5001,dtype=np.uint8)
        with Model(recall_credit=[0,0,1,1]) as m:
            self.assertEqual(m.stats()['checkpoint_version'],12)
            start=0
            for end in (1,16,64,127,511,512,513,2047,4000):
                m.run(data[start:end],data[start+1:end+1]);start=end
                with Model.loads(m.dumps()) as q:
                    self.assertEqual(q.stats()['recall_credit'],[0,0,1,1])
                    self.assertEqual(m.dumps(),q.dumps())
                    with m.clone() as twin:
                        self.assertEqual(twin.run(data[end:end+517],data[end+1:end+518]),q.run(data[end:end+517],data[end+1:end+518]))
                        self.assertEqual(twin.dumps(),q.dumps())
            old=m.dumps();m.evaluate(data)
            self.assertEqual(m.dumps(),old)
            with self.assertRaises(RuntimeError):m.set_recall_credit([0]*4)
            self.assertEqual(m.dumps(),old)
            m.finish();m.set_recall_credit([0]*4)
            self.assertEqual(m.stats()['checkpoint_version'],10)
            with Model.loads(m.dumps()) as q:self.assertEqual(q.dumps(),m.dumps())
        with self.assertRaises(ValueError):Model(recall_credit=[1,2])

if __name__=='__main__':unittest.main()
