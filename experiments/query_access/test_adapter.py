import unittest
from dataclasses import replace
import numpy as np
from .adapter import Adapter, PARAMETERS, prefix_labels
from .tasks import dataset, balanced_offsets, shifted, fold
from experiments.factorized_readout.adapter import Adapter as Previous

class AccessTests(unittest.TestCase):
    def test_capacity_views_and_frozen_state(self):
        for kind in ('l3','split','pool'):
            with Adapter(kind=kind) as a:
                inv=a.inventory();self.assertEqual(inv['readout_parameters'],PARAMETERS)
                self.assertLessEqual(inv['parameters'],500000)
                self.assertEqual(inv['added_persistent_sequence_state_bytes'],0)
                self.assertEqual(inv['normalization_bytes'],5160)
                a.train_episode(dataset(4,'train')[0],129)
                before=a.dumps()
                with a.clone() as f:
                    for t in range(131):f.observe(t%256);f.log_probs()
                    self.assertEqual(f.inventory()['normalization_samples'],2)
                    self.assertLessEqual(f.inventory()['source_count'],80)
                    self.assertAlmostEqual(np.exp(f.log_probs()).sum(),1,places=6)
                self.assertEqual(a.dumps(),before)

    def test_exp01b_l3_numerical_parity(self):
        with Adapter(kind='l3') as a,Previous(kind='local',joint=False,rate=.16) as b:
            for ep in balanced_offsets(dataset(16,'train')):
                self.assertEqual(a.train_episode(ep,999)[:3],b.train_episode(ep,999)[:3])
                np.testing.assert_array_equal(a.log_probs(),b.log_probs())
                np.testing.assert_array_equal(a.diagnostic(),b.diagnostic())

    def test_continuation_every_boundary(self):
        ep=shifted(dataset(4,'train')[0],7);labs=prefix_labels(ep)
        for kind in ('l3','split','pool'):
            for at in (3,4,15,16,63,64,65,127,128):
                with Adapter(kind=kind) as a:
                    a.run(ep.data[:at],ep.data[1:at+1],ep.mask[1:at+1],labs[:at])
                    with Adapter.loads(a.dumps()) as b:
                        args=(ep.data[at:-1],ep.data[at+1:],ep.mask[at+1:],labs[at:])
                        self.assertEqual(a.run(*args),b.run(*args));self.assertEqual(a.dumps(),b.dumps())

    def test_labels_and_future_do_not_enter_inference(self):
        for kind in ('l3','split','pool'):
            with Adapter(kind=kind) as a,Adapter(kind=kind) as b:
                a.run(b'prefix',b'aaaaaa',bytes(6),np.zeros((6,5),dtype=np.int16))
                b.run(b'prefix',b'zzzzzz',bytes(6),np.full((6,5),255,dtype=np.int16))
                self.assertEqual(a.dumps(),b.dumps());np.testing.assert_array_equal(a.log_probs(),b.log_probs())
        ep=dataset(4,'train')[0];changed=bytearray(ep.data)
        for t,mask in enumerate(ep.mask):
            if mask:changed[t]=255
        np.testing.assert_array_equal(prefix_labels(ep),prefix_labels(replace(ep,data=bytes(changed))))

    def test_phase_delay_and_splits(self):
        groups=[]
        for split in ('train','dev','test'):
            cases=dataset(256,split);groups.append({e.split_key for e in cases})
            self.assertTrue(all(fold(e.split_key)==split for e in cases))
            seen=[]
            for base,aug in zip(cases,balanced_offsets(cases)):
                original=base.mask.index(1);now=aug.mask.index(1);seen.append(now%64)
                self.assertEqual(aug.data[now-12:now],base.data[original-12:original])
                delayed=shifted(base,now-original,128)
                self.assertEqual(delayed.mask.index(1)%64,now%64)
            self.assertEqual(set(seen),set(range(64)))
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_invalid_checkpoint_and_policy(self):
        with self.assertRaises(RuntimeError):Adapter(kind='pool',joint=True)
        with Adapter() as a:
            with self.assertRaises(ValueError):Adapter.loads(a.dumps()[:-1])
            with a.clone(intervention='no_partial') as f:
                with self.assertRaises(RuntimeError):f.run(b'a',b'b')
                with Adapter.loads(f.dumps()) as b:np.testing.assert_array_equal(f.log_probs(),b.log_probs())

    def test_separate_query_metrics_match_manual_rollout(self):
        from .run import swapped_queries
        ep=shifted(dataset(4,'dev')[0],31)
        other=swapped_queries(ep)
        self.assertEqual(ep.mask,other.mask)
        self.assertEqual(ep.answer,tuple(reversed(other.answer)))
        for kind in ('l3','split','pool'):
            with Adapter(kind=kind) as a,a.clone() as f:
                scored=f.score(ep,2,2);f.reset();correct=[]
                for i,byte in enumerate(ep.data):
                    if ep.mask[i]:
                        predicted=int(f.log_probs().argmax());correct.append(int(predicted==byte));byte=predicted
                    if i+1<len(ep.data):f.observe(byte)
                self.assertEqual(scored[7:9],correct)
                self.assertEqual(scored[5],int(all(correct)))

if __name__=='__main__':unittest.main()
