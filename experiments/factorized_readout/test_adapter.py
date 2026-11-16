import unittest
from dataclasses import replace
import numpy as np
from .adapter import Adapter,PARAMETERS,prefix_labels
from .tasks import dataset,Spec,fold
from experiments.mad_next.tasks import parse,KEY

class FactorizedTests(unittest.TestCase):
    def test_capacity_and_normalization_accounting(self):
        with Adapter(kind='answer') as a,Adapter(kind='local') as b:
            self.assertEqual(a.inventory(),{**b.inventory(),'kind':'answer'})
            self.assertEqual(a.inventory()['readout_parameters'],PARAMETERS)
            self.assertLessEqual(a.inventory()['parameters'],500000)
            self.assertEqual(a.inventory()['normalization_bytes'],1032)

    def test_prefix_targets_and_answer_causality(self):
        ep=dataset(Spec('binding',vocab=2),4,'train')[0];records,queries=parse(ep)
        labs=prefix_labels(ep);at=np.flatnonzero(np.frombuffer(ep.mask,dtype=np.uint8))
        for i,t in enumerate(at):
            self.assertEqual(labs[t-1].tolist(),[KEY+queries[i],KEY+records[0][1],KEY+records[1][1],records[0][2],records[1][2]])
        changed=bytearray(ep.data)
        for t in at:changed[t]=255
        np.testing.assert_array_equal(labs,prefix_labels(replace(ep,data=bytes(changed))))
        self.assertTrue(np.all(labs[:at[0]-1]==-1))
        # Altering a later query cannot alter any earlier target.
        changed=bytearray(ep.data);changed[at[0]+3]=KEY+47
        np.testing.assert_array_equal(labs[:at[0]],prefix_labels(replace(ep,data=bytes(changed)))[:at[0]])

    def test_exact_partial_continuation(self):
        ep=dataset(Spec('binding',vocab=2),4,'train')[0];labs=prefix_labels(ep)
        for core in ('minimal','full'):
            for joint in (True,False):
                with Adapter(core=core,joint=joint) as a:
                    a.run(ep.data[:67],ep.data[1:68],ep.mask[1:68],labs[:67])
                    with Adapter.loads(a.dumps()) as b:
                        args=(ep.data[67:-1],ep.data[68:],ep.mask[68:],labs[67:])
                        self.assertEqual(a.run(*args),b.run(*args));self.assertEqual(a.dumps(),b.dumps())

    def test_rls_continuation_and_covariance_freeze(self):
        ep=dataset(Spec('binding',vocab=2),4,'train')[0]
        with Adapter(kind='rls',joint=False) as a:
            a.train_episode(ep,129)
            self.assertEqual(a.inventory()['rls_updates'],2)
            self.assertEqual(a.inventory()['optimizer_state_bytes'],33808)
            with Adapter.loads(a.dumps()) as b:
                a.train_episode(ep,129);b.train_episode(ep,129);self.assertEqual(a.dumps(),b.dumps())
            before=a.dumps()
            with a.clone() as f:
                for i in range(80):f.observe(i);f.log_probs()
                self.assertEqual(f.inventory()['rls_updates'],4)
            self.assertEqual(a.dumps(),before)
        with self.assertRaises(RuntimeError):Adapter(kind='rls',joint=True)

    def test_normalizer_and_features_are_frozen_in_inference(self):
        with Adapter() as a:
            a.train_episode(dataset(Spec('binding',vocab=2),4,'train')[0],129);before=a.dumps()
            with a.clone() as f:
                start=f.inventory()['normalization_samples']
                for i in range(100):f.observe(i);f.features();f.log_probs()
                self.assertEqual(f.inventory()['normalization_samples'],start)
                self.assertAlmostEqual(float(np.exp(f.log_probs()).sum()),1,places=6)
                with self.assertRaises(RuntimeError):f.run(b'a',b'b')
            self.assertEqual(a.dumps(),before)

    def test_no_label_effect_on_observation_or_ignored_targets(self):
        with Adapter() as a,Adapter() as b:
            labels=np.zeros((6,5),dtype=np.int16)
            a.run(b'prefix',b'aaaaaa',bytes(6),labels);b.run(b'prefix',b'zzzzzz',bytes(6),labels+255)
            self.assertEqual(a.dumps(),b.dumps());np.testing.assert_array_equal(a.log_probs(),b.log_probs())

    def test_answer_arm_ignores_all_auxiliary_labels(self):
        with Adapter(kind='answer') as a,Adapter(kind='answer') as b:
            a.run(b'prefix',b'aaaaaa',labels=np.zeros((6,5),dtype=np.int16))
            b.run(b'prefix',b'aaaaaa',labels=np.full((6,5),255,dtype=np.int16))
            self.assertEqual(a.dumps(),b.dumps())

    def test_zero_residual_and_adopt(self):
        with Adapter(parameters=500000-PARAMETERS,kind='native') as n,Adapter(joint=False) as a:
            n.run(b'prefix',b'refix!');n.finish();a.adopt(n)
            with n.clone() as f:np.testing.assert_array_equal(a.log_probs(),f.log_probs())

    def test_new_world_splits_and_balanced_queries(self):
        sets=[];spec=Spec('binding',vocab=2)
        for split in ('train','dev','test'):
            # Tiny software validation only; no model scores/selects this fold.
            cases=dataset(spec,16,split);sets.append({e.split_key for e in cases})
            self.assertTrue(all(fold(e.split_key)==split for e in cases))
            for start in range(0,len(cases),4):
                self.assertEqual(len({e.data for e in cases[start:start+4]}),4)
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_invalid_inputs_and_checkpoint(self):
        with Adapter() as a:
            with self.assertRaises(ValueError):a.run(b'a',b'bb')
            with self.assertRaises(ValueError):a.run(b'a',b'b',labels=[1])
            with self.assertRaises(RuntimeError):a.run(b'a',b'b',labels=[[-2]*5])
            with self.assertRaises(ValueError):Adapter.loads(a.dumps()[:-1])
        with self.assertRaises(RuntimeError):a.log_probs()

if __name__=='__main__':unittest.main()
