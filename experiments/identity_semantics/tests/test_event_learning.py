"""Stage-5 directional credit, native MinKron VJP, ownership and continuation."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
sys.path.insert(0,str(ROOT/'src'))
from runtime import fresh_model,Model,core
from reference_acceptance import collect,check_integrity

class EventLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'event_fixture_e{core.E}.so';path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/event_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        cls.fx.event_context_vjp_check.argtypes=[ct.c_void_p,ct.c_int];cls.fx.event_context_vjp_check.restype=ct.c_float
        cls.fx.event_frontier_check.argtypes=[ct.c_void_p,ct.c_int]
        cls.fx.event_content_independence.argtypes=[ct.c_void_p]
    def make(self,**policy):
        m=fresh_model(seed=7,profile='event',event=policy);self.addCleanup(m.close);return m
    def data(self,n):
        raw=b'The river runs through the valley. The king crossed the bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,x,a=0,b=None):
        b=len(x)-1 if b is None else b
        return m.run(x[a:b],x[a+1:b+1],mode='periodic',lr=.04,lp=.04/3,logits=True)
    def test_native_minkron_local_vjp_matches_numerical_oracle(self):
        for heads in (1,2):
            m=self.make(context_heads=heads)
            self.assertLess(self.fx.event_context_vjp_check(m.handle,heads),1e-4)
    def test_old_ancestry_is_consolidated_and_direction_reaches_compact_credit(self):
        m=self.make(context_heads=0)
        self.assertLessEqual(self.fx.event_frontier_check(m.handle,16400),6)
        self.assertGreater(m.event_metrics()['S2']['consolidated_records'],16000)
        n=m.snapshot_copy();self.addCleanup(n.close);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_content_code_does_not_depend_on_context_memory(self):
        self.assertTrue(self.fx.event_content_independence(self.make().handle))
    def test_irregular_chunks_checkpoint_and_snapshot_are_exact(self):
        x=self.data(290)
        for heads in (0,1,2):
            with self.subTest(heads=heads),tempfile.TemporaryDirectory() as tmp:
                a=self.make(context_heads=heads);b=self.make(context_heads=heads)
                _,expected=self.train(a,x);actual=[]
                for lo,hi in ((0,65),(65,129),(129,289)):
                    _,logits=self.train(b,x,lo,hi);actual.append(logits)
                    path=Path(tmp)/'partial.npz';b.save(path);before=b.fingerprint();b.close()
                    b=Model.load(path);self.addCleanup(b.close);self.assertEqual(before,b.fingerprint())
                np.testing.assert_array_equal(expected,np.concatenate(actual));self.assertEqual(a.fingerprint(),b.fingerprint())
                c=b.snapshot_copy();self.addCleanup(c.close);self.assertEqual(b.fingerprint(),c.fingerprint())
                check_integrity(collect(b));self.assertEqual(sum(r['missing'] for r in b.event_metrics().values()),0)
    def test_placement_ablations_and_capabilities(self):
        x=self.data(192)
        for heads,learning in ((1,'ancestry'),(2,'ancestry'),(0,'local'),(1,'local'),(2,'local')):
            m=self.make(context_heads=heads,learning=learning);self.train(m,x)
            c=m.capabilities()['event_features'];self.assertEqual(c['small_head_minkron'],heads>0)
            self.assertEqual(c['full_ancestry_credit'],learning=='ancestry')
            check_integrity(collect(m));self.assertEqual(sum(r['missing'] for r in m.event_metrics().values()),0)
    def test_evaluation_is_causal_and_fitted_parameters_are_frozen(self):
        m=self.make();x=self.data(195);self.train(m,x)
        a=m.evaluation_copy(memory='cold_frozen');b=m.evaluation_copy(memory='cold_frozen')
        self.addCleanup(a.close);self.addCleanup(b.close)
        state=a.event_state(fitted=True);flat=a.flat.copy();other=(x[1:]+7).astype(np.uint8)
        _,l1=a.run(x[:-1],x[1:],mode='eval',logits=True);_,l2=b.run(x[:-1],other,mode='eval',logits=True)
        np.testing.assert_array_equal(l1,l2);np.testing.assert_array_equal(state,a.event_state(fitted=True));np.testing.assert_array_equal(flat,a.flat)
    def test_missing_corrupt_or_nonfinite_extension_rejected_atomically(self):
        m=self.make();self.train(m,self.data(80));before=m.fingerprint();state=m.event_state()
        nan=state.copy();nan[96:100]=np.array([np.nan],np.float32).view(np.uint8)
        bad=state.copy();bad[0]=2
        for invalid in (state[:-1],bad,nan,np.zeros(0,np.uint8)):
            with self.assertRaises(ValueError):m.import_event(invalid)
            self.assertEqual(before,m.fingerprint())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.npz';m.save(path)
            with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
            arrays.pop('event_state');np.savez(path,**arrays)
            with self.assertRaises(ValueError):Model.load(path)
    def test_old_checkpoint_keeps_ancestry_policy(self):
        m=fresh_model(seed=7);self.addCleanup(m.close);x=self.data(96);self.train(m,x)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stage4.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
            self.assertFalse(n.event_enabled());self.assertEqual(m.fingerprint(),n.fingerprint())
            self.train(m,x);self.train(n,x);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_upper_context_and_neural_interventions_reach_decoder(self):
        m=self.make();x=self.data(513);self.train(m,x)
        baseline=m.evaluation_copy(memory='cold_frozen',diagnostic=True);self.addCleanup(baseline.close)
        _,reference=baseline.run(x[:-1],x[1:],mode='eval',logits=True)
        for region in ('L3','E4','V4','G5','EC-C','EC-X','Y'):
            for kind in ('neural_output','retained_input'):
                n=m.evaluation_copy(memory='cold_frozen',diagnostic=True);self.addCleanup(n.close)
                n.configure_contribution(kind,region);before=n.event_state(fitted=True)
                _,actual=n.run(x[:-1],x[1:],mode='eval',logits=True)
                self.assertGreater(float(np.max(np.abs(reference-actual))),0,(region,kind))
                np.testing.assert_array_equal(before,n.event_state(fitted=True))
    def test_real_stream_consolidation_resumes_after_slab_reuse(self):
        m=self.make();self.train(m,self.data(8193))
        self.assertGreater(sum(r['consolidated_records'] for r in m.event_metrics().values()),0)
        check_integrity(collect(m));n=m.snapshot_copy();self.addCleanup(n.close)
        self.assertEqual(m.fingerprint(),n.fingerprint())
        x=self.data(130);_,a=self.train(m,x);_,b=self.train(n,x)
        np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_policy_changes_require_clean_boundary_and_reject_epochs(self):
        m=self.make()
        for policy in ({'context_heads':True},{'context_heads':3},{'frontier_events':0},{'credit_lr':float('nan')}):
            with self.assertRaises(ValueError):m.configure_event(policy)
        with self.assertRaises(ValueError):m.configure_reverse(dict(epoch_bytes=4))
        self.train(m,self.data(65))
        with self.assertRaises(ValueError):m.configure_event(dict(context_heads=2))
        m.reset_context();before=m.event_state(fitted=True).copy();m.configure_event(m.event)
        np.testing.assert_array_equal(before,m.event_state(fitted=True))
    def test_all_regions_receive_task_learning_and_replay_stays_bounded(self):
        m=self.make();x=self.data(513);self.train(m,x)
        metrics=m.event_metrics()
        self.assertTrue(all(metrics[n]['task_exact']>0 for n in core.NAMES[2:]))
        self.assertGreater(sum(r['task_boundary'] for r in metrics.values()),0)
        self.assertGreater(sum(r['task_synthetic'] for r in metrics.values()),0)
        self.assertGreater(sum(r['replay_applied'] for r in metrics.values()),0)
        self.assertLessEqual(sum(r['replay_applied'] for r in metrics.values()),512*2)
        self.assertTrue(all(r['missing']==0 for r in metrics.values()))
    def test_document_drain_and_reset_preserve_fitted_credit(self):
        m=self.make();self.train(m,self.data(190));m.finish_document(mode='periodic',lr=.04,lp=.04/3,reset=False)
        n=m.snapshot_copy();self.addCleanup(n.close);self.assertEqual(m.fingerprint(),n.fingerprint())
        m.reset_context();n.reset_context();self.assertEqual(m.fingerprint(),n.fingerprint())
        self.train(m,self.data(65));self.train(n,self.data(65));self.assertEqual(m.fingerprint(),n.fingerprint())

if __name__=='__main__':unittest.main()
