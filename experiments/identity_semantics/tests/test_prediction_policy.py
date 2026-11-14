"""Action 4: compact forecasting, slow-Y clocks, and exact continuation."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src')]
from runtime import fresh_model,Model,core
from memory_inspection import read_memory_index

class PredictionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'prediction_fixture_e{core.E}.so'
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/prediction_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        cls.fx.prediction_zero_a1_fast.argtypes=[ct.c_void_p]
        cls.fx.prediction_projection_vjp_fixture.argtypes=[ct.c_int];cls.fx.prediction_projection_vjp_fixture.restype=ct.c_float
        cls.fx.prediction_forecast_vjp_fixture.argtypes=[ct.c_void_p];cls.fx.prediction_forecast_vjp_fixture.restype=ct.c_float
        for name in ('teacher','slow_clock','content_order'):getattr(cls.fx,f'prediction_{name}_fixture').argtypes=[ct.c_void_p]
    def make(self,profile='event',**prediction):
        m=fresh_model(seed=7,profile=profile,prediction=dict(compact=True,slow_stride=4,**prediction))
        self.addCleanup(m.close);return m
    def data(self,n=260):
        raw=b'The river runs through the valley. The king crossed the bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,x,a=0,b=None,lr=.04):
        b=len(x)-1 if b is None else b
        return m.run(x[a:b],x[a+1:b+1],mode='periodic',lr=lr,lp=lr/3,logits=True)
    def test_source_order_and_content_context_separation(self):
        self.assertEqual(self.fx.prediction_content_order_fixture(self.make().handle),1)
    def test_native_projection_and_forecast_vjp(self):
        for rank in (8,16,32,64):
            self.assertLess(self.fx.prediction_projection_vjp_fixture(rank),1e-4)
            m=self.make(rank=rank);self.assertLess(self.fx.prediction_forecast_vjp_fixture(m.handle),1e-4)
    def test_next_actual_completed_event_is_the_only_forecast_teacher_even_after_long_gap(self):
        m=self.make();self.assertEqual(self.fx.prediction_teacher_fixture(m.handle),1)
    def test_feature_view_is_immutable_and_slow_clock_has_maximum_age(self):
        m=self.make();self.assertEqual(self.fx.prediction_slow_clock_fixture(m.handle),1)
    def test_checkpoint_chunk_and_worker_count_parity(self):
        x=self.data(290)
        for heads,stride in ((1,4),(2,4),(1,8),(2,8)):
            a=self.make(profile='pipeline');a.reset_context();a.configure_event(dict(a.event,context_heads=heads));a.configure_prediction(dict(compact=True,slow_stride=stride))
            b=self.make(profile='pipeline');b.reset_context();b.configure_event(dict(b.event,context_heads=heads));b.configure_prediction(dict(compact=True,slow_stride=stride));b.configure_pipeline_workers(2)
            _,expected=self.train(a,x);pieces=[]
            with tempfile.TemporaryDirectory() as tmp:
                for lo,hi in ((0,1),(1,3),(3,67),(67,129),(129,289)):
                    _,out=self.train(b,x,lo,hi);pieces.append(out)
                    path=Path(tmp)/'partial.npz';b.save(path);before=b.fingerprint();b.close()
                    b=Model.load(path);self.addCleanup(b.close);self.assertEqual(before,b.fingerprint());b.configure_pipeline_workers(2)
                np.testing.assert_array_equal(expected,np.concatenate(pieces));self.assertEqual(a.fingerprint(),b.fingerprint())
                c=b.snapshot_copy();self.addCleanup(c.close);self.assertEqual(b.fingerprint(),c.fingerprint())
    def test_rank64_forecast_matches_dense_head_numerically(self):
        a=fresh_model(seed=7,profile='event');self.addCleanup(a.close)
        b=fresh_model(seed=7,profile='event',prediction=dict(compact=True,rank=64));self.addCleanup(b.close)
        x=self.data(1025);_,expected=self.train(a,x);_,actual=self.train(b,x)
        np.testing.assert_array_equal(expected,actual)
        original=core.unflatten(a.flat);candidate=core.unflatten(b.flat)
        for key in original:
            if key!='P':np.testing.assert_array_equal(original[key],candidate[key])
        compact=b.prediction_state(fitted=True)[64:64+14*64*64*4].view(np.float32).reshape(14,64,64)
        np.testing.assert_array_equal(original['P'][list(range(2,15))+[17]],compact)
    def test_explicit_finish_preserves_frozen_parameters_with_pipeline(self):
        m=self.make(profile='pipeline');self.train(m,self.data(66));m.finish_prediction()
        self.assertEqual(m.inspect_prediction()['pending_samples'],0)
        self.assertEqual(m.inspect_pipeline()['pending'],0)
        before=m.prediction_state(fitted=True);flat=m.flat.copy();x=self.data(9)
        m.run(x[:-1],x[1:]);np.testing.assert_array_equal(before,m.prediction_state(fitted=True));np.testing.assert_array_equal(flat,m.flat)
    def test_all_ablations_and_ranks_roundtrip(self):
        x=self.data(98)
        for compact,stride in ((True,1),(False,4),(True,4)):
            for rank in (8,32,64):
                m=fresh_model(seed=7,profile='event',prediction=dict(compact=compact,slow_stride=stride,rank=rank));self.addCleanup(m.close)
                self.train(m,x);n=m.snapshot_copy();self.addCleanup(n.close)
                self.assertEqual(m.fingerprint(),n.fingerprint());_,a=self.train(m,x);_,b=self.train(n,x)
                np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_frozen_evaluation_and_generation_are_causal(self):
        m=self.make();self.train(m,self.data())
        a=m.evaluation_copy(memory='cold_frozen');b=m.evaluation_copy(memory='cold_frozen');self.addCleanup(a.close);self.addCleanup(b.close)
        fitted=a.prediction_state(fitted=True);flat=a.flat.copy();x=self.data(98)
        _,la=a.run(x[:-1],x[1:],logits=True);_,lb=b.run(x[:-1],(x[1:]+41).astype(np.uint8),logits=True)
        np.testing.assert_array_equal(la,lb);np.testing.assert_array_equal(fitted,a.prediction_state(fitted=True));np.testing.assert_array_equal(flat,a.flat)
        c=m.evaluation_copy(memory='cold_frozen');self.addCleanup(c.close)
        pieces=[]
        for i in range(len(x)-1):pieces.append(c.run(x[i:i+1],x[i+1:i+2],logits=True)[1])
        np.testing.assert_array_equal(la,np.concatenate(pieces))
    def test_partial_interval_flush_reset_and_rate_transition(self):
        x=self.data(101);a=self.make();b=self.make()
        for lo,hi,lr in ((0,19,.04),(19,47,.02),(47,100,.08)):
            self.train(a,x,lo,hi,lr=lr)
            for i in range(lo,hi):self.train(b,x,i,i+1,lr=lr)
        self.assertEqual(a.fingerprint(),b.fingerprint())
        with self.assertRaises(ValueError):a.run(x[:1],mode='eval')
        a.finish_document(mode='periodic',lr=.08,lp=.08/3);b.finish_document(mode='periodic',lr=.08,lp=.08/3)
        self.assertEqual(a.inspect_prediction()['pending_samples'],0);self.assertEqual(a.fingerprint(),b.fingerprint())
        a.reset_context();b.reset_context();self.assertEqual(a.fingerprint(),b.fingerprint())
        self.train(a,x);self.train(b,x);self.assertEqual(a.fingerprint(),b.fingerprint())
    def test_corrupt_missing_and_wrong_policy_state_rejected_atomically(self):
        m=self.make();self.train(m,self.data(66));state=m.prediction_state();before=m.fingerprint()
        version=state.copy();version[0]=2
        nan=state.copy();nan[64:68]=np.array([np.nan],np.float32).view(np.uint8)
        wrong=state.copy();wrong[24]=32
        for bad in (state[:-1],version,nan,wrong,np.zeros(0,np.uint8)):
            with self.assertRaises(ValueError):m.import_prediction(bad)
            self.assertEqual(before,m.fingerprint())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.npz';m.save(p)
            with np.load(p) as f:arrays={k:f[k].copy() for k in f.files}
            arrays.pop('prediction_state');np.savez(p,**arrays)
            with self.assertRaises(ValueError):Model.load(p)
    def test_controls_and_parameter_accounting(self):
        m=self.make();legacy=fresh_model(seed=7,profile='event');self.addCleanup(legacy.close)
        self.assertEqual(m.parameter_count,legacy.parameter_count-14*64*(64-16)+256)
        for policy in ({'compact':1},{'rank':True},{'rank':7},{'slow_stride':3},{'slow_scale':float('nan')},{'policy':'unknown'}):
            with self.assertRaises(ValueError):m.configure_prediction(policy)
        self.train(m,self.data(9))
        with self.assertRaises(ValueError):m.configure_prediction(dict(compact=True))
        self.assertTrue(m.capabilities()['prediction_features']['fast_slow_y']);self.assertTrue(m.capabilities()['prediction_features']['compact_event_forecast'])
    def test_context_changes_and_long_continuation_keep_live_producers(self):
        m=self.make();m.configure_prediction(dict(compact=True,slow_stride=8));x=self.data(8194);self.train(m,x)
        for row in read_memory_index(m).values():
            self.assertEqual(row['invalid_retained_spans'],0);self.assertEqual(row['missing_content_records'],0)
        n=m.snapshot_copy();self.addCleanup(n.close);self.assertEqual(m.fingerprint(),n.fingerprint())
        changed=np.frombuffer(b'0123456789 NEW CONTEXT! '*8,np.uint8)
        _,a=self.train(m,changed);_,b=self.train(n,changed);np.testing.assert_array_equal(a,b)
        self.assertLessEqual(m.inspect_prediction()['pending_samples'],8)
    def test_whole_a1_source_intervention_includes_fast_correction(self):
        m=self.make();x=self.data(129);self.train(m,x)
        a=m.evaluation_copy(memory='cold_frozen',diagnostic=True);b=m.evaluation_copy(memory='cold_frozen',diagnostic=True)
        self.addCleanup(a.close);self.addCleanup(b.close)
        a.configure_contribution('whole_source','A1');b.configure_contribution('whole_source','A1')
        self.fx.prediction_zero_a1_fast(b.handle)
        _,expected=a.run(x[:-1],x[1:],logits=True);_,actual=b.run(x[:-1],x[1:],logits=True)
        np.testing.assert_array_equal(expected,actual)
    def test_upper_region_and_content_contributions_survive(self):
        m=self.make();x=self.data(513);self.train(m,x)
        baseline=m.evaluation_copy(memory='cold_frozen',diagnostic=True);self.addCleanup(baseline.close)
        _,reference=baseline.run(x[:-1],x[1:],logits=True)
        for region in ('L3','E4','V4','G5','EC-C','EC-X','Y'):
            n=m.evaluation_copy(memory='cold_frozen',diagnostic=True);self.addCleanup(n.close);n.configure_contribution('neural_output',region)
            _,actual=n.run(x[:-1],x[1:],logits=True)
            self.assertGreater(float(np.max(np.abs(reference-actual))),0,region)

if __name__=='__main__':unittest.main(verbosity=2)
