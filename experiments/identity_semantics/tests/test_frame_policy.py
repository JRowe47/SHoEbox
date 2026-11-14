"""Action 5: fixed binding descriptors, learned completion, and continuation."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src'),str(ROOT/'tests')]
from runtime import fresh_model,Model,core
from memory_inspection import read_memory_index
import test_pattern_classes as patterns
import test_paired_memory as paired

class FramePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'frame_fixture_e{core.E}_{os.getpid()}.so'
        subprocess.run(['g++','-O2','-march=native','-std=c++17','-pthread',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/frame_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        for name in ('contract','local_frontier','retention','context'):
            getattr(cls.fx,f'frame_{name}_fixture').argtypes=[ct.c_void_p]
        cls.fx.frame_vjp_fixture.argtypes=[ct.c_void_p,ct.c_int,ct.c_int];cls.fx.frame_vjp_fixture.restype=ct.c_float
    def make(self,profile='event',combined=False,**kwargs):
        m=fresh_model(seed=7,profile=profile,frames='lightweight-frames-v1',
                      prediction=dict(compact=True,rank=16,slow_stride=8) if combined else None,**kwargs)
        self.addCleanup(m.close);return m
    def data(self,n=600):
        raw=b'The river runs through the valley. The king crossed the bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,x,lo=0,hi=None):
        hi=len(x)-1 if hi is None else hi
        return m.run(x[lo:hi],x[lo+1:hi+1],mode='periodic',lr=.04,lp=.04/3,logits=True)[1]
    def test_exact_edges_no_frame_parameters_and_single_frame_learned_completion(self):
        self.assertEqual(self.fx.frame_contract_fixture(self.make(profile=None).handle),1)
    def test_completion_to_child_vjp_through_fixed_frames_and_frozen_recipient(self):
        for frames in (1,2,8):
            for union in (0,1):
                m=self.make(profile=None)
                self.assertLess(self.fx.frame_vjp_fixture(m.handle,frames,union),5e-5)
    def test_fixed_descriptors_are_transparent_to_local_completion_credit(self):
        self.assertEqual(self.fx.frame_local_frontier_fixture(self.make().handle),1)
    def test_content_and_completion_exclude_recurrent_context(self):
        self.assertEqual(self.fx.frame_context_fixture(self.make().handle),1)
    def test_checkpoint_chunks_and_workers_preserve_both_policies(self):
        x=self.data(514)
        for profile in (None,'event','pipeline'):
            for combined in (False,True):
                a=self.make(profile,combined);b=self.make(profile,combined)
                if profile=='pipeline':b.configure_pipeline_workers(2)
                expected=self.train(a,x);pieces=[]
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'partial.npz'
                    for lo,hi in ((0,1),(1,7),(7,129),(129,513)):
                        pieces.append(self.train(b,x,lo,hi));b.save(path);fingerprint=b.fingerprint();b.close()
                        b=Model.load(path);self.addCleanup(b.close)
                        self.assertEqual(b.frames,'lightweight-frames-v1');self.assertEqual(fingerprint,b.fingerprint())
                        if profile=='pipeline':b.configure_pipeline_workers(2)
                    np.testing.assert_array_equal(expected,np.concatenate(pieces));self.assertEqual(a.fingerprint(),b.fingerprint())
                    c=b.snapshot_copy();self.addCleanup(c.close);self.assertEqual(c.fingerprint(),b.fingerprint())
                    a.close();b.close();c.close()
    def test_frozen_causal_generation_and_paired_memory(self):
        m=self.make(combined=True);x=self.data();self.train(m,x)
        a=m.evaluation_copy(memory='warm_frozen');b=m.evaluation_copy(memory='warm_frozen')
        self.addCleanup(a.close);self.addCleanup(b.close)
        before=a.evaluation_memory_info()['store_digest'];fitted=a.flat.copy()
        _,expected=a.run(x[:128],x[1:129],logits=True)
        actual=np.concatenate([b.run(x[i:i+1],np.array([255],np.uint8),logits=True)[1] for i in range(128)])
        np.testing.assert_array_equal(expected,actual);np.testing.assert_array_equal(fitted,a.flat)
        self.assertEqual(before,a.evaluation_memory_info()['store_digest'])
    def test_upper_region_content_and_context_contributions_survive(self):
        from measure_current_baseline import fixtures
        m=self.make(combined=True);x=np.frombuffer(fixtures('natural',2049)[0][0],np.uint8);self.train(m,x)
        a=m.evaluation_copy(memory='cold_frozen',diagnostic=True);self.addCleanup(a.close)
        _,expected=a.run(x[:2048],x[1:2049],logits=True)
        self.assertTrue(all(a.inspect_frames()[r]['publications']>0 for r in core.NAMES))
        for kind,regions in (('neural_output',('L3','E4','V4','G5','EC-C','EC-X','Y')),
                             ('content_expert',('L3','E4','G5'))):
            for region in regions:
                b=m.evaluation_copy(memory='cold_frozen',diagnostic=True);self.addCleanup(b.close)
                b.configure_contribution(kind,region)
                _,actual=b.run(x[:2048],x[1:2049],logits=True)
                self.assertGreater(float(np.max(np.abs(expected-actual))),0,(kind,region))
    def test_wrap_retention_document_drain_and_restart(self):
        m=self.make(combined=True);x=self.data(16386);self.train(m,x)
        self.assertEqual(self.fx.frame_retention_fixture(m.handle),1)
        for row in read_memory_index(m).values():
            self.assertEqual(row['invalid_retained_spans'],0);self.assertEqual(row['missing_content_records'],0)
        n=m.snapshot_copy();self.addCleanup(n.close)
        for model in (m,n):
            model.finish_document(mode='periodic',lr=.04,lp=.04/3)
            self.assertTrue(all(row['pending_frames']==0 for row in model.inspect_frames().values()))
            model.reset_context()
        np.testing.assert_array_equal(self.train(m,self.data(257)),self.train(n,self.data(257)))
        self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_invalid_policy_and_incompatible_checkpoint_rejected(self):
        m=self.make()
        for bad in ('unknown',{},True):
            with self.assertRaises(ValueError):m.configure_frames(bad)
        with self.assertRaises(ValueError):m.configure_content_experts(False)
        with self.assertRaises(ValueError):m.configure_composition({'enabled':False})
        self.train(m,self.data(66))
        with self.assertRaises(ValueError):m.configure_frames(None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';m.save(path)
            with np.load(path) as f:arrays={k:f[k].copy() for k in f.files}
            self.assertEqual(int(arrays['version'][0]),45)
            for key in ('frame_policy','prediction_json','event_json'):
                np.savez(path,**{k:v for k,v in arrays.items() if k!=key})
                with self.assertRaises(ValueError):Model.load(path)
            bad={**arrays,'frame_policy':np.array('learned-frames-v1')};np.savez(path,**bad)
            with self.assertRaises(ValueError):Model.load(path)
        m.reset_context()
        with self.assertRaises(ValueError):m.configure_frames(None)
    def test_frame_policy_adds_no_fitted_parameters(self):
        a=self.make();b=fresh_model(seed=7,profile='event');self.addCleanup(b.close)
        self.assertEqual(a.parameter_count,b.parameter_count)
        self.assertEqual(a.capabilities()['checkpoint_version'],45)

# Exercise the existing recognition and paired-memory oracles with the new
# policy. Legacy-version synthesis is tested on its original representation.
class LightweightPatternTests(patterns.PatternClassTests):
    def make(self):
        m=fresh_model(seed=7,frames='lightweight-frames-v1');self.addCleanup(m.close);return m
    test_v37_checkpoint_preserves_exact_recognition_policy=None

class LightweightPairedTests(paired.PairedMemoryTests):
    def make(self,fixture=True,ways=4):
        m=Model(config=dict(key=0,key_mix=0,center_value=0,threshold=.2,buckets=4,ways=ways),frames='lightweight-frames-v1') if fixture else fresh_model(seed=7,frames='lightweight-frames-v1')
        self.addCleanup(m.close);return m

if __name__=='__main__':unittest.main(verbosity=2)
