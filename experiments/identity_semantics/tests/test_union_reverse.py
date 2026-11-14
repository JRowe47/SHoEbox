"""Union adjoint algebra, explicit clipping, and persisted bank epochs."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core

class UnionReverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'union_fixture_e{core.E}.so'
        subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O3','-march=native','-std=c++17','-pthread',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/union_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path));cls.fx.union_fixture.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]
        cls.fx.union_clipping.argtypes=[ct.c_void_p,ct.c_int]
    def make(self,**policy):
        m=fresh_model(seed=7)
        if policy:m.configure_reverse(policy)
        self.addCleanup(m.close);return m
    def data(self,n):
        raw=b'The river runs through the valley. The king crossed the bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,x,start=0,end=None):
        end=len(x)-1 if end is None else end
        return m.run(x[start:end],x[start+1:end+1],mode='periodic',lr=.04,lp=.04/3,logits=True)
    def test_union_matches_linear_reference_with_mixed_policies_masks_and_frozen_regions(self):
        for mixed in (0,1):
            for frozen in (0,1):
                with self.subTest(mixed=mixed,frozen=frozen):
                    a=self.make(policy='root');b=self.make()
                    self.fx.union_fixture(a.handle,1,mixed,frozen);self.fx.union_fixture(b.handle,2,mixed,frozen)
                    np.testing.assert_allclose(a.flat,b.flat,rtol=1e-5,atol=2e-7)
                    np.testing.assert_allclose(a.semantic_parameters(),b.semantic_parameters(),rtol=1e-5,atol=1e-8)
                    self.assertLess(b.reverse_metrics()['content_vjps'],a.reverse_metrics()['content_vjps'])
                    self.assertEqual(a.reverse_metrics()['roots'],b.reverse_metrics()['roots'])
    def test_clipping_change_is_explicit_and_matches_the_defined_mean_rule(self):
        for policy in (1,2):
            m=self.make();self.fx.union_clipping(m.handle,policy)
            self.assertEqual(m.reverse_metrics()['clipped_coordinates'],1 if policy==1 else 0)
    def test_partial_epochs_survive_irregular_chunks_save_load_and_snapshot_copy(self):
        x=self.data(338)
        for epoch in (4,8,16):
            with self.subTest(epoch=epoch),tempfile.TemporaryDirectory() as tmp:
                a=self.make(epoch_bytes=epoch);b=self.make(epoch_bytes=epoch)
                _,expected=self.train(a,x);actual=[]
                for lo,hi in ((0,65),(65,129),(129,337)):
                    _,logits=self.train(b,x,lo,hi);actual.append(logits)
                    path=Path(tmp)/'partial.npz';b.save(path);before=b.fingerprint();b.close()
                    b=Model.load(path);self.addCleanup(b.close);self.assertEqual(before,b.fingerprint())
                np.testing.assert_array_equal(expected,np.concatenate(actual));self.assertEqual(a.fingerprint(),b.fingerprint())
                self.assertEqual(a.reverse_metrics()['epoch_position'],337%epoch)
                copy=b.snapshot_copy();self.addCleanup(copy.close);self.assertEqual(b.fingerprint(),copy.fingerprint())
                a.finish_learning_epoch();b.finish_learning_epoch();copy.finish_learning_epoch()
                self.assertEqual(a.fingerprint(),b.fingerprint());self.assertEqual(b.fingerprint(),copy.fingerprint())
                self.assertEqual(a.reverse_metrics()['pending_roots'],0)
    def test_evaluation_keeps_pending_epoch_and_uses_visible_weights(self):
        m=self.make(epoch_bytes=16);x=self.data(96);self.train(m,x,0,65)
        state=m.reverse_state().copy();visible=m.visible_parameters()
        self.assertGreater(m.reverse_metrics()['version_bytes'],0)
        self.assertFalse(np.array_equal(visible['W'],m.params['W']))
        frozen=m.frozen_copy();self.addCleanup(frozen.close)
        np.testing.assert_array_equal(frozen.params['W'],visible['W'])
        before=m.flat.copy();m.run(x[:16],x[1:17],mode='eval')
        np.testing.assert_array_equal(state,m.reverse_state());np.testing.assert_array_equal(before,m.flat)
        with self.assertRaises(ValueError):m.configure_reverse(dict(epoch_bytes=4))
        with self.assertRaises(ValueError):m.configure_teaching(np.ones((18,2),np.float32))
        m.finish_learning_epoch();m.configure_reverse(dict(epoch_bytes=4))
    def test_old_checkpoints_keep_root_policy_and_new_checkpoints_require_epoch_state(self):
        m=self.make(policy='root');x=self.data(66);self.train(m,x)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.npz';m.save(path)
            with np.load(path) as a:arrays={k:a[k].copy() for k in a.files}
            arrays.pop('reverse_state');np.savez(path,**arrays)
            with self.assertRaises(ValueError):Model.load(path)
            arrays['version']=np.array([39],np.int32);arrays.pop('reverse_json');np.savez(path,**arrays)
            n=Model.load(path);self.addCleanup(n.close)
            self.assertEqual(n.reverse['policy'],'root');self.assertEqual(m.fingerprint(),n.fingerprint())
            self.train(m,x);self.train(n,x);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_document_drain_commits_epoch_and_serializes_a_valid_boundary(self):
        m=self.make(epoch_bytes=8);x=self.data(80);self.train(m,x)
        m.finish_document(mode='periodic',lr=.04,lp=.04/3,reset=False)
        self.assertEqual(m.reverse_metrics()['epoch_position'],0)
        self.assertEqual(m.reverse_metrics()['pending_roots'],0)
        n=m.snapshot_copy();self.addCleanup(n.close);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_corrupt_epoch_import_is_atomic(self):
        m=self.make(epoch_bytes=8);x=self.data(80);self.train(m,x)
        expected=m.fingerprint();state=m.reverse_state();bad=state.copy();bad[:8]=0
        for invalid in (bad,state[:-1]):
            with self.assertRaises(ValueError):m.import_reverse(invalid)
            self.assertEqual(expected,m.fingerprint())
    def test_reset_context_commits_pending_credit_before_clearing_owners(self):
        m=self.make(epoch_bytes=16);x=self.data(80);self.train(m,x)
        n=m.snapshot_copy();self.addCleanup(n.close);n.finish_learning_epoch()
        m.reset_context();n.reset_context()
        self.assertEqual(m.fingerprint(),n.fingerprint());self.assertEqual(m.reverse_metrics()['pending_roots'],0)

if __name__=='__main__':unittest.main()
