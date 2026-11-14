"""Reusable classes, causal completion evidence, immutable bindings and lifecycle."""
import ctypes as ct
from pathlib import Path
import os,subprocess,sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core,fresh_model

class PatternClassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'pattern_fixture_e{core.E}_{os.getpid()}.so';path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/pattern_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path));p,i=ct.c_void_p,ct.c_int
        for name,args in dict(pattern_fixture_frame=[p,i,i,i,ct.c_int64,i,p],pattern_fixture_transition=[p,i,i],
                             pattern_fixture_prepare_pair=[p,i,i,p],pattern_fixture_relay=[p,p],
                             pattern_fixture_freeze=[p,i],pattern_fixture_eviction=[p,p],
                             pattern_fixture_regions=[p,i,p],pattern_fixture_content=[p,i,p]).items():
            getattr(cls.fx,name).argtypes=args
        cls.fx.pattern_fixture_transition.restype=ct.c_float
        cls.data=np.frombuffer(b'The river runs through the valley. The king crossed the bridge. '*24,np.uint8)

    def make(self):
        m=fresh_model(seed=7);self.addCleanup(m.close);return m

    def frame(self,m,context=0,swap=0,phase=0,begin=0,learn=True):
        out=np.zeros(8,np.uint64)
        self.fx.pattern_fixture_frame(m.handle,context,swap,phase,begin,learn,out.ctypes.data);return out

    def train_pair(self,m):
        scores=[]
        for i in range(6):
            self.frame(m,i,phase=0,begin=8*i)
            self.frame(m,i+10,phase=1,begin=8*i+4)
            scores.append(self.fx.pattern_fixture_transition(m.handle,1,0))
        return scores

    def test_context_changes_reuse_class_and_promote_without_collapsing_exact_bindings(self):
        m=self.make();events=[self.frame(m,i*5,begin=4*i) for i in range(6)]
        self.assertEqual(len({int(v[0]) for v in events}),1)
        self.assertEqual(len({tuple(v[1:3]) for v in events}),6)
        self.assertEqual([int(v[3]) for v in events],[0,0,1,1,1,1])
        self.assertTrue(all(tuple(v[4:6])==(7,3) for v in events))
        self.assertEqual(m.pattern_metrics()['L3']['promoted'],1)
        self.assertEqual(m.pattern_metrics()['L3']['reused'],5)

    def test_primary_role_reversal_and_distinct_content_get_distinct_classes(self):
        m=self.make();a=self.frame(m);b=self.frame(m,swap=1);c=self.frame(m,phase=1)
        self.assertEqual(len({int(v[0]) for v in (a,b,c)}),3)
        self.assertEqual(self.frame(m,context=77)[0],a[0])

    def test_all_completing_regions_reuse_classes_under_changed_singleton_context(self):
        m=self.make();rows=[]
        for context in range(5):
            out=np.zeros((12,4),np.uint64)
            self.fx.pattern_fixture_regions(m.handle,context,out.ctypes.data);rows.append(out)
        for r in range(12):
            with self.subTest(region=core.NAMES[r+3]):
                self.assertEqual(len({int(v[r,0]) for v in rows}),1)
                self.assertEqual([int(v[r,1]) for v in rows],[0,0,1,1,1])
                self.assertGreaterEqual(rows[0][r,3],2)

    def test_learned_child_content_changes_class_without_changing_exact_identity(self):
        m=self.make();a=np.zeros(3,np.uint64);b=a.copy()
        self.fx.pattern_fixture_content(m.handle,0,a.ctypes.data)
        self.fx.pattern_fixture_content(m.handle,1,b.ctypes.data)
        self.assertNotEqual(a[0],b[0]);np.testing.assert_array_equal(a[1:],b[1:])

    def test_transitions_use_prior_evidence_and_content_forecast(self):
        m=self.make();scores=self.train_pair(m)
        self.assertEqual(scores[:3],[0,0,0]);self.assertGreater(scores[-1],.7)
        fitted=m.pattern_state(fitted=True).copy()
        normal=self.fx.pattern_fixture_transition(m.handle,0,0)
        disturbed=self.fx.pattern_fixture_transition(m.handle,0,1)
        self.assertGreater(normal-disturbed,.2)
        np.testing.assert_array_equal(fitted,m.pattern_state(fitted=True))

    def test_learned_completion_joins_changed_context_and_rejects_clock_gaps(self):
        m=self.make();self.train_pair(m)
        for gap in (0,2,-2):
            out=np.zeros(8,np.uint64);self.fx.pattern_fixture_prepare_pair(m.handle,80,gap,out.ctypes.data)
            if gap==0:
                self.assertEqual(tuple(out[:3]),(0,0,2));self.assertEqual(tuple(out[4:]),(1,2,0,8))
            else:self.assertEqual(out[1],1)

    def test_relays_preserve_class_and_eviction_never_aliases_live_assignment(self):
        m=self.make();self.frame(m);out=np.zeros(4,np.uint64)
        self.fx.pattern_fixture_relay(m.handle,out.ctypes.data)
        self.assertEqual(out[0],out[1]);self.assertEqual(out[2],out[3])
        m=self.make();self.fx.pattern_fixture_eviction(m.handle,out.ctypes.data)
        self.assertEqual(out[0],out[1]);self.assertEqual(out[2],0);self.assertGreater(out[3],out[0])
        self.assertGreater(m.pattern_metrics()['L3']['evicted'],0)

    def test_frozen_evaluation_and_region_freeze_do_not_learn_dictionary(self):
        m=self.make();self.train_pair(m);before=m.pattern_state(fitted=True).copy()
        self.frame(m,99,phase=3,learn=False)
        np.testing.assert_array_equal(before,m.pattern_state(fitted=True))
        self.fx.pattern_fixture_freeze(m.handle,6);self.frame(m,101,phase=4)
        np.testing.assert_array_equal(before,m.pattern_state(fitted=True))
        m.reset_context();np.testing.assert_array_equal(before,m.pattern_state(fitted=True))
        self.assertEqual(m.pattern_metrics()['L3']['observed'],0)

    def test_live_checkpoint_snapshot_and_chunked_training_continue_exactly(self):
        m=self.make();m.run(self.data[:431],mode='periodic')
        snapshot=m.snapshot_copy();self.addCleanup(snapshot.close)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';m.save(path);loaded=Model.load(path);self.addCleanup(loaded.close)
            self.assertEqual(m.fingerprint(),loaded.fingerprint());self.assertEqual(m.fingerprint(),snapshot.fingerprint())
            a=m.run(self.data[431:900],mode='periodic',logits=True)[1]
            b=np.concatenate([loaded.run(x,mode='periodic',logits=True)[1] for x in np.array_split(self.data[431:900],5)])
            c=snapshot.run(self.data[431:900],mode='periodic',logits=True)[1]
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
            self.assertEqual(m.fingerprint(),loaded.fingerprint());self.assertEqual(m.fingerprint(),snapshot.fingerprint())

    def test_policy_and_corrupt_state_validation_are_explicit_and_atomic(self):
        m=self.make();self.train_pair(m)
        with self.assertRaisesRegex(ValueError,'fresh stream'):m.configure_pattern_classes(False)
        before=m.fingerprint();state=m.pattern_state()
        for bad in (state[:-1],np.zeros_like(state)):
            with self.assertRaisesRegex(ValueError,'pattern-class state'):m.import_pattern_classes(bad)
            self.assertEqual(before,m.fingerprint())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';m.save(path)
            with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
            for key in ('pattern_class_policy','pattern_class_state'):
                np.savez_compressed(path,**{k:v for k,v in arrays.items() if k!=key})
                with self.assertRaisesRegex(ValueError,'pattern-class policy'):Model.load(path)

    def test_v37_checkpoint_preserves_exact_recognition_policy(self):
        m=self.make();m.configure_reverse(dict(policy="root"));m.configure_pattern_classes(False);m.configure_paired_memory(False);m.run(self.data[:431],mode='periodic')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.npz';m.save(path)
            with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files if not k.startswith(('pattern_class_','paired_memory_'))}
            arrays['version']=np.array([37],np.int32);np.savez_compressed(path,**arrays)
            old=Model.load(path);self.addCleanup(old.close);self.assertFalse(old.pattern_classes)
            np.testing.assert_array_equal(m.run(self.data[431:650],mode='periodic',logits=True)[1],
                                          old.run(self.data[431:650],mode='periodic',logits=True)[1])
            self.assertEqual(m.fingerprint(),old.fingerprint())

if __name__=='__main__':unittest.main(verbosity=2)
