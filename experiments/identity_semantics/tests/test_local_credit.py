"""Objective-specific causal credit, independently checked derivatives and policies."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core,fresh_model

class LocalCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'local_credit_fixture_e{core.E}.so';path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/local_credit_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path));p,i=ct.c_void_p,ct.c_int
        for name,args in dict(local_fixture_setup=[p,i,i,i],local_fixture_teach=[p,ct.c_float,i],
                             local_fixture_snapshot=[p,i,ct.c_int64,p],local_fixture_pending=[p,p],
                             local_fixture_freeze=[p,i],local_fixture_task_credit=[p],
                             local_fixture_task_status=[p,p],
                             local_fixture_live_teach=[p,i],neural_fixture_advance=[p,i,i]).items():
            getattr(cls.fx,name).argtypes=args

    def make(self,depth=2,repeated=1,relay=0,pattern_classes=True):
        m=fresh_model(seed=7);self.addCleanup(m.close)
        m.configure_content_experts(False) # Isolate the original neural-credit derivative.
        m.configure_pattern_classes(pattern_classes)
        m.configure_paired_memory(False)
        self.fx.local_fixture_setup(m.handle,depth,repeated,relay)
        # The controlled native fixture sets these policies; mirror them in
        # checkpoint metadata exactly as a normal constructor would.
        m.lace.update(depth=depth,credit_gain=1,reuse_exponent=.5,dfa_gain=0,credit_features=64)
        m.utility['innovation']=0
        return m

    def snapshot(self,m,r=6,tick=106):
        a=np.zeros(193,np.float32);self.fx.local_fixture_snapshot(m.handle,r,tick,a.ctypes.data)
        return a.astype(np.float64)

    def teach(self,m,rate=.02,relay=0):
        self.fx.local_fixture_teach(m.handle,rate,relay)

    def changed(self,before,m):
        return np.argwhere(np.any(core.unflatten(before)['W']!=m.params['W'],axis=(-2,-1))).tolist()

    def test_local_only_teaches_causal_children_feedback_and_retained_input(self):
        m=self.make();before=m.flat.copy();scales=np.ones((18,2),np.float32);scales[:,0]=0
        m.configure_teaching(scales);self.teach(m)
        self.assertEqual(self.changed(before,m),[[3,2],[4,3],[5,4],[6,2]])
        metrics=m.local_credit_metrics();self.assertGreater(metrics[core.NAMES[5]]['neural_updates'],0)
        self.assertGreater(metrics[core.NAMES[4]]['feedback_deliveries'],0)
        pending=np.zeros(2,np.uint64);self.fx.local_fixture_pending(m.handle,pending.ctypes.data)
        np.testing.assert_array_equal(pending,0)
        self.assertTrue(all(v['missing_producers']==0 for v in metrics.values()))
        observed=m.acceptance_metrics()
        self.assertEqual(sum(v['language_dfa_update_calls'] for v in observed.values()),0)
        self.assertGreater(sum(v['local_update_calls'] for v in observed.values()),0)
        after=m.flat.copy();self.teach(m);np.testing.assert_array_equal(after,m.flat)

    def test_derivatives_match_finite_difference_before_any_mutation(self):
        m=self.make(depth=1);saved=self.snapshot(m);u,h,residual,gate=saved[:64],saved[64:128],saved[128:192],saved[192]
        params={k:v.copy().astype(np.float64) for k,v in m.params.items()}
        w=params['W'][6,2];bias=params['bias'][6,2];p=params['P'][6];pb=params['pbias'][6]
        target=np.array([.04*(j%7-3) for j in range(64)],np.float64)
        error=p@h+pb-target;self.assertLess(np.max(np.abs(error)),1)
        g=p.T@error;pre=g*(1-h*h);du=pre+gate*w.T@pre
        route=np.array([.04*(1 if j%2 else -1) for j in range(64)])
        dcue=(pre@residual)*route
        slot=int(m.fixed['starts'][6]);perm=m.fixed['perm'][slot];sign=m.fixed['signs'][slot]
        expected={3:.3*du,4:np.zeros(64)}
        expected[4][perm]=sign*(.2*du+.6*.8*(.5*.1*du+.1*dcue))
        # Independently perturb the consumed source output through both saved
        # input and response-gate branches, holding hard routing fixed.
        for source in (3,4):
            for coordinate in (0,11,37,63):
                def loss(eps):
                    delta=np.zeros(64);gate_delta=0.
                    if source==3:delta[coordinate]=.3*eps
                    else:
                        j=int(np.flatnonzero(perm==coordinate)[0]);value=sign[j]*eps
                        delta[j]=(.2+.6*.8*.5*.1)*value
                        gate_delta=.6*.8*.1*route[j]*value
                    response=np.tanh(u+delta+bias+(gate+gate_delta)*(w@(u+delta)))
                    err=p@response+pb-target
                    return .5*(err@err)
                numeric=(loss(1e-4)-loss(-1e-4))/2e-4
                self.assertAlmostEqual(numeric,expected[source][coordinate],delta=2e-7)
        child={3:self.snapshot(m,3,101),4:self.snapshot(m,4,102)}
        self.teach(m)
        for source,expert in ((3,2),(4,3)):
            v=child[source];scale=.02/(1+v[:64]@v[:64])
            delta=-scale*expected[source]*(1-v[64:128]**2)
            np.testing.assert_allclose(m.params['bias'][source,expert]-params['bias'][source,expert],delta,rtol=3e-4,atol=3e-9)

    def test_reuse_combines_feedback_branches_and_counts_repeated_roles(self):
        single=self.make(depth=1);repeated=self.make(depth=1,repeated=2)
        before=single.params['bias'].copy();self.teach(single);self.teach(repeated)
        np.testing.assert_allclose(repeated.params['bias'][3,2]-before[3,2],
                                   (single.params['bias'][3,2]-before[3,2])/np.sqrt(2),atol=3e-9)
        np.testing.assert_array_equal(single.params['bias'][4,3],repeated.params['bias'][4,3])
        self.assertEqual(single.local_credit_metrics()[core.NAMES[4]]['neural_deliveries'],1)

    def test_freeze_blocks_mutation_but_not_transmission_and_scales_are_local(self):
        a=self.make();b=self.make();before=a.flat.copy()
        self.fx.local_fixture_freeze(b.handle,6) # frozen issuer consumes without teaching
        self.teach(b);np.testing.assert_array_equal(before,b.flat)
        c=self.make();self.fx.local_fixture_freeze(c.handle,3)
        scales=np.ones((18,2),np.float32);scales[:,0]=0;scales[4,1]=.5
        c.configure_teaching(scales);self.teach(a);self.teach(c)
        self.assertEqual(self.changed(before,c),[[4,3],[5,4],[6,2]])
        np.testing.assert_array_equal(a.params['W'][5],c.params['W'][5])
        initial=core.unflatten(before)['bias'][4,3]
        np.testing.assert_allclose(c.params['bias'][4,3]-initial,.5*(a.params['bias'][4,3]-initial),atol=3e-9)
        for rate,scale in ((0,1),(.02,0)):
            d=self.make();scales=np.ones((18,2),np.float32);scales[6,1]=scale;d.configure_teaching(scales)
            self.teach(d,rate);np.testing.assert_array_equal(before,d.flat)

    def test_depth_and_input_only_edges_do_not_invent_expert_updates(self):
        for depth,expected in [(0,[[6,2]]),(1,[[3,2],[4,3],[6,2]]),(2,[[3,2],[4,3],[5,4],[6,2]])]:
            m=self.make(depth=depth);before=m.flat.copy();self.teach(m)
            self.assertEqual(self.changed(before,m),expected)

    def test_transparent_relay_preserves_credit_without_spending_depth(self):
        a=self.make(depth=1);b=self.make(depth=1,relay=1)
        self.teach(a);self.teach(b,relay=1);np.testing.assert_array_equal(a.flat,b.flat)
        np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
        self.assertGreater(b.local_credit_metrics()['S3']['relay_hops'],0)

    def test_semantic_branch_is_applied_once_and_language_queue_is_untouched(self):
        a=self.make(depth=1);b=self.make(depth=1);b.configure_local_credit(False)
        self.fx.local_fixture_task_credit(a.handle);before=np.zeros(195,np.float32)
        self.fx.local_fixture_task_status(a.handle,before.ctypes.data)
        self.teach(a);self.teach(b)
        np.testing.assert_array_equal(a.semantic_parameters(),b.semantic_parameters())
        after=np.zeros_like(before);self.fx.local_fixture_task_status(a.handle,after.ctypes.data)
        np.testing.assert_array_equal(after,before)

    def test_archived_producers_save_load_and_relay_target_match_immediate(self):
        a=self.make();b=self.make()
        for source in (3,4,5,6,7):self.fx.neural_fixture_advance(b.handle,source,192)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';b.save(path);c=Model.load(path);self.addCleanup(c.close)
            for m in (a,b,c):self.teach(m,relay=1)
            np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(b.flat,c.flat)
            self.assertEqual(b.fingerprint(),c.fingerprint())
            self.assertTrue(all(v['missing_producers']==0 for v in c.local_credit_metrics().values()))

    def test_v35_preserves_old_policy_and_v36_requires_explicit_saved_policy(self):
        m=self.make(pattern_classes=False) # v35 predates learned class recognition.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';m.save(path)
            with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
            arrays.pop('local_credit_policy');np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'local-credit policy'):Model.load(path)
            arrays['version'][0]=35
            arrays['composition_state'][:4]=np.array([9],np.uint32).view(np.uint8)
            arrays.pop('content_expert_policy');arrays.pop('content_expert_state')
            arrays.pop('pattern_class_policy');arrays.pop('pattern_class_state')
            arrays.pop('paired_memory_policy');arrays.pop('paired_memory_state')
            np.savez_compressed(path,**arrays)
            old=Model.load(path);self.addCleanup(old.close);self.assertFalse(old.local_credit)
            baseline=m.snapshot_copy();self.addCleanup(baseline.close);baseline.configure_local_credit(False)
            self.teach(old);self.teach(baseline);self.assertEqual(old.fingerprint(),baseline.fingerprint())
            old.save(path);again=Model.load(path);self.addCleanup(again.close);self.assertFalse(again.local_credit)

    def test_isolated_live_local_objectives_reach_children_in_six_upper_regions(self):
        workloads=[b'The king walked across the street. The river runs through the valley. '*40,
                   (b'A network carries messages between connected regions. Each region learns from the next observation. '
                    b'When the context changes, a remembered event can guide the next decision. '*17)]
        for text in workloads:
            base=fresh_model(seed=7)
            try:
                base.run(np.frombuffer(text,np.uint8),mode='periodic',lr=.02,lp=.02/3)
                for name in ('L3','S3','F3','E4','F4','G5'):
                    with self.subTest(workload=text[:12],region=name):
                        m=base.snapshot_copy()
                        try:
                            scales=np.ones((18,2),np.float32);scales[:,0]=0;m.configure_teaching(scales)
                            before=m.flat.copy();self.fx.local_fixture_live_teach(m.handle,core.NAMES.index(name))
                            changed=self.changed(before,m)
                            self.assertTrue(any(r!=core.NAMES.index(name) for r,e in changed),changed)
                            metrics=m.local_credit_metrics()
                            self.assertEqual(metrics[name]['objectives'],1)
                            self.assertGreater(sum(v['neural_updates'] for v in metrics.values()),0)
                            self.assertTrue(all(v['missing_producers']==0 for v in metrics.values()))
                        finally:m.close()
            finally:base.close()

if __name__=='__main__':unittest.main(verbosity=2)
