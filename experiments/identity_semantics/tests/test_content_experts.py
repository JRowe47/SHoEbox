"""Constituent-only experts: causal effects, derivatives, ownership and checkpoints."""
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

class ContentExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'content_expert_fixture_e{core.E}.so';path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/content_expert_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path));p,i,f=ct.c_void_p,ct.c_int,ct.c_float
        for name,args in dict(content_fixture_build=[p,i,i,i,p,p],content_fixture_teach=[p,p,f,i],
                             content_fixture_history=[p],content_fixture_refresh=[p],content_fixture_freeze=[p,i],
                             content_fixture_route_config=[p,i],message_fixture_weights=[p],
                             completion_fixture_source=[p,i,i,ct.c_int64,ct.c_int64]).items():
            getattr(cls.fx,name).argtypes=args

    def make(self):
        m=fresh_model(seed=7);self.addCleanup(m.close)
        self.fx.message_fixture_weights(m.handle)
        self.fx.content_fixture_route_config(m.handle,64)
        m.lace['sdr_k']=64
        return m

    def build(self,m,same=False,repeated=True,relay=False):
        values=np.zeros((3,322),np.float32);meta=np.zeros((3,10),np.uint64)
        self.fx.content_fixture_build(m.handle,same,repeated,relay,values.ctypes.data,meta.ctypes.data)
        return values,meta

    def teach(self,m,g,rate=.04,teaching=0):
        self.fx.content_fixture_teach(m.handle,g.ctypes.data,rate,teaching)

    def test_selected_weights_change_child_code_and_parent_input_only_at_fixed_identity(self):
        m=self.make();base,meta=self.build(m);expert=int(meta[0,2]);r=3
        m.params['W'][r,expert,17,11]+=.3
        changed,other=self.build(m)
        self.assertGreater(abs(base[0,17]-changed[0,17]),1e-5)
        self.assertGreater(np.linalg.norm(base[1,128:192]-changed[1,128:192]),1e-5)
        np.testing.assert_array_equal(base[:,64:128],changed[:,64:128])
        np.testing.assert_array_equal(meta[:,5:9],other[:,5:9])
        m.params['W'][r,(expert+1)%core.E,19,13]+=.5
        unaffected,_=self.build(m);np.testing.assert_array_equal(changed,unaffected)

    def test_content_routing_is_expert_conditioned_and_excludes_recurrent_history(self):
        m=self.make();base,meta=self.build(m)
        self.fx.content_fixture_history(m.handle);changed,other=self.build(m)
        np.testing.assert_array_equal(base,changed)
        np.testing.assert_array_equal(meta[:,2:],other[:,2:])
        old=int(meta[0,2]);new=(old+1)%core.E
        x=base[0,128:192];m.params['keys'][3].fill(0);m.params['keys'][3,new]=x/np.linalg.norm(x)
        m.params['bias'][3,new]+=.2;self.fx.content_fixture_refresh(m.handle)
        changed,other=self.build(m);self.assertEqual(other[0,2],new)
        self.assertGreater(np.linalg.norm(base[0,:64]-changed[0,:64]),1e-3)
        np.testing.assert_array_equal(meta[:,5:9],other[:,5:9])

    def test_own_and_recursive_weight_bias_derivatives_match_finite_differences(self):
        g=np.linspace(-.2,.3,64,dtype=np.float32)
        for level,r in enumerate((3,7,10)):
            for parameter,indices in [('bias',(17,)),('W',(17,11))]:
                with self.subTest(level=level,parameter=parameter):
                    m=self.make();values,meta=self.build(m);expert=int(meta[level,2]);where=(r,expert,*indices)
                    original=float(m.params[parameter][where]);scores=[]
                    for pulse in (-.003,.003):
                        m.params[parameter][where]=original+pulse
                        out,route=self.build(m);np.testing.assert_array_equal(meta[:,2:4],route[:,2:4])
                        scores.append(float(out[2,:64].astype(float)@g))
                    m.params[parameter][where]=original;self.build(m);self.teach(m,g)
                    derivative=(original-float(m.params[parameter][where]))*(1+values[level,128:192].astype(float)@values[level,128:192])/.04
                    self.assertAlmostEqual(derivative,(scores[1]-scores[0])/.006,delta=1.5e-5)

    def test_route_key_derivative_and_tangent_normalization(self):
        m=self.make();values,meta=self.build(m);r=10;e=int(meta[2,2]);runner=int(meta[2,3]);x=values[2,128:192].astype(float)
        g=np.linspace(-.2,.3,64,dtype=np.float32)
        original=m.params['keys'][r,e].copy();direction=np.sin(np.arange(64)*.7).astype(np.float32);direction/=np.linalg.norm(direction)
        scores=[]
        for pulse in (-.002,.002):
            m.params['keys'][r,e]=original+pulse*direction;self.fx.content_fixture_refresh(m.handle)
            out,route=self.build(m);np.testing.assert_array_equal(meta[:,2:4],route[:,2:4])
            scores.append(float(out[2,:64].astype(float)@g))
        m.params['keys'][r,e]=original;self.fx.content_fixture_refresh(m.handle);self.build(m)
        residual=(values[2,:64]-values[2,64:128])/.25
        local=g*.25*(1-residual**2);factor=float(local@values[2,192:256])*float(values[2,321])
        self.assertAlmostEqual((scores[1]-scores[0])/.004,factor*float(x@direction),delta=1.5e-5)
        before=m.params['keys'].copy();self.teach(m,g)
        for chosen,sign in ((e,1),(runner,-1)):
            expected=before[r,chosen].astype(float)-sign*.04*m.lace['router_task_gain']*factor*x
            expected/=np.sqrt(1e-9+expected@expected)
            np.testing.assert_allclose(m.params['keys'][r,chosen],expected,atol=6e-8,rtol=3e-6)

    def test_shared_bank_accumulation_uses_pre_update_jacobians(self):
        a=self.make();b=self.make();self.build(a,same=True);self.build(b,same=True)
        # Every semantic node uses the same expert: force one positive key.
        for m in (a,b):
            m.params['keys'][3].fill(0);self.fx.content_fixture_refresh(m.handle);self.build(m,same=True)
        g=np.linspace(-.2,.3,64,dtype=np.float32);before=a.params['W'].copy()
        self.teach(a,g,.02);self.teach(b,g,.4)
        np.testing.assert_allclose(b.params['W']-before,20*(a.params['W']-before),rtol=3e-3,atol=2e-7)
        self.assertEqual(np.argwhere(np.any(b.params['W']!=before,axis=(-2,-1))).tolist(),[[3,0]])

    def test_frozen_middle_transmits_child_credit_and_objective_scales(self):
        a=self.make();b=self.make();self.build(a);self.build(b);before=b.flat.copy()
        self.fx.content_fixture_freeze(b.handle,7) # Native derivative check beyond public freeze orders.
        g=np.linspace(-.2,.3,64,dtype=np.float32);self.teach(a,g);self.teach(b,g)
        np.testing.assert_array_equal(a.params['W'][3],b.params['W'][3])
        np.testing.assert_array_equal(core.unflatten(before)['W'][7],b.params['W'][7])
        self.assertGreater(np.linalg.norm(b.params['W'][3]-core.unflatten(before)['W'][3]),0)
        for teaching in (0,1):
            c=self.make();self.build(c);initial=c.params['W'].copy();scales=np.ones((18,2),np.float32)
            scales[3,teaching]=0;c.configure_teaching(scales);self.teach(c,g,teaching=teaching)
            np.testing.assert_array_equal(c.params['W'][3],initial[3])
            self.assertGreater(np.linalg.norm(c.params['W'][7]-initial[7]),0)

    def test_relay_and_source_replacement_preserve_saved_content_producers(self):
        a=self.make();b=self.make();base,meta=self.build(a);other,ids=self.build(b,relay=True)
        np.testing.assert_array_equal(base,other);np.testing.assert_array_equal(meta[:,2:9],ids[:,2:9])
        for i in range(192):self.fx.completion_fixture_source(b.handle,3,100+i,2*i,2*i+2)
        g=np.linspace(-.2,.3,64,dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';b.save(path);c=Model.load(path);self.addCleanup(c.close)
            for m in (a,b,c):self.teach(m,g)
            np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(b.flat,c.flat)
            self.assertEqual(b.fingerprint(),c.fingerprint())
            self.assertTrue(all(v['missing_producers']==0 for v in c.content_expert_metrics().values()))

    def test_required_checkpoint_policy_and_eligibility_are_validated_atomically(self):
        m=self.make();self.build(m)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';m.save(path)
            with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
            for key in ('content_expert_policy','content_expert_state'):
                damaged={k:v for k,v in arrays.items() if k!=key};np.savez_compressed(path,**damaged)
                with self.assertRaisesRegex(ValueError,'content-expert policy'):Model.load(path)
            baseline=m.fingerprint();saved=m.content_expert_state()
            for damaged in (saved[:-1],np.array([1,0],np.uint64).view(np.uint8)):
                with self.assertRaisesRegex(ValueError,'eligibility'):m.import_content_experts(damaged)
                self.assertEqual(baseline,m.fingerprint())
            m.reset_context();self.assertEqual(m.content_expert_state().size,16)
            self.assertTrue(all(v['encodes']==0 for v in m.content_expert_metrics().values()))

    def test_production_local_and_language_credit_reach_saved_content_experts(self):
        data=np.frombuffer(b'A network carries messages between regions. Each region learns from its children. '*16,np.uint8)
        for local,task in ((1,0),(0,1)):
            m=fresh_model(seed=7);self.addCleanup(m.close)
            scales=np.ones((18,2),np.float32);scales[:,0]=task;scales[:,1]=local;m.configure_teaching(scales)
            m.run(data,mode='periodic');m.finish_document(mode='periodic')
            metrics=m.content_expert_metrics()
            self.assertGreater(sum(v['local_updates' if local else 'task_updates'] for v in metrics.values()),0)
            self.assertTrue(all(v['missing_producers']==0 for v in metrics.values()))
            self.assertTrue(all(v['expert_constituent_content'] for v in [m.capabilities()['features']]))

if __name__=='__main__':unittest.main(verbosity=2)
