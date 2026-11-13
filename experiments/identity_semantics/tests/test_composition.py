import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core,fresh_model,engine

def feed(m,data,train=False):
    a=np.frombuffer(data,np.uint8)
    return m.run(a,mode='periodic' if train else 'eval',lr=0,lp=0,route=0)

def trace(m,data,region='L2',finish=False):
    rows=[];previous=0
    for byte in data:
        m.run([byte]);e=m.content_events()[region]
        if e['seq']!=previous:
            previous=e['seq']
            if previous:rows.append(e)
    if finish:
        m.finish_document();e=m.content_events()[region]
        if e['seq']!=previous and e['seq']:rows.append(e)
    return rows

class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(4)
        cls.words=[b'abcd',b'efgh',b'ijkl',b'mnop']
        cls.grammar=fresh_model()
        data=b''.join(bytes(rng.integers(160,240,size=rng.integers(1,6),dtype=np.uint8))+cls.words[int(rng.integers(4))] for _ in range(2000))
        feed(cls.grammar,data,True)
        cls.grammar.finish_document()
        cls.fixture=ROOT/'results/native_fixture.so'
        subprocess.run(['g++','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(ROOT/'tests/native_fixture.cpp'),'-o',str(cls.fixture)],check=True)
        cls.fx=ct.CDLL(str(cls.fixture))
        cls.fx.fixture_memory.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_float]
        cls.fx.fixture_flush.argtypes=[ct.c_void_p,ct.c_int]
        cls.fx.fixture_credit_clear.argtypes=[ct.c_void_p]
        cls.fx.fixture_lace_parents.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
    @classmethod
    def tearDownClass(cls):cls.grammar.close()
    def make(self):
        m=fresh_model();self.addCleanup(m.close);return m
    def frozen(self,m=None):
        c=(m or self.grammar).frozen_copy();self.addCleanup(c.close);return c
    def snapshot(self,m):
        c=m.snapshot_copy();self.addCleanup(c.close);return c
    def credit(self,m,r):
        a=np.zeros((16,136),np.float64)
        m.lib.composition_credit_snapshot.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p]
        m.lib.composition_credit_snapshot(m.handle,r,a.ctypes.data);return a
    def inject(self,m,r,g,input_only=0):
        g=np.ascontiguousarray(g,np.float32)
        m.lib.composition_credit_probe.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]
        m.lib.composition_credit_probe(m.handle,r,g.ctypes.data,input_only,0)
    def test_01_identity_is_context_and_alignment_invariant(self):
        identities=[];codes=[]
        for prefix in (b'X',b'YZ',b'!QZ',bytes([200,201,202,203,204])):
            rows=trace(self.frozen(),prefix+b'abcd'+b'!')
            matches=[e for e in rows if e['span']==(len(prefix),len(prefix)+4)]
            self.assertEqual(len(matches),1)
            self.assertTrue(matches[0]['recognized']);identities.append(matches[0]['id']);codes.append(matches[0]['code'])
        self.assertEqual(len(set(identities)),1)
        for code in codes[1:]:np.testing.assert_array_equal(code,codes[0])
    def test_02_hold_does_not_republish_or_reconsume(self):
        m=self.frozen();feed(m,b'ab')
        before=m.content_events();seq=before['L2']['seq'];down=m.composition_metrics()['L3']['consumed']
        feed(m,b'c');after=m.content_events()
        self.assertEqual(after['L2']['seq'],seq)
        self.assertEqual(m.composition_metrics()['L3']['consumed'],down)
        self.assertEqual(after['L2']['pending'],3)
    def test_03_order_and_unknown_recognition(self):
        m=self.frozen();rows=trace(m,b'!abcd?dcba!')
        ab=[e for e in rows if e['span']==(1,5)][0]
        self.assertTrue(ab['recognized'])
        reversed_rows=[e for e in rows if e['span'][0]>=6]
        self.assertTrue(reversed_rows)
        self.assertTrue(all(e['id']!=ab['id'] for e in reversed_rows))
        self.assertTrue(all(not e['recognized'] for e in reversed_rows))
    def test_04_new_combinations_of_familiar_units(self):
        m=self.frozen();frozen=np.zeros(18,np.int32);frozen[:6]=1
        m.configure_training_policy(frozen)
        rng=np.random.default_rng(12)
        data=b''.join(bytes(rng.integers(160,240,size=rng.integers(2,7),dtype=np.uint8))+(b'abcdefgh' if rng.integers(2)==0 else b'ijklmnop') for _ in range(400))
        feed(m,data,True)
        # Only AB and CD were taught here. AD and DA are novel relations.
        a=trace(self.frozen(m),b'!abcdmnop?',region='L3')
        b=trace(self.frozen(m),b'!mnopabcd?',region='L3')
        m1=[e for e in a if e['span']==(1,9)]
        m2=[e for e in b if e['span']==(1,9)]
        # One extra observation delivers the final completed lower event.
        if not m1 or not m2:
            a=trace(self.frozen(m),b'!abcdmnop?!',region='L3')
            b=trace(self.frozen(m),b'!mnopabcd?!',region='L3')
            m1=[e for e in a if e['span']==(1,9)];m2=[e for e in b if e['span']==(1,9)]
        self.assertEqual(len(m1),1);self.assertEqual(len(m2),1)
        self.assertEqual(m1[0]['count'],2);self.assertFalse(m1[0]['recognized'])
        self.assertTrue(all(m1[0]['children'][:,8]))
        self.assertNotEqual(m1[0]['id'],m2[0]['id'])
        self.assertGreater(np.linalg.norm(m1[0]['code']-m2[0]['code']),.1)
    def test_05_spans_cover_stream_once(self):
        data=b'!abcd?efgh#ijkl$mnop@abc'
        rows=trace(self.frozen(),data,finish=True)
        self.assertEqual(rows[0]['span'][0],0);self.assertEqual(rows[-1]['span'][1],len(data))
        for left,right in zip(rows,rows[1:]):self.assertEqual(left['span'][1],right['span'][0])
        for event in rows:
            children=event['children']
            self.assertEqual(len(set(map(int,children[:,2]))),len(children))
            if len(children)>1:self.assertTrue(np.all(children[1:,3]==children[:-1,4]))
    def test_06_frozen_eval_does_not_learn_dictionaries(self):
        m=self.frozen();before=m.composition_state(fitted=True)
        feed(m,b'novel sequence not present in dictionary'*20);m.finish_document()
        np.testing.assert_array_equal(m.composition_state(fitted=True),before)
    def test_07_resume_is_exact(self):
        m=self.make();x=np.arange(200,dtype=np.uint8);m.run(x,mode='periodic',lr=.012,lp=.004)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
            self.assertEqual(m.fingerprint(),n.fingerprint())
            a=m.run(x,mode='periodic',lr=.012,lp=.004,logits=True)
            b=n.run(x,mode='periodic',lr=.012,lp=.004,logits=True)
            self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_08_chunk_boundaries_are_not_events(self):
        a=self.frozen();b=self.frozen();data=b'!abcd?efgh#ijkl$mnop@'*9
        feed(a,data)
        for lo in range(0,len(data),7):feed(b,data[lo:lo+7])
        np.testing.assert_array_equal(a.composition_state(),b.composition_state())
        np.testing.assert_array_equal(a.publications(),b.publications())
    def test_09_targets_do_not_change_inference(self):
        a=self.frozen();b=self.frozen();x=np.frombuffer(b'!abcdefgh?ijklmnop#',np.uint8)
        out1=a.run(x,np.zeros_like(x),logits=True)[1]
        out2=b.run(x,np.full_like(x,255),logits=True)[1]
        np.testing.assert_array_equal(out1,out2)
        np.testing.assert_array_equal(a.composition_state(),b.composition_state())
    def test_10_zero_local_scale_freezes_all_error_driven_paths(self):
        m=self.make();feed(m,b'abcdefghijklmno')
        scale=np.ones((18,2),np.float32);scale[17,1]=0;m.configure_teaching(scale)
        before=m.flat.copy();extra=m.extra_flat.copy();g=np.ones(64,np.float32)
        m.lib.verification_expert_update.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_float,ct.c_int]
        m.lib.verification_expert_update(m.handle,17,g.ctypes.data,.1,1)
        np.testing.assert_array_equal(m.flat,before);np.testing.assert_array_equal(m.extra_flat,extra)
    def test_11_quarter_scale_also_scales_routing(self):
        m=self.make();feed(m,b'abcdefghijklmno');a=self.snapshot(m);b=self.snapshot(m)
        scales=np.ones((18,2),np.float32);scales[17,1]=.25;b.configure_teaching(scales)
        g=np.linspace(-.8,1.1,64,dtype=np.float32)
        for v in (a,b):
            v.lib.verification_expert_update.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_float,ct.c_int]
            v.lib.verification_expert_update(v.handle,17,g.ctypes.data,.01,1)
        da=np.linalg.norm(a.params['keys'][17]-m.params['keys'][17]);db=np.linalg.norm(b.params['keys'][17]-m.params['keys'][17])
        self.assertGreater(da,1e-9);self.assertAlmostEqual(db/da,.25,delta=.015)
    def test_12_retention_targets_input_not_previous_expert(self):
        m=self.make();feed(m,b'0123456789abcdef');self.fx.fixture_credit_clear(m.handle)
        self.inject(m,17,np.ones(64,np.float32),input_only=1)
        snapshot=self.credit(m,17)
        self.assertEqual(snapshot[1,4],1)
        np.testing.assert_allclose(snapshot[1,8:72],.5,atol=1e-7)
        np.testing.assert_array_equal(snapshot[1,72:],0)
        before=m.flat.copy();self.fx.fixture_flush(m.handle,17)
        np.testing.assert_array_equal(m.flat,before)
    def test_13_retention_gradient_matches_finite_difference(self):
        base=self.make();feed(base,b'0123456789abcdef')
        m=self.snapshot(base);feed(m,b'g');self.fx.fixture_credit_clear(m.handle)
        g=np.linspace(-.4,.7,64,dtype=np.float32);self.inject(m,17,g)
        analytical=self.credit(m,17)[1,8:72]
        numerical=[]
        for j in (0,7,29,63):
            values=[]
            for delta in (-.002,.002):
                c=self.snapshot(base);self.fx.fixture_memory(c.handle,17,j,delta);feed(c,b'g')
                values.append(float(np.dot(g,c.publications()[17])));c.close()
            numerical.append((values[1]-values[0])/.004)
        np.testing.assert_allclose(analytical[[0,7,29,63]],numerical,rtol=.004,atol=1e-5)
    def test_14_context_edges_name_actual_child_publications(self):
        m=self.frozen();feed(m,b'!abcd?')
        event=m.content_events()['L2'];self.assertEqual(event['span'],(1,5))
        rows=np.zeros((16,4),np.int64);weights=np.zeros(16,np.float32)
        self.fx.fixture_lace_parents(m.handle,2,rows.ctypes.data,weights.ctypes.data)
        actual={(int(row[0]),int(row[1])) for row in rows if row[2]<=-100}
        expected={(int(c[6]),int(c[5])) for c in event['children']}
        self.assertEqual(actual,expected)
        self.assertTrue(all(weights[i]>0 for i,row in enumerate(rows) if row[2]<=-100))
    def test_15_legacy_disabled_matches_original(self):
        a=fresh_model(kernel='native');self.addCleanup(a.close)
        b=self.make();b.configure_composition(dict(enabled=False))
        x=np.frombuffer(b'a short matched training stream. '*4,np.uint8)
        ya=a.run(x,mode='periodic',lr=.012,lp=.004,logits=True)
        yb=b.run(x,mode='periodic',lr=.012,lp=.004,logits=True)
        self.assertEqual(ya[0],yb[0]);np.testing.assert_array_equal(ya[1],yb[1])
        np.testing.assert_array_equal(a.flat,b.flat);np.testing.assert_array_equal(a.extra_flat,b.extra_flat)
    def test_16_invalid_policy_and_incompatible_mode(self):
        m=self.make()
        for policy in ({'threshold':float('nan')},{'max_children':9},{'min_count':0},{'max_children':2.5}):
            with self.assertRaises(ValueError):m.configure_composition(policy)
        with self.assertRaises(ValueError):m.run([1],mode='spatial')
        with self.assertRaises(ValueError):m.import_composition(np.zeros(4,np.uint8))
    def test_17_document_reset_preserves_learned_codes(self):
        m=self.frozen();before=m.composition_state(fitted=True);feed(m,b'abcdef')
        m.finish_document(reset=True)
        np.testing.assert_array_equal(before,m.composition_state(fitted=True))
        self.assertTrue(all(e['seq']==0 and e['pending']==0 for e in m.content_events().values()))
    def test_18_singletons_keep_neural_provenance_without_tape_entries(self):
        m=self.make();feed(m,bytes(range(30)))
        self.assertGreater(m.content_events()['L3']['seq'],0)
        self.assertEqual(m.lace_metrics()['L3']['publications'],0)
        rows=np.zeros((16,4),np.int64);weights=np.zeros(16,np.float32)
        self.fx.fixture_lace_parents(m.handle,17,rows.ctypes.data,weights.ctypes.data)
        origins=[int(row[0]) for row in rows if row[2]<=-100]
        self.assertIn(0,origins)
        self.assertNotIn(6,origins)
    def test_19_region_ablation_removes_content_evidence(self):
        m=self.frozen();m.ablate(10) # Remove A1 as a source.
        feed(m,b'!abcd?efgh!')
        self.assertEqual(m.content_events()['L2']['seq'],0)
        m.finish_document()

if __name__=='__main__':unittest.main(verbosity=2)
