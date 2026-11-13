"""Causal channel isolation, production parity, and explicit memory policies."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,core


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'results').mkdir(exist_ok=True)
        path=ROOT/'results/contribution_fixture.so'
        subprocess.run(['g++','-O2','-march=native','-std=c++17','-shared','-fPIC',
                        str(ROOT/'tests/contribution_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        for name,args in {
            'check_fixture_memory':[ct.c_void_p,ct.c_int,ct.c_float],
            'check_fixture_vectors':[ct.c_void_p,ct.c_int,ct.c_void_p],
            'check_fixture_relay':[ct.c_void_p,ct.c_int,ct.c_int],
            'check_fixture_atom':[ct.c_void_p,ct.c_int,ct.c_int],
            'check_fixture_symbol':[ct.c_void_p,ct.c_int],
            'check_fixture_zero_expert':[ct.c_void_p,ct.c_int],
        }.items():getattr(cls.fx,name).argtypes=args
        cls.original=fresh_model(kernel='relays')
        cls.text=np.frombuffer(b'To be, or not to be: that is the question. '*16,np.uint8).copy()
        cls.original.run(cls.text,mode='periodic',lr=.04,lp=.013,route=.002)

    @classmethod
    def tearDownClass(cls):cls.original.close()

    def make(self,kind='none',region=None,memory='cold_adaptive',diagnostic=True):
        m=self.original.evaluation_copy(memory=memory,diagnostic=diagnostic)
        self.addCleanup(m.close)
        if diagnostic:m.configure_contribution(kind,region)
        return m

    def vectors(self,m,r):
        a=np.zeros((5,64),np.float32);self.fx.check_fixture_vectors(m.handle,r,a.ctypes.data);return a

    def test_01_disabled_checks_match_production_exactly(self):
        source=self.original.fingerprint()
        for memory in ('cold_adaptive','cold_frozen','warm_frozen','warm_adaptive'):
            a=self.make(memory=memory,diagnostic=False);b=self.make(memory=memory)
            la,za=a.run(self.text,logits=True);lb,zb=b.run(self.text,logits=True)
            self.assertEqual(la,lb);np.testing.assert_array_equal(za,zb)
            self.assertEqual(a.fingerprint(),b.fingerprint())
            self.assertEqual(a.evaluation_memory_info(),b.evaluation_memory_info())
        self.assertEqual(source,self.original.fingerprint())

    def test_02_channel_checks_preserve_entire_event_stream(self):
        base=self.make();base.run(self.text);events=base.contribution_metrics()
        for kind,r in [('neural_output',6),('content_descriptor',6),('relay_context',2),
                       ('retained_input',10),('semantic_residual',6),('expert_transform',6)]:
            with self.subTest(kind=kind):
                m=self.make(kind,r);before=m.composition_state(fitted=True).copy()
                flat=m.flat.copy();extra=m.extra_flat.copy();m.run(self.text)
                for name,row in m.contribution_metrics().items():
                    for key in ('event_hash','publications','neural_executions','span_sum','max_span'):
                        self.assertEqual(row[key],events[name][key],(kind,name,key))
                self.assertGreater(m.contribution_metrics()[core.NAMES[r]]['intervention_applications'],0)
                np.testing.assert_array_equal(before,m.composition_state(fitted=True))
                np.testing.assert_array_equal(flat,m.flat);np.testing.assert_array_equal(extra,m.extra_flat)

    def test_03_neural_mute_preserves_inherited_relay_context(self):
        a=self.make();b=self.make('neural_output',2)
        for m in (a,b):
            self.fx.check_fixture_atom(m.handle,0,65)
            self.fx.check_fixture_relay(m.handle,0,2)
        np.testing.assert_array_equal(self.vectors(a,2),self.vectors(b,2))
        self.assertEqual(b.contribution_metrics()['L2']['intervention_applications'],0)

    def test_04_relay_mute_preserves_code_recognition_origin_and_memory(self):
        a=self.make();b=self.make('relay_context',2)
        for m in (a,b):
            self.fx.check_fixture_memory(m.handle,2,.375)
            self.fx.check_fixture_atom(m.handle,0,65)
            self.fx.check_fixture_relay(m.handle,0,2)
        va,vb=self.vectors(a,2),self.vectors(b,2)
        for i in (0,2,4):np.testing.assert_array_equal(va[i],vb[i])
        self.assertGreater(np.linalg.norm(va[3]),0);self.assertEqual(np.linalg.norm(vb[3]),0)
        np.testing.assert_array_equal(a.content_events()['L2']['code'],b.content_events()['L2']['code'])
        self.assertEqual(a.relay_event_status()['L2'],b.relay_event_status()['L2'])
        self.assertEqual(a.contribution_metrics()['L2']['event_hash'],b.contribution_metrics()['L2']['event_hash'])

    def test_05_descriptor_mute_leaves_context_and_metadata(self):
        a=self.make();b=self.make('content_descriptor',0)
        for m in (a,b):self.fx.check_fixture_atom(m.handle,0,65)
        va,vb=self.vectors(a,0),self.vectors(b,0)
        np.testing.assert_array_equal(va[3],vb[3]);self.assertGreater(np.linalg.norm(va[2]),0)
        self.assertEqual(np.linalg.norm(vb[2]),0);self.assertEqual(np.linalg.norm(b.content_events()['A1']['code']),0)
        self.assertEqual(a.contribution_metrics()['A1']['event_hash'],b.contribution_metrics()['A1']['event_hash'])

    def test_06_retained_filter_preserves_incoming_scale(self):
        a=self.make();b=self.make('retained_input',0)
        self.fx.check_fixture_memory(b.handle,0,3.)
        x=np.array([65],np.uint8)
        np.testing.assert_array_equal(a.run(x,logits=True)[1],b.run(x,logits=True)[1])
        np.testing.assert_array_equal(self.vectors(a,0),self.vectors(b,0))

    def test_07_neural_mute_preserves_private_bottom_memory(self):
        a=self.make();b=self.make('neural_output',0);x=np.array([65],np.uint8)
        a.run(x);b.run(x);va,vb=self.vectors(a,0),self.vectors(b,0)
        for i in (0,2,4):np.testing.assert_array_equal(va[i],vb[i])
        self.assertGreater(np.linalg.norm(va[1]),0);self.assertEqual(np.linalg.norm(vb[1]),0)

    def test_08_semantic_residual_mute_preserves_descriptor(self):
        a=self.make();b=self.make('semantic_residual',6)
        for m in (a,b):self.fx.check_fixture_symbol(m.handle,6);m.run(self.text)
        np.testing.assert_array_equal(self.vectors(a,6)[2],self.vectors(b,6)[2])
        np.testing.assert_array_equal(b.content_events()['L3']['code'],self.vectors(b,6)[2])
        self.assertGreater(np.linalg.norm(a.content_events()['L3']['code']-self.vectors(a,6)[2]),0)

    def test_09_all_regions_and_control_types_are_isolated_per_instance(self):
        for region in core.NAMES:
            a=self.make('neural_output',region);b=self.make()
            a.run(self.text[:24]);b.run(self.text[:24])
            self.assertTrue(all(r['intervention_applications']==0 for r in b.contribution_metrics().values()))
            a.close();b.close()

    def test_10_chunking_is_exact_with_an_intervention(self):
        a=self.make('neural_output','L3');b=self.make('neural_output','L3')
        za=a.run(self.text,logits=True)[1]
        zb=np.concatenate([b.run(x,logits=True)[1] for x in np.array_split(self.text,7)])
        np.testing.assert_array_equal(za,zb);self.assertEqual(a.fingerprint(),b.fingerprint())
        self.assertEqual(a.contribution_metrics(),b.contribution_metrics())

    def test_11_memory_policies_preserve_or_clear_store_without_stream_leakage(self):
        source=self.original.evaluation_memory_info();self.assertGreater(sum(source['occupied']),0)
        for policy in ('cold_adaptive','cold_frozen','warm_frozen','warm_adaptive'):
            m=self.make(memory=policy);before=m.evaluation_memory_info()
            self.assertTrue(all(e['seq']==0 for e in m.content_events().values()))
            self.assertTrue(all(t['pending_tick']==0 for t in m.relay_teaching_status().values()))
            if policy.startswith('warm'):
                self.assertEqual(before['store_digest'],source['store_digest']);self.assertEqual(before['clock'],source['clock'])
            else:self.assertEqual(sum(before['occupied']),0);self.assertEqual(before['clock'],0)
            m.run(self.text);after=m.evaluation_memory_info()
            if policy.endswith('frozen'):self.assertEqual(before['store_digest'],after['store_digest'])
            else:self.assertGreater(m.persistent_metrics()['write_considered'],0)

    def test_12_whole_source_removes_direct_decoder_paths(self):
        base=self.make();z=base.run(self.text[:20],logits=True)[1]
        for r in (0,17):
            m=self.make('whole_source',r);self.assertGreater(np.max(np.abs(z-m.run(self.text[:20],logits=True)[1])),1e-5)

    def test_13_reject_bad_controls_training_and_serialization(self):
        m=self.make()
        for kind,region in [('bad',2),('none',2),('neural_output',True),('neural_output',1.2),
                            ('neural_output',-1),('neural_output',18),('neural_output','bad'),('semantic_residual',0)]:
            with self.assertRaises(ValueError):m.configure_contribution(kind,region)
        with self.assertRaises(ValueError):m.run(self.text,mode='periodic')
        with self.assertRaises(ValueError):m.finish_document(mode='periodic')
        with self.assertRaises(ValueError):m.ablate(16)
        with self.assertRaises(ValueError):m.snapshot_copy()
        with self.assertRaises(ValueError):m.frozen_copy()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):m.save(Path(tmp)/'bad.npz')
        m.run(self.text[:2])
        with self.assertRaises(ValueError):m.configure_contribution('neural_output','L3')
        with self.assertRaises(ValueError):self.original.evaluation_copy(memory='unknown')

    def test_14_expert_transform_control_keeps_skip_and_recall(self):
        for r in (0,6,15,17):
            a=self.make();b=self.make('expert_transform',r)
            self.fx.check_fixture_zero_expert(a.handle,r)
            np.testing.assert_array_equal(a.run(self.text,logits=True)[1],b.run(self.text,logits=True)[1])

    def test_15_document_drain_handles_removed_sources(self):
        for kind,r in [('none',None),('whole_source',0),('whole_source',2),('neural_output',6)]:
            m=self.make(kind,r);flat=m.flat.copy();fitted=m.composition_state(fitted=True).copy()
            m.run(self.text);m.finish_document(reset=True)
            self.assertTrue(all(e['seq']==0 for e in m.content_events().values()))
            np.testing.assert_array_equal(flat,m.flat);np.testing.assert_array_equal(fitted,m.composition_state(fitted=True))


if __name__=='__main__':unittest.main()
