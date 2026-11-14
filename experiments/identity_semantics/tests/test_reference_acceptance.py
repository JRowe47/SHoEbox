"""Acceptance failures must be detected, not hidden by activity or coverage."""
import copy
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src')]
from runtime import Model,core,fresh_model
from reference_acceptance import (REGIONS,ZERO_COUNTERS,collect,check_integrity,
                                  check_hierarchy,check_contribution,check_training,require)


class ReferenceAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'acceptance_fixture_e{core.E}_{os.getpid()}.so'
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/acceptance_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        cls.fx.acceptance_fixture_publish.argtypes=[ct.c_void_p,ct.c_int]+[ct.c_int64]*4+[ct.c_int]*4
        cls.fx.acceptance_fixture_hash.argtypes=[ct.c_int64,ct.c_int64]
        cls.fx.acceptance_fixture_hash.restype=ct.c_uint64
        # All fixture calls use diagnostic instances: their appended hash fields exist.
        cls.data=np.frombuffer(b'The river runs through the valley. The king crossed the bridge. '*10,np.uint8)

    def make(self,diagnostic=False):
        m=fresh_model(seed=7)
        if diagnostic:
            other=m.evaluation_copy(diagnostic=True);m.close();m=other
        self.addCleanup(m.close);return m

    def publish(self,m,**kwargs):
        fields=dict(r=6,begin=0,end=100,primary_begin=10,primary_end=14,kind=4,count=3,recognized=0,relay=0)
        fields.update(kwargs);self.fx.acceptance_fixture_publish(m.handle,*fields.values())

    def test_primary_distribution_ignores_held_coverage_and_binding_arity(self):
        m=self.make(True);self.publish(m)
        self.publish(m,end=200,primary_begin=14,primary_end=18,recognized=1)
        self.publish(m,end=300,primary_begin=18,primary_end=26,kind=1,count=2,recognized=1)
        self.publish(m,end=400,primary_begin=18,primary_end=26,kind=1,count=2,recognized=1,relay=1)
        v=m.acceptance_metrics()['L3']
        self.assertEqual((v['publications'],v['primary_span_sum'],v['coverage_span_sum']),(4,24,1000))
        self.assertEqual(v['primary_span_histogram'][3:5],[2,2])
        self.assertEqual(v['sequence_length_histogram'],[0,2,1,0,0,0,0,0,0])
        self.assertEqual(v['recognized_sequence_length_histogram'],[0,1,1,0,0,0,0,0,0])
        self.assertEqual((v['formed'],v['recognized'],v['provisional'],v['relays']),(3,2,1,1))
        check_integrity(collect(m))

    def test_primary_observations_do_not_depend_on_class_learning(self):
        m=self.make(True);m.configure_pattern_classes(False);self.publish(m)
        self.assertEqual(m.acceptance_metrics()['L3']['primary_span_sum'],4)
        self.assertEqual(m.pattern_metrics()['L3']['primary_span_sum'],0)
        check_integrity(collect(m))

    def test_clock_hash_distinguishes_same_coverage_with_different_primary_clock(self):
        self.assertNotEqual(self.fx.acceptance_fixture_hash(10,14),self.fx.acceptance_fixture_hash(10,18))
        self.assertEqual(self.fx.acceptance_fixture_hash(10,14),self.fx.acceptance_fixture_hash(10,14))

    def test_partial_ec_fusion_without_primary_is_explicit_not_a_temporal_span(self):
        m=self.make(True);self.publish(m,r=15,primary_begin=0,primary_end=0)
        v=m.acceptance_metrics()['EC-C']
        self.assertEqual((v['unclocked_fusions'],v['invalid_primary_spans'],v['primary_span_histogram'][0]),(1,0,1))
        check_integrity(collect(m))
        m=self.make(True);self.publish(m,r=15,primary_begin=2,primary_end=2)
        with self.assertRaisesRegex(AssertionError,'invalid_primary'):check_integrity(collect(m))

    def test_invalid_spans_and_arity_fail_closed(self):
        for kwargs,reason in ((dict(primary_begin=20,primary_end=19),'invalid_primary'),
                              (dict(end=0),'invalid_coverage'),
                              (dict(primary_end=101),'primary_outside'),
                              (dict(kind=1,count=0),'invalid sequence')):
            m=self.make(True);self.publish(m,**kwargs)
            with self.assertRaisesRegex(AssertionError,reason):check_integrity(collect(m))

    def test_every_lost_credit_counter_and_missing_region_is_rejected(self):
        m=self.make();row=collect(m);check_integrity(row)
        for group,fields in ZERO_COUNTERS.items():
            for field in fields:
                broken=copy.deepcopy(row);broken[group]['G5'][field]=1
                with self.subTest(group=group,field=field),self.assertRaisesRegex(AssertionError,field):
                    check_integrity(broken)
        row['local_credit'].pop('G5')
        with self.assertRaisesRegex(AssertionError,'missing regional'):check_integrity(row)

    def test_old_coverage_success_cannot_pass_primary_hierarchy(self):
        rows={r:dict(publications=100//(i+1),primary_span_sum=100//(i+1),coverage_span_sum=1000*(i+1))
              for i,r in enumerate(('L2','L3','F4','G5'))}
        with self.assertRaisesRegex(AssertionError,'primary span'):check_hierarchy(rows)

    def test_hierarchy_uses_declared_primary_driver_not_a_larger_side_source(self):
        m=self.make();edges=collect(m)['primary_edges']
        self.assertIn(('E4','G5'),edges);self.assertNotIn(('F4','G5'),edges)
        rows={'E4':dict(publications=10,primary_span_sum=100),
              'G5':dict(publications=5,primary_span_sum=100),
              'F4':dict(publications=2,primary_span_sum=200)}
        check_hierarchy(rows,[('E4','G5')])

    def test_disabled_reference_mechanism_cannot_pass_training_gate(self):
        m=self.make();m.configure_local_credit(False)
        with self.assertRaisesRegex(AssertionError,'Full reference'):check_training(collect(m))

    def test_zero_errors_do_not_pass_an_unexercised_reference(self):
        m=self.make()
        with self.assertRaisesRegex(AssertionError,'unexercised'):check_training(collect(m))

    def test_counter_reset_and_checkpoint_leave_computation_bitwise_exact(self):
        m=self.make();m.run(self.data[:257],mode='periodic')
        self.assertGreater(m.acceptance_metrics()['L3']['publications'],0)
        before=m.fingerprint();m.reset_acceptance_metrics();self.assertEqual(before,m.fingerprint())
        self.assertTrue(all(v['publications']==0 for v in m.acceptance_metrics().values()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reference.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
            self.assertEqual(m.fingerprint(),n.fingerprint())
            a=m.run(self.data[257:449],mode='periodic',logits=True)[1]
            b=np.concatenate([n.run(chunk,mode='periodic',logits=True)[1] for chunk in np.array_split(self.data[257:449],3)])
            np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
            self.assertEqual(m.acceptance_metrics(),n.acceptance_metrics())
        m.reset_context();self.assertTrue(all(v['publications']==0 for v in m.acceptance_metrics().values()))

    def test_objective_calls_respect_zero_rate_and_freeze(self):
        for rates in ((0,.02),(.02,0),(0,0)):
            m=self.make();m.run(self.data[:321],mode='periodic',lr=rates[0],lp=rates[1]);m.finish_document(mode='periodic',lr=rates[0],lp=rates[1])
            v=m.acceptance_metrics()
            for rate,field in zip(rates,('language_dfa_update_calls','local_update_calls')):
                self.assertEqual(sum(x[field] for x in v.values())>0,rate>0)
        m=self.make();frozen=np.zeros(18,np.int32);frozen[:6]=1
        m.configure_training_policy(frozen);m.run(self.data[:161],mode='periodic')
        self.assertTrue(all(m.acceptance_metrics()[r]['language_dfa_update_calls']==m.acceptance_metrics()[r]['local_update_calls']==0 for r in REGIONS[:6]))

    def test_contribution_activity_and_effect_are_not_quality_claims(self):
        row=dict(baseline_nll=2.,intervention_nll=1.9,delta_nll=-.1,logit_rms=.01,applications=1,
                 fitted_unchanged=True,source_unchanged=True,events_identical=True)
        self.assertEqual(check_contribution(row.copy(),require_effect=True,preserve_events=True)['effect'],'harmful')
        for key,value in (('applications',0),('logit_rms',0),('events_identical',False),('fitted_unchanged',False),('delta_nll',float('nan'))):
            broken={**row,key:value}
            with self.subTest(key=key),self.assertRaises(AssertionError):
                check_contribution(broken,require_effect=True,preserve_events=True)

    def test_acceptance_checks_survive_optimized_python(self):
        code="from reference_acceptance import require; require(False, 'sentinel')"
        result=subprocess.run([sys.executable,'-O','-c',code],env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('sentinel',result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
