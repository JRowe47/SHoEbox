"""Minimal baseline contracts, including real partial-state continuation."""
import unittest
import numpy as np
from shoe_multiscale import Model, fresh_model


class MinimalCoreTests(unittest.TestCase):
    def test_default_structure_and_all_four_clocks(self):
        with fresh_model(parameters=500000) as m:
            m.train(bytes(range(256))*3)
            s=m.stats()
            self.assertEqual(s['core'],'minimal')
            self.assertEqual(s['checkpoint_version'],19)
            self.assertEqual(s['level_kinds'],['expert','expert','summary','summary'])
            self.assertEqual(s['experts'][2:],[0,0])
            self.assertEqual(s['events'],[767,191,47,11])
            self.assertEqual(s['native_targets'][2:],[0,0])
            self.assertEqual(s['balance_policy'],'warmup')
            self.assertGreater(s['address_energy'][3],0)
            self.assertGreater(s['retained_units'][3],0)
            self.assertGreater(s['reconstruction_targets'][3],0)
            self.assertEqual(m.routing_stats()['levels'][2]['events'],0)

    def test_streaming_every_boundary_and_frozen_state(self):
        data=np.random.default_rng(19).integers(0,256,1110,dtype=np.uint8)
        for levels in (1,2,3,4):
            with Model(levels=levels,parameters=250000) as m, m.clone() as whole:
                expected=whole.run(data[:-1],data[1:])
                total,start=0.,0
                for end in (1,4,8,15,16,63,64,65,511,512,513,1109):
                    total+=m.run(data[start:end],data[start+1:end+1])*(end-start)
                    raw=m.dumps()
                    q=Model.loads(raw)
                    self.assertEqual(q.dumps(),raw)
                    m.close();m=q;start=end
                self.assertAlmostEqual(total/1109,expected,places=12)
                self.assertEqual(m.dumps(),whole.dumps())
                before=m.dumps()
                self.assertTrue(np.isfinite(m.evaluate(data)))
                self.assertEqual(m.dumps(),before)
                with m.inference_copy() as f:
                    for b in data[:71]:f.observe(int(b))
                    with Model.loads(f.dumps()) as g:
                        np.testing.assert_array_equal(f.observe(17),g.observe(17))
                m.close()

    def test_extensions_require_full_and_failed_changes_are_atomic(self):
        with Model() as m:
            before=m.dumps()
            for change in (lambda:m.configure_loops(),lambda:m.configure_route_lr(),
                           lambda:m.set_balance_policy('legacy'),
                           lambda:m.set_language_credit([1,1,2,2]),
                           lambda:m.set_recall_credit([0,0,1,1])):
                with self.assertRaises(RuntimeError):change()
                self.assertEqual(before,m.dumps())
        for options in ({'loops':{}},{'route_lr':{}},{'routing_arm':'D'},
                        {'associations':True},{'balance_policy':'legacy'}):
            with self.assertRaises(ValueError):Model(core='minimal',**options)
            with Model(**options) as m:self.assertEqual(m.stats()['core'],'full')

    def test_budget_and_exact_external_store(self):
        for budget in (250000,500000,1000000,5000000):
            with Model(parameters=budget) as m:
                self.assertLessEqual(m.parameter_count,budget)
                self.assertLess(budget-m.parameter_count,4224)
                h=m.bind(b'key',b'value')
                with Model.loads(m.dumps()) as q:
                    self.assertEqual(q.lookup(b'key'),h)
                    self.assertEqual(q.payload(h),b'value')

if __name__=='__main__':unittest.main()
