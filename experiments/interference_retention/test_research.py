import unittest
import numpy as np
from .backend import Model
from .tasks import worlds,cases,make,manifest,excluded,interpret
from .diagnose import trace
from experiments.joint_cold_gate.adapter import Adapter,prefix_labels

class HarnessTests(unittest.TestCase):
    def test_disjoint_fresh_families(self):
        sets=[{w.group_key for w in worlds(p,n)} for p,n in [('train',512),('dev',128),('audit',256)]]
        for i,a in enumerate(sets):
            self.assertFalse(any(excluded(k) for k in a))
            for b in sets[i+1:]:self.assertFalse(a&b)
    def test_interpreter_ignores_answers(self):
        for cond in ['pad0','diverse64','compete16','overwrite16']:
            for c in cases('train',8,condition=cond):
                from dataclasses import replace
                x=bytes(255 if c.ep.mask[i] else b for i,b in enumerate(c.ep.data))
                self.assertEqual(interpret(replace(c.ep,data=x)),c.expected)
    def test_labels_causal_and_no_future_answer(self):
        from dataclasses import replace
        ep=cases('train',1,condition='overwrite16')[0].ep
        full=prefix_labels(ep)
        for t in range(1,len(ep.data)-1):
            altered=replace(ep,data=ep.data[:t]+bytes([238])*(len(ep.data)-t))
            np.testing.assert_array_equal(prefix_labels(altered)[:t],full[:t])
    def test_instrumentation_numerically_exact_e8(self):
        ep=cases('train',1)[0].ep;n=len(ep.data)-1
        with Model(seed=7) as a,Adapter(seed=7) as b:
            self.assertEqual(a.dumps(),b.dumps())
            self.assertEqual(a.run(ep.data[:-1],ep.data[1:],ep.mask[1:],prefix_labels(ep)),b.run(ep.data[:-1],ep.data[1:],ep.mask[1:],prefix_labels(ep)))
            self.assertEqual(a.dumps(),b.dumps());before=a.dumps()
            a.state();a.decoded();a.diagnostic();self.assertEqual(before,a.dumps())
            with a.clone() as x,b.clone() as y:
                for byte in bytes(range(256)):
                    x.observe(byte);y.observe(byte);np.testing.assert_array_equal(x.log_probs(),y.log_probs())
            self.assertEqual(before,a.dumps())
    def test_freeze_frontier_completion_all_phases(self):
        w=worlds('train',1)[0]
        with Model(seed=7) as a,a.clone() as f:
            for offset in range(64):
                c=make(w,'diverse16',offset=offset)
                rows,n=trace(f,c,True)
                self.assertTrue(all(r['present'] for r in rows))
    def test_a1_cold_equivalence_and_policy_identity(self):
        with Model('e8',seed=7) as a,Model('a1',seed=7) as b:
            data=bytes(range(256))*2
            self.assertEqual(a.run(data,data),b.run(data,data))
            self.assertEqual(a.core_dumps(),b.core_dumps())
            with self.assertRaises(RuntimeError):Model.loads(a.dumps(),'a1')
            with self.assertRaises(RuntimeError):Model.loads(b.dumps(),'e8')
            with Model.loads(b.dumps(),'a1') as c:self.assertEqual(c.dumps(),b.dumps())
    def test_interrupted_cursor_and_finish_boundary(self):
        from .stream import advance,new_cursor
        for policy in ('e8','a1'):
            with Model(policy,seed=7) as a:
                s,_=advance(a,new_cursor(),1001);raw=a.dumps()
                with Model.loads(raw,policy) as b:
                    t,_=advance(a,s,1000)
                    u,_=advance(b,s,371)
                    with Model.loads(b.dumps(),policy) as c:
                        v,_=advance(c,u,629)
                        self.assertEqual(t,v);self.assertEqual(a.dumps(),c.dumps())
    def test_readonly_lifecycle(self):
        with Model() as a:
            with self.assertRaises(RuntimeError):a.freeze_span(b'abc')
            raw=a.dumps()
            with Model.loads(raw) as b:self.assertEqual(raw,b.dumps())
            with a.clone() as b:
                with self.assertRaises(RuntimeError):b.run(b'a',b'b')
                b.freeze_span(b'abc');self.assertFalse(b.inventory()['training'])
                np.testing.assert_allclose(np.exp(b.log_probs()).sum(),1,rtol=1e-5)
            self.assertEqual(raw,a.dumps())
        with self.assertRaises(RuntimeError):a.state()

if __name__=='__main__':unittest.main()
