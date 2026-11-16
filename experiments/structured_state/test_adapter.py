import random
import unittest
import numpy as np
from experiments.structured_state.adapter import Adapter
from experiments.structured_state.tasks import episode,suite,verify

class Tests(unittest.TestCase):
    def test_interpreter_and_splits(self):
        train={};rng=random.Random(19)
        for i in range(1000):
            ep=episode(rng,('tracking','composition')[i%2],1+i%4,'train');self.assertTrue(verify(ep));train[ep.template]=ep
        for fold in ('dev','audit'):
            for ep in suite(23,500,fold,lengths=(1,2,4,8,12)):
                self.assertTrue(verify(ep));self.assertNotIn(ep.template,train)
        for ep in suite(30,100,'audit',renamed=True):self.assertTrue(verify(ep))
    def test_causality_clocks_and_continuation(self):
        ep=suite(6,1)[0]
        with Adapter(arm='state_aux') as a:
            text=ep.text
            for j in range(2):
                a.run(text[j*64:(j+1)*64],text[j*64+1:(j+1)*64+1]);before=a.info();x,_=a.read()
                a.supervise(ep.states[j],True);after=a.info();np.testing.assert_array_equal(x,a.read()[0]);self.assertEqual(before[1:3],after[1:3]);self.assertEqual(before[6:14],after[6:14])
            with Adapter.loads(a.dumps()) as b:
                for m in (a,b):
                    for j in range(2,len(ep.records)):
                        m.run(text[j*64:(j+1)*64],text[j*64+1:(j+1)*64+1]);m.supervise(ep.states[j],True)
                    labels=[-1]*4;labels[ep.query]=ep.answer;m.supervise(labels);m.finish()
                self.assertEqual(a.dumps(),b.dumps())
            with a.clone(frozen=True) as b:
                saved=a.dumps();b.run(text[:-1],text[1:]);self.assertEqual(a.dumps(),saved)
                with self.assertRaises(RuntimeError):b.supervise([0,0,0,0])
    def test_head_only_is_detached_and_state_credit_is_effective(self):
        ep=suite(11,1)[0];a=Adapter(arm='final_only');b=Adapter(arm='head_only');c=Adapter(arm='state_aux')
        try:
            for m in (a,b,c):m.run(ep.text[:64],ep.text[1:65])
            b.supervise(ep.states[0],True);c.supervise(ep.states[0],True)
            np.testing.assert_array_equal(b.read()[1],c.read()[1])
            for m in (a,b,c):m.finish();m.reset();m.run(ep.text[:64],ep.text[1:65])
            np.testing.assert_array_equal(a.read()[0],b.read()[0])
            self.assertGreater(np.max(np.abs(a.read()[0]-c.read()[0])),0)
        finally:
            for m in (a,b,c):m.close()
    def test_native_baseline_equivalence(self):
        from shoe_multiscale import Model
        with Adapter() as a,Model(seed=7) as b:
            b.configure_route_lr(lower=.1,upper=1,anchor=1,amplitude=0)
            b.set_learning_anchor(.1,over_targets=3000000)
            data=bytes((i*17)%256 for i in range(1025))
            self.assertEqual(a.run(data[:-1],data[1:]),b.run(data[:-1],data[1:]))
    def test_epoch_retirement_and_chunking(self):
        ep=suite(71,1,'audit',lengths=(12,))[0]
        with Adapter(arm='state_aux') as a,Adapter(arm='state_aux') as b:
            text=ep.text
            for j in range(len(ep.records)):
                start=j*64
                a.run(text[start:start+64],text[start+1:start+65])
                for k in range(0,64,7):
                    end=min(64,k+7);b.run(text[start+k:start+end],text[start+k+1:start+end+1])
                a.supervise(ep.states[j],True);b.supervise(ep.states[j],True)
                self.assertEqual(a.dumps(),b.dumps())
                if j in (7,11):
                    with Adapter.loads(a.dumps()) as resumed:self.assertEqual(a.dumps(),resumed.dumps())
            a.finish();b.finish();self.assertEqual(a.dumps(),b.dumps())

    def test_validation(self):
        with Adapter() as a:
            with self.assertRaises(RuntimeError):a.supervise([0,1,2,3])
            a.run(b' '*64,b' '*64)
            for labels in ([8,0,0,0],[-2,0,0,0],[-1]*4):
                with self.assertRaises(RuntimeError):a.supervise(labels)
            with self.assertRaises(RuntimeError):a.permute(2)
            raw=a.dumps()
            with self.assertRaises(ValueError):Adapter.loads(raw[:-1])
            with self.assertRaises(ValueError):Adapter.loads(raw[:32]+b'x'+raw[33:])

if __name__=='__main__':unittest.main()
