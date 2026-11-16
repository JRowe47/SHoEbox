"""E8 public lifecycle, causality, parity and explicit format separation."""
import hashlib
import struct
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shoe_multiscale import Model
from multiscale.e8.model import E8Model
from multiscale.e8.backend import source
from experiments.joint_cold_gate.adapter import Adapter, prefix_labels
from experiments.joint_cold_gate.tasks import dataset
from experiments.joint_request_gate.adapter import Adapter as E7


class PublicE8Tests(unittest.TestCase):
    def assert_payload(self, public, research):
        self.assertEqual(public.dumps()[48:], research.dumps()[40:])

    def test_frozen_qualified_source(self):
        root=Path(__file__).resolve().parents[2]
        qualified=(root/'experiments/joint_cold_gate/readout.hpp').read_text()
        public=(root/'multiscale/e8/readout.hpp').read_text()
        self.assertEqual(public,qualified.replace('0x38455041514853ULL','0x3850454f4853ULL'))
        self.assertEqual((root/'multiscale/e8/api.cpp').read_bytes(),
                         (root/'experiments/joint_cold_gate/api.cpp').read_bytes())

    def test_promoted_default_matches_explicit_e8(self):
        from shoe_multiscale import fresh_model
        for seed in (1,7,19):
            with fresh_model(parameters=500000,seed=seed) as m, Model(policy='e8',parameters=500000,seed=seed) as explicit:
                self.assertEqual(m.dumps(),explicit.dumps())
                data=bytes(range(256))*3
                for start,stop in ((0,1),(1,63),(63,64),(64,517),(517,767)):
                    self.assertEqual(m.run(data[start:stop],data[start+1:stop+1]),
                                     explicit.run(data[start:stop],data[start+1:stop+1]))
                    self.assertEqual(m.dumps(),explicit.dumps())

    def test_initialization_and_budget(self):
        for budget in (250000,500000,1000000,5000000):
            with Model(policy='e8',parameters=budget,seed=7) as m, Adapter(parameters=budget,seed=7) as a:
                self.assertIsInstance(m, Model)
                self.assert_payload(m,a)
                s=m.stats()
                self.assertEqual(s['readout_parameters'],7670)
                self.assertEqual(s['added_persistent_sequence_state_bytes'],0)
                self.assertEqual(s['optimizer_state_bytes'],0)
                self.assertLessEqual(m.parameter_count,budget)
                self.assertEqual(s['level_kinds'],['expert','expert','summary','summary'])
                self.assertEqual(s['checkpoint_version'],'e8-public-1')

    def test_public_training_exact_and_partial_boundaries(self):
        ep=dataset(8,'train')[3];x,y=ep.data[:-1],ep.data[1:];labels=prefix_labels(ep)
        for local in (False,True):
            with Model(policy='e8',parameters=500000,seed=7) as m, Adapter(seed=7) as a:
                start=0
                for stop in sorted(set([1,4,8,15,16,63,64,65,len(x)])):
                    stop=min(stop,len(x))
                    if stop<=start:continue
                    lab=labels[start:stop] if local else None
                    mask=ep.mask[start+1:stop+1]
                    r=m.run_objectives(x[start:stop],y[start:stop],mask=mask,local_labels=lab)
                    loss,targets=a.run(x[start:stop],y[start:stop],mask,lab)
                    self.assertEqual((r['loss_sum'],r['targets']),(loss,targets))
                    self.assert_payload(m,a)
                    np.testing.assert_array_equal(m.logits(),a.log_probs())
                    with Model.loads(m.dumps()) as q:
                        self.assertEqual(q.dumps(),m.dumps())
                    start=stop
                m.finish();a.finish();self.assert_payload(m,a)
                m.reset_context(clear_memory=True);a.reset();self.assert_payload(m,a)

    def test_generic_heads_frozen(self):
        with Model(policy='e8') as m:
            before=m.dumps();s=m.readout_stats()
            m.train(bytes(range(256))*2)
            after=m.dumps();t=m.readout_stats()
            def heads(raw):
                p=raw[40:];n=struct.unpack_from('=Q',p,12)[0];at=20+n
                return p[at:at+7605*4],p[at+7605*4+34:at+7605*4+34+260]
            self.assertEqual(heads(before),heads(after))
            for k in ('gate_targets','local_projection_steps','local_presence_steps'):
                self.assertEqual(s[k],t[k])
            self.assertNotEqual(before,after)

    def test_frozen_lifecycle_generation_and_memory(self):
        with Model(policy='e8',parameters=250000) as m:
            m.train(b'alpha beta gamma'*5);h=m.bind(b'key',b'value');before=m.dumps()
            with self.assertRaises(RuntimeError):m.observe(1)
            with m.inference_copy(cold=False) as f:
                self.assertFalse(f.stats()['training'])
                with f.clone() as c:
                    self.assertFalse(c.stats()['training'])
                    self.assertEqual(c.dumps(),f.dumps())
                f.observe(8)
                with Model.loads(f.dumps()) as g:
                    np.testing.assert_array_equal(f.observe(19),g.observe(19))
                with self.assertRaises(RuntimeError):f.train(b'ab')
                with self.assertRaises(RuntimeError):f.run(b'a',b'b',local_labels=[[-1]*6])
                self.assertTrue(np.isfinite(f.run(b'abc',b'bcd')))
            with m.inference_copy() as c:
                self.assertEqual(c.stats()['bytes'],0);self.assertEqual(c.lookup(b'key'),h)
            with m.inference_copy(clear_memory=True) as c:self.assertIsNone(c.lookup(b'key'))
            self.assertEqual(m.dumps(),before)
            self.assertTrue(np.isfinite(m.evaluate(b'abcd')))
            self.assertEqual(m.generate(b'x',length=7,seed=1),m.generate(b'x',length=7,seed=1))
            self.assertEqual(m.dumps(),before)
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'model.shoe';m.save(p)
                with Model.load(p) as q:self.assertEqual(q.dumps(),before)
            m.close();m.close()
            for call in (m.stats,m.logits,m.dumps,m.clone,lambda:m.observe(0)):
                with self.assertRaises(RuntimeError):call()

    def test_import_rejects_other_formats_and_controls(self):
        with Adapter() as a, E7() as old:
            with Model.import_e8_research(a.dumps())[0] as m:self.assert_payload(m,a)
            with self.assertRaises(ValueError):Model.loads(a.dumps())
            with self.assertRaises(ValueError):Model.import_e8_research(old.dumps())
            with Model(policy='e8') as m:
                with self.assertRaises(ValueError):Adapter.loads(m.dumps())
                raw=bytearray(m.dumps());raw[-1]^=1
                with self.assertRaises(ValueError):Model.loads(raw)
        for options in ({'kind':'native'},{'kind':'paired_gated'},{'core':'full'},{'joint':False}):
            with Adapter(**options) as a:
                with self.assertRaises(RuntimeError):Model.import_e8_research(a.dumps())

    def test_cli_explicit_e8_and_exact_resume(self):
        data=b'Exact public CLI continuation. '*30
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'first').write_bytes(data[:518]);(d/'next').write_bytes(data[517:])
            cmd=[sys.executable,'-m','multiscale.cli']
            subprocess.run(cmd+['train','--policy','e8','--data',str(d/'first'),'--out',str(d/'m.shoe')],check=True,capture_output=True)
            with Model(policy='e8') as m:
                m.train(data[:518]);self.assertEqual(m.dumps(),(d/'m.shoe').read_bytes())
                subprocess.run(cmd+['train','--resume',str(d/'m.shoe'),'--data',str(d/'next'),'--out',str(d/'m.shoe')],check=True,capture_output=True)
                m.train(data[517:]);self.assertEqual(m.dumps(),(d/'m.shoe').read_bytes())
            r=subprocess.run(cmd+['inspect','--checkpoint',str(d/'m.shoe')],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(r.stdout)['checkpoint_version'],'e8-public-1')
            r=subprocess.run(cmd+['train','--policy','v19','--resume',str(d/'m.shoe'),'--data',str(d/'next'),'--out',str(d/'m.shoe')],capture_output=True)
            self.assertNotEqual(r.returncode,0)

    def test_nominal_budget_cap_precedes_readout_deduction(self):
        with self.assertRaises(RuntimeError):
            Model(policy='e8',parameters=100_000_001)
        with self.assertRaises(RuntimeError):
            Model(policy='v19',parameters=100_000_001)

    def test_invalid_inputs_atomic(self):
        with Model(policy='e8') as m:
            before=m.dumps()
            for labels in ([[1.2]*6],[[2**40]*6],[[0]*5],[[0,0,0,0,0,2]]):
                with self.assertRaises(ValueError):m.run(b'a',b'b',local_labels=labels)
                self.assertEqual(m.dumps(),before)
            for args in ((b'',b''),(b'a',b'bc')):
                with self.assertRaises(ValueError):m.run(*args)
            with self.assertRaises(ValueError):m.run(b'a',b'b',mask=b'\x02')
            for change in (lambda:m.configure_loops(),lambda:m.configure_route_lr(),
                           lambda:m.set_balance_policy('legacy')):
                with self.assertRaises(RuntimeError):change()
                self.assertEqual(m.dumps(),before)
        for options in ({'levels':3},{'core':'full'},{'routing_arm':'D'}):
            with self.assertRaises(ValueError):Model(policy='e8',**options)

    def test_causal_full_distribution(self):
        with Model(policy='e8',parameters=250000) as m, m.inference_copy() as a, m.inference_copy() as b:
            for byte in b'causal prefix':
                np.testing.assert_array_equal(a.observe(byte),b.observe(byte))
                self.assertAlmostEqual(float(np.exp(a.logits()).sum()),1.,places=6)
            snapshot=a.dumps();b.observe(255)
            self.assertEqual(a.dumps(),snapshot)
            for target in range(256):
                with a.clone() as c:
                    expected=-float(c.observe(42)[target])
                with a.clone() as c:
                    self.assertAlmostEqual(c.run(b'*',bytes([target])),expected,places=5)

    def test_explicit_historical_meanings(self):
        for kwargs,version in (({'core':'minimal'},19),({'policy':'v19'},19),({'core':'full'},10),
                               ({'routing_arm':'C'},9),({'routing_arm':'D'},9)):
            with Model(parameters=250000,**kwargs) as m:
                self.assertEqual(m.stats()['checkpoint_version'],version)
                m.train(b'abcde'*7)
                with Model.loads(m.dumps()) as q:self.assertEqual(q.dumps(),m.dumps())

if __name__=='__main__':unittest.main()
