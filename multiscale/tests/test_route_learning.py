"""Optimizer lifecycle, frozen evaluation, resumption and public controls."""
import json
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
from shoe_multiscale import Model


class RouteLearningTests(unittest.TestCase):
    def test_partial_checkpoint_chunking_and_credit_extensions(self):
        data=np.random.default_rng(419).integers(0,256,3001,dtype=np.uint8)
        for levels in (1,2,3,4):
            with Model(levels=levels,route_lr={'period':17},
                       language_credit=[2]*levels,recall_credit=[1]*levels) as m:
                m.set_learning_anchor(.75,over_targets=1501)
                m.run(data[:513],data[1:514])
                self.assertEqual(m.stats()['checkpoint_version'],17)
                with Model.loads(m.dumps()) as q:
                    self.assertEqual(m.dumps(),q.dumps())
                    m.run(data[513:2000],data[514:2001])
                    for a,b in ((513,514),(514,800),(800,1025),(1025,2000)):
                        q.run(data[a:b],data[a+1:b+1])
                    self.assertEqual(m.dumps(),q.dumps())
                    self.assertEqual(m.route_lr_stats()['anchor'],.75)
                state=m.dumps();m.evaluate(data)
                self.assertEqual(m.dumps(),state)
                with m.inference_copy() as frozen:
                    stats=frozen.route_lr_stats(include_routes=True)
                    frozen.run(data[:-1],data[1:]);frozen.reset_context()
                    self.assertEqual(frozen.route_lr_stats(include_routes=True),stats)
                    with self.assertRaises(RuntimeError):frozen.set_learning_anchor(1.)
                    with self.assertRaises(RuntimeError):frozen.configure_route_lr()

    def test_clocks_reset_pause_and_anchor_retarget(self):
        data=b'route-local learning clocks retain their history. '*29
        with Model(route_lr={'period':19}) as m:
            m.train(data);m.finish()
            before=m.route_lr_stats(include_routes=True)
            for i,l in enumerate(before['levels']):
                self.assertEqual(sum(l['visits']),(len(data)-1)//(4**i))
                self.assertTrue(all(.5<=x<=1.5 for x in l['multipliers']))
            m.reset_context();self.assertEqual(before,m.route_lr_stats(include_routes=True))
            m.set_learning_anchor(.6,over_targets=100)
            m.run(data[:50],data[1:51]);self.assertAlmostEqual(m.route_lr_stats()['anchor'],.8,places=6)
            m.set_learning_anchor(1.2,over_targets=100)
            m.run(data[:50],data[1:51]);self.assertAlmostEqual(m.route_lr_stats()['anchor'],1.,places=6)
            m.finish();m.configure_route_lr(enabled=False)
            paused=m.route_lr_stats(include_routes=True)
            m.train(data);m.finish();self.assertEqual(paused,m.route_lr_stats(include_routes=True))
            m.configure_route_lr(period=19)
            self.assertEqual([l['visits'] for l in m.route_lr_stats(include_routes=True)['levels']],
                             [l['visits'] for l in paused['levels']])

    def test_validation_and_failure_atomicity(self):
        with Model() as m:
            old=m.dumps()
            for kwargs in ({'lower':-1},{'upper':.2},{'anchor':2},{'amplitude':1.01},
                           {'period':0},{'period':1.2},{'period':True},{'period':2**64},
                           {'anchor':float('nan')},{'upper':float('inf')},{'enabled':2}):
                with self.assertRaises((ValueError,RuntimeError)):m.configure_route_lr(**kwargs)
                self.assertEqual(m.dumps(),old)
            m.configure_route_lr();m.train(b'pending credit')
            old=m.dumps()
            with self.assertRaises(RuntimeError):m.configure_route_lr(period=9)
            self.assertEqual(m.dumps(),old)
            for value in (-1,2,float('nan')):
                with self.assertRaises((ValueError,RuntimeError)):m.set_learning_anchor(value)
                self.assertEqual(m.dumps(),old)
            with self.assertRaises(ValueError):m.configure(slow_rate=float(np.finfo(np.float32).max))
            self.assertEqual(m.dumps(),old)
        with self.assertRaises(ValueError):Model(memory_profile='legacy',route_lr={})
        with self.assertRaises(ValueError):Model(semantic_laces=True,route_lr={})

    def test_full_width_clock_inspection(self):
        with Model(route_lr={'period':2**64-1}) as m:
            m.set_learning_anchor(.7,over_targets=2**64-1)
            self.assertEqual(m.route_lr_stats()['period'],2**64-1)
            with Model.loads(m.dumps()) as q:
                self.assertEqual(q.route_lr_stats()['ramp_targets'],2**64-1)

    def test_semantically_invalid_checkpoint_controller(self):
        with Model(route_lr={}) as m:
            original=m.dumps();counts=m.stats()['experts']
            extension=len(original)-56-sum(8+8*n for n in counts)
            for offset,fmt,value in ((0,'I',2),(4,'f',float('nan')),(12,'f',9.),
                                     (24,'Q',0),(40,'Q',1),(56,'Q',counts[0]+1)):
                data=bytearray(original);struct.pack_into('='+fmt,data,extension+offset,value)
                data[8:40]=hashlib.sha256(data[40:]).digest()
                with self.assertRaises(RuntimeError):Model.loads(bytes(data))

    def test_cli_anchor_resume(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);data=td/'data';a=td/'a.shoe';b=td/'b.shoe'
            data.write_bytes(b'bounded clocks '*80)
            cmd=[sys.executable,'-m','multiscale.cli','train','--data',str(data)]
            subprocess.run(cmd+['--out',str(a),'--targets','513','--route-lr','--lr-period','17',
                '--lr-anchor-target','.75','--lr-anchor-targets','1000'],check=True,capture_output=True)
            result=subprocess.run(cmd+['--resume',str(a),'--out',str(b),'--targets','513'],check=True,capture_output=True)
            stats=json.loads(result.stdout)['route_learning']
            self.assertEqual(stats['training_targets'],1026)
            self.assertEqual(stats['anchor'],.75)
            self.assertEqual(stats['period'],17)
            with Model.load(a) as m,Model.load(b) as q:
                raw=data.read_bytes();m.run(raw[:513],raw[1:514]);self.assertEqual(m.dumps(),q.dumps())
            subprocess.run(cmd+['--resume',str(a),'--out',str(b),'--targets','513','--route-lr',
                '--lr-period','23','--learning-rate','.001'],check=True,capture_output=True)
            with Model.load(a) as m,Model.load(b) as q:
                m.finish();m.configure(fast_rate=.001)
                m.configure_route_lr(anchor=m.route_lr_stats()['anchor'],period=23)
                m.run(raw[:513],raw[1:514]);self.assertEqual(m.dumps(),q.dumps())

if __name__=='__main__':unittest.main()
