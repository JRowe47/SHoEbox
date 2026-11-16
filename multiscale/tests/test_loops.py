"""Selective depth, causal completion, serialization and public controls."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from shoe_multiscale import Model


def base_payload(data):
    raw=data[40:]
    if struct.unpack_from('=I',raw,8)[0]==18:
        size=struct.unpack_from('=Q',raw,16)[0]
        return raw[24:24+size]
    return raw


class LoopTests(unittest.TestCase):
    def test_zero_depth_exact_control(self):
        data=np.random.default_rng(7).integers(0,256,1201,dtype=np.uint8)
        with Model(core="full") as base,Model(loops={'steps':[0,0,0,0]}) as loops:
            self.assertEqual(base.parameter_count,loops.parameter_count)
            self.assertEqual(base.run(data[:-1],data[1:]),loops.run(data[:-1],data[1:]))
            self.assertEqual(base_payload(base.dumps()),base_payload(loops.dumps()))
            base.finish();loops.finish()
            self.assertEqual(base_payload(base.dumps()),base_payload(loops.dumps()))

    def test_continuation_all_phases_and_credit_modes(self):
        data=np.random.default_rng(17).integers(0,256,1300,dtype=np.uint8)
        for levels in (1,2,3,4):
            with Model(levels=levels,loops={},route_lr={'period':19},
                       language_credit=[2]*levels,recall_credit=[1]*levels) as whole:
                whole.set_learning_anchor(.8,over_targets=1000)
                with whole.clone() as split:
                    whole.run(data[:-1],data[1:])
                    start=0
                    for end in (1,3,4,7,8,15,16,63,64,511,512,513,1024,1299):
                        split.run(data[start:end],data[start+1:end+1])
                        old=split.dumps()
                        restored=Model.loads(old)
                        self.assertEqual(restored.dumps(),old)
                        split.close();split=restored;start=end
                    self.assertEqual(whole.dumps(),split.dumps())
                    split.close()

    def test_real_time_not_loop_time(self):
        data=b'Completed chunks alone advance native time. '*25
        with Model(loops={'steps':[0,1,2,4],'tolerance':0},route_lr={}) as m:
            m.train(data);m.finish();s=m.stats()
            for i,l in enumerate(s['loops']['levels']):
                events=(len(data)-1)//4**i
                self.assertEqual(s['events'][i],events)
                self.assertEqual(sum(m.route_lr_stats(include_routes=True)['levels'][i]['visits']),events)
                self.assertEqual(l['passes'],events*[0,1,2,4][i])
                if i:self.assertEqual(s['clock_ticks'][i],events)
            self.assertTrue(all(l['pending_traces']==0 for l in s['loops']['levels']))

    def test_target_causality_and_frozen_source(self):
        prefix=b'Past inputs determine the next constituent state.'*3
        with Model(loops={}) as m,m.clone() as q:
            m.run(prefix,b'a'*len(prefix));q.run(prefix,b'a'*(len(prefix)-1)+b'b')
            for a,b in zip(m.loop_stats()['levels'],q.loop_stats()['levels']):
                a.pop('allocated_tape_bytes');b.pop('allocated_tape_bytes');self.assertEqual(a,b)
            self.assertEqual(m.stats()['state_energy'],q.stats()['state_energy'])
            old=m.dumps();m.evaluate(prefix);self.assertEqual(old,m.dumps())
            with m.inference_copy() as frozen:
                frozen.configure_loops(steps=[0,4,4,4],tolerance=1)
                frozen.run(prefix,prefix)
                for i,l in enumerate(frozen.loop_stats()['levels']):
                    self.assertEqual(l['allocated_tape_bytes'],0)
                    self.assertEqual(l['pending_traces'],0)
                    if i:self.assertLess(l['passes'],4*l['events'])
                saved=frozen.dumps()
                with Model.loads(saved) as resumed:
                    self.assertEqual(resumed.dumps(),saved)
                    for byte in prefix:
                        np.testing.assert_array_equal(frozen.observe(byte),resumed.observe(byte))
                    self.assertEqual(frozen.dumps(),resumed.dumps())

    def test_validation_is_atomic(self):
        with Model(core="full") as m:
            old=m.dumps()
            for kw in ({'steps':[1,1,1,1]},{'steps':[0,5,2,2]},
                       {'steps':[0,1,2]},{'steps':[0.,1.,2.,2.]},
                       {'steps':[False]*4},{'strength':0},{'strength':1.1},
                       {'tolerance':-1},{'tolerance':float('nan')},{'strength':float('inf')}):
                with self.assertRaises((ValueError,RuntimeError)):m.configure_loops(**kw)
                self.assertEqual(m.dumps(),old)
            m.train(b'pending constituent credit')
            old=m.dumps()
            with self.assertRaises(RuntimeError):m.configure_loops()
            self.assertEqual(m.dumps(),old)
            m.finish();m.configure_loops();m.train(b'subsequent completed constituents '*30)
            m.finish();m.configure_loops(steps=[0]*4)
            with Model.loads(m.dumps()) as q:self.assertEqual(m.dumps(),q.dumps())
        with self.assertRaises(ValueError):Model(semantic_laces=True,loops={})
        with self.assertRaises(ValueError):Model(memory_profile='legacy',loops={})

    def test_malformed_extensions(self):
        with Model(loops={}) as m:
            m.train(b'pending loops '*10);original=m.dumps()
            base_size=struct.unpack_from('=Q',original,56)[0]
            extension=64+base_size
            # Policy, level-zero stats/tape size; outer hash recomputed to test
            # semantic validation independently from the checksum.
            for offset,fmt,value in ((0,'i',1),(4,'i',5),(16,'f',float('nan')),
                                     (20,'f',-1.),(24+40,'Q',1)):
                bad=bytearray(original);struct.pack_into('='+fmt,bad,extension+offset,value)
                bad[8:40]=hashlib.sha256(bad[40:]).digest()
                with self.assertRaises(RuntimeError):Model.loads(bytes(bad))
            # L1 tape begins after policy, L0 stats/count, L1 stats/count.
            trace_steps=extension+24+48+48
            bad=bytearray(original);struct.pack_into('=i',bad,trace_steps,5)
            bad[8:40]=hashlib.sha256(bad[40:]).digest()
            with self.assertRaises(RuntimeError):Model.loads(bytes(bad))

    def test_cli_resume_and_inference_override(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);data=td/'data';a=td/'a.shoe';b=td/'b.shoe'
            raw=b'Loop only completed constituents. '*50;data.write_bytes(raw)
            cmd=[sys.executable,'-m','multiscale.cli','train','--data',str(data)]
            subprocess.run(cmd+['--out',str(a),'--targets','511','--loop-steps','0','1','2','2'],check=True,capture_output=True)
            subprocess.run(cmd+['--resume',str(a),'--out',str(b),'--targets','513'],check=True,capture_output=True)
            with Model.load(a) as m,Model.load(b) as q:
                m.run(raw[:513],raw[1:514]);self.assertEqual(m.dumps(),q.dumps())
            old=a.read_bytes()
            subprocess.run([sys.executable,'-m','multiscale.cli','generate','--checkpoint',str(a),
                            '--length','16','--loop-steps','0','2','3','4','--loop-tolerance','0'],check=True,capture_output=True)
            self.assertEqual(a.read_bytes(),old)


if __name__=='__main__':unittest.main()
