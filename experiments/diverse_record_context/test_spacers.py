"""Independent literal spacer interpreter and occurrence/cursor contracts."""
import hashlib, struct, unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from .spacers import spacer, augment
from .tasks import worlds,make,interpret
from .stream import new_cursor,advance
from .backend import Model
from experiments.joint_cold_gate.adapter import prefix_labels

def reference(group, episode, physical, n):
    result=[];block=0
    while len(result)<n:
        message=b'SHoEbox-EXP0305-diverse-record-context-v7-spacer'+bytes([0])+group+bytes([0])
        message+=struct.pack('<QBI',episode,physical,block)
        raw=hashlib.sha256(message).digest()
        result.extend(160+int(x)%96 for x in raw if int(x)<192)
        block+=1
    return bytes(result[:n])

class SpacerTests(unittest.TestCase):
    def test_reference_all_lengths_orders_occurrences(self):
        for w in worlds('train',16):
            for episode in (0,1,63,64,511,65537,2**32+1):
                for physical in (0,1):
                    for length in range(8):
                        self.assertEqual(spacer(w.group_key,episode,physical,length),reference(w.group_key,episode,physical,length))

    def test_rejection_threshold_block_and_zero(self):
        calls=[]
        class Digest:
            def __init__(self,raw):self.raw=raw
            def digest(self):return self.raw
        def fake(message):
            calls.append(message)
            return Digest(bytes([192,255])*16 if len(calls)==1 else bytes([0,95,96,191,192,255,1,97])*4)
        with patch('experiments.diverse_record_context.spacers.hashlib.sha256',fake):
            self.assertEqual(spacer(b'group',19,1,0),b'');self.assertEqual(calls,[])
            self.assertEqual(spacer(b'group',19,1,7),bytes([160,255,160,255,161,161,160]))
        self.assertEqual(calls[0][-13:],struct.pack('<QBI',19,1,0))
        self.assertEqual(calls[1][-13:],struct.pack('<QBI',19,1,1))

    def test_only_spacers_labels_masks_phase_interpreter(self):
        w=worlds('train',1)[0]
        for condition in ('pad0','repeat16','diverse16','compete16','overwrite16'):
            for variant in range(4):
                for lengths in ((0,0),(0,7),(7,0),(7,7),(3,5)):
                    for changed in (0,1):
                        c=make(w,condition,variant,13,lengths,changed)
                        ep,positions=augment(c.ep,w,149,13,lengths)
                        self.assertEqual(len(ep.data),len(c.ep.data));self.assertEqual(ep.mask,c.ep.mask)
                        self.assertEqual(ep.answer,c.ep.answer);self.assertEqual(interpret(ep),c.expected)
                        np.testing.assert_array_equal(prefix_labels(ep),prefix_labels(c.ep))
                        self.assertEqual([i for i,(a,b) in enumerate(zip(ep.data,c.ep.data)) if a!=b],positions)
                        self.assertTrue(all(160<=ep.data[i]<=255 for i in positions))
                        for t in (2,15,len(ep.data)//2,len(ep.data)-3):
                            future=replace(ep,data=ep.data[:t]+bytes([239])*(len(ep.data)-t))
                            np.testing.assert_array_equal(prefix_labels(future)[:t],prefix_labels(ep)[:t])

    def test_partial_spacer_resume_and_policy_guard(self):
        with Model('a7',seed=7) as m:
            cursor=new_cursor()
            # Stop within an actual nonempty spacer using the exact selected cursor.
            s,_=advance(m,cursor,1);cur=s['current']
            pos=2+cur['offset']+5
            self.assertGreater(cur['spacers'][0],1)
            s,_=advance(m,s,pos+1-s['model_bytes'])
            self.assertEqual(s['current']['used'],pos+1)
            with Model.loads(m.dumps(),'a7') as other:
                a,_=advance(m,s,101);b,_=advance(other,s,17)
                with Model.loads(other.dumps(),'a7') as resumed:
                    c,_=advance(resumed,b,84)
                    self.assertEqual(a,c);self.assertEqual(m.dumps(),resumed.dumps())
            with self.assertRaises(ValueError):advance(m,new_cursor(policy='a6'),0)

if __name__=='__main__':unittest.main()
