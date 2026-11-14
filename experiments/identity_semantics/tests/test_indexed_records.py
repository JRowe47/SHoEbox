"""Incremental retention versus the independent full-mark Action-1 oracle."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,core

class IndexedRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'indexed_records_fixture_e{core.E}.so'
        subprocess.run(['g++','-O2','-march=native','-std=c++17','-pthread',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/indexed_records_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        for name in ('indexed_verify','indexed_semantic_verify','indexed_overlap','indexed_context_lookup'):
            getattr(cls.fx,name).argtypes=[ct.c_void_p]
        cls.fx.indexed_storage.argtypes=[ct.c_void_p,ct.c_void_p]
    def make(self,profile='reference',**event):
        m=fresh_model(seed=7,profile=profile,event=event or None);self.addCleanup(m.close);return m
    def test_archive_recycling_has_bounded_idle_storage(self):
        self.assertEqual(self.fx.indexed_pool_bounds(),1)
    def test_overlapping_owners_preserve_depth_and_release(self):
        self.assertEqual(self.fx.indexed_overlap(self.make().handle),1)
    def test_hot_archived_detached_and_stale_context_lookup(self):
        self.assertEqual(self.fx.indexed_context_lookup(self.make().handle),1)
    def test_full_mark_matches_across_wrap_drain_restart_and_reset(self):
        raw=np.frombuffer((b'The river runs through the valley. The king crossed the bridge. '*100),np.uint8)
        for profile,heads in (('reference',0),('event',1),('event',2),('pipeline',1)):
            with self.subTest(profile=profile,heads=heads):
                m=self.make(profile,context_heads=heads)
                lo=0
                for count in (1,63,65,127,512,1024,1023):
                    hi=lo+count;m.run(raw[lo:hi],raw[lo+1:hi+1],mode='periodic')
                    self.assertEqual(self.fx.indexed_verify(m.handle),1)
                    self.assertEqual(self.fx.indexed_semantic_verify(m.handle),1)
                    lo=hi
                n=m.snapshot_copy();self.addCleanup(n.close)
                self.assertEqual(self.fx.indexed_verify(n.handle),1)
                self.assertEqual(self.fx.indexed_semantic_verify(n.handle),1)
                self.assertEqual(m.fingerprint(),n.fingerprint())
                for model in (m,n):
                    model.finish_document(mode='periodic')
                    self.assertEqual(self.fx.indexed_verify(model.handle),1)
                self.assertEqual(m.fingerprint(),n.fingerprint())
                m.reset_context();self.assertEqual(self.fx.indexed_verify(m.handle),1)
    def test_runtime_eligibility_is_head_sized_and_wire_is_legacy(self):
        raw=np.frombuffer(b'The river runs through the valley. '*12,np.uint8)
        for heads in (0,1,2):
            m=self.make('event',context_heads=heads);m.run(raw[:-1],raw[1:],mode='periodic')
            stats=np.zeros(6,np.uint64);self.fx.indexed_storage(m.handle,stats.ctypes.data)
            self.assertEqual(stats[0],stats[2]*heads*320)
            self.assertEqual(stats[0],stats[1]);self.assertEqual(stats[5],648)
            before=m.event_state();m.import_event(before);np.testing.assert_array_equal(before,m.event_state())
    def test_reducing_heads_releases_inactive_capacity(self):
        raw=np.frombuffer(b'The river runs through the valley. '*5,np.uint8)
        m=self.make('event',context_heads=2)
        for heads in (2,1,0):
            if heads!=2:
                m.reset_context();m.configure_event({'context_heads':heads})
                stats=np.zeros(6,np.uint64);self.fx.indexed_storage(m.handle,stats.ctypes.data)
                self.assertEqual(stats[1],0)
            m.run(raw[:-1],raw[1:],mode='periodic')
            stats=np.zeros(6,np.uint64);self.fx.indexed_storage(m.handle,stats.ctypes.data)
            self.assertEqual(stats[0],stats[2]*heads*320)
            self.assertEqual(stats[0],stats[1])

if __name__=='__main__':unittest.main(verbosity=2)
