"""Shared publications and sparse pending work preserve reference behavior."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,core


class PublicationStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'publication_fixture_e{core.E}.so'
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread',
                        f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',
                        str(ROOT/'tests/publication_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        cls.fx.publication_fixture_handles.argtypes=[ct.c_void_p]

    def make(self):
        m=fresh_model(seed=7);self.addCleanup(m.close);return m

    def test_handles_detach_on_changes_and_cache_uses_payload_versions(self):
        self.assertEqual(self.fx.publication_fixture_handles(self.make().handle),1)

    def test_sparse_gradients_are_owned_and_wire_roundtrip_is_exact(self):
        self.assertEqual(self.fx.publication_fixture_sparse(),1)

    def test_stream_continuation_sharing_and_gradient_lifecycle(self):
        m=self.make();text=np.frombuffer((b'The river runs through the valley. '*70)[:2050],np.uint8)
        m.run(text[:1024],text[1:1025],mode='periodic')
        before=m.fingerprint();stats=m.publication_storage()
        self.assertLess(stats['event_handle_bytes'],stats['legacy_event_bytes']/5)
        self.assertGreater(stats['content_cache_hits'],0)
        self.assertGreater(stats['context_cache_hits'],0)
        self.assertGreater(stats['neural_gradient_blocks'],0)
        archive=m.completion_storage()['archived_records']
        self.assertLess(stats['neural_gradient_blocks']+stats['input_gradient_blocks'],archive)
        # Observation must never allocate empty gradients or modify checkpoint data.
        self.assertEqual(before,m.fingerprint())
        clone=m.snapshot_copy();self.addCleanup(clone.close)
        self.assertEqual(before,clone.fingerprint())
        self.assertEqual(clone.publication_storage()['content_cache_hits'],0)
        for model in (m,clone):model.run(text[1024:1153],text[1025:1154],mode='eval')
        for model in (m,clone):model.run(text[1153:1282],text[1154:1283],mode='periodic')
        self.assertEqual(m.fingerprint(),clone.fingerprint())
        for model in (m,clone):
            model.reset_context();after=model.publication_storage()
            self.assertEqual(after['gradient_payload_bytes'],0)
            self.assertEqual(after['unique_content_payloads'],0)
            self.assertEqual(after['unique_context_payloads'],0)
            self.assertEqual(after['role_cache_bytes'],0)
            self.assertEqual(after['content_cache_hits'],0)


if __name__=='__main__':unittest.main(verbosity=2)
