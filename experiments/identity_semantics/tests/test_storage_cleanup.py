"""Compact completion records, historical wire compatibility and adapter parity."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core


class StorageCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'storage_fixture_e{core.E}.so'
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread',
                        f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',
                        str(ROOT/'tests/storage_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        cls.fx.storage_fixture_encode.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]

    def make(self,kernel='completion_checks'):
        m=fresh_model(seed=7,kernel=kernel);self.addCleanup(m.close);return m

    def test_removed_adapter_has_exact_final_values_and_diagnostic_counts(self):
        for muted in (0,1):
            for experts in (0,1):
                with self.subTest(muted=muted,experts=experts):
                    results=[]
                    for old in (0,1):
                        m=self.make();out=np.zeros(193,np.float32)
                        self.fx.storage_fixture_encode(m.handle,old,muted,experts,out.ctypes.data)
                        results.append((out,m.content_expert_state(),m.composition_state()))
                    for a,b in zip(*results):np.testing.assert_array_equal(a,b)
                    self.assertEqual(results[0][0][-1],muted)

    def test_legacy_input_keeps_all_consumed_eligibility(self):
        self.assertEqual(self.fx.storage_fixture_legacy_input(),1)

    def test_storage_reduction_and_bottom_temporal_lifecycle(self):
        m=self.make('completion');text=np.frombuffer(b'The river runs through the valley. '*40,np.uint8)
        m.run(text[:1025],mode='periodic')
        stats=m.completion_storage()
        self.assertEqual(stats['legacy_input_bytes'],5488)
        self.assertEqual(stats['legacy_record_bytes'],11424)
        self.assertLess(stats['input_bytes'],stats['legacy_input_bytes']/3)
        self.assertLess(stats['upper_record_bytes'],stats['legacy_record_bytes']/2)
        self.assertGreater(stats['temporal_records'],0)
        self.assertGreater(stats['archived_records'],stats['temporal_records'])
        # Wire ABI remains readable by existing v39/v10/v1 checkpoints.
        self.assertEqual(int.from_bytes(m.composition_state()[:4].tobytes(),'little'),10)
        self.assertEqual(int.from_bytes(m.neural_state()[:8].tobytes(),'little'),1)
        copy=m.snapshot_copy();self.addCleanup(copy.close)
        self.assertEqual(m.fingerprint(),copy.fingerprint())
        self.assertEqual(m.completion_storage(),copy.completion_storage())
        for model in (m,copy):model.run(text[:129],mode='periodic')
        self.assertEqual(m.fingerprint(),copy.fingerprint())
        copy.import_composition(m.composition_state(fitted=True),fitted=True)
        self.assertEqual(copy.completion_storage()['temporal_records'],0)
        m.reset_context();self.assertEqual(m.completion_storage()['temporal_records'],0)
        m.run(text[:129],mode='periodic');m.reset()
        self.assertEqual(m.completion_storage()['temporal_records'],0)


if __name__=='__main__':unittest.main(verbosity=2)
