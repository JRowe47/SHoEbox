"""Reassociated factors preserve request nonlinearities and update scopes."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,core,Model


class FactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'factor_fixture_e{core.E}.so'
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O3','-march=native','-std=c++17','-pthread',
                        f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',
                        str(ROOT/'tests/factor_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        cls.fx.factor_fixture.argtypes=[ct.c_void_p]+[ct.c_int]*4
        cls.fx.factor_cancellation.argtypes=[ct.c_void_p]

    def make(self):
        model=fresh_model(seed=7);self.addCleanup(model.close);return model

    def test_repeated_factors_match_scalar_clipping_masks_rates_and_frozen_banks(self):
        for sparse in (0,1):
            for frozen in (0,1):
                with self.subTest(sparse=sparse,frozen=frozen):
                    scalar=self.make();grouped=self.make();initial=scalar.flat.copy()
                    self.fx.factor_fixture(scalar.handle,0,3,sparse,frozen)
                    self.fx.factor_fixture(grouped.handle,1,3,sparse,frozen)
                    np.testing.assert_allclose(grouped.flat,scalar.flat,rtol=1e-5,atol=2e-7)
                    drift=np.linalg.norm(grouped.flat.astype(float)-scalar.flat)
                    update=np.linalg.norm(scalar.flat.astype(float)-initial)
                    self.assertLess(drift/update,1e-6)
                    metrics=grouped.content_factor_metrics()
                    self.assertEqual(metrics['scopes'],3)
                    self.assertLess(metrics['weight_products'],metrics['weight_requests']/2)
                    self.assertLess(metrics['route_products'],metrics['route_requests']/4)

    def test_parallel_rows_are_bitwise_identical_after_grouping(self):
        cpus=sorted(os.sched_getaffinity(0))
        if len(cpus)<2:self.skipTest('Two CPUs required')
        serial=self.make();parallel=self.make()
        parallel.configure_content_workers(2,cpus[:2])
        self.fx.factor_fixture(serial.handle,1,2,1,0)
        self.fx.factor_fixture(parallel.handle,1,2,1,0)
        np.testing.assert_array_equal(serial.flat,parallel.flat)
        self.assertEqual(serial.content_factor_metrics(),parallel.content_factor_metrics())

    def test_cancelled_routes_still_normalize_and_recycled_slots_start_fresh(self):
        m=self.make();self.fx.factor_cancellation(m.handle)
        self.assertEqual(m.content_factor_metrics()['scopes'],2)

    def test_save_load_chunk_boundaries_and_frozen_evaluation(self):
        data=np.frombuffer((b'The river runs through the valley. The king crossed the bridge. '*40)[:1026],np.uint8)
        a=self.make();b=self.make()
        a.run(data[:1025],data[1:1026],mode='periodic',lr=.04,lp=.04/3)
        with tempfile.TemporaryDirectory(prefix='shoebox-factors-') as temporary:
            path=Path(temporary)/'checkpoint.npz'
            for start,end in ((0,321),(321,777),(777,1025)):
                b.run(data[start:end],data[start+1:end+1],mode='periodic',lr=.04,lp=.04/3)
                b.save(path);b.close();b=Model.load(path);self.addCleanup(b.close)
            self.assertEqual(a.fingerprint(),b.fingerprint())
            before=a.fingerprint();a.reset_content_factor_metrics()
            self.assertEqual(before,a.fingerprint())
            nll,la=a.run(data[:128],data[1:129],mode='eval',logits=True)
            nllb,lb=b.run(data[:128],data[1:129],mode='eval',logits=True)
            self.assertEqual(nll,nllb);np.testing.assert_array_equal(la,lb)
            self.assertEqual(a.content_factor_metrics()['weight_products'],0)
            a.save(path);c=Model.load(path);self.addCleanup(c.close)
            a.run(data[:128],data[1:129],mode='periodic',lr=.04,lp=.04/3)
            c.run(data[:128],data[1:129],mode='periodic',lr=.04,lp=.04/3)
            self.assertEqual(a.fingerprint(),c.fingerprint())


if __name__=='__main__':unittest.main()
