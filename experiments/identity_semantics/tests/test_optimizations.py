"""Bitwise oracles for bank batching, pending order and transient lifecycle."""
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


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'optimization_fixture_e{core.E}.so'
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O3','-march=native','-std=c++17','-pthread',
                        f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',
                        str(ROOT/'tests/optimization_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        for name,args in {
            'optimization_vjp':[ct.c_void_p,ct.c_int,ct.c_void_p],
            'optimization_allocate':[ct.c_void_p,ct.c_int,ct.c_void_p],
            'optimization_gradients':[ct.c_void_p,ct.c_int,ct.c_int],
            'optimization_credit_order':[ct.c_void_p,ct.c_int,ct.c_int],
            'optimization_cache_status':[ct.c_void_p,ct.c_void_p],
        }.items():getattr(cls.fx,name).argtypes=args

    def make(self):
        model=fresh_model(seed=7);self.addCleanup(model.close);return model

    def test_ready_derivative_batches_equal_scalar(self):
        m=self.make()
        for count in (1,3,4,9,16):
            with self.subTest(count=count):
                values=np.zeros((2,count,64),np.float32)
                self.fx.optimization_vjp(m.handle,count,values.ctypes.data)
                np.testing.assert_array_equal(values[0],values[1])

    def test_bank_tiles_preserve_clipping_masks_and_request_order(self):
        for sparse in (0,1):
            with self.subTest(sparse=sparse):
                scalar=self.make();batched=self.make()
                self.fx.optimization_gradients(scalar.handle,0,sparse)
                self.fx.optimization_gradients(batched.handle,1,sparse)
                np.testing.assert_array_equal(scalar.flat,batched.flat)

    def test_parallel_bank_rows_equal_scalar_and_workers_can_be_replaced(self):
        cpus=sorted(os.sched_getaffinity(0))
        if len(cpus)<2:self.skipTest('Two CPUs required for the parallel oracle')
        for sparse in (0,1):
            for count in sorted({2,min(4,len(cpus))}):
                with self.subTest(sparse=sparse,workers=count):
                    scalar=self.make();parallel=self.make()
                    parallel.configure_content_workers(count,cpus[:count])
                    self.fx.optimization_gradients(scalar.handle,0,sparse)
                    self.fx.optimization_gradients(parallel.handle,1,sparse)
                    np.testing.assert_array_equal(scalar.flat,parallel.flat)
                    parallel.configure_content_workers(1)
                    parallel.configure_content_workers(2,cpus[:2])
                    parallel.configure_content_workers(1)

    def test_pending_order_matches_full_scan_through_delayed_cycles(self):
        for frozen in (0,1):
            with self.subTest(frozen=frozen):
                scan=self.make();indexed=self.make()
                self.fx.optimization_credit_order(scan.handle,0,frozen)
                self.fx.optimization_credit_order(indexed.handle,1,frozen)
                self.assertEqual(scan.fingerprint(),indexed.fingerprint())

    def test_import_and_reset_invalidate_warmed_caches(self):
        m=self.make()
        data=np.frombuffer((b'The river runs through the valley. '*20)[:513],np.uint8)
        m.run(data[:-1],data[1:],mode='periodic',lr=.04,lp=.04/3)
        def status():
            out=np.zeros(4,np.uint64);self.fx.optimization_cache_status(m.handle,out.ctypes.data);return out
        self.assertTrue(np.all(status()[:2]>0))
        composition=m.composition_state();content=m.content_expert_state()
        m.import_composition(composition);m.import_content_experts(content)
        np.testing.assert_array_equal(status(),0)
        m.run(data[:-1],data[1:],mode='periodic',lr=.04,lp=.04/3)
        self.assertGreater(status()[0],0)
        m.reset();np.testing.assert_array_equal(status(),0)

    def test_free_inventory_continues_exactly_after_collection_and_reload(self):
        m=self.make();status=np.zeros(4,np.uint64)
        self.fx.optimization_allocate(m.handle,16384+79,status.ctypes.data)
        self.assertGreater(status[0],status[1])
        self.assertGreater(status[2],10000)
        with tempfile.TemporaryDirectory(prefix='shoebox-free-inventory-') as temporary:
            path=Path(temporary)/'checkpoint.npz';m.save(path)
            n=Model.load(path);self.addCleanup(n.close)
            self.assertEqual(m.fingerprint(),n.fingerprint())
            self.fx.optimization_allocate(m.handle,211,status.ctypes.data)
            for count in (7,61,143):self.fx.optimization_allocate(n.handle,count,status.ctypes.data)
            self.assertEqual(m.fingerprint(),n.fingerprint())

    def test_worker_configuration_rejects_invalid_requests(self):
        m=self.make()
        for count in (0,33,True,2.5):
            with self.subTest(count=count),self.assertRaises(ValueError):m.configure_content_workers(count)
        for cpus in ([-1,0],[0,0],[],[0,999999],[True,0]):
            with self.subTest(cpus=cpus),self.assertRaises(ValueError):m.configure_content_workers(2,cpus)


if __name__=='__main__':unittest.main()
