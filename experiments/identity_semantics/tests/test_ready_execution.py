"""Execution indexes, exact packed adjoints, and invalidation boundaries."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import core,fresh_model

class ReadyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'ready_execution_fixture_e{core.E}.so'
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-pthread',
                        f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',
                        str(ROOT/'tests/ready_execution_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path))
        for name in ('adjoint','scheduler','reverse'):
            getattr(cls.fx,f'execution_{name}_fixture').argtypes=[ct.c_void_p]

    def make(self):
        m=fresh_model(seed=7);self.addCleanup(m.close);return m

    def test_indexed_accumulators_match_ordered_map_across_growth_reset_and_wrap(self):
        self.assertEqual(self.fx.execution_accumulator_fixture(),1)

    def test_dense_packets_are_bitwise_equal_to_scalar_derivatives(self):
        self.assertEqual(self.fx.execution_adjoint_fixture(self.make().handle),1)

    def test_scheduler_retains_late_publications_and_drains_once_in_order(self):
        self.assertEqual(self.fx.execution_scheduler_fixture(self.make().handle),1)

    def test_teaching_keys_slab_generation_frontier_and_replay_invalidate_plans(self):
        self.assertEqual(self.fx.execution_reverse_fixture(self.make().handle),1)

if __name__=='__main__':unittest.main(verbosity=2)
