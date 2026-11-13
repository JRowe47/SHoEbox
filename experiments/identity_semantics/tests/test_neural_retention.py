"""Neural ownership across cache wrap, credit branches, relays and persistence."""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from runtime import Model, core, fresh_model


class NeuralRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'results' / f'neural_fixture_e{core.E}.so'
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++', '-O2', '-march=native', '-std=c++17',
                        f'-DSHOEBOX_EXPERTS={core.E}', '-shared', '-fPIC',
                        str(ROOT / 'tests/neural_fixture.cpp'), '-o', str(path)], check=True)
        cls.fx = ct.CDLL(str(path))
        p, i, l = ct.c_void_p, ct.c_int, ct.c_int64
        for name, args in dict(neural_fixture_setup=[p,i,i], neural_fixture_advance=[p,i,i],
                               neural_fixture_status=[p,i,l,p], neural_fixture_update=[p,i,i],
                               neural_fixture_release=[p], neural_fixture_depth=[p,p], neural_fixture_revive=[p,i,i],
                               neural_fixture_semantic_input=[p], neural_fixture_recycle_semantics=[p,p]).items():
            getattr(cls.fx, name).argtypes = args

    def make(self):
        m = fresh_model(seed=7)
        self.addCleanup(m.close)
        return m

    def status(self, m, source=3, tick=101):
        result = np.zeros(6, np.uint64)
        self.fx.neural_fixture_status(m.handle, source, tick, result.ctypes.data)
        return result

    def test_all_live_owner_types_survive_256_source_executions_and_release(self):
        for owner, source in [(0,3),(1,3),(2,3),(3,3),(4,3),(5,17)]:
            with self.subTest(owner=owner):
                m = self.make()
                self.fx.neural_fixture_setup(m.handle, source, owner)
                self.fx.neural_fixture_advance(m.handle, source, 256)
                np.testing.assert_array_equal(self.status(m, source), [1,1,2,1,1,1])
                self.fx.neural_fixture_release(m.handle)
                self.assertEqual(self.status(m, source)[0], 0)
                self.assertGreater(sum(v['reclaimed'] for v in m.neural_metrics().values()), 0)

    def test_language_local_and_feedback_update_original_expert_and_router(self):
        for objective in range(3):
            with self.subTest(objective=objective):
                immediate, delayed = self.make(), self.make()
                for m in (immediate, delayed):
                    self.fx.neural_fixture_setup(m.handle, 3, 0)
                self.fx.neural_fixture_advance(delayed.handle, 3, 256)
                restored = delayed.snapshot_copy()
                self.addCleanup(restored.close)
                before = immediate.flat.copy()
                for m in (immediate, delayed, restored):
                    self.fx.neural_fixture_update(m.handle, 3, objective)
                np.testing.assert_array_equal(immediate.flat, delayed.flat)
                np.testing.assert_array_equal(delayed.flat, restored.flat)
                weights_before = core.unflatten(before)['W']
                weights_after = core.unflatten(delayed.flat)['W']
                changed = np.any(weights_before != weights_after, axis=(-2,-1))
                self.assertEqual(np.argwhere(changed).tolist(), [[3,2]])
                self.assertGreater(delayed.lace_metrics()['S2']['router_updates'], 0)

    def test_archived_inputs_keep_semantic_eligibility_during_pool_collection(self):
        m = self.make()
        self.fx.neural_fixture_setup(m.handle, 3, 0)
        self.fx.neural_fixture_semantic_input(m.handle)
        self.fx.neural_fixture_advance(m.handle, 3, 256)
        result = np.zeros(2, np.uint64)
        self.fx.neural_fixture_recycle_semantics(m.handle, result.ctypes.data)
        np.testing.assert_array_equal(result, [1,16384])
        before = m.semantic_parameters()
        self.fx.neural_fixture_update(m.handle, 3, 1)
        self.assertGreater(np.linalg.norm(before[1] - m.semantic_parameters()[1]), 0)
        self.assertTrue(all(v['expired_semantic_handles'] == 0 for v in m.completion_metrics().values()))

    def test_bottom_temporal_eligibility_survives_wrap(self):
        m = self.make()
        self.fx.neural_fixture_setup(m.handle, 0, 0)
        self.fx.neural_fixture_advance(m.handle, 0, 256)
        np.testing.assert_array_equal(self.status(m, 0), [1,1,2,1,1,1])
        before = m.flat.copy()
        self.fx.neural_fixture_update(m.handle, 0, 0)
        self.assertFalse(np.array_equal(before, m.flat))
        self.assertEqual(m.lace_metrics()['A1']['missing_parent'], 0)

    def test_pending_credit_checkpoint_continuation_and_stale_handle_rejection(self):
        m = self.make()
        self.fx.neural_fixture_setup(m.handle, 3, 2)
        self.fx.neural_fixture_advance(m.handle, 3, 256)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'held.npz'
            m.save(path)
            n = Model.load(path)
            self.addCleanup(n.close)
            self.assertEqual(m.fingerprint(), n.fingerprint())
            for model in (m,n):
                self.fx.neural_fixture_update(model.handle, 3, 3)
            np.testing.assert_array_equal(m.flat, n.flat)
            self.assertEqual(m.fingerprint(), n.fingerprint())
            self.assertEqual(self.status(n, 3, 1)[0], 0)  # same slot hint, wrong generation
            before = n.fingerprint()
            with self.assertRaises(ValueError):
                n.import_neural(n.neural_state()[:-1])
            self.assertEqual(before, n.fingerprint())
            n.reset_context()
            self.assertTrue(all(v['live_archived'] == 0 for v in n.neural_metrics().values()))

    def test_freeze_and_expert_revival_respect_retained_producers(self):
        m = self.make()
        self.fx.neural_fixture_setup(m.handle, 0, 0)
        self.fx.neural_fixture_advance(m.handle, 0, 256)
        self.assertEqual(self.fx.neural_fixture_revive(m.handle, 0, 2), 0)
        frozen = np.zeros(18, np.int32)
        frozen[0] = 1
        m.configure_training_policy(frozen)
        before = m.flat.copy()
        self.fx.neural_fixture_update(m.handle, 0, 0)
        np.testing.assert_array_equal(before, m.flat)
        m.configure_training_policy(np.zeros(18, np.int32))
        self.fx.neural_fixture_release(m.handle)
        self.assertEqual(self.fx.neural_fixture_revive(m.handle, 0, 2), 1)

    def test_credit_depth_reclaims_only_out_of_budget_ancestors(self):
        m = self.make()
        result = np.zeros(4, np.uint64)
        self.fx.neural_fixture_depth(m.handle, result.ctypes.data)
        np.testing.assert_array_equal(result, [1,1,0,0])

    def test_repeated_text_has_no_lost_neural_credit_and_drains(self):
        m = self.make()
        text = np.frombuffer(b'The king walked across the street. ' * 120, np.uint8)
        m.run(text, mode='periodic')
        m.finish_document(mode='periodic')
        for metrics in m.lace_metrics().values():
            self.assertEqual(metrics['missing_parent'], 0)
            self.assertEqual(metrics['overwritten_credit'], 0)
        for metrics in m.message_metrics().values():
            self.assertEqual(metrics['missing_feedback_producers'], 0)
        for metrics in m.neural_metrics().values():
            self.assertEqual(metrics['missing_live_dependencies'], 0)
        for metrics in m.composition_metrics().values():
            self.assertEqual(metrics['expired_input_edges'], 0)
        m.finish_document(mode='periodic', reset=True)
        self.assertTrue(all(v['live_archived'] == 0 for v in m.neural_metrics().values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
