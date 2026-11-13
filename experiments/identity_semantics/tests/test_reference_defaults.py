"""Reference defaults, opt-outs, effective capabilities and migration safety."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'src'))
from runtime import Model, fresh_model
from convert_to_completion import convert


class ReferenceDefaultsTests(unittest.TestCase):
    def test_public_and_low_level_constructors_enable_the_reference(self):
        a = fresh_model()
        b = Model()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        for model in (a, b):
            self.assertEqual(model.runtime_kernel, 'completion')
            self.assertTrue(all(model.capabilities()['features'].values()))
            self.assertEqual(model.capabilities()['composition_abi'], 10)
            self.assertEqual(model.capabilities()['reference_profile'], 'reference-v1')
            self.assertEqual(model.lace['depth'], 3)
            self.assertEqual(model.state_config['temporal_features'], 64)
            self.assertTrue(model.capabilities()['unavailable_experiment_controls'])
        for group in ('config', 'lace', 'state_config', 'probe', 'utility', 'messages'):
            self.assertEqual(getattr(a, group), getattr(b, group))
        np.testing.assert_array_equal(a.flat, b.flat)
        np.testing.assert_array_equal(a.extra_flat, b.extra_flat)

    def test_historical_comparators_and_explicit_ablation_remain_explicit(self):
        a = fresh_model(kernel='relays')
        b = fresh_model(kernel='completion', profile='historical')
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        self.assertEqual(a.lace['dfa_gain'], 0)
        self.assertEqual(b.state_config['temporal_features'], 16)
        self.assertFalse(b.local_credit)
        self.assertFalse(b.content_experts)
        self.assertFalse(b.pattern_classes)
        self.assertFalse(b.paired_memory)
        self.assertEqual(b.capabilities()['reference_profile'], 'custom')
        b.configure_messages({'payload_gain': 0})
        self.assertFalse(b.capabilities()['features']['recursive_constituent_content'])
        with self.assertRaises(ValueError):
            fresh_model(kernel='relays', profile='reference')
        with self.assertRaises(ValueError):
            fresh_model(profile='typo')

    def test_unimplemented_port_controls_cannot_silently_claim_activation(self):
        for name in ('port_mode', 'port_gain', 'balance'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'no port_mode'):
                Model(probe={name: 1})

    def test_default_checkpoint_resumes_exactly_with_saved_policy(self):
        m = fresh_model()
        self.addCleanup(m.close)
        text = np.frombuffer(b'The river runs through the valley. ' * 35, np.uint8)
        m.run(text[:377], mode='periodic')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reference.npz'
            m.save(path)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(int(saved['version'][0]), 39)
                self.assertEqual(saved['composition_state'][:4].view(np.uint32)[0], 10)
            n = Model.load(path)
            self.addCleanup(n.close)
            self.assertEqual(m.fingerprint(), n.fingerprint())
            expected = m.run(text[377:], mode='periodic', logits=True)[1]
            actual = np.concatenate([n.run(part, mode='periodic', logits=True)[1]
                                     for part in np.array_split(text[377:], 5)])
            np.testing.assert_array_equal(expected, actual)
            self.assertEqual(m.fingerprint(), n.fingerprint())

    def test_v33_requires_explicit_fitted_conversion(self):
        m = fresh_model(profile='historical')
        self.addCleanup(m.close)
        m.run(np.frombuffer(b'Old clocks must not resume. ' * 15, np.uint8), mode='periodic')
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'v33.npz', Path(tmp) / 'v35.npz'
            m.save(source)
            with np.load(source, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            # v33 and v34 have identical fitted layouts, but incompatible clocks.
            arrays['version'][0] = 33
            arrays['composition_state'][:4] = np.array([7], np.uint32).view(np.uint8)
            np.savez_compressed(source, **arrays)
            with self.assertRaisesRegex(ValueError, 'coverage clocks'):
                Model.load(source)
            result = convert(source, target)
            self.assertEqual((result['version'], result['abi']), (39, 10))
            n = Model.load(target)
            self.addCleanup(n.close)
            for old, new in ((m.flat, n.flat), (m.extra_flat, n.extra_flat),
                             (m.semantic_parameters(), n.semantic_parameters()),
                             (m.completion_gates(), n.completion_gates()),
                             (m.means(), n.means()), (m.ymean(), n.ymean())):
                np.testing.assert_array_equal(old, new)
            self.assertEqual(m.lace, n.lace)
            self.assertTrue(all(event['seq'] == 0 for event in n.content_events().values()))
            self.assertEqual(n.persistent_metrics()['occupied'], 0)

    def test_v34_requires_explicit_fitted_conversion(self):
        m = fresh_model(profile='historical')
        self.addCleanup(m.close)
        m.run(np.frombuffer(b'Old clocks must not resume. ' * 15, np.uint8), mode='periodic')
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'v34.npz', Path(tmp) / 'v35.npz'
            m.save(source)
            with np.load(source, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            # Fitted arrays are unchanged; v34 lacks owned neural eligibility.
            arrays['version'][0] = 34
            arrays['composition_state'][:4] = np.array([8], np.uint32).view(np.uint8)
            arrays.pop('neural_state', None)
            np.savez_compressed(source, **arrays)
            with self.assertRaisesRegex(ValueError, 'expiring neural producers'):
                Model.load(source)
            result = convert(source, target)
            self.assertEqual((result['version'], result['abi']), (39, 10))
            n = Model.load(target)
            self.addCleanup(n.close)
            for old, new in ((m.flat, n.flat), (m.extra_flat, n.extra_flat),
                             (m.semantic_parameters(), n.semantic_parameters()),
                             (m.completion_gates(), n.completion_gates()),
                             (m.means(), n.means()), (m.ymean(), n.ymean())):
                np.testing.assert_array_equal(old, new)
            self.assertEqual(m.lace, n.lace)
            self.assertTrue(all(event['seq'] == 0 for event in n.content_events().values()))
            self.assertEqual(n.persistent_metrics()['occupied'], 0)

    def test_repository_entry_point_uses_reference_without_torch(self):
        command = 'from shoebox_reference import fresh_model; m=fresh_model(); '
        command += 'assert m.runtime_kernel=="completion"; '
        command += 'assert all(m.capabilities()["features"].values()); m.close()'
        subprocess.run([sys.executable, '-c', command], cwd=ROOT.parents[1],
                       env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'}, check=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
