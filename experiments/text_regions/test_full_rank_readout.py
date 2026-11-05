"""Full-matrix refits, frozen provenance and deployable-head equivalence."""
import copy
from dataclasses import asdict, replace
import unittest

import torch

from .compiled_readout import compile_readout, restore_compiled
from .full_rank_readout import (SETTINGS, FullRankHead, fit_full_rank, record_head,
    restore_head, verify_resume)
from .readout_diagnostic import ByteDecoder, fit_byte_decoder
from .residual_adapter import (balanced_weights, base_fingerprint, initialize_residual,
    select_strength)
from .test_residual_adapter import fixture

torch.set_num_threads(2)


class FullRankChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = fixture('train'); cls.validation = fixture('validation')
        cls.base = fit_byte_decoder(cls.training['features']['final/raw'], cls.training['answers'],
            sample_weights=balanced_weights(cls.training), max_iter=100)

    def test_unpenalized_refit_reaches_existing_convex_solution(self):
        h = fit_full_rank(self.training, self.base, max_iter=300)
        self.assertAlmostEqual(h.refit.fit['objective'], self.base.fit['objective'], places=7)
        self.assertTrue(torch.equal(h.predict(self.training['features']['final/raw']), self.training['answers']))
        self.assertEqual(h.refit.weight.shape, (13, 256))

    def test_repeatable_fitting_preserves_base_input_and_rng(self):
        before = base_fingerprint(self.base); rng = torch.get_rng_state().clone()
        x = self.training['features']['final/raw'].clone()
        a = fit_full_rank(self.training, self.base, max_iter=20)
        b = fit_full_rank(self.training, self.base, max_iter=20)
        self.assertTrue(torch.equal(a.refit.weight, b.refit.weight))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(x, self.training['features']['final/raw']))
        self.assertEqual(before, base_fingerprint(self.base))

    def test_training_preservation_protects_conflicting_shared_answers(self):
        data = fixture('train'); data['features']['final/raw'] = torch.zeros(12, 1)
        data['answers'] = torch.tensor([65]*4+[66]*8)
        weight = torch.zeros(2, 256, dtype=torch.double); weight[-1, 65] = 1
        base = ByteDecoder(torch.zeros(1, dtype=torch.double), torch.ones(1, dtype=torch.double), weight, {})
        a = fit_full_rank(data, base, penalty=0, max_iter=100)
        b = fit_full_rank(data, base, penalty=10, max_iter=100)
        self.assertEqual(a.refit.fit['preservation']['baseline_correct_answers_lost'], 4)
        self.assertEqual(b.refit.fit['preservation']['baseline_correct_answers_lost'], 0)

    def test_rejects_wrong_split_invalid_settings_and_features(self):
        for args in ({'l2': 0}, {'penalty': -1}, {'penalty': float('nan')}, {'max_iter': 0}, {'margin_cap': 0}):
            with self.assertRaises(ValueError): fit_full_rank(self.training, self.base, **args)
        with self.assertRaises(ValueError): fit_full_rank(self.validation, self.base)
        bad = copy.deepcopy(self.training); bad['features']['final/raw'][0, 0] = float('nan')
        with self.assertRaises(ValueError): fit_full_rank(bad, self.base)

    def test_validation_selection_serialization_and_frozen_fallback(self):
        h = fit_full_rank(self.training, self.base, max_iter=20)
        h = select_strength(h, self.validation, fit_identities=self.training['identities'])
        restored = restore_head(record_head(h), self.base)
        x = self.validation['features']['final/raw']
        self.assertTrue(torch.equal(h.logits(x), restored.logits(x)))
        self.assertTrue(torch.equal(replace(h, strength=0).logits(x), self.base.logits(x)))
        bad = copy.deepcopy(record_head(h)); bad['refit']['mean'][0] += 1
        with self.assertRaises(ValueError): restore_head(bad, self.base)
        overlap = copy.deepcopy(self.validation); overlap['identities'][0] = self.training['identities'][0]
        with self.assertRaises(ValueError): select_strength(h, overlap, fit_identities=self.training['identities'])

    def test_resume_rejects_changed_inputs_sources_settings_or_base(self):
        invocation = {'settings': SETTINGS, 'source_sha256': {'code': 'one'}, 'input_sha256': {'x': 'two'}}
        saved = {'schema_version': 1, 'invocation': copy.deepcopy(invocation), 'base': asdict(self.base), 'heads': {}}
        verify_resume(saved, invocation, self.base)
        for key in ('settings', 'source_sha256', 'input_sha256'):
            altered = copy.deepcopy(invocation); altered[key] = {'changed': True}
            with self.assertRaises(ValueError): verify_resume(saved, altered, self.base)
        changed = replace(self.base, weight=self.base.weight+1)
        with self.assertRaises(ValueError): verify_resume(saved, invocation, changed)


class CompiledChecks(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(42)
        self.x = torch.randn(19, 7, generator=g, dtype=torch.double)
        self.base = ByteDecoder(torch.randn(7, generator=g, dtype=torch.double),
            torch.rand(7, generator=g, dtype=torch.double)+.2,
            torch.randn(8, 256, generator=g, dtype=torch.double), {})
        self.g = g

    def assert_equivalent(self, head):
        c = compile_readout(head); z = head.logits(self.x)
        self.assertTrue(torch.allclose(c(self.x), z, atol=1e-10, rtol=1e-12))
        self.assertTrue(torch.equal(c.predict(self.x), z.argmax(-1)))
        self.assertEqual(list(c.parameters()), [])
        restored = restore_compiled(c.state_dict())
        self.assertTrue(torch.equal(c(self.x), restored(self.x)))
        return c

    def test_normalization_folds_into_one_affine_map(self):
        c = self.assert_equivalent(self.base)
        self.assertEqual(c.cost()['stored_scalars'], 8*256)
        self.assertEqual(c.cost()['matrix_multiplications'], 1)
        c.weight.zero_()
        self.assertFalse(torch.equal(c.weight, self.base.weight[:-1]))

    def test_full_rank_interpolation_and_all_256_bytes(self):
        refit = replace(self.base, weight=self.base.weight*2)
        for strength in (0., .125, .5, 1.):
            self.assert_equivalent(FullRankHead(self.base, refit, strength, {}))
        weight = torch.zeros_like(self.base.weight); weight[-1, 233] = 100
        c = compile_readout(replace(self.base, weight=weight))
        self.assertEqual(c.predict(self.x).tolist(), [233]*len(self.x))

    def test_identity_and_gelu_nonzero_corrections(self):
        for activation in ('identity', 'gelu'):
            h = initialize_residual(self.base, hidden=4, activation=activation, seed=91)
            correction = replace(h.correction,
                b1=torch.randn(4, generator=self.g, dtype=torch.double),
                w2=torch.randn(4, 256, generator=self.g, dtype=torch.double),
                b2=torch.randn(256, generator=self.g, dtype=torch.double))
            h = replace(h, correction=correction, strength=.25)
            c = self.assert_equivalent(h)
            self.assertEqual(c.cost()['matrix_multiplications'], 1 if activation == 'identity' else 3)
            self.assert_equivalent(replace(h, strength=0))

    def test_invalid_input_and_export_are_rejected(self):
        c = compile_readout(self.base)
        for x in (torch.zeros(3, 6), torch.full((3, 7), float('inf'))):
            with self.assertRaises(ValueError): c(x)
        with self.assertRaises(ValueError): restore_compiled({'weight': c.weight})
        with self.assertRaises(ValueError): compile_readout(FullRankHead(self.base, self.base, float('nan'), {}))


if __name__ == '__main__':
    unittest.main()
