"""Training-only preservation, feature-only prediction, and validation selection."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from .readout_diagnostic import ByteDecoder, fit_byte_decoder
from .residual_adapter import (balanced_weights, base_fingerprint, fit_residual,
    load_residuals, save_residuals, score_residuals, select_strength)
from .shared_preservation import (answer_margins, shared_anchors, preservation_loss,
    fit_preserving, fit_candidates, choose_penalty)
from .test_residual_adapter import fixture

torch.set_num_threads(2)


class MarginChecks(unittest.TestCase):
    def test_competitor_includes_every_byte_and_excludes_true_byte(self):
        z = torch.zeros(2, 256, dtype=torch.double)
        z[0, 65] = 4; z[0, 233] = 3; z[1, 255] = 5
        self.assertEqual(answer_margins(z, torch.tensor([65, 0])).tolist(), [1., -5.])

    def test_only_correct_shared_training_answers_anchor_and_confidence_is_capped(self):
        z = torch.zeros(4, 256, dtype=torch.double)
        z[:, 65] = torch.tensor([4., .2, 4., -1.])
        mask, target, weight = shared_anchors(z, torch.tensor([65]*4),
            ['shared', 'shared', 'left', 'shared'], torch.ones(4)/4)
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertTrue(torch.allclose(target[:2], torch.tensor([1., .2], dtype=torch.double)))
        self.assertEqual(weight.tolist(), [.5, .5, 0., 0.])

    def test_loss_is_zero_when_margins_hold_and_ignores_private_and_wrong_baseline(self):
        z = torch.zeros(3, 256, dtype=torch.double); z[:, 65] = 2
        y = torch.tensor([65, 65, 66])
        _, target, weights = shared_anchors(z, y, ['shared', 'left', 'shared'], torch.ones(3)/3)
        changed = z.clone(); changed[1:, 65] = -100
        self.assertEqual(float(preservation_loss(changed, y, target, weights)), 0.)
        changed[0, 65] = -.5
        self.assertAlmostEqual(float(preservation_loss(changed, y, target, weights)), 2.25)

    def test_gradient_pushes_answer_up_and_strongest_rival_down(self):
        z = torch.zeros(1, 256, dtype=torch.double); z[0, 233] = 2
        z.requires_grad_(True)
        preservation_loss(z, torch.tensor([65]), torch.ones(1), torch.ones(1)).backward()
        self.assertLess(z.grad[0, 65], 0)
        self.assertGreater(z.grad[0, 233], 0)
        self.assertEqual(z.grad[0, 66], 0)

    def test_no_correct_shared_anchors_is_finite_zero(self):
        z = torch.zeros(3, 256, dtype=torch.double); z[:, 66] = 2
        y = torch.tensor([65]*3)
        mask, target, weights = shared_anchors(z, y, ['shared']*3, torch.ones(3)/3)
        self.assertFalse(mask.any())
        self.assertEqual(float(preservation_loss(z, y, target, weights)), 0.)


class FittingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = fixture('train')
        cls.validation = fixture('validation')
        cls.base = fit_byte_decoder(cls.training['features']['final/raw'], cls.training['answers'],
            sample_weights=balanced_weights(cls.training), max_iter=30)

    def test_zero_penalty_reproduces_prior_optimizer_exactly(self):
        mass = balanced_weights(self.training)
        old = fit_residual(self.training['features']['final/raw'], self.training['answers'], self.base,
            sample_weights=mass, seed=7, max_iter=30)
        new = fit_preserving(self.training, self.base, penalty=0, seed=7, max_iter=30)
        for field in ('w1', 'b1', 'w2', 'b2'):
            self.assertTrue(torch.equal(getattr(old.correction, field), getattr(new.correction, field)))
        self.assertEqual(old.correction.fit['objective'], new.correction.fit['objective'])

    def test_reproducibility_frozen_baseline_and_unchanged_rng(self):
        before = base_fingerprint(self.base); rng = torch.get_rng_state().clone()
        a = fit_preserving(self.training, self.base, max_iter=20)
        b = fit_preserving(self.training, self.base, max_iter=20)
        self.assertTrue(torch.equal(a.logits(self.training['features']['final/raw']),
                                    b.logits(self.training['features']['final/raw'])))
        self.assertEqual(base_fingerprint(self.base), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_preservation_reduces_shared_regression_under_conflicting_training_targets(self):
        # Identical features cannot separate shared A from private B. This tests
        # the actual optimization tradeoff rather than merely its loss formula.
        data = fixture('train')
        data['features']['final/raw'] = torch.zeros(12, 1)
        data['answers'] = torch.tensor([65]*4 + [66]*8)
        w = torch.zeros(2, 256, dtype=torch.double); w[-1, 65] = 1.
        base = ByteDecoder(torch.zeros(1, dtype=torch.double), torch.ones(1, dtype=torch.double), w, {})
        unprotected = fit_preserving(data, base, penalty=0, max_iter=60)
        protected = fit_preserving(data, base, penalty=10, max_iter=60)
        self.assertEqual(unprotected.correction.fit['preservation']['baseline_correct_answers_lost'], 4)
        self.assertEqual(protected.correction.fit['preservation']['baseline_correct_answers_lost'], 0)

    def test_rejects_nontraining_anchors_invalid_penalties_and_overlap(self):
        for penalty in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                fit_preserving(self.training, self.base, penalty=penalty)
        with self.assertRaises(ValueError):
            fit_preserving(self.validation, self.base)
        bad = copy.deepcopy(self.validation); bad['identities'][0] = self.training['identities'][0]
        with self.assertRaises(ValueError):
            fit_candidates(self.training, bad, self.base)
        with self.assertRaises(ValueError):
            fit_candidates(self.training, self.validation, self.base, penalties=(1, 1))

    def test_matched_controls_and_roundtrip_keep_feature_only_interface(self):
        bundles, selected = fit_candidates(self.training, self.validation, self.base,
            seeds=(7,), hidden=4, max_iter=15)
        for bundle in bundles.values():
            a, b = (bundle['heads'][f'{kind}/7'].correction.fit for kind in ('identity', 'gelu'))
            self.assertEqual(a['initial_tensor_sha256'], b['initial_tensor_sha256'])
            self.assertEqual(a['parameters'], b['parameters'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'selected.pt'; save_residuals(path, selected)
            restored = load_residuals(path)
            data = fixture('test'); _, rows = score_residuals(data, restored)
            data['queries'] = ['right']*12; data['controls'] = ['b_only']*12
            data['answers'] = data['answers'].roll(1)
            _, changed = score_residuals(data, restored)
            self.assertEqual([r['predictions'] for r in rows], [r['predictions'] for r in changed])
            for name in selected['heads']:
                self.assertTrue(torch.equal(selected['heads'][name].logits(data['features']['final/raw']),
                    restored['heads'][name].logits(data['features']['final/raw'])))

    def test_joint_selection_uses_validation_objective_and_stable_ties(self):
        heads = {}
        for penalty in (1., 10.):
            h = fit_preserving(self.training, self.base, penalty=penalty, max_iter=5)
            heads[penalty] = select_strength(h, self.validation, fit_identities=self.training['identities'])
        # Equal scores prefer lower strength even when it has the higher penalty.
        for p in heads:
            h = heads[p]
            heads[p] = replace(h, strength=.5 if p == 1 else .25,
                selection={**h.selection, 'selected_weighted_nll': 1.})
        chosen = choose_penalty(heads)
        self.assertEqual(chosen.correction.fit['preservation']['penalty'], 10.)
        heads[1.] = replace(heads[1.], strength=.25)
        self.assertEqual(choose_penalty(heads).correction.fit['preservation']['penalty'], 1.)
        bad = {1.: replace(heads[1.], selection={'split': 'test'})}
        with self.assertRaises(ValueError):
            choose_penalty(bad)


if __name__ == '__main__':
    unittest.main()
