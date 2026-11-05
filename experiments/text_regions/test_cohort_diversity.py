"""Nested-cohort coverage, frozen normalization, split and resume safeguards."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .cohort_diversity import (cohort_coverage, nested_prefix, validate_splits, head_record,
    restore_head, verify_resume, fit_key, SETTINGS)
from .readout_diagnostic import ByteDecoder, SUBSETS, controlled_batch, visible
from .readout_compare import expected_cohort
from .residual_adapter import base_fingerprint, save_tensors, select_strength
from .shared_preservation import fit_preserving
from .view_eval import FACTORS
from .test_residual_adapter import fixture

torch.set_num_threads(2)


def fake_capture(count):
    protocol = {'split': 'train', 'count_per_cell': count, 'seed': 20261120,
        'gap': 8, 'query_every': 4, 'batch_size': 64, 'segment': 16,
        'model_tensor_sha256': 'fixture', 'policy': {}, 'frozen_verified': True}
    answers = []; ids = []; queries = []; controls = []; groups = []; cursor = 0
    for c in SUBSETS:
        for q in FACTORS:
            if not visible(c, q): continue
            b = controlled_batch('train', 0, count, control=c, query=q,
                seed=20261120, gap=8, query_every=4)
            answers.append(b.answers); ids.extend(b.identities)
            queries.extend([q]*count); controls.extend([c]*count)
            groups.append({'control': c, 'query': q, 'start': cursor, 'stop': cursor+count,
                'expert_executions': count*25*84, 'external_events': count*25,
                'information_present': True})
            cursor += count
    expected = expected_cohort(protocol)
    protocol.update({k: expected[k] for k in ('case_sha256', 'inputs_sha256', 'episodes')})
    protocol.update(expert_executions=cursor*25*84, external_events=cursor*25)
    x = torch.arange(cursor*64, dtype=torch.float32).reshape(cursor, 64)/cursor
    y = torch.cat(answers)
    return {'protocol': protocol, 'features': {'final/raw': x, 'final/mixed': x[:, :32].clone()},
        'answers': y, 'identities': ids, 'queries': queries, 'controls': controls, 'groups': groups,
        'direct_predictions': y.clone(), 'direct_nll_per_case': torch.zeros(cursor)}


class CohortChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.large = fake_capture(256)

    def test_coverage_matches_preserved_protocol(self):
        for c, n, ids, signatures in [(64, 960, 192, 869), (256, 3840, 768, 2360)]:
            r = cohort_coverage({**self.large['protocol'], 'count_per_cell': c})
            self.assertEqual((r['cases'], r['distinct_episode_ids'], r['observed_template_signatures']),
                             (n, ids, signatures))
            self.assertEqual(r['full_fact_combinations'], 32)
            self.assertEqual(r['traces_without_id_headers'], n)

    def test_small_arm_is_exact_prefix_of_every_cell(self):
        small = nested_prefix(self.large, 64)
        self.assertEqual(small['protocol']['episodes'], 960)
        self.assertEqual(small['protocol']['expert_executions'], 960*25*84)
        for a, b in zip(small['groups'], self.large['groups']):
            self.assertTrue(torch.equal(small['features']['final/raw'][a['start']:a['stop']],
                self.large['features']['final/raw'][b['start']:b['start']+64]))
        expected = expected_cohort(small['protocol'])
        self.assertEqual(small['protocol']['inputs_sha256'], expected['inputs_sha256'])
        self.assertEqual(small['protocol']['case_sha256'], expected['case_sha256'])

    def test_prefix_owns_tensors_and_full_prefix_is_identical(self):
        before = self.large['features']['final/raw'].clone()
        small = nested_prefix(self.large, 64); small['features']['final/raw'].zero_()
        self.assertTrue(torch.equal(before, self.large['features']['final/raw']))
        full = nested_prefix(self.large, 256)
        self.assertTrue(torch.equal(before, full['features']['final/raw']))
        self.assertEqual(full['protocol'], self.large['protocol'])

    def test_invalid_counts_and_malformed_groups_fail(self):
        for count in (0, 33, 512, 64.0):
            with self.assertRaises(ValueError): nested_prefix(self.large, count)
        bad = copy.deepcopy(self.large); bad['groups'][0]['stop'] -= 1
        with self.assertRaises(ValueError): nested_prefix(bad, 64)

    def test_validation_and_evaluation_identity_leakage_fails(self):
        train, valid = fixture('train'), fixture('validation')
        evaluations = {s: fixture(s) for s in ('test', 'independent')}
        validate_splits(train, valid, evaluations)
        for split in evaluations:
            bad = copy.deepcopy(evaluations); bad[split]['identities'][0] = train['identities'][0]
            with self.assertRaises(ValueError): validate_splits(train, valid, bad)
        bad = copy.deepcopy(valid); bad['identities'][0] = train['identities'][0]
        with self.assertRaises(ValueError): validate_splits(train, bad, evaluations)

    def test_split_policy_or_missing_evaluation_fails(self):
        train, valid = fixture('train'), fixture('validation')
        evaluations = {s: fixture(s) for s in ('test', 'independent')}
        evaluations['test']['protocol']['model_tensor_sha256'] = 'different'
        with self.assertRaises(ValueError): validate_splits(train, valid, evaluations)
        with self.assertRaises(ValueError): validate_splits(train, valid, {})


class ResumeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = fake_capture(64)
        cls.base = ByteDecoder(torch.zeros(64, dtype=torch.double), torch.ones(64, dtype=torch.double),
                               torch.zeros(65, 256, dtype=torch.double), {})
        head = fit_preserving(data, cls.base, penalty=0., activation='identity', max_iter=3)
        valid = copy.deepcopy(data); valid['protocol']['split'] = 'validation'
        valid['identities'] = ['validation/'+x for x in data['identities']]
        head = select_strength(head, valid, fit_identities=data['identities'])
        cls.record = head_record(head)
        cls.invocation = {'settings': 'unit-fixture', 'source_sha256': 'fixed'}
        cls.saved = {'invocation': cls.invocation, 'capture_sha256': 'capture', 'base': asdict(cls.base),
            'heads': {fit_key(64, 0., 'identity', 20261121): cls.record}}

    def test_expanded_data_does_not_refit_normalization(self):
        h = restore_head(self.record, self.base)
        self.assertEqual(base_fingerprint(h.base), base_fingerprint(self.base))
        self.assertTrue(torch.equal(h.correction.mean, self.base.mean))
        self.assertTrue(torch.equal(h.correction.scale, self.base.scale))

    def test_partial_fit_roundtrip_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'fitted.pt'; save_tensors(p, self.saved)
            restored = torch.load(p, weights_only=True)
            with patch.dict(SETTINGS, {'max_iter': 3}):
                verify_resume(restored, self.invocation, 'capture', self.base)
                with self.assertRaises(ValueError): verify_resume(restored, self.invocation, 'changed', self.base)
                with self.assertRaises(ValueError): verify_resume(restored, {'settings': 'changed'}, 'capture', self.base)
            self.assertEqual(restore_head(restored['heads'][next(iter(restored['heads']))], self.base).strength,
                             self.record['strength'])

    def test_corrupt_tensor_or_decision_is_rejected(self):
        for kind in ('shape', 'nonfinite', 'decision'):
            r = copy.deepcopy(self.record)
            if kind == 'shape': r['correction']['w1'] = r['correction']['w1'][:1]
            elif kind == 'nonfinite': r['correction']['w1'][0, 0] = float('nan')
            else: r['strength'] = .3
            with self.assertRaises(ValueError): restore_head(r, self.base)


if __name__ == '__main__':
    unittest.main()
