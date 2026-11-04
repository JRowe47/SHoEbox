"""Regional captures, training-only fits, valid subsets, and no mutation."""
from dataclasses import asdict, replace
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .config import GraphConfig
from .model import RegionLanguageModel
from .transfer_eval import (capture_transfer, fit_transfer_readouts, evaluate_transfer,
                            load_transfer_readouts)
from .view_data import make_view_batch
from .view_eval import evaluation_mode
from .view_train import tensor_fingerprint

torch.set_num_threads(2)


class PortModel(nn.Module):
    """Small causal byte memory; no factor labels or query metadata are inputs."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.child = nn.Dropout()
        self.register_buffer('utility_memory', torch.tensor([.4, -.7]))
        self.cfg = GraphConfig.tiny()

    def forward(self, inputs, *, state=None, reset=None, availability=None):
        b, t = inputs['A'].shape
        if state is None:
            memories = {name: torch.zeros(b, 4) for name in 'ABC'}
            known = {name: torch.zeros(b, dtype=torch.bool) for name in 'ABC'}
        else:
            memories = {name: value.clone() for name, value in state.memories.items()}
            known = {name: value.clone() for name, value in state.known.items()}
        for event in range(t):
            for name in 'ABC':
                memories[name][reset[:, event]] = 0
                known[name][reset[:, event]] = False
                value = inputs[name][:, event]-65
                write = availability[name][:, event] & (value >= 0) & (value < 4)
                memories[name][write] = F.one_hot(value[write], 4).float()
                known[name] |= write
        ports = {}
        for name, sources in {'A': 'A', 'B': 'B', 'C': 'C', 'U': 'AB', 'P': 'AB', 'V': 'BC', 'Q': 'BC'}.items():
            ports[name] = SimpleNamespace(payload=sum(memories[source] for source in sources),
                                          valid=torch.stack([known[source] for source in sources]).any(0))
        for name in 'ABC':
            ports['@'+name] = SimpleNamespace(payload=memories[name], valid=availability[name][:, -1])
        logits = torch.zeros(b, t, 256); logits[..., 65] = 1
        return {'state': SimpleNamespace(memories=memories, known=known, ports=SimpleNamespace(latest=ports)),
                'logits': logits, 'expert_executions': b*t*7, 'external_events': b*t}


class TransferCaptureChecks(unittest.TestCase):
    def test_actual_ports_and_chunk_splits_preserve_outputs_work_and_model(self):
        torch.manual_seed(612)
        model = RegionLanguageModel(GraphConfig.tiny(evidence_gated_ports=True)).train()
        batch = make_view_batch('test', 0, 2, gap=0, query_type='shared')
        fingerprint = tensor_fingerprint(model)
        capture = capture_transfer(model, batch, segment=5)
        with evaluation_mode(model):
            direct = model(batch.inputs, reset=batch.reset, availability=batch.availability)
            prefix = batch.slice(0, batch.description_end)
            before_query = model(prefix.inputs, reset=prefix.reset, availability=prefix.availability)
        torch.testing.assert_close(capture['logits'], direct['logits'][:, -1], rtol=0, atol=0)
        self.assertEqual(capture['expert_executions'], direct['expert_executions'])
        for moment, result in (('prequery', before_query), ('final', direct)):
            for name, record in capture['snapshots'][moment].items():
                port = result['state'].ports.latest[name]
                expected = torch.where(port.valid[:, None], port.payload, torch.zeros_like(port.payload))
                torch.testing.assert_close(record['payload'], expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(record['valid'], port.valid))
        self.assertEqual(tensor_fingerprint(model), fingerprint)
        self.assertTrue(model.training)

    def test_invalid_publications_are_zero_and_not_scored_as_retained_facts(self):
        model = PortModel()
        batch = make_view_batch('independent', 0, 4, gap=0, control='b_only', query_type='shared')
        result = capture_transfer(model, batch)
        for moment in ('prequery', 'final'):
            self.assertFalse(bool(result['snapshots'][moment]['A']['valid'].any()))
            self.assertEqual(float(result['snapshots'][moment]['A']['payload'].abs().sum()), 0)
        self.assertTrue(bool(result['snapshots']['final']['@B']['valid'].all()))


class TransferFitChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PortModel().train()
        cls.model.child.eval()
        cls.readouts = fit_transfer_readouts(cls.model, count=96, batch_size=17, segment=7, gap=0)

    def test_matching_training_controls_and_supported_factors_only(self):
        self.assertEqual(self.readouts.fit_split, 'train')
        self.assertEqual(len(self.readouts.fit_episode_ids), 96)
        self.assertEqual(len(set(self.readouts.fit_expert_executions.values())), 1)
        for key in self.readouts.fits:
            control, moment, node, factor = key.split('|')
            if control == 'b_only':
                self.assertEqual(factor, 'shared')
            if node == '@B':
                self.assertEqual(factor, 'shared')
        missing = 'b_only|final|A|shared'
        self.assertEqual(self.readouts.fits[missing]['valid_episodes'], 0)
        self.assertNotIn(missing, self.readouts.heads)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.child.training)
        with self.assertRaises(ValueError):
            fit_transfer_readouts(self.model, split='test', count=96)

    def test_full_factorial_controls_and_query_matched_scores(self):
        fingerprint = tensor_fingerprint(self.model)
        rng = torch.get_rng_state().clone()
        result = evaluate_transfer(self.model, self.readouts, count_per_query=64, batch_size=23, segment=7, gap=0)
        self.assertTrue(result['model_tensors_unchanged'])
        self.assertEqual(len({row['expert_executions'] for row in result['controls'].values()}), 1)
        self.assertEqual(len({tuple(row['episode_ids']) for row in result['controls'].values()}), 1)
        for row in result['controls'].values():
            self.assertEqual(row['direct_output']['exact_accuracy'], .25)
        b = result['controls']['b_only']['publications']['final']
        self.assertEqual(b['A']['valid_episodes'], 0)
        self.assertIsNone(b['A']['factors']['shared']['pooled'])
        score = b['@B']['factors']['shared']
        self.assertEqual(score['pooled']['episodes'], 192)
        self.assertEqual(score['query_matched']['episodes'], 64)
        self.assertEqual(score['pooled']['exact_accuracy'], 1.)
        self.assertEqual(score['query_matched']['exact_accuracy'], 1.)
        self.assertEqual(score['query_matched_fit']['exact_accuracy'], 1.)
        self.assertEqual(score['fit']['query_matched_fit']['episodes'], 32)
        self.assertEqual(result['inference_policy']['evidence_gated_ports'], False)
        self.assertEqual(tensor_fingerprint(self.model), fingerprint)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_heldout_identity_model_match_and_safe_artifact_roundtrip(self):
        result = evaluate_transfer(self.model, self.readouts, split='test', count_per_query=64, gap=0)
        self.assertEqual(tuple(result['controls']), ('all',))
        with self.assertRaises(ValueError):
            evaluate_transfer(self.model, self.readouts, split='test', controls=('b_only',), count_per_query=64)
        with self.assertRaises(ValueError):
            evaluate_transfer(self.model, replace(self.readouts, model_tensor_sha256='wrong'), count_per_query=64)
        original_config = self.model.cfg
        self.model.cfg = replace(original_config, evidence_gated_ports=True)
        try:
            self.assertEqual(tensor_fingerprint(self.model), self.readouts.model_tensor_sha256)
            with self.assertRaises(ValueError):
                evaluate_transfer(self.model, self.readouts, count_per_query=64)
        finally:
            self.model.cfg = original_config
        overlap = make_view_batch('test', 0, 1, query_type='shared').identities
        with self.assertRaises(ValueError):
            evaluate_transfer(self.model, replace(self.readouts, fit_episode_ids=overlap), split='test', count_per_query=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'readouts.pt'
            payload = {'transfer_readout_schema': 1, **asdict(self.readouts)}
            torch.save(payload, path)
            restored = load_transfer_readouts(path)
            self.assertEqual(restored.fit_episode_ids, self.readouts.fit_episode_ids)
            key = 'b_only|final|@B|shared'
            features = torch.eye(4)
            self.assertTrue(torch.equal(restored.heads[key].predict(features), self.readouts.heads[key].predict(features)))
            self.assertTrue(torch.equal(restored.heads[key+'|matched'].predict(features),
                                        self.readouts.heads[key+'|matched'].predict(features)))
            malformed = []
            bad = copy.deepcopy(payload); bad['heads'][key]['weight'] = torch.full_like(bad['heads'][key]['weight'], float('nan'))
            malformed.append(bad)
            bad = copy.deepcopy(payload); bad['heads'][key]['weight'] = bad['heads'][key]['weight'][:-1]
            malformed.append(bad)
            malformed.append({**payload, 'fit_split': 'test'})
            malformed.append({**payload, 'fit_episode_ids': ('duplicate',)*96})
            malformed.append({**payload, 'transfer_readout_schema': 2})
            for bad in malformed:
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    load_transfer_readouts(path)


if __name__ == '__main__':
    unittest.main()
