import copy
import io
import unittest
import torch
from torch.nn import functional as F
from model import Config, SHoEGPT, semantic_reads


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = SHoEGPT().eval()
        self.tokens = torch.randint(65, (2, 23))

    def test_size_and_membership(self):
        summary = self.model.parameter_summary()
        self.assertEqual(summary['total'], 1000732)
        self.assertEqual(sum(summary['expert_counts']), 1024)
        for block, count in zip(self.model.blocks, self.model.cfg.expert_counts):
            ids = block.router.membership.flatten()
            self.assertTrue(torch.equal(ids[ids >= 0].sort().values, torch.arange(count)))

    def test_gradients_reach_mechanisms(self):
        result = self.model(self.tokens, self.tokens.roll(-1, 1))
        (result['ce'] + .01*result['aux']).backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_task_loss_trains_laces_and_addresses(self):
        self.model(self.tokens, self.tokens.roll(-1, 1))['ce'].backward()
        for name in ['lace.template_logits', 'lace.context_query.weight',
                     'blocks.0.router.prototypes', 'blocks.1.router.expert_residual',
                     'blocks.2.router.key.weight']:
            self.assertGreater(float(dict(self.model.named_parameters())[name].grad.abs().sum()), 0, name)

    def test_causal_suffix_and_prefix(self):
        changed = self.tokens.clone()
        changed[:, 13:] = torch.randint(65, changed[:, 13:].shape)
        with torch.no_grad():
            full = self.model(self.tokens)['logits'][:, :13]
            altered = self.model(changed)['logits'][:, :13]
            short = self.model(self.tokens[:, :13])['logits']
        torch.testing.assert_close(full, altered, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(full, short, atol=2e-6, rtol=2e-6)

    def test_postings_match_reference(self):
        assignments = torch.randint(5, (2, 41))
        groups = torch.rand(2, 41, 5).topk(2, -1).indices
        reads, valid = semantic_reads(assignments, groups, 7, 3)
        for b in range(2):
            for t in range(41):
                expected = list(range(t, max(-1, t-7), -1))
                for group in groups[b, t]:
                    matches = [j for j in range(t-7, -1, -1) if assignments[b, j] == group][:3]
                    expected += matches
                actual = reads[b, t][valid[b, t]].tolist()
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), len(set(actual)))
                self.assertTrue(all(j <= t for j in actual))

    def test_sparse_attention_matches_dense_reference(self):
        block = self.model.blocks[0]
        x = torch.randn(2, 23, 100)
        prior = torch.full((2, 23, 32), 1/32)
        route = block.router(x, prior)
        actual, mem = block.attention(x, route)
        q, k, v = block.attention.qkv(x).reshape(2, 23, 3, 4, 25).unbind(2)
        score = torch.einsum('bthd,bshd->btsh', q, k)/5
        score += .25*torch.einsum('btd,bsd->bts', route['query'], route['key'])[..., None]
        source_groups = route['assignments'][:, None].expand(-1, 23, -1)
        score += .1*route['group_scores'].gather(-1, source_groups)[..., None]
        mask = torch.zeros(2, 23, 23, dtype=torch.bool)
        for b in range(2):
            for t in range(23):
                mask[b, t, mem['reads'][b, t][mem['valid'][b, t]]] = True
        weights = score.masked_fill(~mask[..., None], -torch.inf).softmax(2)
        expected = block.attention.output(torch.einsum('btsh,bshd->bthd', weights, v).reshape(2,23,100))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_selected_expert_arithmetic_and_inactive_gradients(self):
        bank = self.model.blocks[0].experts
        x = torch.randn(1, 2, 100)
        ids = torch.tensor([[[0, 9], [9, 42]]])
        gates = torch.tensor([[[.3, .7], [.4, .6]]])
        actual = bank(x, ids, gates)
        expected = torch.stack([sum(F.gelu(x[0,t] @ bank.up[e].T) @ bank.down[e]*gates[0,t,i]
                                    for i,e in enumerate(ids[0,t]))/2**.5 for t in range(2)])[None]
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        active = bank.up.grad.abs().sum((1,2)).nonzero().flatten().tolist()
        self.assertEqual(active, [0,9,42])

    def test_cross_layer_message_intervention(self):
        evidence = torch.randn(1, 5, 3, 32)
        context = torch.randn(1, 5, 100)
        prior, _ = self.model.lace(evidence, context)
        changed = evidence.clone()
        changed[:, :, 2, 7] += 12
        other, _ = self.model.lace(changed, context)
        self.assertGreater(float((prior[:, :, 0]-other[:, :, 0]).abs().max().detach()), 1e-3)

    def test_context_changes_routes_for_same_current_token(self):
        a = torch.zeros(1, 23, dtype=torch.long)
        b = torch.ones(1, 23, dtype=torch.long)
        a[:, -1] = b[:, -1] = 17
        with torch.no_grad():
            aa, bb = self.model(a, trace=True), self.model(b, trace=True)
        self.assertGreater(float((aa['trace'][0]['group_scores'][:, -1]-bb['trace'][0]['group_scores'][:, -1]).abs().max()), .01)

    def test_ablation_and_roundtrip(self):
        with torch.no_grad():
            a = self.model(self.tokens, lace_enabled=False)['logits']
            b = self.model(self.tokens, lace_enabled=False, permutation=torch.randperm(32))['logits']
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            stream = io.BytesIO()
            torch.save(self.model.state_dict(), stream)
            stream.seek(0)
            other = SHoEGPT()
            other.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(self.model(self.tokens)['logits'], other(self.tokens)['logits'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
