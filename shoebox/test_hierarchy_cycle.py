"""Freeze/thaw must preserve temporal execution, credit and exact continuation."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
import torch
from .checkpoint import save, load
from .hierarchy import HierarchyModel
from .hierarchy_training import CycleConfig, RegionalCycle, parameter_role, regional_optimizer
from .test_hierarchy import tiny
from .train_fineweb import optimizer_for
from .train_hierarchy import train_update, evaluate


class RegionalCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.model = HierarchyModel(tiny()).set_execution_backend('cpu')
        self.raw = torch.randint(256, (2, 65))

    def cycle(self, **kwargs):
        cycle = RegionalCycle(CycleConfig(mode='cycle', warmup=0, upper=2, lower=2, joint=2, **kwargs))
        cycle.observe(dict(mid_nll=5.5, high_nll=5.5))
        for _ in range(4):
            cycle.observe(dict(mid_nll=3.5, high_nll=3.5))
        return cycle

    def update(self, model, opt, state=None, start=0, scale=1.):
        return train_update(model, opt, self.raw[:, start:start+16], self.raw[:, start+1:start+17],
                            state, self.raw[:, start+1:start+32], decoder_local_scale=scale)

    def test_frozen_weights_moments_and_clocks_then_thaw(self):
        opt = regional_optimizer(self.model, .001)
        state, _ = self.update(self.model, opt)
        cycle = self.cycle()
        for step, active in ((0, {'upper', 'heads', 'reader'}), (2, {'fast'})):
            cycle.apply(self.model, opt, step, .001)
            before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
            moments = {n: deepcopy(opt.state[p]) for n, p in self.model.named_parameters() if p in opt.state}
            previous_clocks = {n: s.clock for n, s in state.nodes.items()}
            state, _ = self.update(self.model, opt, state)
            changed = set()
            for name, p in self.model.named_parameters():
                role = parameter_role(name)
                if role not in active:
                    torch.testing.assert_close(p, before[name], rtol=0, atol=0, msg=name)
                    self.assertIsNone(p.grad, name)
                    for key, value in moments.get(name, {}).items():
                        torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0, msg=name)
                elif not torch.equal(p, before[name]):
                    changed.add(role)
            self.assertEqual(changed, active)
            self.assertTrue(all(s.clock > previous_clocks[n] for n, s in state.nodes.items()))
        upper = self.model.regions['L3'].publish.weight
        age = opt.state[upper]['step'].clone()
        cycle.apply(self.model, opt, 4, .001)
        self.update(self.model, opt, state)
        self.assertEqual(float(opt.state[upper]['step']), float(age)+1)

    def test_frozen_upper_transmits_future_loss_to_lower_inputs(self):
        opt = regional_optimizer(self.model, .001)
        self.cycle().apply(self.model, opt, 2, .001)
        out = self.model(self.raw[:, :32], self.raw[:, 1:33], future_targets=self.raw[:, 1:48])
        out['span_losses']['high'].backward()
        for name in ('A1', 'B1'):
            self.assertGreater(float(self.model.regions[name].publish.weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.regions['L3'].publish.weight.grad)
        self.assertIsNone(self.model.future_heads['high'].weight.grad)

    def test_frozen_lower_transmits_feedback_credit_to_upper(self):
        opt = regional_optimizer(self.model, .001)
        self.cycle().apply(self.model, opt, 0, .001)
        out = self.model(self.raw[:, :32])
        # This loss sees no decoder or future head. Credit must pass through
        # the frozen A1 computation into its learned L2 top-down context.
        out['state'].latest['A1'].value.square().mean().backward()
        self.assertGreater(float(self.model.regions['L2'].publish.weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.regions['A1'].publish.weight.grad)

    def test_decoder_attenuation_preserves_producers_and_temporal_execution(self):
        with torch.no_grad():
            a = self.model(self.raw[:, :32])
            b = self.model(self.raw[:, :32], decoder_local_scale=.5)
        for name in ('A1', 'B1', 'L2', 'S2', 'L3'):
            torch.testing.assert_close(a['state'].latest[name].value, b['state'].latest[name].value, rtol=0, atol=0)
        self.assertFalse(torch.equal(a['logits'], b['logits']))
        self.assertEqual(a['work'], b['work'])

    def test_full_upper_intervention_includes_causal_feedback(self):
        with torch.no_grad():
            a = self.model(self.raw[:, :32])
            b = self.model(self.raw[:, :32], feedback_permutation=torch.tensor([1, 0]))
        torch.testing.assert_close(a['logits'][:, :4], b['logits'][:, :4], rtol=0, atol=0)
        self.assertFalse(torch.equal(a['state'].latest['A1'].value, b['state'].latest['A1'].value))
        self.assertFalse(torch.equal(a['state'].latest['L3'].value, b['state'].latest['L3'].value))

    def test_legacy_optimizer_migration_preserves_next_update(self):
        old = optimizer_for(self.model, .001, .01)
        state, _ = self.update(self.model, old)
        other = HierarchyModel(tiny()).set_execution_backend('cpu')
        other.load_state_dict(self.model.state_dict())
        opt = regional_optimizer(other, .001, legacy_state=deepcopy(old.state_dict()))
        self.update(self.model, old, state, 16)
        self.update(other, opt, state, 16)
        for (name, p), q in zip(self.model.named_parameters(), other.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0, msg=name)
            if p in old.state:
                for key in old.state[p]:
                    torch.testing.assert_close(old.state[p][key], opt.state[q][key], rtol=0, atol=0)

    def test_exact_resume_across_all_phase_and_attenuation_boundaries(self):
        opt = regional_optimizer(self.model, .001)
        cycle = self.cycle(attenuation_every=2)
        cycle.apply(self.model, opt, 0, .001)
        state, row = self.update(self.model, opt)
        cycle.observe(row)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cycle.pt'
            save(path, self.model, state, opt, step=1, metadata=dict(cycle=cycle.state_dict()))
            other, restored, _, record = load(path)
            other.set_execution_backend('cpu')
            other_opt = regional_optimizer(other, .001)
            other_opt.load_state_dict(record['optimizer'])
            other_cycle = RegionalCycle(cycle.config)
            other_cycle.load_state_dict(record['metadata']['cycle'])
            for step in range(1, 7):
                plan = cycle.apply(self.model, opt, step, .001)
                self.assertEqual(plan, other_cycle.apply(other, other_opt, step, .001))
                state, a = self.update(self.model, opt, state, scale=plan['local_scale'])
                restored, b = self.update(other, other_opt, restored, scale=plan['local_scale'])
                cycle.observe(a); other_cycle.observe(b)
                self.assertEqual(a['nll'], b['nll'])
                self.assertEqual(cycle.state_dict(), other_cycle.state_dict())
                for p, q in zip(self.model.parameters(), other.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                torch.testing.assert_close(state.latest['L3'].value, restored.latest['L3'].value, rtol=0, atol=0)

    def test_lower_phase_waits_for_both_span_objectives(self):
        cycle = RegionalCycle(CycleConfig(mode='cycle', warmup=0, upper=2, lower=2, joint=2))
        cycle.observe(dict(mid_nll=5., high_nll=5.))
        for _ in range(10):
            cycle.observe(dict(mid_nll=3., high_nll=5.))
        self.assertEqual(cycle.plan(2)['phase'], 'joint_wait')
        for _ in range(10):
            cycle.observe(dict(mid_nll=3., high_nll=3.))
        self.assertEqual(cycle.plan(2)['phase'], 'lower')
        self.assertEqual(cycle.plan(4)['fast_lr_scale'], .5)

    def test_span_evaluation_matches_document_weighted_scores(self):
        docs = [self.raw[0, :39], self.raw[1, :25]]
        separate = [evaluate(self.model, [d], context=16, span_metrics=True) for d in docs]
        combined = evaluate(self.model, docs, context=16, span_metrics=True)
        for scale in ('mid', 'high'):
            count = sum(r['span_targets'][scale] for r in separate)
            expected = sum(r['span_nll'][scale]*r['span_targets'][scale] for r in separate)/count
            self.assertEqual(combined['span_targets'][scale], count)
            self.assertAlmostEqual(combined['span_nll'][scale], expected, places=5)


if __name__ == '__main__':
    unittest.main()
