"""Optional retention policies preserve the original stream and state contracts."""
from dataclasses import replace
import io
import unittest
import torch
from .ports import tensor_list
from .task_memory import TaskMemoryConfig, TaskMemoryModel


torch.set_num_threads(1)


def config(**options):
    values = dict(width=32, owners=3, experts_per_block=32, shared_per_block=16,
                  groups=4, address_dim=8, laces=16, hat_symbols=16)
    values.update(options)
    return TaskMemoryConfig(**values)


def model(**options):
    torch.manual_seed(932)
    return TaskMemoryModel(config(**options))


class RetentionPolicyChecks(unittest.TestCase):
    def same(self, first, second):
        a, b = tensor_list(first), tensor_list(second)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_original_checkpoint_and_reference_predictions(self):
        original = model()
        legacy_config = original.cfg.to_dict()
        for name in ('write_mode', 'memory_aware_write', 'bootstrap_empty'):
            legacy_config.pop(name)
        buffer = io.BytesIO()
        torch.save(dict(config=legacy_config, weights=original.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = TaskMemoryModel(TaskMemoryConfig.from_dict(saved['config']))
        restored.load_state_dict(saved['weights'], strict=True)
        inputs = torch.tensor([list(b'A x=1;'), list(b'B x=3;')])
        with torch.no_grad():
            first, last = original(inputs), restored(inputs)
        self.same(first['logits'], last['logits'])
        self.same(first['state'], last['state'])
        # Captured from released fd4d1ac with this seed/config/input, before the
        # optional retention modes. This fixture does not require git access.
        expected = torch.tensor([
            [-.05267607048153877, .06863126158714294, .012961536645889282, .0332932323217392,
             .02629110962152481, .02329445630311966, -.08350804448127747, -.055951446294784546],
            [.0379847027361393, .026030302047729492, -.23558282852172852, .18188154697418213,
             -.11386363953351974, .021879110485315323, .08717295527458191, .2950882911682129]])
        observed = torch.stack((first['logits'][0, 0, :8], first['logits'][1, -1, :8]))
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(any(k.startswith('memory_policy_features') for k in original.state_dict()))

    def test_optional_memory_features_preserve_common_initialization_and_rng(self):
        base = model()
        after_base = torch.random.get_rng_state()
        aware = model(memory_aware_write=True)
        self.same(after_base, torch.random.get_rng_state())
        for name, value in base.state_dict().items():
            self.same(value, aware.state_dict()[name])
        self.assertEqual(float(aware.memory_policy_features.weight.detach().abs().sum()), 0)
        inputs = torch.tensor([list(b'A x=1;')])
        with torch.no_grad():
            first, second = base(inputs), aware(inputs)
        self.same(first['logits'], second['logits'])
        self.same(first['write_probs'], second['write_probs'])
        self.same(first['state'], second['state'])

    def test_competition_selects_one_owner_or_null_without_parameters(self):
        learner = model(write_mode='competitive')
        self.assertEqual(set(model().state_dict()), set(learner.state_dict()))
        with torch.no_grad():
            learner.write_policy.zero_()
            learner.write_policy[:, -1] = -4
            empty = learner(torch.tensor([[65, 66]]))
            self.assertFalse(bool(empty['write_actions'].any()))
            self.assertTrue(bool((empty['write_probs'].sum(-1) < 1).all()))
            learner.write_policy[1, -1] = 4
            selected = learner(torch.tensor([[65, 66]]))
        self.assertTrue(bool(selected['write_actions'].sum(-1).eq(1).all()))
        self.assertTrue(bool(selected['write_actions'][:, :, 1].all()))
        self.assertEqual(selected['write_mode'], 'competitive')

    def test_bootstrap_recruits_dead_owners_once_and_excludes_policy_credit(self):
        for mode in ('independent', 'competitive'):
            learner = model(write_mode=mode, bootstrap_empty=True)
            with torch.no_grad():
                learner.write_policy.zero_()
                learner.write_policy[:, -1] = -20
            result = learner(torch.tensor([[65, 66, 67, 68, 69]]))
            self.same(result['write_actions'][0, :3], torch.eye(3, dtype=torch.bool))
            self.assertFalse(bool(result['write_actions'][:, 3:].any()))
            self.assertEqual(result['bootstrap_mask'].tolist(), [[True, True, True, False, False]])
            self.assertFalse(bool(result['write_eligible'][:, :3].any()))
            self.assertTrue(bool(result['memory_valid'][:, -1].all()))
            single = learner(torch.tensor([[65]]))
            single['state'].memory_tensor().sum().backward()
            self.assertEqual(float(learner.write_policy.grad.abs().sum()), 0)

    def test_bootstrap_is_per_stream_and_override_hold_preserves_empty_state(self):
        learner = model(write_mode='competitive', bootstrap_empty=True)
        initial = learner.initial_state(2)
        held = learner(torch.tensor([[65, 66], [67, 68]]), state=initial, write_override='hold')
        self.same(initial.publications, held['state'].publications)
        self.assertFalse(bool(held['bootstrap_mask'].any()))
        self.assertFalse(bool(held['write_eligible'].any()))
        with torch.no_grad():
            filled = learner(torch.tensor([[65], [66]]), write_override='always')['state']
            mixed = filled.reset(torch.tensor([True, False]))
            learner.write_policy.zero_()
            learner.write_policy[:, -1] = -20
            result = learner(torch.tensor([[67], [68]]), state=mixed)
        self.assertEqual(result['bootstrap_mask'].tolist(), [[True], [False]])
        self.assertEqual(result['write_actions'].sum(-1).tolist(), [[1], [0]])
        self.assertFalse(bool(result['write_eligible'][0].any()))
        self.same(mixed.publications['M1'], result['state'].publications['M1'])

    def test_memory_conditioning_reads_detached_owned_state_and_does_not_mutate(self):
        learner = model(memory_aware_write=True, bootstrap_empty=True)
        with torch.no_grad():
            prefix = learner(torch.tensor([list(b'A x=1;'), list(b'B x=2;')]),
                             write_override='always')['state'].detach()
            learner.memory_policy_features.weight.fill_(.01)
        leaf = prefix.memory_tensor().detach().requires_grad_()
        state = prefix.with_memory_tensor(leaf)
        snapshot = state.clone()
        result = learner(torch.tensor([[63], [63]]), state=state, control='shuffle')
        result['write_features'].sum().backward()
        self.assertIsNone(leaf.grad)
        self.assertGreater(float(learner.memory_policy_features.weight.grad.abs().sum()), 0)
        self.same(state, snapshot)
        # Read interventions do not change what the writer knows about its own
        # publications, even when the read values come from another stream.
        with torch.no_grad():
            intact = learner(torch.tensor([[63], [63]]), state=state)
        self.same(result['write_features'], intact['write_features'])

    def test_policy_and_bootstrap_metadata_survive_disabled_capture(self):
        learner = model(write_mode='competitive', bootstrap_empty=True, memory_aware_write=True)
        with torch.no_grad():
            result = learner(torch.tensor([[65]]), capture=False)
        self.assertEqual(result['write_mode'], 'competitive')
        self.assertTrue(result['bootstrap_empty'])
        self.assertTrue(result['memory_aware_write'])
        self.assertEqual(result['bootstrap_mask'].tolist(), [[True]])
        self.assertNotIn('write_probs', result)

    @torch.no_grad()
    def test_fresh_reads_current_experts_independent_of_publications(self):
        learner = model(write_mode='competitive', bootstrap_empty=True, memory_aware_write=True)
        learner.memory_policy_features.weight.fill_(.03)
        prefix = learner(torch.tensor([list(b'A x=1;'), list(b'B x=2;')]),
                         write_override='always')['state'].detach()
        original = prefix.clone()
        changed = prefix.with_memory_tensor(prefix.memory_tensor()*-7+3)
        changed = replace(changed, publications={name: replace(pub, valid=torch.zeros_like(pub.valid))
                                                for name, pub in changed.publications.items()})
        query = torch.tensor([list(b'A x?;'), list(b'B x?;')])
        a = learner(query, state=prefix, control='fresh')
        b = learner(query, state=changed, control='fresh')
        self.same(a['logits'], b['logits'])
        self.same(prefix, original)
        self.assertEqual(a['read_source'], 'candidate')
        self.assertTrue(bool(a['read_valid'].all()))
        self.assertTrue(bool(a['read_is_current'].all()))
        self.assertTrue(bool(a['read_versions'].eq(-1).all()))
        intact = learner(query, state=prefix)
        self.assertEqual(a['expert_executions'], intact['expert_executions'])
        self.assertEqual(a['external_events'], intact['external_events'])

    def test_invalid_write_mode_rejected(self):
        with self.assertRaises(ValueError):
            TaskMemoryModel(config(write_mode='unknown'))


if __name__ == '__main__':
    unittest.main()
