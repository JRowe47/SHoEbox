"""Delayed payload credit: independent autograd oracle and live training checks."""
import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from .ports import tensor_list
from .task_memory import TaskMemoryModel
from .task_payload import PayloadAdapter, PayloadEligibility, make_eligibility
from .test_task_retention_integration import model_config

torch.set_num_threads(1)


class PayloadChecks(unittest.TestCase):
    def model(self, rank=4):
        torch.manual_seed(51)
        return TaskMemoryModel(replace(model_config(), payload_rank=rank))

    def test_identity_preserves_legacy_forward_initialization_and_rng(self):
        old = self.model(0)
        old_rng = torch.get_rng_state().clone()
        new = self.model()
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        inputs = torch.tensor([list(b'A x=2; A x?'), list(b'B x=3; B x?')])
        a, b = old(inputs), new(inputs)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        for left, right in zip(tensor_list(a['state']), tensor_list(b['state'])):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(a['expert_executions'], b['expert_executions'])
        basis = new.payload_adapter.basis
        torch.testing.assert_close(basis @ basis.T, torch.eye(4), rtol=1e-6, atol=1e-6)

    def test_factored_trace_matches_full_history_and_adamw_update(self):
        torch.manual_seed(13)
        raw = torch.randn(6, 2, 2, 8, dtype=torch.float64)
        actions = torch.tensor([[[1, 1], [1, 1]], [[0, 0], [1, 0]],
                                [[1, 0], [0, 0]], [[0, 1], [0, 0]],
                                [[0, 0], [0, 0]], [[0, 0], [1, 0]]]).bool()
        models = [PayloadAdapter(2, 8, 3).double() for _ in range(3)]
        gradients, updates = [], []
        # Full graph, detached + delayed, detached without delayed.
        for mode, adapter in enumerate(models):
            memory = torch.zeros(2, 2, 8, dtype=torch.float64)
            trace = PayloadEligibility.zeros(2, ('A', 'B'), 3, dtype=torch.float64)
            versions = torch.zeros(2, 2, dtype=torch.long)
            receipt = None
            for t in range(6):
                if t == 3 and mode:
                    from .task_payload import PayloadWriteBoundary
                    from .task_credit import CreditBoundary
                    leaf = torch.cat((memory, torch.zeros(2, 2, 2, dtype=memory.dtype)), -1).detach().requires_grad_(True)
                    receipt = PayloadWriteBoundary(CreditBoundary(leaf, torch.zeros(2, 2, 10, 1)),
                                                   trace.features.clone())
                    memory = leaf[..., :8]
                candidate, features = adapter(raw[t])
                memory = torch.where(actions[t][..., None], candidate, memory)
                versions = versions + actions[t].long()
                trace = trace.advance(features, actions[t], versions)
            loss = (memory.sin() * torch.arange(1, 9, dtype=memory.dtype)).sum()
            loss.backward()
            if receipt is not None:
                receipt.apply_payload(adapter, enabled=mode == 1)
                with self.assertRaises(RuntimeError):
                    receipt.apply_payload(adapter)
            gradients.append(adapter.coefficients.grad.clone())
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01)
            optimizer.step()
            updates.append(adapter.coefficients.detach().clone())
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(updates[0], updates[1], rtol=1e-12, atol=1e-12)
        self.assertGreater(float((gradients[1] - gradients[2]).norm()), .01)
        self.assertGreater(float((updates[1] - updates[2]).norm()), .001)

    def test_actual_model_held_publication_gradient_matches_live_publication_graph(self):
        initial = self.model()
        gradients, logits = [], []
        for delayed in (False, True):
            model = copy.deepcopy(initial)
            prefix = model(torch.tensor([list(b'A x=1;'), list(b'A x=3;')]), write_override='always')
            state = prefix['state'].detach()
            if delayed:
                trace = make_eligibility(model, 2).advance_chunk(prefix)
                boundary = trace.capture(state.memory_tensor())
                state = state.with_memory_tensor(boundary.memory)
            else:
                state = replace(state, publications=prefix['state'].publications)
            result = model(torch.tensor([list(b'A x?')] * 2), state=state, write_override='hold')
            result['logits'][:, -1].square().sum().backward()
            if delayed:
                self.assertGreater(boundary.apply_payload(model.payload_adapter), 0)
            gradients.append(model.payload_adapter.coefficients.grad)
            logits.append(result['logits'])
        torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-5, atol=1e-8)

    def test_replacement_reset_bounds_and_serialization(self):
        model = self.model()
        trace = make_eligibility(model, 2)
        result = model(torch.tensor([[65, 66], [67, 68]]), write_override='always')
        trace = trace.advance_chunk(result)
        torch.testing.assert_close(trace.payload.features, result['payload_features'][:, -1])
        held = model(torch.tensor([[69], [70]]), state=result['state'].detach(), write_override='hold')
        after = trace.advance_chunk(held)
        torch.testing.assert_close(trace.payload.features, after.payload.features, rtol=0, atol=0)
        self.assertEqual(trace.payload.storage_bytes(), 2 * model.cfg.owners * (4*4 + 8))
        restored = make_eligibility(model, 2, trace.to_dict())
        torch.testing.assert_close(restored.payload.features, trace.payload.features, rtol=0, atol=0)
        reset = trace.reset(torch.tensor([True, False]))
        self.assertEqual(int(reset.payload.features[0].count_nonzero()), 0)
        self.assertEqual(int(reset.payload.versions[0].count_nonzero()), 0)
        torch.testing.assert_close(reset.payload.features[1], trace.payload.features[1], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            trace.payload.advance(trace.payload.features, torch.ones_like(trace.payload.versions), trace.payload.versions)
        bad = trace.to_dict(); bad['payload']['versions'] += 1
        with self.assertRaises(ValueError):
            make_eligibility(model, 2, bad)
        bad = trace.to_dict(); bad['payload']['features'][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            make_eligibility(model, 2, bad)

    def test_resume_with_payload_credit_across_stage_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
                      '--stream-preset', 'retention', '--curriculum', 'acquisition',
                      '--acquire-steps', '2', '--bridge-steps', '2', '--truncate-every', '8',
                      '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
                      '--payload-rank', '4', '--payload-credit',
                      '--width', '32', '--batch', '2', '--context', '16', '--eval-count', '2',
                      '--eval-every', '2', '--seed', '9', '--data-seed', '42']
            for out, steps, extra in [('full', 5, []), ('split', 3, []),
                                     ('resumed', 5, ['--resume', str(root/'split/last.pt')])]:
                result = subprocess.run(common + ['--out', str(root/out), '--steps', str(steps)] + extra,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            full, resumed = [torch.load(root/name/'last.pt', weights_only=False) for name in ('full', 'resumed')]
            for key in ('weights', 'optimizer', 'state', 'credit', 'rng'):
                for a, b in zip(tensor_list(full[key]), tensor_list(resumed[key])):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in ('payload_gradient_norm_sum', 'payload_update_norm_sum', 'payload_credit_steps'):
                self.assertEqual(full['totals'][key], resumed['totals'][key])
                self.assertGreater(full['totals'][key], 0)
            invalid = subprocess.run([x for x in common if x != '--payload-credit'] +
                ['--out', str(root/'resumed'), '--steps', '6', '--resume', str(root/'resumed/last.pt')],
                capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('Resume payload_credit mismatch', invalid.stderr)


if __name__ == '__main__':
    unittest.main()
