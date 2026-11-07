"""Recomputation must preserve full credit and leave no hooks or extra updates."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch

from . import test_content_credit as fixtures
from .test_probe import assert_tree_equal
from .content_credit import content_window_forward
from .state_credit import make_state_graph_eligibility, state_window_forward
from .model import RegionLanguageModel
from .runtime import _training_arithmetic, _cache


class ContentCheckpointChecks(unittest.TestCase):
    def test_large_banks_and_batched_stages_keep_training_arithmetic(self):
        fixture = fixtures.ContentCreditChecks()
        cfg = replace(fixture.model().cfg, experts_per_block=64,
                      shared_per_block=32, expert_hidden=64)
        for dtype in (torch.float64, torch.float32):
            torch.manual_seed(37)
            model = RegionLanguageModel(cfg).to(dtype)
            self.assertTrue(model._large_expert_banks)
            self.assertTrue(model.batch_independent)
            chunk = fixture.chunk([[65,66]])
            state, credit = model.initial_state(1), make_state_graph_eligibility(model, 1)
            expected = None
            for checkpointed in (False, True):
                model.zero_grad(set_to_none=True)
                result, advanced, receipts = content_window_forward(model, chunk, state, credit,
                    1, checkpoint_segments=checkpointed)
                (result['logits'].square().mean()+.001*result['aux']).backward()
                fixture.apply(model, receipts)
                gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
                if expected is None:
                    expected = (result, advanced, gradients)
                else:
                    torch.testing.assert_close(result['logits'], expected[0]['logits'], rtol=0, atol=0)
                    assert_tree_equal(self, advanced.to_dict(), expected[1].to_dict())
                    for a, b in zip(gradients, expected[2]):
                        if b is None: self.assertIsNone(a)
                        else: torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                self.assertFalse(_training_arithmetic.get())
                self.assertIsNone(_cache.get())

    def test_full_and_truncated_gradients_traces_resets_and_partial_segments(self):
        fixture = fixtures.ContentCreditChecks()
        for cut in (False, True):
            model = fixture.model()
            state, credit = model.initial_state(2), make_state_graph_eligibility(model, 2)
            warm, credit, _ = state_window_forward(model,
                fixture.chunk([[65,66],[67,68]]), state, credit, 1)
            chunk = fixture.chunk([[69,70,71,72],[73,74,75,76]], reset=False)
            reset = chunk.reset.clone(); reset[0, 2] = True
            chunk = replace(chunk, reset=reset)
            reference = None
            for checkpointed in (False, True):
                model.zero_grad(set_to_none=True)
                if cut:
                    result, advanced, receipts = state_window_forward(model, chunk,
                        warm['state'], credit, 2, checkpoint_segments=checkpointed)
                else:
                    result, advanced, receipts = content_window_forward(model, chunk,
                        warm['state'], credit, 3, checkpoint_segments=checkpointed)
                loss = result['logits'].square().mean()+.001*result['aux']
                loss.backward(); fixture.apply(model, receipts)
                gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
                actual = (result, advanced, gradients)
                if reference is None:
                    reference = actual
                else:
                    torch.testing.assert_close(reference[0]['logits'], result['logits'], rtol=0, atol=0)
                    fixture.assert_gradients(gradients, reference[2])
                    assert_tree_equal(self, reference[0]['state'].to_dict(), result['state'].to_dict())
                    assert_tree_equal(self, reference[1].to_dict(), advanced.to_dict())
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_checkpointed_source_credit_and_exact_resume(self):
        fixture = fixtures.ContentCreditChecks()
        run = fixture.run_model()
        run.protocol.update(checkpoint_segments=True, audit_coverage=True)
        run.advance()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resume.pt'; run.save(path)
            expected = run.advance()
            restored = fixture.run_model()
            restored.protocol.update(checkpoint_segments=True, audit_coverage=True)
            restored.restore(path)
            actual = restored.advance()
            self.assertEqual(expected, actual)
            assert_tree_equal(self, run.model.state_dict(), restored.model.state_dict())
            assert_tree_equal(self, run.optimizer.state_dict(), restored.optimizer.state_dict())
            for source in run.states:
                assert_tree_equal(self, run.states[source].to_dict(), restored.states[source].to_dict())
                assert_tree_equal(self, run.credits[source].to_dict(), restored.credits[source].to_dict())

    def test_failed_recomputation_removes_recorder_hooks(self):
        fixture = fixtures.ContentCreditChecks(); model = fixture.model()
        calls = 0
        def fail(module, args):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise RuntimeError('injected recomputation failure')
        handle = model.readout.register_forward_pre_hook(fail)
        result, _, _ = content_window_forward(model, fixture.chunk([[65,66]]),
            model.initial_state(1), make_state_graph_eligibility(model, 1), 1,
            checkpoint_segments=True)
        with self.assertRaisesRegex(RuntimeError, 'injected recomputation'):
            result['logits'].square().mean().backward()
        handle.remove()
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
        self.assertFalse(_training_arithmetic.get())
        self.assertIsNone(_cache.get())


if __name__ == '__main__':
    unittest.main()
