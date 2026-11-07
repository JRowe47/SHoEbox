"""Exact cross-chunk gradients, entry credit, resets and optimizer recovery."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch

from .config import GraphConfig
from .model import RegionLanguageModel
from .state_credit import make_state_graph_eligibility, state_window_forward
from .content_credit import content_window_forward
from .content_credit_train import ContentCreditRun
from .task_stream import TaskStreamConfig, TaskChunk
from .ports import tensor_list
from .test_probe import assert_tree_equal

torch.set_num_threads(1)


class ContentCreditChecks(unittest.TestCase):
    def model(self):
        torch.manual_seed(37)
        cfg = GraphConfig.tiny(task_publication=('A','B','C','U','V'), publication_payload_rank=2,
            publication_state_commit='publication', recall_slots=2, rounds=2,
            recurrent_feedback=True, seed=37)
        return RegionLanguageModel(cfg).double()

    def chunk(self, values, reset=True):
        values = torch.tensor(values)
        ignored = torch.full_like(values, -100)
        resets = torch.zeros_like(values, dtype=torch.bool)
        resets[:, 0] = reset
        return TaskChunk(values, ignored, ignored, ignored.ne(-100), resets, ignored, ignored)

    def apply(self, model, boundaries):
        for boundary in boundaries:
            boundary.apply(model.publisher.policy)
            boundary.apply_payload(model.publisher.payload_adapter)
            boundary.apply_state(model.publisher.policy)

    def gradients(self, model, chunk, state, credit, segment, historical=False):
        model.zero_grad(set_to_none=True)
        result, advanced, receipts = content_window_forward(model, chunk, state, credit, segment)
        (result['logits'][:, -1].square().mean()+.001*result['aux']).backward()
        if historical:
            self.apply(model, receipts)
        return result, advanced, receipts, [None if p.grad is None else p.grad.clone() for p in model.parameters()]

    def assert_gradients(self, actual, expected):
        for a, b in zip(actual, expected):
            if b is None: self.assertIsNone(a)
            else: torch.testing.assert_close(a, b, rtol=2e-7, atol=1e-8)

    def test_chunks_match_full_window_logits_gradients_state_and_credit(self):
        model = self.model(); chunk = self.chunk([[65,32,120,61,48]])
        state, credit = model.initial_state(1), make_state_graph_eligibility(model, 1)
        full, fc, fr, fg = self.gradients(model, chunk, state, credit, segment=5)
        part, pc, pr, pg = self.gradients(model, chunk, state, credit, segment=2)
        torch.testing.assert_close(full['logits'], part['logits'], rtol=0, atol=0)
        torch.testing.assert_close(full['aux'], part['aux'], rtol=1e-14, atol=1e-14)
        self.assert_gradients(pg, fg)
        assert_tree_equal(self, full['state'].to_dict(), part['state'].to_dict())
        assert_tree_equal(self, fc.to_dict(), pc.to_dict())
        self.assertEqual(len(fr), 1); self.assertEqual(len(pr), 1)
        for key in ('external_events','expert_executions'):
            self.assertEqual(full[key], part[key])
        self.assertTrue(all(not t.requires_grad and t.grad_fn is None for t in tensor_list(part['state'])))

    def test_early_observation_gradient_matches_full_and_old_cut_is_zero(self):
        model = self.model(); chunk = self.chunk([[65,32,120,61,48,59]])
        source_gradients = []
        for mode in ('full','connected','cut'):
            leaves = []
            def remember(module, args, output):
                output.retain_grad(); leaves.append(output)
            hook = model.hat.byte_embedding.register_forward_hook(remember)
            state, credit = model.initial_state(1), make_state_graph_eligibility(model, 1)
            if mode == 'cut': result, _, _ = state_window_forward(model, chunk, state, credit, 2)
            else: result, _, _ = content_window_forward(model, chunk, state, credit, 6 if mode == 'full' else 2)
            result['logits'][:, -1, 49].sum().backward()
            source_gradients.append(None if leaves[0].grad is None else leaves[0].grad.clone())
            hook.remove(); model.zero_grad(set_to_none=True)
        self.assertTrue(source_gradients[2] is None or not bool(source_gradients[2].ne(0).any()))
        self.assertGreater(float(source_gradients[0].norm()), 0.)
        torch.testing.assert_close(source_gradients[1], source_gradients[0], rtol=1e-9, atol=1e-10)

    def test_existing_older_credit_is_applied_once_at_window_entry(self):
        model = self.model()
        state, credit = model.initial_state(1), make_state_graph_eligibility(model, 1)
        warm, credit, _ = state_window_forward(model, self.chunk([[65,66,67,68]]), state, credit, 2)
        chunk = self.chunk([[69,70,71,72]], reset=False)
        full, fc, _, fg = self.gradients(model, chunk, warm['state'], credit, 4, historical=True)
        part, pc, receipts, pg = self.gradients(model, chunk, warm['state'], credit, 2, historical=True)
        self.assertEqual(len(receipts), 1)
        self.assert_gradients(pg, fg)
        assert_tree_equal(self, fc.to_dict(), pc.to_dict())
        with self.assertRaises(RuntimeError):
            self.apply(model, receipts)

    def test_document_reset_severs_only_reset_rows(self):
        model = self.model(); chunk = self.chunk([[65,66,67,68],[69,70,71,72]])
        reset = chunk.reset.clone(); reset[0, 2] = True
        chunk = replace(chunk, reset=reset)
        first = []
        def remember(module, args, output):
            output.retain_grad(); first.append(output)
        hook = model.hat.byte_embedding.register_forward_hook(remember)
        result, _, _ = content_window_forward(model, chunk, model.initial_state(2),
            make_state_graph_eligibility(model, 2), 2)
        result['logits'][:, -1, 49].sum().backward(); hook.remove()
        self.assertEqual(float(first[0].grad[0].norm()), 0.)
        self.assertGreater(float(first[0].grad[1].norm()), 0.)

    def test_entry_detaches_any_incoming_live_graph(self):
        model = self.model()
        state, credit = model.initial_state(1), make_state_graph_eligibility(model, 1)
        # Use a live readout-node state to ensure entry detaches fields outside
        # the producer/publication capture routines as well.
        probe = torch.ones_like(state.nodes['P'].blocks[0], requires_grad=True)
        nodes = dict(state.nodes); old = nodes['P']
        nodes['P'] = replace(old, blocks=(probe, *old.blocks[1:]))
        state = replace(state, nodes=nodes)
        result, _, _ = content_window_forward(model, self.chunk([[65,66]], reset=False), state, credit, 1)
        result['logits'].square().sum().backward()
        self.assertIsNone(probe.grad)

    def run_model(self):
        return ContentCreditRun(b'Ordinary text between facts.\n'*200, self.model().cfg,
            TaskStreamConfig(entities='AB', delays=(4,24), cycles=1),
            dict(seed=37, data_seed=583, batch=1, context=4, documents=8, lr=.001,
                syntax_weight=.1, truncate_every=0, execution_segment=2, cut_at_segments=False,
                eval_chunks=1, eval_count=2, delayed_credit=True, state_credit=True,
                gradient_mode='sum', gradient_cap=1., natural_context=True))

    def test_exact_resume_includes_optimizer_sources_and_conditional_traces(self):
        run = self.run_model()
        untouched = run.credits['memory'].to_dict()
        run.source_gradient('language')
        assert_tree_equal(self, run.credits['memory'].to_dict(), untouched)
        run.advance()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resume.pt'; run.save(path)
            expected = run.advance()
            restored = self.run_model(); restored.restore(path)
            actual = restored.advance()
            self.assertEqual(expected, actual)
            assert_tree_equal(self, run.model.state_dict(), restored.model.state_dict())
            assert_tree_equal(self, run.optimizer.state_dict(), restored.optimizer.state_dict())
            for source in run.credits:
                assert_tree_equal(self, run.credits[source].to_dict(), restored.credits[source].to_dict())
                assert_tree_equal(self, run.states[source].to_dict(), restored.states[source].to_dict())

    def test_bad_horizon_or_execution_segment_is_rejected(self):
        model = self.model()
        with self.assertRaises(ValueError):
            content_window_forward(model, self.chunk([[65]]), model.initial_state(1),
                make_state_graph_eligibility(model, 1), 0)
        run = self.run_model()
        with self.assertRaises(ValueError):
            run.forward_window(run.model, self.chunk([[65,66]]), run.states['language'],
                run.credits['language'], truncate_every=1)


if __name__ == '__main__':
    unittest.main()
