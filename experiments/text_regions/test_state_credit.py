"""State-Jacobian transport, temporal credit, source isolation and recovery."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch

from ..hat_minkron.kron import MinKron, compact_chunk
from .config import GraphConfig
from .model import RegionLanguageModel
from .ports import tensor_list
from .state_credit import (recurrence_factors, transport, commit_sensitivity, ProducerEligibility,
    ProducerRecorder, ProducerBoundary, make_state_graph_eligibility, state_window_forward)
from .state_credit_train import StateCreditRun
from .graph_credit import make_graph_eligibility, graph_window_forward
from .task_stream import TaskStreamConfig, TaskChunk

torch.set_num_threads(1)


class StateCreditChecks(unittest.TestCase):
    def model(self, backend='optimized'):
        torch.manual_seed(19)
        cfg = GraphConfig.tiny(task_publication=('A','B','C','U','V'), publication_payload_rank=4,
            publication_state_commit='publication', recall_slots=2, rounds=2, recurrent_feedback=True, seed=19)
        return RegionLanguageModel(cfg, backend)

    def run_model(self, enabled=True):
        return StateCreditRun(b'Natural text between facts.\n'*200, self.model().cfg,
            TaskStreamConfig(entities='ABCD', delays=(4,24), cycles=1, interference=True),
            dict(seed=19, data_seed=483, batch=2, context=4, documents=8, lr=.001,
                 syntax_weight=.1, truncate_every=2, eval_chunks=1, eval_count=2,
                 delayed_credit=True, state_credit=enabled, gradient_mode='sum', gradient_cap=1., natural_context=True))

    def same(self, a, b, atol=0, rtol=0):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, atol=atol, rtol=rtol)

    def test_transport_matches_autograd_local_state_jacobian(self):
        torch.manual_seed(4)
        module = MinKron(8, 2, 3, 4, safe_zero=True).double()
        x = torch.randn(2, 1, 8, dtype=torch.float64)
        state = torch.randn(2, 2, 3, 4, dtype=torch.float64)
        directions = torch.randn(*state.shape, 5, dtype=torch.float64)
        for reset in (False, True):
            expected = torch.stack([torch.autograd.functional.jvp(
                lambda s: module(x, s, backend='step', reset_each=reset)[1], state, directions[..., i])[1]
                for i in range(5)], -1)
            actual = transport(directions, recurrence_factors(module, x, reset))
            self.same(actual, expected, atol=1e-12, rtol=1e-12)

    def test_conditional_credit_matches_untruncated_gate_gradient(self):
        # A fixed-input recurrent owner isolates the advertised conditional
        # derivative, including holds, replacements and forced bootstrapping.
        torch.manual_seed(12)
        module = MinKron(8, 2, 3, 4, safe_zero=True).double()
        inputs = [torch.randn(2, 1, 8, dtype=torch.float64) for _ in range(7)]
        features = [torch.randn(2, 3, dtype=torch.float64).tanh() for _ in inputs]
        policy = torch.randn(4, dtype=torch.float64, requires_grad=True)
        state = module.initial_state(2, 'cpu', torch.float64)
        trace = torch.zeros(*state.shape, 4, dtype=torch.float64)
        for t, (x, f) in enumerate(zip(inputs, features)):
            candidate = module(x, state, backend='step')[1]
            p = (torch.cat((f, torch.ones(2, 1, dtype=f.dtype)), -1) @ policy).sigmoid()
            actions = torch.tensor([t % 3 != 1, t % 2 == 0])
            eligible = torch.full((2,), t != 0)
            st = (p-p.detach()) * eligible
            actual = torch.where(actions[:, None, None, None], candidate, state)
            actual = actual + st[:, None, None, None]*(candidate-state)
            trace = commit_sensitivity(trace, transport(trace, recurrence_factors(module, x)),
                state.detach(), candidate.detach(), f, p.detach(), actions, eligible, torch.ones(2, dtype=torch.bool))
            state = actual
        weight = torch.randn_like(state)
        expected = torch.autograd.grad((state*weight).sum(), policy)[0]
        leaf = state.detach().requires_grad_(True)
        boundary = ProducerBoundary(('A',), ((leaf,),), ((trace,),))
        (leaf*weight).sum().backward()
        target = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        self.assertGreater(boundary.apply(target), 0.)
        self.same(target.grad[0], expected, atol=1e-11, rtol=1e-11)
        with self.assertRaisesRegex(RuntimeError, 'consumed'):
            boundary.apply(target)

    def test_credit_survives_replacement_and_disabled_receipt_has_no_effect(self):
        model = self.model()
        trace = ProducerEligibility.zeros(model, 2)
        values = tuple(tuple(torch.ones_like(t) for t in row) for row in trace.sensitivity)
        trace = replace(trace, sensitivity=values)
        state = model.initial_state(2)
        with ProducerRecorder(model, trace) as recorder:
            result = model(torch.tensor([[4],[8]]), state=state,
                publication_overrides={n:'always' for n in trace.owners})
            advanced = recorder.finish(result['state'])
        self.assertTrue(all(float(t.norm()) > 0 for row in advanced.sensitivity for t in row))
        _, receipt = advanced.capture(result['state'])
        policy = model.publisher.policy
        policy.grad = torch.ones_like(policy)
        self.assertEqual(receipt.apply(policy, False), 0.)
        self.same(policy.grad, torch.ones_like(policy))

    def test_recording_preserves_values_and_has_bounded_storage(self):
        model = self.model()
        x = torch.tensor([[2,3,4],[5,6,7]])
        state = model.initial_state(2)
        plain = model(x, state=state)
        trace = ProducerEligibility.zeros(model, 2)
        with ProducerRecorder(model, trace) as recorder:
            tracked = model(x, state=state)
            advanced = recorder.finish(tracked['state'])
        self.same(plain['logits'], tracked['logits'])
        self.same(plain['state'], tracked['state'])
        self.assertEqual(trace.storage_bytes(), advanced.storage_bytes())
        self.assertTrue(all(t.grad_fn is None and not t.requires_grad for row in advanced.sensitivity for t in row))
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))

    def test_partial_rows_holds_and_resets_match_fresh_row(self):
        model = self.model()
        state = model(torch.tensor([[2],[5]]))['state'].detach()
        zero = ProducerEligibility.zeros(model, 2)
        trace = replace(zero, clocks=torch.stack([state.nodes[n].clock for n in zero.owners],1),
            sensitivity=tuple(tuple(torch.ones_like(t) for t in row) for row in zero.sensitivity))
        masks = {n: torch.tensor([[False],[True]]) for n in model.nodes}
        with ProducerRecorder(model, trace) as recorder:
            result = model(torch.tensor([[3],[6]]), state=state, node_masks=masks,
                publication_overrides={n:'hold' for n in trace.owners})
            held = recorder.finish(result['state'])
        self.same(trace.sensitivity, held.sensitivity)
        with ProducerRecorder(model, held) as recorder:
            result = model(torch.tensor([[7],[8]]), state=result['state'], reset=torch.tensor([True,False]),
                publication_overrides={n:'hold' for n in trace.owners})
            reset = recorder.finish(result['state'])
        self.assertTrue(all(not t[0].any() and bool(t[1].eq(1).all()) for row in reset.sensitivity for t in row))
        self.same(reset.epochs[:,0], torch.tensor([1,0]))

    def test_reference_and_optimized_trace_agree(self):
        outputs = []
        for backend in ('reference','optimized'):
            model = self.model(backend)
            with ProducerRecorder(model, ProducerEligibility.zeros(model,2)) as recorder:
                result = model(torch.tensor([[2,3],[5,6]]))
                outputs.append(recorder.finish(result['state']).to_dict())
        self.same(*outputs, atol=1e-5, rtol=1e-4)

    def test_hooks_clean_up_after_exception_and_stage_batching_remains_available(self):
        model = self.model(); model.batch_independent = True
        trace = ProducerEligibility.zeros(model, 2)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with ProducerRecorder(model, trace):
                model(torch.tensor([[2],[3]]))
                raise RuntimeError('intentional')
        self.assertTrue(model.batch_independent)
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        # Hooks make the existing stage planner skip tracked contextualizers.
        with ProducerRecorder(model, trace) as recorder:
            result = model(torch.tensor([[2],[3]]))
            recorder.finish(result['state'])

    def test_disabled_credit_matches_legacy_gradients(self):
        model = self.model()
        plain, tracked = deepcopy(model), deepcopy(model)
        x = torch.tensor([[2,3,4,5],[6,7,8,9]])
        y = torch.full_like(x, -100)
        chunk = TaskChunk(x,y,y,y.ne(-100),torch.zeros_like(x,dtype=torch.bool),y,y)
        gradients = []
        for m, factory, forward in ((plain,make_graph_eligibility,graph_window_forward),
                                     (tracked,make_state_graph_eligibility,state_window_forward)):
            result, credit, receipts = forward(m,chunk,m.initial_state(2),factory(m,2),2)
            result['logits'].square().mean().backward()
            for receipt in receipts:
                receipt.apply(m.publisher.policy)
                receipt.apply_payload(m.publisher.payload_adapter)
                if hasattr(receipt,'apply_state'):
                    receipt.apply_state(m.publisher.policy,False)
            gradients.append({n:p.grad for n,p in m.named_parameters()})
        self.same(*gradients)

    def test_identity_and_serialized_sensitivity_validation(self):
        model = self.model(); trace = ProducerEligibility.zeros(model,2)
        state = model.initial_state(2)
        with self.assertRaisesRegex(ValueError,'clock/epoch'):
            trace.capture(replace(state, epochs=state.epochs+1))
        saved = trace.to_dict(); saved['sensitivity'][0][0][0,0,0,0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'sensitivity'):
            ProducerEligibility.from_dict(saved,model,2)
        saved = trace.to_dict(); saved['owners'] = tuple(reversed(saved['owners']))
        with self.assertRaisesRegex(ValueError,'owners'):
            ProducerEligibility.from_dict(saved,model,2)

    def test_source_isolation_and_exact_resume(self):
        run = self.run_model()
        untouched = deepcopy(run.credits['memory'].to_dict())
        _, row = run.source_gradient('language')
        self.same(untouched, run.credits['memory'].to_dict())
        self.assertGreater(row['state_gradient_norm'],0.)
        run.advance()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'; run.save(path)
            run.advance()
            resumed=self.run_model(); resumed.restore(path); resumed.advance()
            with self.assertRaisesRegex(ValueError,'protocol'):
                self.run_model(False).restore(path)
        self.same(run.model.state_dict(),resumed.model.state_dict())
        self.same(run.optimizer.state_dict(),resumed.optimizer.state_dict())
        for source in run.states:
            self.same(run.states[source],resumed.states[source])
            self.same(run.credits[source].to_dict(),resumed.credits[source].to_dict())
            self.assertEqual(run.streams[source].state_dict(),resumed.streams[source].state_dict())
        self.assertEqual(run.totals,resumed.totals)


if __name__ == '__main__':
    unittest.main()
