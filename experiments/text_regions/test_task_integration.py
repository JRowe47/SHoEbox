"""Causal integration and immutable-state contracts for task memory."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from .ports import tensor_list
from .task_memory import TaskMemoryConfig, TaskMemoryModel, TaskMemoryState
from .task_credit import WriteEligibility
from .task_stream import TaskByteStreams, TaskStreamConfig, make_documents
from .task_memory_train import checkpoint, objective, evaluate, validate_resume


torch.set_num_threads(1)


def tiny_model():
    torch.manual_seed(932)
    return TaskMemoryModel(TaskMemoryConfig(width=32, owners=3, experts_per_block=32,
        shared_per_block=16, groups=4, address_dim=8, laces=16, hat_symbols=16))


class TaskIntegrationChecks(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.inputs = torch.tensor([list(b'A x=1;'), list(b'B x=3;')])

    def same(self, left, right):
        a, b = tensor_list(left), tensor_list(right)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    @torch.no_grad()
    def test_writes_commit_after_prediction_and_reads_report_old_versions(self):
        prefix = self.model(self.inputs, write_override='always')['state']
        query = torch.full((2, 1), ord('?'), dtype=torch.long)
        held = self.model(query, state=prefix, write_override='hold')
        written = self.model(query, state=prefix, write_override='always')
        self.same(held['logits'], written['logits'])
        old_versions = torch.stack([p.version for p in prefix.publications.values()], 1)
        self.same(written['read_versions'][:, 0], old_versions)
        self.same(written['memory_versions'][:, 0], old_versions+1)
        self.same(held['state'].publications, prefix.publications)
        self.assertTrue(bool((held['state'].producer.clock > prefix.producer.clock).all()))

    @torch.no_grad()
    def test_matched_intervention_is_immutable_and_has_identical_expert_budget(self):
        prefix = self.model(self.inputs, write_override='always')['state']
        snapshot = prefix.clone()
        query = torch.full((2, 1), ord('?'), dtype=torch.long)
        intact = self.model(query, state=prefix)
        blocked = self.model(query, state=prefix, control='blocked')
        shuffled = self.model(query, state=prefix, control='shuffle')
        self.same(prefix, snapshot)
        for intervention in (blocked, shuffled):
            self.assertEqual(intact['expert_executions'], intervention['expert_executions'])
            self.assertEqual(intact['external_events'], intervention['external_events'])
            self.same(intact['state'].producer, intervention['state'].producer)
            self.same(intact['state'].publications, intervention['state'].publications)
        self.assertGreater(float((intact['logits']-blocked['logits']).abs().max()), 1e-6)

    @torch.no_grad()
    def test_full_and_detached_chunk_execution_are_identical(self):
        whole = self.model(self.inputs)
        first = self.model(self.inputs[:, :3])
        last = self.model(self.inputs[:, 3:], state=first['state'].detach())
        self.same(whole['logits'], torch.cat((first['logits'], last['logits']), 1))
        self.same(whole['state'], last['state'])
        self.assertEqual(whole['external_events'], 12)
        self.assertEqual(first['external_events']+last['external_events'], 12)

    @torch.no_grad()
    def test_later_inputs_cannot_change_earlier_predictions(self):
        altered = self.inputs.clone()
        altered[:, 3:] = ord('Z')
        first = self.model(self.inputs)
        other = self.model(altered)
        self.same(first['logits'][:, :3], other['logits'][:, :3])

    @torch.no_grad()
    def test_serialization_continues_exactly_without_source_mutation(self):
        state = self.model(self.inputs)['state'].detach()
        buffer = io.BytesIO()
        torch.save(dict(config=self.model.cfg.to_dict(), weights=self.model.state_dict(),
                        state=state.to_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = TaskMemoryModel(TaskMemoryConfig.from_dict(saved['config']))
        restored.load_state_dict(saved['weights'])
        restored_state = TaskMemoryState.from_dict(saved['state'])
        query = torch.tensor([list(b'A x?'), list(b'B x?')])
        a, b = self.model(query, state=state), restored(query, state=restored_state)
        self.same(a['logits'], b['logits'])
        self.same(a['state'], b['state'])
        self.assertFalse(any(t.requires_grad for t in restored_state.tensors()))

    @torch.no_grad()
    def test_reset_clears_only_selected_stream_and_preserves_empty_holds(self):
        state = self.model(self.inputs, write_override='always')['state']
        reset = state.reset(torch.tensor([True, False]))
        self.assertEqual(reset.events_seen.tolist(), [0, 6])
        self.assertEqual(reset.epochs.tolist(), [1, 0])
        for name, old in state.publications.items():
            new = reset.publications[name]
            self.assertFalse(bool(new.valid[0]))
            self.assertEqual(int(new.version[0]), 0)
            for left, right in zip(tensor_list(old), tensor_list(new)):
                self.same(left[1:], right[1:])
        held = self.model(torch.tensor([[ord('?')], [ord('?')]]), state=reset,
                          write_override='hold')
        self.same(held['state'].publications, reset.publications)
        self.assertTrue(bool(held['read_weights'][0, :, -1].eq(1).all()))

    def test_detached_memory_leaf_receives_task_gradient_only_when_read(self):
        with torch.no_grad():
            prefix = self.model(self.inputs, write_override='always')['state'].detach()
        for control in ('intact', 'blocked'):
            leaf = prefix.memory_tensor().detach().requires_grad_()
            result = self.model(torch.full((2, 1), ord('?'), dtype=torch.long),
                                state=prefix.with_memory_tensor(leaf), control=control,
                                write_override='hold')
            result['logits'].square().sum().backward()
            if control == 'intact':
                self.assertGreater(float(leaf.grad.abs().sum()), 0)
            else:
                self.assertEqual(float(leaf.grad.abs().sum()), 0)
            self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def test_blocking_removes_all_current_publication_influence(self):
        prefix = self.model(self.inputs, write_override='always')['state']
        changed = prefix.with_memory_tensor(torch.randn_like(prefix.memory_tensor())*3)
        query = torch.full((2, 1), ord('?'), dtype=torch.long)
        ordinary = self.model(query, state=prefix, control='blocked')
        altered = self.model(query, state=changed, control='blocked')
        self.same(ordinary['logits'], altered['logits'])
        first = self.model(query, state=prefix)
        other = self.model(query, state=changed)
        self.assertGreater(float((first['logits']-other['logits']).abs().max()), 1e-6)

    def test_no_memory_evaluation_keeps_reads_blocked(self):
        config = TaskStreamConfig(cycles=1, queries_per_cycle=2, delays=(0,),
                                  semantic_fraction=0, contextual_questions=False)
        with patch.object(self.model, 'forward', wraps=self.model.forward) as calls:
            summary, records = evaluate(self.model, config, 12, count=2,
                                         interventions=True, control='blocked')
        self.assertTrue(calls.call_args_list)
        self.assertTrue(all(call.kwargs['control'] == 'blocked' for call in calls.call_args_list))
        self.assertEqual(summary['control'], 'blocked')
        self.assertEqual(summary['effective_read_mass'], 0.)
        self.assertEqual(summary['interventions'], {})
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record['effective_weighted_publication_age'] is None for record in records))

    def test_evaluation_reports_actual_publication_age_and_owner_utilization(self):
        config = TaskStreamConfig(cycles=1, queries_per_cycle=2, delays=(0,),
                                  semantic_fraction=0, contextual_questions=False)
        original = self.model.forward

        def always_write(*args, **kwargs):
            return original(*args, **kwargs, write_override='always')

        with patch.object(self.model, 'forward', side_effect=always_write):
            summary, records = evaluate(self.model, config, 12, count=2)
        # Recurrent summaries replaced every byte have age one regardless of
        # the task's earlier observation-to-query delay.
        for record in records:
            self.assertEqual(record['read_source_owners'], list(self.model.owner_names))
            self.assertEqual(record['read_ages'], [1]*self.model.cfg.owners)
            self.assertEqual(record['read_occurrences'], [record['position']-1]*self.model.cfg.owners)
            self.assertAlmostEqual(record['weighted_publication_age'], 1., places=6)
        for owner in summary['owners'].values():
            self.assertEqual(owner['write_fraction'], 1.)
            self.assertAlmostEqual(owner['valid_fraction'], 1-2/summary['external_bytes'])
        self.assertAlmostEqual(summary['query_mean_weighted_publication_age'], 1., places=6)
        json.dumps(dict(summary=summary, records=records), allow_nan=False)

    def test_resume_rejects_changed_training_objective(self):
        config = TaskStreamConfig()
        args = SimpleNamespace(seed=1, data_seed=2, arm='integrated', batch=2, context=9,
            width=self.model.cfg.width, owners=self.model.cfg.owners, lr=.001,
            text_weight=.1, threads=1, eval_count=2, steps=3)
        saved = dict(config=self.model.cfg.to_dict(), data_config=config.to_dict(),
                     args=vars(args).copy(), step=2)
        validate_resume(saved, args, self.model, config)
        args.text_weight = .2
        with self.assertRaisesRegex(ValueError, 'text_weight'):
            validate_resume(saved, args, self.model, config)

    def test_checkpoint_resumes_optimizer_stream_memory_and_credit_exactly(self):
        cfg = TaskStreamConfig()
        docs = make_documents(cfg, seed=52, count=4)
        streams = TaskByteStreams(docs, 2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        state = self.model.initial_state(2)
        credit = WriteEligibility.zeros(2, self.model.owner_names, self.model.memory_dim,
                                        self.model.cfg.policy_features)

        def step(model, optim, current, trace, stream):
            chunk = stream.next_chunk(9)
            optim.zero_grad(set_to_none=True)
            boundary = trace.capture(current.memory_tensor())
            result = model(chunk.values, state=current.with_memory_tensor(boundary.memory), reset=chunk.reset)
            loss, _, _ = objective(result, chunk)
            loss.backward()
            boundary.apply(model.write_policy)
            trace = trace.advance_chunk(result)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            return result['state'].detach(), trace, loss.detach()

        state, credit, _ = step(self.model, optimizer, state, credit, streams)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'resume.pt'
            checkpoint(path, self.model, optimizer, state, credit, streams,
                       SimpleNamespace(arm='integrated', context=9), cfg, 1, 9.)
            expected_state, expected_credit, expected_loss = step(self.model, optimizer, state, credit, streams)
            saved = torch.load(path, weights_only=True)
            resumed = TaskMemoryModel(TaskMemoryConfig.from_dict(saved['config']))
            resumed.load_state_dict(saved['weights'])
            resumed_optim = torch.optim.AdamW(resumed.parameters(), lr=.001)
            resumed_optim.load_state_dict(saved['optimizer'])
            resumed_stream = TaskByteStreams(docs, 2)
            resumed_stream.load_state_dict(saved['stream'])
            torch.set_rng_state(saved['rng'])
            got_state, got_credit, got_loss = step(resumed, resumed_optim,
                TaskMemoryState.from_dict(saved['state']), WriteEligibility.from_dict(saved['credit']),
                resumed_stream)
            self.same(got_loss, expected_loss)
            self.same(got_state, expected_state)
            self.same(got_credit.to_dict(), expected_credit.to_dict())
            self.same(resumed.state_dict(), self.model.state_dict())
            self.same(resumed_optim.state_dict(), optimizer.state_dict())
            self.assertEqual(resumed_stream.state_dict(), streams.state_dict())


if __name__ == '__main__':
    unittest.main()
