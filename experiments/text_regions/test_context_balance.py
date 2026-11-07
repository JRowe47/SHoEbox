"""Paired update math, source isolation, corpus boundaries and exact recovery."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import gradient_balance
from .config import GraphConfig
from .context_balance import ContextBalancedRun, parameter_groups, retention_metrics, final_evaluation
from .mixed_gradients import clip_mixed_gradients
from .ports import tensor_list
from .task_stream import TaskStreamConfig, make_documents, TaskByteStreams

torch.set_num_threads(1)


class ContextBalanceChecks(unittest.TestCase):
    def run_model(self, mode='group_clip', natural=True, parent=None):
        config = GraphConfig.tiny(task_publication=('A','B','C','U','V'), publication_payload_rank=4,
            publication_state_commit='publication', recall_slots=2, rounds=1, recurrent_feedback=True, seed=19)
        return ContextBalancedRun(b'Natural language to carry between the facts.\n'*150,
            config, TaskStreamConfig(entities='ABCD', delays=(4,24), queries_per_cycle=4,
                                    cycles=1, interference=True, acquisition_queries=True),
            dict(seed=19, data_seed=383, batch=2, context=8, documents=8, lr=.001,
                 syntax_weight=.1, truncate_every=4, eval_chunks=1, eval_count=2,
                 delayed_credit=True, gradient_mode=mode, gradient_cap=1., natural_context=natural), parent)

    def same(self, a, b):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_groups_partition_every_parameter_and_interfaces(self):
        run = self.run_model()
        groups = parameter_groups(run.model)
        self.assertEqual(sorted(i for ids in groups.values() for i in ids), list(range(len(run.parameters))))
        self.assertTrue(all(groups.values()))
        names = list(dict(run.model.named_parameters()))
        self.assertTrue(all(names[i].startswith('publisher.') for i in groups['publication']))

    def test_sum_update_matches_manual_equal_weight_objectives(self):
        run, reference = self.run_model('sum'), self.run_model('sum')
        left, _ = reference.source_gradient('language')
        right, _ = reference.source_gradient('memory')
        for p, a, b in zip(reference.parameters, left, right):
            p.grad = (None if a is None and b is None else b if a is None else a if b is None else a+b)
        clip_mixed_gradients(reference.parameters, 1.)
        reference.optimizer.step()
        record = run.advance()
        self.same(run.model.state_dict(), reference.model.state_dict())
        self.same(run.optimizer.state_dict(), reference.optimizer.state_dict())
        self.assertEqual(record['step'], 1)

    def test_both_sources_see_same_weights_and_one_optimizer_step(self):
        run = self.run_model()
        observed = []
        original = run.source_gradient
        def capture(source):
            observed.append(deepcopy(run.model.state_dict()))
            return original(source)
        with patch.object(run, 'source_gradient', side_effect=capture), patch.object(
                run.optimizer, 'step', wraps=run.optimizer.step) as step:
            record = run.advance()
        self.same(*observed)
        self.assertEqual(step.call_count, 1)
        for row in record['gradient_groups'].values():
            self.assertLessEqual(row['text_norm']*row['text_scale'], 1.+1e-10)
            self.assertLessEqual(row['memory_norm']*row['memory_scale'], 1.+1e-10)

    def test_source_state_demand_credit_and_cursor_isolation(self):
        run = self.run_model()
        state = deepcopy(run.states['memory'])
        credit = deepcopy(run.credits['memory'].to_dict())
        cursor = deepcopy(run.streams['memory'].state_dict())
        run.source_gradient('language')
        self.same(state, run.states['memory'])
        self.same(credit, run.credits['memory'].to_dict())
        self.assertEqual(cursor, run.streams['memory'].state_dict())

    def test_parent_continues_language_cursor_with_fresh_runtime(self):
        run = self.run_model(); run.advance()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'parent.pt'; run.save(path)
            child = self.run_model(parent=path)
        self.same(run.model.state_dict(), child.model.state_dict())
        self.assertEqual(run.streams['language'].state_dict(), child.streams['language'].state_dict())
        self.assertEqual(child.step, 0)
        self.assertFalse(child.optimizer.state)
        self.same(child.states['language'], child.model.initial_state(2))
        self.assertEqual(child.streams['memory'].positions, [0,0])

    def test_exact_resume_includes_both_sources_and_optimizer(self):
        run = self.run_model()
        run.advance()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'; run.save(path)
            run.advance(); run.advance()
            resumed = self.run_model(); resumed.restore(path)
            resumed.advance(); resumed.advance()
            wrong_mode = self.run_model('sum')
            with self.assertRaisesRegex(ValueError, 'protocol'):
                wrong_mode.restore(path)
        self.same(run.model.state_dict(), resumed.model.state_dict())
        self.same(run.optimizer.state_dict(), resumed.optimizer.state_dict())
        for source in run.states:
            self.same(run.states[source], resumed.states[source])
            self.same(run.credits[source].to_dict(), resumed.credits[source].to_dict())
            self.assertEqual(run.streams[source].state_dict(), resumed.streams[source].state_dict())
        self.assertEqual(run.totals, resumed.totals)

    def test_natural_bytes_follow_corpus_split_and_stream_fingerprint(self):
        run = self.run_model()
        run.train_raw, run.validation_raw, run.test_raw = b'T'*100, b'V'*100, b'S'*100
        run.data_config = replace(run.data_config, delays=(4,96))
        for split, expected in [('train', b'T'), ('validation', b'V'), ('test', b'S'), ('long_delay', b'S')]:
            docs = run.documents(split, 4)
            events = [e for d in docs for e in d.events if e.kind == 'natural_distractor']
            self.assertTrue(events)
            self.assertTrue(all(set(e.text.strip()) == set(expected) for e in events))
        a = TaskByteStreams(run.documents('train', 4), 2)
        b = TaskByteStreams(run.documents('validation', 4), 2)
        with self.assertRaises(ValueError):
            b.load_state_dict(a.state_dict())

    def test_natural_context_preserves_fact_versions_and_answer_targets(self):
        cfg = TaskStreamConfig(entities='ABCD', delays=(4,96), queries_per_cycle=4,
                               interference=True, acquisition_queries=True)
        docs = make_documents(cfg, count=8, distractor_bytes=b'Words between facts. ')
        self.assertEqual(docs, make_documents(cfg, count=8, distractor_bytes=b'Words between facts. '))
        for doc in docs:
            current, sources, versions = {}, {}, {}
            for event in doc.events:
                if event.kind in ('observation', 'correction'):
                    entity, binding = event.text.decode().removesuffix(';').split(' ')
                    attribute, value = binding.split('='); pair = (entity, attribute)
                    current[pair], sources[pair] = value, event.stop-2
                    versions[pair] = versions.get(pair, 0)+1
                elif event.kind == 'natural_distractor':
                    self.assertTrue(all(x == -100 for x in doc.answer_targets[event.start:event.stop]))
                elif event.kind == 'query':
                    q = doc.queries[event.query_index]
                    if q.kind == 'context':
                        continue
                    pair = (q.entity, q.attribute)
                    expected = (ord('H' if cfg.values.index(current[pair]) >= len(cfg.values)//2 else 'L')
                                if q.kind == 'semantic' else ord(current[pair]))
                    self.assertEqual(q.answer, expected)
                    self.assertEqual(q.source_position, sources[pair])
                    self.assertEqual(q.source_version, versions[pair])
                    self.assertEqual(doc.answer_targets[q.position], expected)
                    self.assertEqual(q.delay, q.position-q.source_position)

    def test_distractor_default_is_exact_and_bad_corpus_rejected(self):
        self.assertEqual(make_documents(count=2), make_documents(count=2, distractor_bytes=None))
        for raw in (b'', 'text'):
            with self.assertRaisesRegex(ValueError, 'nonempty bytes'):
                make_documents(count=2, distractor_bytes=raw)

    def test_publication_age_separates_producers_from_other_ports(self):
        traces = [dict(position=20, reads={'Q':dict(edges=['A__Q__forward','P__Q__lateral'],
            weights=[[[1.,0.,0.],[0.,1.,0.]]], valid=[[True,True]], occurrences=[[2,19]])})]
        result = retention_metrics(traces, ('A',), 4)
        self.assertEqual(result['all']['mean_age'], 9.5)
        self.assertEqual(result['producers']['mean_age'], 18.)
        self.assertEqual(result['producers']['older_than_segment_fraction'], 1.)

    def test_evaluation_cache_reuses_weights_and_invalidates_changed_weights(self):
        run = self.run_model()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            run.save(folder/'best.pt'); run.save(folder/'last.pt')
            with patch.object(run, 'language_metrics', return_value={'nll':2.}), patch.object(
                    run, 'memory_metrics', return_value=({'answer_nll':1.}, [])) as memory, patch(
                    'experiments.text_regions.context_balance.evaluate_memory', return_value=({'answer_nll':1.}, [])):
                first = final_evaluation(run, folder)
                calls = memory.call_count
                self.assertEqual(calls, 6)
                self.assertEqual(first, final_evaluation(run, folder))
                self.assertEqual(memory.call_count, calls)
                with torch.no_grad():
                    next(run.model.parameters()).add_(.001)
                run.save(folder/'best.pt')
                final_evaluation(run, folder)
                self.assertEqual(memory.call_count, calls*2)


if __name__ == '__main__':
    unittest.main()
