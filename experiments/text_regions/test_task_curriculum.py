"""Payload learning paths, matched truncation, and exact stage continuation."""
import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from .ports import tensor_list
from .task_credit import CompetitiveWriteEligibility
from .task_curriculum import SourceCoverage, stage_config, stage_for_step, window_forward
from .task_memory import TaskMemoryModel
from .task_stream import TaskByteStreams, TaskStreamConfig, make_documents
from .test_task_retention_integration import model_config

torch.set_num_threads(1)


class CurriculumChecks(unittest.TestCase):
    def model(self):
        torch.manual_seed(131)
        return TaskMemoryModel(model_config())

    def test_acquisition_answers_and_observation_coverage(self):
        cfg = stage_config(TaskStreamConfig(entities='ABCD', queries_per_cycle=4,
            delays=(2, 48), interference=True), 'acquire')
        docs = make_documents(cfg, seed=13, count=4)
        for doc in docs:
            queries = [q for q in doc.queries if q.cycle == -1]
            self.assertEqual(len(queries), 4)
            for q in queries:
                self.assertEqual(q.kind, 'exact')
                self.assertEqual(q.delay, 5)
                self.assertEqual(doc.data[q.source_position], q.answer)
                event = next(e for e in doc.events if e.query_index == doc.queries.index(q))
                self.assertNotIn(q.answer, event.text)
                self.assertEqual(doc.data[q.position+1], ord(';'))
        stream = TaskByteStreams(docs, 2)
        chunk = stream.next_chunk(32)
        coverage = SourceCoverage(docs)
        full, short = coverage.count(chunk), coverage.count(chunk, 16)
        self.assertEqual(full['exact_queries'], short['exact_queries'])
        self.assertGreater(full['observation_connected'], short['observation_connected'])
        self.assertGreaterEqual(full['source_value_connected'], full['observation_connected'])

    def test_stage_boundaries_and_legacy_config(self):
        self.assertEqual([stage_for_step(i, 'acquisition', 2, 2) for i in range(7)],
                         ['acquire']*3 + ['bridge']*2 + ['target']*2)
        cfg = TaskStreamConfig()
        old = cfg.to_dict(); old.pop('acquisition_queries')
        self.assertEqual(TaskStreamConfig.from_dict(old), cfg)
        self.assertEqual(stage_config(cfg, 'target'), cfg)
        with self.assertRaises(ValueError):
            stage_for_step(1, 'invalid')

    def test_forward_budget_and_values_match_but_historical_gradient_differs(self):
        model = self.model()
        docs = make_documents(TaskStreamConfig(acquisition_queries=True), seed=4, count=2)
        chunk = TaskByteStreams(docs, 2).next_chunk(16)
        results, gradients = [], []
        for span in (0, 8):
            current = copy.deepcopy(model)
            outputs = []
            handle = current.producer.register_forward_hook(lambda module, args, out: outputs.append(out['latent']))
            trace = CompetitiveWriteEligibility.zeros(2, current.owner_names, current.memory_dim,
                                                       current.cfg.policy_features)
            result, _, _ = window_forward(current, chunk, current.initial_state(2), trace,
                                          truncate_every=span)
            handle.remove()
            gradient = torch.autograd.grad(result['logits'][:, -1].square().sum(), outputs[0],
                                           allow_unused=True)[0]
            gradients.append(0. if gradient is None else float(gradient.abs().sum()))
            results.append(result)
        self.assertGreater(gradients[0], 1e-12)
        self.assertEqual(gradients[1], 0.)
        torch.testing.assert_close(results[0]['logits'], results[1]['logits'], rtol=1e-5, atol=1e-6)
        self.assertEqual(results[0]['external_events'], results[1]['external_events'])
        self.assertEqual(results[0]['expert_executions'], results[1]['expert_executions'])
        for a, b in zip(tensor_list(results[0]['state']), tensor_list(results[1]['state'])):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_held_payload_trains_acquisition_producer_without_recurrent_path(self):
        gradients = []
        for detached in (False, True):
            model = self.model()
            produced = []
            handle = model.producer.register_forward_hook(lambda module, args, out: produced.append(out['latent']))
            acquisition = model(torch.tensor([list(b'A x=1;'), list(b'A x=3;')]), write_override='always')
            handle.remove()
            state = acquisition['state'].detach()
            if not detached:
                state = replace(state, publications=acquisition['state'].publications)
            query = model(torch.tensor([list(b'A x?')]*2), state=state, write_override='hold')
            loss = F.cross_entropy(query['logits'][:, -1], torch.tensor([ord('1'), ord('3')]))
            gradient = torch.autograd.grad(loss, produced[-1], allow_unused=True)[0]
            gradients.append(0. if gradient is None else float(gradient.abs().sum()))
        self.assertGreater(gradients[0], 1e-12)
        self.assertEqual(gradients[1], 0.)

    def test_resume_across_stage_transition_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
                      '--stream-preset', 'retention', '--curriculum', 'acquisition',
                      '--acquire-steps', '2', '--bridge-steps', '2', '--truncate-every', '8',
                      '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
                      '--width', '32', '--batch', '2', '--context', '16', '--eval-count', '2',
                      '--eval-every', '2', '--seed', '9', '--data-seed', '42']
            for out, steps, extra in [('full', 5, []), ('split', 2, []),
                                      ('resumed', 5, ['--resume', str(root/'split/last.pt')])]:
                result = subprocess.run(common+['--out', str(root/out), '--steps', str(steps)]+extra,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            full, split = [torch.load(root/name/'last.pt', weights_only=False) for name in ('full', 'resumed')]
            for key in ('weights', 'optimizer', 'state', 'credit', 'rng'):
                a, b = tensor_list(full[key]), tensor_list(split[key])
                self.assertEqual(len(a), len(b))
                for left, right in zip(a, b):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(full['stream'], split['stream'])
            self.assertEqual(full['totals']['stages'], split['totals']['stages'])
            self.assertTrue((root/'resumed/acquire.pt').exists())
            self.assertTrue((root/'resumed/bridge.pt').exists())
            invalid = subprocess.run(common+['--out', str(root/'resumed'), '--steps', '6',
                '--resume', str(root/'resumed/last.pt'), '--truncate-every', '0'], text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('Resume truncate_every mismatch', invalid.stderr)


if __name__ == '__main__':
    unittest.main()
