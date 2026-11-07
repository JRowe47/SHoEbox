"""Detect constant-answer shortcuts and audit real gradient-window coverage."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import torch
from torch import nn

from .content_credit_audit import window_coverage
from .task_stream import TaskByteStreams, TaskDocument, TaskQuery, TaskStreamConfig, make_documents
from .value_recall import value_cases, evaluate_values


class ByteMemory(nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(views=('A',))
        self.nodes = {'A': None}; self.constant = constant

    def forward(self, values, state=None, reset=None, availability=None, node_masks=None):
        if state is None:
            state = torch.full_like(values[:, 0], 49)
        outputs = []
        for t in range(values.shape[1]):
            state = torch.where(reset[:, t], 49, state)
            active = availability['A'][:, t]
            digit = (values[:, t] >= 48) & (values[:, t] <= 51) & active
            if not self.constant:
                state = torch.where(digit, values[:, t], state)
            logits = torch.zeros(len(values), 256)
            logits.scatter_(1, state[:, None], 4.)
            outputs.append(logits)
        return dict(logits=torch.stack(outputs, 1), state=state)


class ValueRecallChecks(unittest.TestCase):
    def cases(self):
        docs = make_documents(TaskStreamConfig(entities='ABCD', acquisition_queries=True),
                              split='test', seed=53, count=2)
        return docs, value_cases(docs, acquisition=True)

    def test_only_the_observed_value_changes_and_targets_are_balanced(self):
        docs, cases = self.cases()
        self.assertEqual(len(cases), 8)
        for case in cases:
            q = case['query']; original = docs[case['document']].data[:q['position']+1]
            self.assertEqual(case['answers'], list(b'0123'))
            for data, answer in zip(case['variants'], case['answers']):
                differences = [i for i, (a, b) in enumerate(zip(original, data)) if a != b]
                self.assertEqual(differences, [] if answer == q['answer'] else [q['source_position']])
                self.assertEqual(data[q['source_position']], answer)

    def test_constant_answer_is_chance_without_value_sensitivity(self):
        _, cases = self.cases()
        result = evaluate_values(ByteMemory(constant=True), cases)
        self.assertEqual(result['accuracy'], .25)
        for key in ('complete_set_accuracy','answer_switch_fraction'):
            self.assertEqual(result[key], 0.)
        self.assertAlmostEqual(result['value_margin'], 0., delta=1e-6)

    def test_correct_value_recall_survives_padding_chunking_and_restores_mode(self):
        _, cases = self.cases()
        model = ByteMemory(); model.train()
        a = evaluate_values(model, cases, context=1, cases_per_batch=3)
        b = evaluate_values(model, cases, context=13, cases_per_batch=8)
        self.assertEqual(a, b)
        self.assertTrue(model.training)
        self.assertEqual(a['accuracy'], 1.)
        self.assertEqual(a['complete_set_accuracy'], 1.)
        self.assertEqual(a['answer_switch_fraction'], 1.)
        self.assertGreater(a['value_margin'], 0.)

    def test_coverage_uses_actual_boundaries_and_reset_occurrences(self):
        queries = (TaskQuery(9, 48, 'exact', 'A', 'x', 3, 1, 6, False, None, 0, -1),
                   TaskQuery(11, 49, 'exact', 'B', 'x', 8, 1, 3, False, None, 0, 0))
        targets = [-100]*12
        for q in queries: targets[q.position] = q.answer
        document = TaskDocument(b'abc0defg1???', tuple(targets), queries, (), 'test', 0)
        stream = TaskByteStreams([document], 1)
        first = stream.next_chunk(8)
        self.assertEqual(window_coverage(first, [document], 4)['all']['answers'], 0)
        # First query's source is in the preceding window; second is local.
        second = stream.next_chunk(4)
        counts = window_coverage(second, [document], 4)
        self.assertEqual(counts['exact'], dict(answers=2, within_window=1, within_segment=1))
        self.assertEqual(counts['acquisition']['within_window'], 0)
        # A fresh full document includes both sources; only B fits one segment.
        whole = stream.next_chunk(12)
        counts = window_coverage(whole, [document], 4)
        self.assertEqual(counts['exact'], dict(answers=2, within_window=2, within_segment=1))
        reset = whole.reset.clone(); reset[0, 6] = True
        counts = window_coverage(replace(whole, reset=reset), [document], 4)
        self.assertEqual(counts['exact']['within_window'], 1)


if __name__ == '__main__':
    unittest.main()
