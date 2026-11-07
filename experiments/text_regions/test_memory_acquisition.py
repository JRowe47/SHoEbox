"""Guard balanced source interventions and masked variable-length prefixes."""
import unittest
import torch
from .task_stream import TaskStreamConfig
from .memory_acquisition import acquisition_cases, pack_cases, delayed_cases


class AcquisitionChecks(unittest.TestCase):
    def cases(self, split='train', start=0):
        return acquisition_cases(TaskStreamConfig(entities='ABCD', acquisition_queries=True),
                                 split=split, seed=683, start=start, count=4)

    def test_only_source_value_changes_and_labels_stay_out_of_input(self):
        for case in self.cases():
            source = case['query']['source_position']
            self.assertEqual(case['answers'], list(b'0123'))
            for sequence, answer in zip(case['variants'], case['answers']):
                self.assertEqual(sequence[source], answer)
                self.assertEqual(sequence[-1:], b'?')
                reference = case['variants'][0]
                self.assertEqual(sequence[:source]+sequence[source+1:], reference[:source]+reference[source+1:])

    def test_padding_and_query_separators_are_excluded_from_syntax_loss(self):
        cases = self.cases(); x, ordinary, answers, valid, reset = pack_cases(cases)
        self.assertTrue(bool((ordinary[~valid] == -100).all()))
        self.assertTrue(bool((ordinary[:, -1] == -100).all()))
        self.assertTrue(bool((reset.sum(-1) == 1).all()))
        self.assertEqual(answers.tolist(), list(b'0123')*4)
        for i, case in enumerate(cases):
            for j, sequence in enumerate(case['variants']):
                row = i*4+j; offset = x.shape[1]-len(sequence)
                self.assertEqual(bytes(x[row, offset:].tolist()), sequence)
                self.assertTrue(bool(reset[row, offset]))
                for position in case['query_positions']:
                    self.assertEqual(int(ordinary[row, offset+position]), -100)

    def test_document_cursor_advances_and_splits_are_independent(self):
        first, second = self.cases(), self.cases(start=4)
        self.assertFalse({c['document'] for c in first} & {c['document'] for c in second})
        self.assertNotEqual([c['variants'] for c in first], [c['variants'] for c in self.cases('validation')])

    def test_delay_insertion_keeps_source_and_answer_causal(self):
        cases=self.cases()
        later=delayed_cases(cases, 24, b'Fresh held-out ordinary words. ')
        for original, delayed in zip(cases,later):
            self.assertEqual(delayed['query']['delay'], original['query']['delay']+24)
            self.assertEqual(delayed['answers'], original['answers'])
            source=delayed['query']['source_position']
            for sequence,answer in zip(delayed['variants'],delayed['answers']):
                self.assertEqual(sequence[source],answer)
                self.assertEqual(sequence[-1:],b'?')
                self.assertEqual(len(sequence),len(original['variants'][0])+24)
            self.assertEqual(delayed['query_positions'][-1], len(delayed['variants'][0])-1)


if __name__ == '__main__':
    unittest.main()
