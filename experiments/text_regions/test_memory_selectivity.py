"""Check causality, nuisance balance and matched query interventions."""
import re
import unittest
import hashlib
import io
import torch

from .task_stream import TaskStreamConfig
from .memory_selectivity import MODES, selective_cases, training_cases
from .memory_acquisition import pack_cases
from .config import GraphConfig
from .model import RegionLanguageModel
from .data import OrderedByteStreams, split_bytes
from .test_probe import assert_tree_equal
from .test_value_recall import ByteMemory
from .value_recall import evaluate_values
from scripts.run_memory_selectivity import SelectivityRun
from scripts.evaluate_memory_selectivity import paired_queries

torch.set_num_threads(1)


class SelectivityChecks(unittest.TestCase):
    cfg = TaskStreamConfig(entities='ABCD', acquisition_queries=True)

    def cases(self, mode, **kwargs):
        options = dict(split='train', seed=683, start=384, count=4, mode=mode,
                       gap=8, distractor_bytes=b'ordinary prose with no answers ')
        options.update(kwargs)
        return selective_cases(self.cfg, **options)

    def test_latest_matching_write_determines_every_target(self):
        for mode in MODES[1:]:
            for case in self.cases(mode):
                query = case['query']
                for sequence, answer in zip(case['variants'], case['answers']):
                    events = list(re.finditer(rb'([ABCD]) x=([0123]);', sequence))
                    matching = [event for event in events if event[1].decode() == query['entity']]
                    self.assertEqual(matching[-1][2][0], answer)
                    self.assertEqual(matching[-1].start(2), query['source_position'])
                    self.assertEqual(sequence[-4:], f"{query['entity']} x?".encode())
                    self.assertEqual(query['position'], len(sequence)-1)
                    self.assertEqual(query['delay'], query['position']-query['source_position'])
                    source = query['source_position']
                    self.assertEqual(sequence[:source]+sequence[source+1:],
                                     case['variants'][0][:source]+case['variants'][0][source+1:])

    def test_target_and_nuisance_have_full_cartesian_balance(self):
        for mode in MODES[1:]:
            cases = self.cases(mode, count=1)
            pairs = [(target, case['nuisance_value']) for case in cases for target in case['answers']]
            self.assertEqual(set(pairs), {(a,b) for a in b'0123' for b in b'0123'})
            self.assertEqual(len(pairs), 16)

    def test_query_switch_preserves_the_event_history(self):
        older, newer = self.cases('older'), self.cases('newer')
        for base in range(4):
            for a in range(4):
                for b in range(4):
                    old = older[base*4+b]['variants'][a]
                    new = newer[base*4+a]['variants'][b]
                    self.assertEqual(old[:-4], new[:-4])
                    self.assertNotEqual(old[-4:], new[-4:])

    def test_train_schedule_and_packing_exclude_answer_inputs(self):
        cursor = 384
        seen = set()
        for step in range(20):
            cases, mode, gap, cursor = training_cases(self.cfg, step=step, start=cursor,
                seed=683, gaps=[0,4,8,16,32], distractor_bytes=b'Natural text. ')
            indices = {case['document'] for case in cases}
            self.assertFalse(seen & indices)
            seen.update(indices)
            x, syntax, answers, valid, reset = pack_cases(cases)
            self.assertEqual(len(answers), 16)
            self.assertEqual(answers.tolist(), list(b'0123')*4)
            self.assertTrue(bool((x[:, -1] == ord('?')).all()))
            self.assertTrue(bool((syntax[:, -1] == -100).all()))
            self.assertTrue(bool((reset.sum(-1) == 1).all()))
            self.assertEqual(mode, MODES[step % 4])
            self.assertEqual(gap, [0,4,8,16,32][step % 5])

    def test_splits_are_distinct_and_reproducible(self):
        self.assertEqual(self.cases('older'), self.cases('older'))
        self.assertNotEqual(self.cases('older'), self.cases('older', split='validation'))
        self.assertNotEqual(self.cases('older'), self.cases('older', split='test'))

    def test_latest_value_shortcut_fails_query_pair_metric(self):
        model = ByteMemory()
        older = evaluate_values(model, self.cases('older'))
        newer = evaluate_values(model, self.cases('newer'))
        corrected = evaluate_values(model, self.cases('correction'))
        self.assertEqual(older['accuracy'], .25)
        self.assertEqual(newer['accuracy'], 1.)
        self.assertEqual(corrected['accuracy'], 1.)
        self.assertEqual(paired_queries(older, newer), dict(pairs=48, both_correct=0, accuracy=0.))

    def test_resume_preserves_next_update_and_both_source_cursors(self):
        torch.manual_seed(31)
        raw = b'Natural language is learned alongside observed facts.\n'*100
        cfg = GraphConfig.tiny(task_publication=('A','B','C','U','V'),
            publication_state_commit='publication', recall_slots=2, recurrent_feedback=True)
        model = RegionLanguageModel(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.9,.95), weight_decay=.01)
        streams = OrderedByteStreams.from_bytes(split_bytes(raw)[0], 1)
        parent = dict(kind='balanced_acquisition', config=cfg.to_dict(), data_config=self.cfg.to_dict(),
            raw_sha256=hashlib.sha256(raw).hexdigest(), weights=model.state_dict(),
            optimizer=optimizer.state_dict(), language_stream=streams.state_dict(),
            language_state=model.initial_state(1).to_dict(), rng=torch.get_rng_state(),
            step=96, next_document=384,
            protocol=dict(data_seed=683, language_batch=1, language_context=2))
        run = SelectivityRun(parent, raw, validation_cases=1)
        run.update()
        buffer = io.BytesIO(); torch.save(run.state_dict(), buffer); buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        expected_row = run.update()
        expected = io.BytesIO(); torch.save(run.state_dict(), expected); expected.seek(0)
        restored = SelectivityRun(parent, raw, validation_cases=1); restored.restore(checkpoint)
        actual_row = restored.update()
        for key in ('seconds', 'peak_mib'):
            expected_row.pop(key); actual_row.pop(key)
        self.assertEqual(expected_row, actual_row)
        assert_tree_equal(self, torch.load(expected, weights_only=True), restored.state_dict())
        # A completed selective phase can seed a fresh declared budget while
        # carrying learned state and both source cursors, without replay.
        continued = SelectivityRun(restored.state_dict(), raw, validation_cases=1)
        self.assertEqual(continued.step, 0)
        self.assertEqual(continued.next_document, restored.next_document)
        assert_tree_equal(self, continued.model.state_dict(), restored.model.state_dict())
        assert_tree_equal(self, continued.optimizer.state_dict(), restored.optimizer.state_dict())
        assert_tree_equal(self, continued.language_state.to_dict(), restored.language_state.to_dict())
        self.assertEqual(continued.streams.state_dict(), restored.streams.state_dict())
        altered = SelectivityRun(parent, raw, lr=.001, validation_cases=1)
        with self.assertRaises(ValueError):
            altered.restore(checkpoint)


if __name__ == '__main__':
    unittest.main()
