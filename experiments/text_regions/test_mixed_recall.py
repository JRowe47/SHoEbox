"""Mixed-stream isolation, exact optimizer/credit resume, and corpus guards."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch

from .mixed_recall_train import MixedRun, evaluate_language
from .task_memory import TaskMemoryConfig
from .task_stream import TaskStreamConfig
from .ports import tensor_list

torch.set_num_threads(1)


class MixedRecallChecks(unittest.TestCase):
    def run_model(self, raw=None):
        return MixedRun(raw or b'A small language stream.\n' * 150,
            TaskMemoryConfig(width=32, experts_per_block=32, shared_per_block=16,
                groups=4, address_dim=8, laces=16, hat_symbols=16, owners=3,
                recall_slots=2, payload_rank=4, memory_aware_write=True,
                write_mode='competitive', bootstrap_empty=True, state_commit='publication'),
            TaskStreamConfig(delays=(0,2), cycles=1, acquisition_queries=True),
            dict(seed=35, data_seed=236, batch=2, context=16, documents=8,
                 lr=.002, syntax_weight=.1, truncate_every=8, eval_chunks=1, eval_count=2))

    def same(self, a, b):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_stream_state_credit_and_cursor_isolation(self):
        run = self.run_model()
        memory = deepcopy(run.states['memory'].to_dict())
        credit = deepcopy(run.credits['memory'].to_dict())
        cursor = deepcopy(run.streams['memory'].state_dict())
        step = run.advance()
        self.assertEqual(step['source'], 'language')
        self.same(memory, run.states['memory'].to_dict())
        self.same(credit, run.credits['memory'].to_dict())
        self.assertEqual(cursor, run.streams['memory'].state_dict())
        self.assertEqual(run.totals['language']['answers'], 0)
        language = deepcopy(run.states['language'].to_dict())
        self.assertEqual(run.advance()['source'], 'memory')
        self.same(language, run.states['language'].to_dict())
        for _ in range(4):
            run.advance()
        for source in ('language', 'memory'):
            self.assertEqual(run.totals[source]['bytes'], 96)
            self.assertEqual(run.totals[source]['experts'], 96*4*12)
            self.assertGreater(run.totals[source]['delayed_gradient_norm_sum'], 0.)
        self.assertGreater(run.totals['memory']['answers'], 0)

    def test_resume_exactly_restores_both_streams_optimizer_and_credit(self):
        run = self.run_model()
        for _ in range(3):
            run.advance()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'last.pt'
            run.save(checkpoint)
            for _ in range(3):
                run.advance()
            continued = self.run_model()
            continued.restore(checkpoint)
            for _ in range(3):
                continued.advance()
            self.same(run.model.state_dict(), continued.model.state_dict())
            self.same(run.optimizer.state_dict(), continued.optimizer.state_dict())
            self.same(run.states, continued.states)
            self.same({k:v.to_dict() for k,v in run.credits.items()},
                      {k:v.to_dict() for k,v in continued.credits.items()})
            self.assertEqual(run.totals, continued.totals)
            for source in run.streams:
                self.assertEqual(run.streams[source].state_dict(), continued.streams[source].state_dict())
            with self.assertRaisesRegex(ValueError, 'sha256'):
                self.run_model(raw=b'A changed language stream.\n'*150).restore(checkpoint)
            different = self.run_model(); different.protocol['lr'] = .003
            with self.assertRaisesRegex(ValueError, 'protocol'):
                different.restore(checkpoint)

    def test_evaluation_isolated_and_respects_actual_target_count(self):
        run = self.run_model()
        run.advance()
        state = deepcopy(run.states)
        rng = torch.get_rng_state()
        metrics = evaluate_language(run.model, run.validation_raw, 2, 16, 2)
        self.assertEqual(metrics['targets'], 64)
        self.assertTrue(0 < metrics['nll'] < 10)
        self.assertTrue(run.model.training)
        self.same(state, run.states)
        self.same(rng, torch.get_rng_state())


if __name__ == '__main__':
    unittest.main()
