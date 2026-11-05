"""Joint learning: one decoder, causal labels, preserved state and exact resume."""
import contextlib
import copy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .composition_train import build_model
from .config import GraphConfig
from .integrated_eval import evaluate_views, selection_key
from .integrated_train import main, stream_loss
from . import test_composition_continue as continuation_tests
from .test_probe import assert_tree_equal
from .view_data import make_view_batch

torch.set_num_threads(1)


def progress(minimum=.5, preserved=True, qualified=False):
    return {'gate_failures': [] if qualified else ['left'],
            'composition_preserved': preserved, 'minimum_accuracy': minimum,
            'macro_accuracy': minimum, 'macro_nats': 1.-minimum,
            'scores': {'composition': .95, 'left': minimum}}


class IntegrationChecks(unittest.TestCase):
    def test_labels_never_change_logits_or_state(self):
        torch.manual_seed(801)
        model = build_model(GraphConfig.tiny(readout_mode='concat', rounds=2,
                                            recurrent_feedback=True), .5, 1.)
        batch = make_view_batch('train', 0, 4, gap=0)
        changed = replace(batch, targets=torch.where(batch.targets.ne(-100),
                                                     65+(batch.targets-64)%4, -100))
        _, a = stream_loss(model, batch, None)
        _, b = stream_loss(model, changed, None)
        assert_tree_equal(self, a['state'].to_dict(), b['state'].to_dict())
        self.assertTrue(torch.equal(a['logits'], b['logits']))
        self.assertFalse(torch.equal(a['ce'], b['ce']))

    def test_selection_requires_composition_and_does_not_hide_a_weak_task(self):
        self.assertGreater(selection_key(progress(.5, True)), selection_key(progress(.9, False)))
        self.assertGreater(selection_key(progress(.7)), selection_key(progress(.6)))
        self.assertGreater(selection_key(progress(.9, True, True)), selection_key(progress(.95)))

    def run_cli(self, parent, out, extras=(), scores=None):
        args = ['integrated', '--parent', str(parent), '--out', str(out), '--steps', '2',
                '--stream-batch', '4', '--composition-batch', '4', '--validate-every', '1',
                '--save-every', '1', *extras]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            with patch('experiments.text_regions.integrated_train.evaluate_capabilities',
                       side_effect=scores or (lambda *a, **kw: progress())):
                main()

    def make_parent(self, path):
        # Reuse the exercised composition parent format; set inference defaults
        # to the two-round execution already used by its training objective.
        continuation_tests.ContinuationChecks().make_parent(path)
        data = torch.load(path, weights_only=True)
        data['config'].update(rounds=2, recurrent_feedback=True)
        torch.save(data, path)

    def test_joint_cli_exact_resume_including_both_runtime_states(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent = root/'parent.pt'; self.make_parent(parent)
            self.run_cli(parent, root/'full')
            self.run_cli(parent, root/'split', ('--stop-after', '1'))
            self.run_cli(parent, root/'split', ('--resume',))
            a = torch.load(root/'full/last.pt', weights_only=True)
            b = torch.load(root/'split/last.pt', weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng', 'states', 'streams'):
                assert_tree_equal(self, a[key], b[key])
            for key in a['totals']:
                if key != 'training_seconds': self.assertEqual(a['totals'][key], b['totals'][key])
            self.assertNotEqual(a['states']['correction']['input_mode'], a['states']['views']['input_mode'])
            initial = torch.load(root/'full/initial.pt', weights_only=True)
            old = torch.load(parent, weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng'):
                assert_tree_equal(self, old[key], initial[key])
            self.assertEqual(a['totals']['composition_episodes'], 8)
            self.assertEqual(a['totals']['views_final_queries'], 4)
            self.assertEqual(a['totals']['correction_final_queries'], 0)

    def test_regressed_model_does_not_replace_preserved_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent = root/'parent.pt'; self.make_parent(parent)
            self.run_cli(parent, root/'run', scores=[progress(.3), progress(.5), progress(.9, False)])
            selected = torch.load(root/'run/best.pt', weights_only=True)
            self.assertEqual(selected['step'], 1)
            with self.assertRaises(ValueError):
                from .integrated_train import load_integrated
                bad = copy.deepcopy(selected['protocol']); bad['lr'] *= 2
                load_integrated(root/'run/best.pt', bad)

    def test_view_evaluation_masks_preserve_work_and_use_256_logits(self):
        torch.manual_seed(802)
        model = build_model(GraphConfig.tiny(readout_mode='concat', rounds=2,
                                            recurrent_feedback=True), .5, 1.)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        results = evaluate_views(model, 'independent', 4, 4, gap=0,
                                  controls=('all', 'b_only', 'reset_query'))
        self.assertTrue(model.training)
        for query in ('shared', 'left', 'right'):
            rows = [results[control][query] for control in results]
            self.assertEqual(len({r['expert_executions'] for r in rows}), 1)
            self.assertEqual(len({tuple(r['episode_ids']) for r in rows}), 1)
            self.assertEqual(rows[0]['episodes'], 4)
            self.assertEqual(rows[0]['majority_class_chance'], .25)
        assert_tree_equal(self, before, model.state_dict())


if __name__ == '__main__': unittest.main()
