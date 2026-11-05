"""Credit-path isolation, objective equivalence and exact recovery."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch

from .episode_data import make_batch
from .fact_supervision import attach_heads
from .gradient_horizon import episode_forward, episode_loss, batch_digest
from .horizon_train import main
from .integrated_train import make_streams, save_integrated, stream_loss
from .test_fact_supervision import model_fixture
from .test_integrated import progress
from .test_probe import assert_tree_equal
from .view_data import make_view_batch

torch.set_num_threads(1)


class GradientHorizonChecks(unittest.TestCase):
    def fixture(self):
        model, optimizer = model_fixture(); attach_heads(model, optimizer)
        return model, optimizer

    def test_detachment_preserves_logits_state_losses_and_budgets(self):
        model, _ = self.fixture()
        for batch in (make_batch('correction', 'train', 0, 1, gap=0),
                      make_view_batch('train', 0, 1, gap=0)):
            a, ar = episode_loss(model, batch, 8, False)
            b, br = episode_loss(model, batch, 8, True)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(ar['logits'], br['logits']))
            assert_tree_equal(self, ar['state'].to_dict(), br['state'].to_dict())
            for key in ('expert_executions', 'source_occurrences', 'external_events', 'retention'):
                self.assertEqual(ar[key], br[key])

    def test_full_chunks_match_unsliced_forward_loss_and_gradient(self):
        model, _ = self.fixture()
        # FP64 isolates graph equivalence from cancellation in the large
        # normalization gradients of this untrained fixture.
        model.double()
        batch = make_view_batch('train', 0, 1, gap=0)
        full, fr = stream_loss(model, batch, None, 1.)
        selected = tuple(model.parameters())
        full_grad = torch.autograd.grad(full, selected, allow_unused=True)
        chunked, cr = episode_loss(model, batch, 8, False)
        chunk_grad = torch.autograd.grad(chunked, selected, allow_unused=True)
        self.assertTrue(torch.equal(fr['logits'], cr['logits']))
        torch.testing.assert_close(full, chunked, rtol=1e-6, atol=1e-6)
        for a, b in zip(full_grad, chunk_grad):
            if a is None or b is None: self.assertIs(a, b)
            else: torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)

    def test_only_full_horizon_connects_final_query_to_earlier_write(self):
        model, _ = self.fixture()
        batch = make_batch('recall', 'train', 0, 1, gap=8)
        gradients = []
        for detach in (False, True):
            embedded = []
            def remember(module, inputs, output):
                output.retain_grad(); embedded.append(output)
            hook = model.hat.byte_embedding.register_forward_hook(remember)
            model.zero_grad(set_to_none=True)
            result = episode_forward(model, batch, 16, detach, False)
            result['logits'][:, -1].square().sum().backward()
            hook.remove()
            # '=value' is at event 8, strictly before the cut at event 16.
            gradients.append(0. if embedded[8].grad is None else float(embedded[8].grad.norm()))
        self.assertGreater(gradients[0], 0.)
        self.assertEqual(gradients[1], 0.)

    def test_loss_labels_never_enter_forward_and_hash_tracks_observations(self):
        model, _ = self.fixture()
        batch = make_view_batch('train', 0, 1, gap=0)
        changed = replace(batch, targets=torch.where(batch.targets.ne(-100), 65+(batch.targets-64)%4, -100),
                          factors={k:65+(v-64)%4 for k,v in batch.factors.items()})
        a, ar = episode_loss(model, batch, 8)
        b, br = episode_loss(model, changed, 8)
        self.assertTrue(torch.equal(ar['logits'], br['logits']))
        self.assertNotEqual(float(a.detach()), float(b.detach()))
        self.assertNotEqual(batch_digest(batch), batch_digest(changed))
        altered = replace(batch, inputs={k:v.clone() for k,v in batch.inputs.items()})
        altered.inputs['A'][0, 0] ^= 1
        self.assertNotEqual(batch_digest(batch), batch_digest(altered))

    def test_exact_recovery_and_matched_initialization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent = root/'parent.pt'
            model, optimizer = self.fixture()
            for p in model.parameters(): p.grad = torch.ones_like(p) * .001
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            protocol = dict(stream_batch=1, composition_batch=4, segment=16, memory_gap=0, view_gap=0,
                            seed=507, lr=.0005, threads=1, validation_count=64,
                            task_weights={'composition':1., 'correction':1., 'views':1.},
                            fact_supervision={'weight':0., 'head_seed':20261108})
            streams = make_streams(protocol)
            save_integrated(parent, model, optimizer, random.Random(93), streams,
                            {k:None for k in streams}, 1, protocol, [], {})
            def run(out, extras=()):
                args = ['horizon', '--parent', str(parent), '--out', str(root/out), '--steps','2',
                        '--stream-batch','1','--composition-batch','4','--validate-every','1',
                        '--save-every','1', *extras]
                with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()), \
                     patch('experiments.text_regions.horizon_train.evaluate_capabilities', return_value=progress()):
                    main()
            run('full'); run('resumed', ('--stop-after','1')); run('resumed', ('--resume',))
            a = torch.load(root/'full/last.pt', weights_only=True)
            b = torch.load(root/'resumed/last.pt', weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng', 'streams', 'states'):
                assert_tree_equal(self, a[key], b[key])
            run('truncated', ('--stop-after','1','--detach-between'))
            x = torch.load(root/'full/initial.pt', weights_only=True)
            y = torch.load(root/'truncated/initial.pt', weights_only=True)
            original = torch.load(parent, weights_only=True)
            for key in ('weights','optimizer','rng','torch_rng'):
                assert_tree_equal(self, x[key], y[key]); assert_tree_equal(self, x[key], original[key])
            truncated = torch.load(root/'truncated/last.pt', weights_only=True)
            for key in ('batch_sha256','composition_nats','correction_nats','views_nats',
                        'expert_executions','source_occurrences','external_events'):
                self.assertEqual(a['history'][1][key], truncated['history'][1][key])
            first = torch.load(root/'full/step0001.pt', weights_only=True)
            self.assertTrue(any(not torch.equal(value, truncated['weights'][name])
                                for name, value in first['weights'].items()))


if __name__ == '__main__': unittest.main()
