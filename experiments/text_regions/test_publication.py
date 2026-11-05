"""Publication semantics, causal data, and recoverable learning checks."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch
from ..hat_minkron.train import optimizer_for
from .composition_train import atomic_save
from .config import GraphConfig, NodeSpec
from .model import RegionLanguageModel
from .ports import tensor_list
from .publication_data import make_batch, causal_labels, batch_digest
from .publication_train import forward_batch, update, load
from .state import GraphState

torch.set_num_threads(1)


def tiny(selective=True):
    torch.manual_seed(177)
    return RegionLanguageModel(GraphConfig.tiny(nodes=(NodeSpec('A', 'R0', 'A'),), edges=(),
        readout=('A',), readout_mode='concat', selective_publication=('A',) if selective else (),
        evidence_gated_ports=True))


class PublicationChecks(unittest.TestCase):
    def test_hold_preserves_all_publication_fields_but_advances_node_state(self):
        model = tiny()
        first = model({'A': torch.tensor([[65], [66]])})['state']
        snapshot = [t.clone() for t in tensor_list(first.ports.latest['A'])]
        result = model({'A': torch.tensor([[67], [68]])}, state=first,
                       publication_overrides={'A': 'hold'}, return_publication_activity=True)
        for old, new, frozen in zip(tensor_list(first.ports.latest['A']),
                                    tensor_list(result['state'].ports.latest['A']), snapshot):
            torch.testing.assert_close(new, old, rtol=0, atol=0)
            torch.testing.assert_close(old, frozen, rtol=0, atol=0)
        self.assertEqual(result['state'].nodes['A'].clock.tolist(), [2, 2])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            first.nodes['A'].blocks, result['state'].nodes['A'].blocks)))
        self.assertFalse(bool(result['publication_activity']['A']['written'].any()))

    def test_learned_threshold_and_inactive_rows(self):
        model = tiny()
        with torch.no_grad(): model.nodes['A'].publication_gate.bias.fill_(-10.)
        x = {'A': torch.tensor([[65], [66]])}
        result = model(x)
        self.assertFalse(bool(result['state'].ports.latest['A'].valid.any()))
        with torch.no_grad(): model.nodes['A'].publication_gate.bias.fill_(10.)
        result = model(x, node_masks={'A': torch.tensor([[True], [False]])})
        self.assertEqual(result['state'].ports.latest['A'].version.tolist(), [1, 0])
        self.assertEqual(result['state'].nodes['A'].clock.tolist(), [1, 0])

    def test_hold_does_not_gain_validity_from_arriving_evidence(self):
        result = tiny()({'A': torch.tensor([[65]])}, publication_overrides={'A': 'hold'})
        self.assertTrue(bool(result['state'].ports.latest['@A'].valid.all()))
        self.assertFalse(bool(result['state'].ports.latest['A'].valid.any()))

    def test_absent_source_cannot_validate_a_new_publication(self):
        result = tiny()({'A': torch.tensor([[65]])}, availability={'A': torch.tensor([[False]])})
        self.assertFalse(bool(result['state'].ports.latest['A'].valid.any()))
        self.assertEqual(result['source_occurrences'], 0)

    def test_selective_reset_and_serialization(self):
        model = tiny(); state = model({'A': torch.tensor([[65], [66]])})['state'].detach()
        state = GraphState.from_dict(state.to_dict())
        result = model({'A': torch.tensor([[67], [68]])}, state=state, reset=torch.tensor([True, False]),
                       publication_overrides={'A': 'hold'})
        self.assertEqual(result['state'].ports.latest['A'].valid.tolist(), [False, True])
        torch.testing.assert_close(result['state'].ports.latest['A'].payload[1], state.ports.latest['A'].payload[1], rtol=0, atol=0)
        self.assertFalse(any(t.requires_grad for t in result['state'].detach().tensors()))

    def test_always_override_matches_legacy_and_same_expert_budget(self):
        model = tiny(); legacy = tiny(False)
        legacy.load_state_dict({k:v for k,v in model.state_dict().items() if '.publication_gate.' not in k})
        x = {'A': torch.tensor([[65, 66, 67], [68, 65, 66]])}
        old, new = legacy(x), model(x, publication_overrides={'A': 'always'})
        torch.testing.assert_close(old['logits'], new['logits'], rtol=0, atol=0)
        for a, b in zip(old['state'].tensors(), new['state'].tensors()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        held = model(x, publication_overrides={'A': 'hold'})
        self.assertEqual(old['expert_executions'], new['expert_executions'])
        self.assertEqual(old['expert_executions'], held['expert_executions'])

    def test_round_capture_and_config_compatibility(self):
        model = tiny()
        r = model({'A': torch.tensor([[65, 66]])}, rounds=2, return_publication_activity=True)
        self.assertEqual(tuple(r['publication_activity']['A']['logit'].shape), (1, 2, 2))
        cfg = model.cfg.to_dict(); cfg.pop('selective_publication')
        self.assertEqual(GraphConfig.from_dict(cfg).selective_publication, ())
        self.assertEqual(GraphConfig.from_dict(model.cfg.to_dict()).selective_publication, ('A',))
        for bad in (('X',), ('A', 'A')):
            with self.assertRaises(ValueError): RegionLanguageModel(replace(model.cfg, selective_publication=bad))
        with self.assertRaises(ValueError): model({'A': torch.tensor([[65]])}, publication_overrides={'X': 'hold'})

    def test_later_inputs_do_not_change_earlier_logits(self):
        model = tiny(); a = torch.tensor([[65, 66, 67, 68]])
        b = a.clone(); b[:, 2:] = 70
        torch.testing.assert_close(model({'A':a})['logits'][:, :2], model({'A':b})['logits'][:, :2], rtol=0, atol=0)


class PublicationDataChecks(unittest.TestCase):
    def test_bare_values_invalid_markers_and_reset(self):
        x = torch.tensor([list(b'A=x=BCC=D')])
        reset = torch.zeros_like(x, dtype=torch.bool); reset[:, -1] = True
        available = torch.ones_like(reset)
        writes, retained = causal_labels(x, reset, available)
        self.assertEqual(writes.nonzero()[:, 1].tolist(), [4])
        self.assertEqual(retained.tolist(), [[-100,-100,-100,-100,66,66,66,66,-100]])
        available[:, 3] = False
        self.assertFalse(bool(causal_labels(x, reset, available)[0].any()))

    def test_withheld_values_and_hidden_inputs_never_become_labels(self):
        for control in ('carried', 'withheld', 'no_writes', 'reset', 'absent'):
            batch = make_batch('test', 0, 32, 12, 16, 'correction', control)
            if control in ('no_writes', 'absent'):
                self.assertFalse(bool(batch.writes.any()))
                self.assertTrue(bool(batch.retained.eq(-100).all()))
            elif control == 'reset': self.assertTrue(bool(batch.retained[:, -1].eq(-100).all()))
            else:
                self.assertTrue(bool(batch.retained[:, -1].eq(batch.answers).all()))
                self.assertTrue(bool(batch.writes.sum(1).eq(1 if control == 'withheld' else 2).all()))

    def test_seed_split_and_row_timing(self):
        a = make_batch('train', 0, 32, 2)
        b = make_batch('train', 0, 32, 2)
        self.assertEqual(batch_digest(a), batch_digest(b))
        self.assertNotEqual(batch_digest(a), batch_digest(make_batch('test', 0, 32, 2)))
        self.assertGreater(a.writes.int().argmax(1).unique().numel(), 1)

    def test_labels_are_not_forward_inputs_and_chunk_boundaries_match(self):
        model = tiny(); batch = make_batch('train', 0, 2, 2, 0)
        changed = replace(batch, answers=batch.answers.roll(1), retained=torch.zeros_like(batch.retained), writes=~batch.writes)
        a = forward_batch(model, batch, 'learned')
        b = forward_batch(model, changed, 'learned', segment=7)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        for x, y in zip(a['state'].tensors(), b['state'].tensors()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_exact_optimizer_continuation(self):
        model = tiny(); optimizer = optimizer_for(model, .003)
        batch = make_batch('train', 0, 2, 2, 0)
        update(model, optimizer, batch, 'learned')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            atomic_save(path, dict(schema='shoebox-publication-v1', config=model.cfg.to_dict(), weights=model.state_dict(),
                optimizer=optimizer.state_dict(), protocol={'lr':.003}, torch_rng=torch.get_rng_state()))
            restored, second, _ = load(path)
            batch = make_batch('train', 2, 2, 2, 0)
            a = update(model, optimizer, batch, 'learned')
            b = update(restored, second, batch, 'learned')
            self.assertEqual(a, b)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            for key, state in optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, second.state_dict()['state'][key][name], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
