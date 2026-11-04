"""Engine invariants, legacy parity, and exact training continuation."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest
import torch
from torch.nn import functional as F
from ..hat_minkron.events import ByteBatch
from ..hat_minkron.model import HATConfig, HATSHoE
from ..hat_minkron.train import optimizer_for
from .config import EdgeSpec, GraphConfig, NodeSpec
from .data import OrderedByteStreams
from .graph import EdgeSubstitution
from .model import RegionLanguageModel
from .ports import tensor_list
from .probes import make_episode
from .state import GraphState
from .train import evaluate, load_checkpoint, save_checkpoint

torch.set_num_threads(2)


def legacy_execute(self, payload, prior, states, backend='chunk', memory_enabled=True, trace=False):
    """Pinned b4bb7cf execution body: parity reference, not production code."""
    x, aux, actual, final, details = payload, payload.new_zeros(()), [], [], []
    for layer, block in enumerate(self.blocks):
        x, memory, route = block(x, prior[:, :, layer], states[layer], backend, memory_enabled)
        final.append(memory)
        aux = aux + route['aux']
        actual.append(route['group_scores'])
        if trace:
            details.append({k: v.detach() for k, v in route.items()})
    logits = self.decoder(self.final_norm(x))
    with torch.no_grad():
        observed, lace = self.lace(torch.stack(actual, -2).detach(), F.layer_norm(payload.detach(), (self.cfg.width,)))
    return {'logits': logits, 'state': tuple(final), 'aux': aux/len(self.blocks),
            'feedback_prior': observed, 'feedback_lace_ids': lace['lace_ids'], 'trace': details}


def same_tensors(test, first, second, exact=True):
    a, b = tensor_list(first), tensor_list(second)
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0 if exact else 1e-4, atol=0 if exact else 1e-5)


class RegionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(819)
        self.model = RegionLanguageModel(GraphConfig.tiny()).eval()
        self.x = torch.tensor([[65, 66, 67, 68], [71, 72, 73, 74]])

    def test_01_legacy_logits_state_and_gradients_are_identical(self):
        c = HATConfig(width=32, experts=192, groups=8, experts_per_group=2, address_dim=8,
                      laces=16, hat_symbols=16, kron_heads=2, kron_key=4, kron_value=4, exploration=0)
        new = HATSHoE(c).eval()
        old = copy.deepcopy(new)
        for child in old.children_blocks:
            child.execute = types.MethodType(legacy_execute, child)
        a = new(self.x, self.x, forced_child=0, force_discovery=True)
        b = old(self.x, self.x, forced_child=0, force_discovery=True)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        same_tensors(self, a['state'], b['state'])
        a['loss'].backward()
        b['loss'].backward()
        for p, q in zip(new.parameters(), old.parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_02_shared_weights_have_separate_state_and_no_local_decoder(self):
        state = self.model.initial_state(2)
        for depth in range(3):
            self.assertIs(self.model.nodes['A'].core.blocks[depth].experts.shared,
                          self.model.nodes['B'].core.blocks[depth].experts.shared)
            self.assertNotEqual(state.nodes['A'].blocks[depth].data_ptr(), state.nodes['B'].blocks[depth].data_ptr())
        self.assertIsNone(self.model.nodes['P'].core.decoder)
        m = RegionLanguageModel()
        p = m.parameter_summary()
        self.assertEqual(p['physical_experts'], 1536)
        self.assertEqual(p['expert_matrix_parameters'], 1228800)
        self.assertEqual(p['expert_calls_per_round_per_external_event'], 168)
        self.assertTrue(2_000_000 < p['total_unique_parameters'] < 3_000_000)

    def test_03_snapshot_read_survives_replacement_without_mutation(self):
        old = self.model(self.x)['state'].detach()
        before = old.clone()
        reader = old.ports.snapshot()['A']
        newer = self.model(self.x, state=old)['state']
        same_tensors(self, old, before)
        self.assertIs(old.ports.latest['A'], reader)
        self.assertFalse(torch.equal(reader.payload, newer.ports.latest['A'].payload))
        self.assertTrue(bool((newer.ports.latest['A'].version > reader.version).all()))

    def test_04_detach_and_serialization_cover_every_tensor(self):
        state = self.model(self.x, rounds=2, feedback=True)['state']
        self.assertTrue(any(x.grad_fn is not None for x in state.tensors()))
        detached = state.detach()
        self.assertTrue(all(x.grad_fn is None and not x.requires_grad for x in detached.tensors()))
        restored = GraphState.from_dict(detached.to_dict())
        same_tensors(self, detached, restored)
        self.assertEqual(detached.storage_bytes(), restored.storage_bytes())
        self.assertTrue(state.ports.latest['A'].header.memberships.requires_grad)

    def test_05_inactive_node_preserves_all_state_and_has_no_private_gradient(self):
        old = self.model(self.x)['state'].detach()
        masks = {'B': torch.zeros_like(self.x, dtype=torch.bool)}
        result = self.model(self.x, self.x, state=old, node_masks=masks)
        same_tensors(self, old.nodes['B'], result['state'].nodes['B'])
        same_tensors(self, old.ports.latest['B'], result['state'].ports.latest['B'])
        result['loss'].backward()
        for block in self.model.nodes['B'].core.blocks:
            self.assertIsNone(block.experts.private.up.grad)
        expected = 6*3*2*2*self.x.numel()
        self.assertEqual(result['expert_executions'], expected)

    def test_06_selective_row_reset_matches_fresh_and_carried_rows(self):
        old = self.model(self.x)['state'].detach()
        result = self.model(self.x, state=old, reset=torch.tensor([True, False]))
        fresh = self.model(self.x[:1])
        carry = self.model(self.x, state=old)
        torch.testing.assert_close(result['logits'][:1], fresh['logits'], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(result['logits'][1:], carry['logits'][1:], rtol=0, atol=0)
        self.assertEqual(result['state'].epochs.tolist(), [1, 0])
        for name in self.model.nodes:
            self.assertEqual(result['state'].nodes[name].clock.tolist(), [4, 8])

    def test_07_chunked_unchunked_and_single_event_inference_agree(self):
        with torch.no_grad():
            full = self.model(self.x, rounds=2, feedback=True)
            left = self.model(self.x[:, :1], rounds=2, feedback=True)
            right = self.model(self.x[:, 1:], state=left['state'], rounds=2, feedback=True)
            torch.testing.assert_close(full['logits'], torch.cat((left['logits'], right['logits']), 1), rtol=0, atol=0)
            same_tensors(self, full['state'], right['state'])

    def test_08_future_suffix_and_targets_never_change_earlier_predictions(self):
        a = self.model(self.x, rounds=2, feedback=True)
        changed = self.x.clone()
        changed[:, 2:] += 10
        b = self.model(changed, rounds=2, feedback=True)
        c = self.model(self.x, changed, rounds=2, feedback=True)
        torch.testing.assert_close(a['logits'][:, :2], b['logits'][:, :2], rtol=0, atol=0)
        torch.testing.assert_close(a['logits'], c['logits'], rtol=0, atol=0)
        same_tensors(self, a['state'], c['state'])

    def test_09_delayed_edges_only_consume_previous_round(self):
        result = self.model(self.x[:, :1], rounds=2, feedback=True, trace=True)
        for record in result['trace']:
            for dependency in record['dependencies']:
                if dependency.edge.endswith(('lateral', 'descending')):
                    self.assertEqual(record['round'], 1)
                    self.assertTrue(bool((dependency.version == 1).all()))
                    self.assertTrue(bool((dependency.round_index == 0).all()))
                elif dependency.edge.endswith('forward'):
                    self.assertTrue(bool((dependency.round_index == record['round']).all()))
        self.assertTrue(any(d.edge == 'P__Q__lateral' for r in result['trace'] for d in r['dependencies']))

    def test_10_independent_stage_order_is_irrelevant(self):
        a = self.model(self.x, rounds=2, feedback=True)
        b = self.model(self.x, rounds=2, feedback=True, reverse_independent=True)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        same_tensors(self, a['state'], b['state'])

    def test_11_occurrences_count_once_and_every_round_runs_every_pipeline(self):
        counts = {'adapter': 0, 'block': 0}
        def count(which):
            def hook(*_):
                counts[which] += 1
            return hook
        hooks = [self.model.nodes['A'].adapter.register_forward_hook(count('adapter')),
                 self.model.nodes['A'].core.blocks[0].register_forward_hook(count('block'))]
        result = self.model(self.x, rounds=4, feedback=True, trace=True)
        for hook in hooks:
            hook.remove()
        self.assertEqual(counts, {'adapter': 16, 'block': 16})
        self.assertEqual(result['state'].events_seen.tolist(), [4, 4])
        self.assertEqual(result['state'].rounds_seen.tolist(), [16, 16])
        self.assertEqual(result['state'].ingress['text'].clocks[0].tolist(), [4, 4])
        self.assertEqual(result['source_occurrences'], self.x.numel())
        self.assertEqual(result['expert_executions'], 4*84*self.x.numel())
        records = [r for r in result['trace'] if r['node'] == 'A' and r['event'] == 0]
        self.assertTrue(any(not torch.equal(records[0]['routing']['experts'], r['routing']['experts']) for r in records[1:]))

    def test_12_downstream_gradients_cross_nodes_and_earlier_rounds(self):
        captured = []
        def keep(module, args, result):
            result['latent'].retain_grad()
            captured.append(result['latent'])
        hook = self.model.nodes['A'].register_forward_hook(keep)
        result = self.model(self.x[:, :1], self.x[:, :1], rounds=2, feedback=True)
        result['loss'].backward()
        hook.remove()
        self.assertGreater(float(captured[0].grad.abs().sum()), 0)
        for name in ('A__U__forward', 'U__P__forward', 'P__Q__lateral'):
            self.assertGreater(float(self.model.connections.edges[name].value.weight.grad.abs().sum()), 0, name)

    def test_13_absent_view_is_independent_of_its_bytes_and_differs_from_zero(self):
        views = {v: self.x.clone() for v in ('A', 'B', 'C')}
        masks = {'B': torch.zeros_like(self.x, dtype=torch.bool)}
        a = self.model(views, availability=masks)
        views['B'] = views['B']+30
        b = self.model(views, availability=masks)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        self.assertEqual(a['state'].ingress['B'].clocks, {})
        views['B'] = torch.zeros_like(self.x)
        zero = self.model(views)
        self.assertFalse(torch.equal(a['logits'], zero['logits']))
        self.assertEqual(a['expert_executions'], zero['expert_executions'])
        self.assertEqual(a['source_occurrences'], 2*self.x.numel())

    def test_14_edge_mask_and_substitution_have_only_reachable_effects(self):
        a = self.model(self.x)
        mask = {'A__U__forward': torch.zeros_like(self.x, dtype=torch.bool)}
        b = self.model(self.x, edge_masks=mask)
        sub = EdgeSubstitution(torch.randn(2, 4, 32), torch.zeros(2, 4, 3, 4))
        c = self.model(self.x, substitutions={'A__U__forward': sub})
        for result in (b, c):
            self.assertEqual(result['expert_executions'], a['expert_executions'])
            for n in ('A', 'B', 'C', 'V', 'Q'):
                same_tensors(self, a['state'].nodes[n], result['state'].nodes[n])
            for n in ('U', 'P'):
                self.assertFalse(torch.equal(a['state'].ports.latest[n].payload, result['state'].ports.latest[n].payload))
        dependency = next(d for d in b['state'].ports.latest['U'].dependencies if d.edge == 'A__U__forward')
        self.assertFalse(bool(dependency.consumed.any()))

    def test_15_custom_topology_and_invalid_cycle(self):
        nodes = (NodeSpec('X', 'R0', 'view'), NodeSpec('Y', 'R1'), NodeSpec('Z', 'R1'))
        edges = (EdgeSpec('X', 'Y'), EdgeSpec('Y', 'Z'), EdgeSpec('Z', 'X', 'descending'))
        cfg = GraphConfig.tiny(nodes=nodes, edges=edges, readout=('Z',), rounds=2, recurrent_feedback=True)
        model = RegionLanguageModel(cfg)
        out = model(self.x)
        self.assertEqual(set(out['state'].nodes), {'X', 'Y', 'Z'})
        self.assertEqual(out['expert_executions'], 3*3*4*2*self.x.numel())
        with self.assertRaises(ValueError):
            RegionLanguageModel(replace(cfg, edges=edges+(EdgeSpec('Z', 'X'),)))

    def test_16_invalidation_and_row_reset_do_not_change_existing_reader(self):
        state = self.model(self.x)['state']
        p = state.ports.latest['A']
        ports = state.ports.invalidate('A', torch.tensor([True, False]))
        self.assertEqual(ports.latest['A'].valid.tolist(), [False, True])
        self.assertEqual(p.valid.tolist(), [True, True])
        reset = state.reset(torch.tensor([True, False]))
        self.assertFalse(bool(reset.ports.latest['A'].valid[0]))
        self.assertEqual(float(reset.ports.latest['A'].header.memberships[0].detach().abs().sum()), 0)

    def test_17_stream_boundaries_and_cursor_resume(self):
        streams = OrderedByteStreams([b'ABC', b'XYZ'], 2)
        chunk = streams.next_chunk(3)
        self.assertEqual(chunk.reset.tolist(), [[True, False, True], [True, False, True]])
        self.assertEqual(chunk.values.tolist(), [[65, 66, 65], [88, 89, 88]])
        saved = streams.state_dict()
        expected = streams.next_chunk(7)
        restored = OrderedByteStreams([b'ABC', b'XYZ'], 2)
        restored.load_state_dict(saved)
        actual = restored.next_chunk(7)
        same_tensors(self, actual, expected)
        with self.assertRaises(ValueError):
            OrderedByteStreams([b'ABx', b'XYZ'], 2).load_state_dict(saved)

    def test_18_boundary_inside_chunk_matches_explicit_reset(self):
        masks = torch.zeros_like(self.x, dtype=torch.bool)
        masks[:, 2] = True
        whole = self.model(self.x, reset=masks)
        fresh = self.model(self.x[:, 2:])
        torch.testing.assert_close(whole['logits'][:, 2:], fresh['logits'], rtol=0, atol=0)

    def test_19_checkpoint_continues_optimizer_rng_cursors_and_state_exactly(self):
        streams = OrderedByteStreams([b'ABCDEF', b'UVWXYZ'], 2)
        model = self.model.train()
        optimizer = optimizer_for(model, .002)
        def step(m, opt, state, cursor):
            c = cursor.next_chunk(2)
            opt.zero_grad(set_to_none=True)
            result = m(c.values, c.targets, state=state, reset=c.reset)
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
            return result['state'].detach(), result['logits'].detach()
        state, _ = step(model, optimizer, None, streams)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            save_checkpoint(path, model, optimizer, state, streams, 1, {'lr': .002})
            rng_expected = torch.rand(3)
            expected, logits = step(model, optimizer, state, streams)
            cursor = OrderedByteStreams([b'ABCDEF', b'UVWXYZ'], 2)
            loaded, opt, carried, saved = load_checkpoint(path, cursor)
            torch.testing.assert_close(torch.rand(3), rng_expected, rtol=0, atol=0)
            actual, actual_logits = step(loaded, opt, carried, cursor)
            torch.testing.assert_close(logits, actual_logits, rtol=0, atol=0)
            same_tensors(self, expected, actual)
            for p, q in zip(model.parameters(), loaded.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            for p, q in zip(optimizer.state.values(), opt.state.values()):
                for key in p:
                    torch.testing.assert_close(p[key], q[key], rtol=0, atol=0)

    def test_20_long_stream_has_bounded_state_and_dependencies(self):
        with torch.no_grad():
            state = self.model(self.x, rounds=2, feedback=True)['state']
            size = state.tensor_bytes()
            for _ in range(5):
                state = self.model(self.x, state=state, rounds=2, feedback=True)['state']
                self.assertEqual(state.tensor_bytes(), size)
                self.assertLessEqual(max(len(p.dependencies) for p in state.ports.latest.values()), 4)

    def test_21_validation_preserves_training_state_and_rng(self):
        state = self.model(self.x)['state'].detach()
        before = state.clone()
        rng = torch.get_rng_state().clone()
        buffers = {n: p.clone() for n, p in self.model.named_buffers()}
        self.model.train()
        loss = evaluate(self.model, b'ABCD\n'*20, batch_size=2, context=3, chunks=1)
        self.assertTrue(self.model.training)
        self.assertTrue(0 < loss < 20)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        same_tensors(self, state, before)
        for n, p in self.model.named_buffers():
            torch.testing.assert_close(p, buffers[n], rtol=0, atol=0)

    def test_22_split_gradients_cross_chunk_boundaries_without_detach(self):
        other = copy.deepcopy(self.model)
        full = self.model(self.x, rounds=2, feedback=True)
        left = other(self.x[:, :2], rounds=2, feedback=True)
        right = other(self.x[:, 2:], state=left['state'], rounds=2, feedback=True)
        full['logits'].square().mean().backward()
        torch.cat((left['logits'], right['logits']), 1).square().mean().backward()
        for (name, p), (_, q) in zip(self.model.named_parameters(), other.named_parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)

    def test_23_fixed_probe_splits_and_fresh_identifiers(self):
        for kind in ('recall', 'shared_private', 'composition'):
            a = make_episode(kind, 'train', 0)
            self.assertEqual(a, make_episode(kind, 'train', 0))
            self.assertNotEqual(a.seed, make_episode(kind, 'validation', 0).seed)
            self.assertNotEqual(a.answer, make_episode(kind, 'train', 1).answer)

    def test_24_provenance_and_bad_masks_are_checked(self):
        events = ByteBatch.stream(self.x, link=2, slot=7)
        result = self.model(events)
        ingress = result['state'].ports.latest['@text']
        self.assertEqual(ingress.links.tolist(), [2, 2])
        self.assertEqual(ingress.slots.tolist(), [7, 7])
        self.assertEqual(ingress.occurrence.tolist(), [3, 3])
        self.assertEqual(result['state'].ports.latest['B'].header.block_namespaces,
                         (('B', 0), ('B', 1), ('B', 2)))
        with self.assertRaises(ValueError):
            self.model(self.x, node_masks={'unknown': torch.ones_like(self.x, dtype=torch.bool)})
        with self.assertRaises(ValueError):
            self.model(self.x, availability={'A': torch.ones_like(self.x)})

    def test_25_transport_requires_a_second_complete_round(self):
        model = RegionLanguageModel(GraphConfig.tiny(readout=('Q',), recurrent_feedback=True)).eval()
        views = {v: self.x[:, :1] for v in ('A', 'B', 'C')}
        changed = {**views, 'A': views['A']+50}
        with torch.no_grad():
            a1, b1 = model(views, rounds=1), model(changed, rounds=1)
            a2, b2 = model(views, rounds=2), model(changed, rounds=2)
        torch.testing.assert_close(a1['logits'], b1['logits'], rtol=0, atol=0)
        self.assertFalse(torch.equal(a2['logits'], b2['logits']))
        self.assertEqual(a2['expert_executions'], b2['expert_executions'])

    def test_26_carried_state_does_not_retain_the_input_window_storage(self):
        with torch.no_grad():
            values = self.x.repeat(1, 8)
            views = {v: values for v in ('A', 'B', 'C')}
            masks = {'A': torch.ones_like(values, dtype=torch.bool)}
            state = self.model(views, availability=masks, rounds=1)['state']
            for tensor in state.tensors():
                self.assertLessEqual(tensor.untyped_storage().nbytes(), tensor.numel()*tensor.element_size())

    def test_27_default_two_round_segment_has_finite_gradients(self):
        # Regression for the closed-loop gradient failure at gates initialized
        # to 0.5. Do not shorten this segment: the failure appeared at 128 bytes.
        torch.manual_seed(20261104)
        model = RegionLanguageModel(GraphConfig(rounds=2, recurrent_feedback=True)).train()
        x = torch.randint(0, 256, (1, 128))
        result = model(x, x.roll(-1, 1))
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        self.assertTrue(bool(torch.isfinite(norm)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RegionChecks))
    raise SystemExit(0 if result.wasSuccessful() else 1)
