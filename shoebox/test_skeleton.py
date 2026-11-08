"""Behavioral shakedown: causality, framing, gradients, state and exact resume."""
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
import torch
from .config import SkeletonConfig, REGIONS, NAMES
from .model import CorticalModel
from .transport import Frame, byte_frame, publish
from .checkpoint import save, load

torch.set_num_threads(1)


class SkeletonChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfg = SkeletonConfig()
        self.model = CorticalModel(self.cfg)
        self.x = torch.tensor([[65, 66, 67], [67, 68, 69]])

    def test_topology_and_removed_controllers(self):
        self.assertEqual(len(NAMES), 18)
        self.assertEqual(len(set(id(n) for n in self.model.regions.values())), 18)
        for spec in REGIONS:
            for edge in spec.inputs:
                if not edge.delayed:
                    self.assertLess(NAMES.index(edge.source), NAMES.index(spec.name))
        forbidden = ('lace_head', 'anchor_reader', 'publication_gate', 'memory_context',
                     'retention', 'query_bias', 'capsule')
        self.assertFalse(any(any(word in name for word in forbidden) for name, _ in self.model.named_parameters()))
        state = self.model.initial_state(2)
        self.assertEqual(set(vars(state.nodes['L2'])), {'hat', 'blocks', 'clock', 'bytes_seen'})
        self.assertNotEqual(state.nodes['A1'].blocks[0].data_ptr(), state.nodes['B1'].blocks[0].data_ptr())

    def test_once_only_ingress_and_snapshot_feedback(self):
        with torch.no_grad():
            out = self.model(self.x, trace=True)
        state = out['state']
        self.assertEqual(state.event, 3)
        self.assertEqual(state.memory.writes, 3)
        for name in NAMES:
            self.assertEqual(state.nodes[name].clock, 3 if name in ('A1', 'B1') else 6)
        self.assertEqual(state.nodes['A1'].bytes_seen, 3)
        for record in out['trace']:
            spec = next(s for s in REGIONS if s.name == record['region'])
            delayed = {edge.source for edge in spec.inputs if edge.delayed}
            for frame in record['inputs']:
                if frame['source'] in delayed:
                    self.assertLess((frame['event'], frame['revision']), (record['event'], record['round']))
                if frame['source'].startswith('HC-'):
                    self.assertIn(record['region'], ('EC-C', 'EC-X'))
                    self.assertTrue(all(e < record['event'] for e in frame['episode_events']))
                    self.assertEqual(frame['origin_event'], frame['episode_events'])
                    self.assertTrue(all(r == self.cfg.rounds-1 for r in frame['origin_revision']))
                if frame['source'].startswith('external:'):
                    self.assertIn(record['region'], ('A1', 'B1'))
                    self.assertEqual(record['round'], 0)
            if record['region'] == 'L2' and record['event'] == 2:
                a1 = [f for f in record['inputs'] if f['source'] == 'A1']
                self.assertEqual([f['event'] for f in a1], [1, 2])
                self.assertEqual([f['width'] for f in a1], [4, 4])
        self.assertTrue(all(len(history) <= 2 for history in state.ports.values()))

    def test_chunking_causality_and_batch_independence(self):
        self.model.eval()
        with torch.no_grad():
            whole = self.model(self.x)
            first = self.model(self.x[:, :1])
            rest = self.model(self.x[:, 1:], state=first['state'])
            torch.testing.assert_close(whole['logits'], torch.cat((first['logits'], rest['logits']), 1), rtol=0, atol=0)
            changed = self.x.clone(); changed[:, -1] = 90
            altered = self.model(changed)
            torch.testing.assert_close(whole['logits'][:, :-1], altered['logits'][:, :-1], rtol=0, atol=0)
            alone = self.model(self.x[:1])
            torch.testing.assert_close(whole['logits'][:1], alone['logits'], atol=1e-5, rtol=1e-4)

    def test_complete_ports_and_order_are_observable(self):
        region = self.model.regions['L2']
        state = region.initial_state(2, 'cpu', torch.float32)
        one = Frame('A1', 0, 0, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        two = Frame('A1', 1, 0, one.values.flip(1))
        with torch.no_grad():
            a = region.hat([one, two], state.hat)[0]
            b = region.hat([two, one], state.hat)[0]
            c = region.hat([replace(one, source='B1'), two], state.hat)[0]
        self.assertGreater(float((a-b).abs().max()), 1e-7)
        self.assertGreater(float((a-c).abs().max()), 1e-7)
        logits = torch.randn(2, 4, 256, requires_grad=True)
        frame = byte_frame('A1', 2, 1, logits)
        torch.testing.assert_close(frame.distribution.sum(-1), torch.ones(2, 4), atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(frame.distribution)), 8)
        region.hat([frame], state.hat)[0].square().sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        ports = publish({}, 'A1', {4: frame})
        updated = publish(ports, 'A1', {4: replace(frame, revision=2)})
        self.assertEqual(len(updated['A1:4']), 1)
        self.assertEqual(ports['A1:4'][0].revision, 1)

    def test_task_loss_reaches_every_region_and_ec_boundary(self):
        out = self.model(self.x, targets=self.x.roll(-1, 1))
        out['ce'].backward()
        for name, region in self.model.regions.items():
            norms = [p.grad.abs().sum() for p in region.parameters() if p.grad is not None]
            self.assertTrue(norms, name)
            self.assertTrue(all(torch.isfinite(n) for n in norms), name)
            self.assertGreater(float(sum(norms)), 0., name)
            expert = [p.grad for n, p in region.named_parameters() if 'experts.private.up' in n and p.grad is not None]
            self.assertGreater(sum(float(g.abs().sum()) for g in expert), 0., name)
        self.assertTrue(torch.isfinite(out['loss']))
        self.assertGreater(float(self.model.hippocampus.address.weight.grad.abs().sum()), 0.)

    def test_memory_capacity_provenance_and_intervention(self):
        model = CorticalModel(replace(self.cfg, episode_capacity=2))
        with torch.no_grad():
            first = model(self.x)
            on = model(self.x[:, :1], state=first['state'], trace=True)
            off = model(self.x[:, :1], state=first['state'], memory_enabled=False, trace=True)
        self.assertEqual([e.event for e in first['state'].memory.episodes], [1, 2])
        self.assertEqual(first['state'].memory.writes, 3)
        self.assertEqual(off['state'].memory.writes, 3)
        self.assertGreater(float((on['logits']-off['logits']).abs().max()), 0.)
        for record in on['trace']:
            for frame in record['inputs']:
                if frame['source'] == 'HC-C':
                    self.assertEqual(frame['origin_source'], 'EC-C')
                if frame['source'] == 'HC-X':
                    self.assertEqual(frame['origin_source'], 'EC-X')
        self.assertFalse(any(f['source'].startswith('HC-') for r in off['trace'] for f in r['inputs']))
        with self.assertRaises(ValueError):
            e = first['state'].memory.episodes[-1]
            model.hippocampus.write(e.content, e.context, e.features[:, :16], e.features[:, 16:], first['state'].memory)

    def test_emitted_bytes_are_not_external_observations(self):
        with torch.no_grad():
            a = self.model(self.x[:, :1], continuation=torch.tensor([[10], [11]]), trace=True)
            b = self.model(self.x[:, :1], continuation=torch.tensor([[12], [13]]), trace=True)
        for name in ('A1', 'B1'):
            torch.testing.assert_close(a['state'].latents[name], b['state'].latents[name], atol=0, rtol=0)
        self.assertGreater(float((a['logits']-b['logits']).abs().max()), 0.)
        self.assertEqual(a['state'].nodes['A1'].bytes_seen, 1)
        self.assertTrue(all(r['region'] == 'Y' for r in a['trace'] if any(f['source'] == 'Y-output' for f in r['inputs'])))

    def test_exact_optimizer_and_state_resume(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=.001)
        out = self.model(self.x[:, :1], targets=self.x[:, 1:2])
        out['loss'].backward(); opt.step(); opt.zero_grad(set_to_none=True)
        state = out['state'].detach()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resume.pt'
            save(path, self.model, state, opt, step=1)
            clone, restored, opt2, data = load(path, optimizer_factory=lambda p: torch.optim.AdamW(p, lr=9.))
        self.assertEqual(data['step'], 1)
        for model, optimizer, previous in ((self.model, opt, state), (clone, opt2, restored)):
            result = model(self.x[:, 1:2], targets=self.x[:, 2:3], state=previous)
            result['loss'].backward(); optimizer.step()
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(restored.memory.writes, state.memory.writes)
        self.assertEqual(restored.storage_bytes(), state.storage_bytes())

    def test_invalid_inputs_and_configuration(self):
        for x in (torch.tensor([[256]]), torch.tensor([[1.5]]), torch.empty(1, 0, dtype=torch.long)):
            with self.assertRaises(ValueError): self.model(x)
        with self.assertRaises(ValueError): self.model(self.x, state=self.model.initial_state(1))
        with self.assertRaises(ValueError): self.model(self.x, targets=torch.full_like(self.x, -100))
        with self.assertRaises(ValueError): self.model(self.x, source=16)
        with self.assertRaises(ValueError): CorticalModel(replace(self.cfg, rounds=0))
        with self.assertRaises(ValueError): CorticalModel(replace(self.cfg, episode_active=100))


if __name__ == '__main__':
    unittest.main()
