"""Wide-port capacity, terminal recurrence, long gradients and compact state."""
import copy
from dataclasses import replace
import unittest
import weakref
import torch
from torch.nn import functional as F
from experiments.hat_minkron.kron import MinKron
from . import CorticalModel, SkeletonConfig, cortical_wide
from .runtime import minkron_last
from .diagnostics import PortMonitor, SavedTensorAccounting
from .transport import Frame


torch.set_num_threads(1)


def small_wide():
    return cortical_wide(width=16,expert_hidden=4,experts=48,groups=4,
                         hat_symbols=16,hat_active_symbols=4,laces=8,active_laces=2,
                         address_dim=4,kron_heads=1,kron_key=4,kron_value=4,
                         region_overrides={'A1':{'width':24},'B1':{'width':24},
                                           'L2':{'width':20},'EC-X':{'width':20}})


class WideChecks(unittest.TestCase):
    def test_accounting_does_not_retain_saved_tensor_owner(self):
        accounting = SavedTensorAccounting()
        x = torch.tensor([.3,.7],requires_grad=True)
        y = x.sin()
        owner = weakref.ref(y)
        packed = accounting.pack(y)
        self.assertIsNone(packed.grad_fn)
        self.assertEqual(packed.untyped_storage().data_ptr(),y.untyped_storage().data_ptr())
        del y
        self.assertIsNone(owner())
        with torch.autograd.graph.saved_tensors_hooks(accounting.pack,accounting.unpack):
            loss = x.sin().square().sum()
        loss.backward()
        torch.testing.assert_close(x.grad,2*x.detach().sin()*x.detach().cos())

    def test_entropy_separates_uncertainty_from_code_diversity(self):
        model = CorticalModel(SkeletonConfig())
        uniform = PortMonitor(model)
        fixed = torch.zeros(4,1,dtype=torch.long)
        uniform('L3',0,0,{'probabilities':{1:torch.full((4,1,256),1/256)},
                         'frames':{1:Frame('L3',0,0,fixed)},'experts':torch.zeros(4,3,1,dtype=torch.long)})
        a = uniform.snapshot()['ports']['L3:1']
        self.assertEqual(a['conditional_entropy_bits'],8.)
        self.assertEqual(a['conditioning_js_bits'],0.)
        self.assertEqual(a['hard_byte_entropy_bits'],0.)
        self.assertEqual(a['unique_complete_frames'],1)
        varying = PortMonitor(model)
        values = torch.arange(4).reshape(4,1)
        varying('L3',0,0,{'probabilities':{1:F.one_hot(values,256).float()},
                         'frames':{1:Frame('L3',0,0,values)},'experts':torch.zeros(4,3,1,dtype=torch.long)})
        b = varying.snapshot()['ports']['L3:1']
        self.assertEqual(b['conditional_entropy_bits'],0.)
        self.assertEqual(b['conditioning_js_bits'],2.)
        self.assertEqual(b['hard_byte_entropy_bits'],2.)
        self.assertEqual(b['unique_complete_frames'],4)

    def test_capacity_and_port_recipes(self):
        model = CorticalModel(cortical_wide())
        self.assertEqual(sum(p.numel() for p in model.parameters()),14976882)
        self.assertEqual(model.regions['A1'].hat.cfg.width,128)
        self.assertEqual(model.regions['L2'].hat.cfg.width,96)
        self.assertEqual(model.regions['G5'].hat.cfg.width,64)
        for spec in model.specs:
            for edge in spec.inputs:
                self.assertIn(edge.width,(4,8,16))
        self.assertEqual(set(model.regions['Y'].outputs),{'1'})
        self.assertEqual(set(model.regions['EC-C'].outputs),{'4','8','16'})
        block = model.regions['A1'].core.blocks[0]
        self.assertEqual(block.experts.private.up.shape,(32,32,128))

    def test_terminal_recurrence_value_and_gradient_parity(self):
        torch.manual_seed(112)
        for length in (1,3,9):
            original = MinKron(8,2,3,5,chunk_size=4,safe_zero=True).double()
            fast = copy.deepcopy(original)
            x = torch.randn(2,length,8,dtype=torch.float64,requires_grad=True)
            x2 = x.detach().clone().requires_grad_()
            state = torch.randn(2,2,3,5,dtype=torch.float64,requires_grad=True)
            state2 = state.detach().clone().requires_grad_()
            all_reads,end = original(x,state)
            last,end2 = minkron_last(fast,x2,state2)
            torch.testing.assert_close(last,all_reads[:,-1:],atol=1e-11,rtol=1e-10)
            torch.testing.assert_close(end,end2,atol=1e-11,rtol=1e-10)
            do,ds = torch.randn_like(last),torch.randn_like(end)
            ga = torch.autograd.grad((all_reads[:,-1:],end),(*original.parameters(),x,state),(do,ds))
            gb = torch.autograd.grad((last,end2),(*fast.parameters(),x2,state2),(do,ds))
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)

    def test_heterogeneous_graph_backend_parity(self):
        torch.manual_seed(113)
        reference = CorticalModel(small_wide()).double()
        optimized = copy.deepcopy(reference).set_execution_backend('optimized')
        x = torch.tensor([[65,66,67],[71,72,73]])
        a = reference(x,targets=x.roll(-1,1))
        b = optimized(x,targets=x.roll(-1,1))
        torch.testing.assert_close(a['logits'],b['logits'],atol=1e-8,rtol=1e-7)
        torch.testing.assert_close(a['loss'],b['loss'],atol=1e-9,rtol=1e-8)
        a['loss'].backward();b['loss'].backward()
        for (name,p),(_,q) in zip(reference.named_parameters(),optimized.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=2e-6,msg=name)

    def test_live_history_and_exact_compaction(self):
        torch.manual_seed(114)
        model = CorticalModel(small_wide()).double().set_execution_backend('optimized')
        taps = []
        def observe(name,event,revision,result):
            if name == 'A1' and event == 0:
                taps.append(result['latent'])
        x = torch.tensor([[65,66,67,68,69,70,71,72]])
        out = model(x,observer=observe)
        loss = F.cross_entropy(out['logits'][:,-1],torch.tensor([73]))
        grad, = torch.autograd.grad(loss,taps)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0.)
        dense = out['state'].detach()
        compact = out['state'].detach(compact=True)
        self.assertLess(compact.storage_bytes(),dense.storage_bytes())
        self.assertTrue(all(f.distribution is None for history in compact.ports.values() for f in history))
        with torch.no_grad():
            a = model(torch.tensor([[73,74]]),state=dense)
            b = model(torch.tensor([[73,74]]),state=compact)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
        self.assertEqual(compact.event,8)
        self.assertEqual(compact.memory.writes,8)


if __name__ == '__main__':
    unittest.main()
