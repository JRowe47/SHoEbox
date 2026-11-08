"""Actual sparse work, per-stream causality, routing credit and CPU equivalence."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from . import CorticalModel
from .test_wide import small_wide
from .config import ROUTING_GROUPS, NAMES, SPINE
from .checkpoint import save, load
from .cpu_transport import hard_embedding
from .transport import Frame
from .sparse_state import publish_rows
from .runtime import minkron_last

torch.set_num_threads(1)


def model():
    torch.manual_seed(921)
    return CorticalModel(replace(small_wide(),region_topk=1,routing_noise=0.)).double().set_execution_backend('cpu')


def choose(m,name):
    index = next(i for i,group in enumerate(ROUTING_GROUPS) if name in group)
    with torch.no_grad():
        m.graph_router.region_keys.fill_(.1)
        m.graph_router.region_bias.zero_()
        m.graph_router.region_bias[index] = 3.
        m.graph_router.edge_keys['Y'].fill_(.1)
        m.graph_router.edge_bias['Y'].zero_()
        m.graph_router.edge_bias['Y'][0] = 3. # L3, with S2 always retained.


class SparseCPUChecks(unittest.TestCase):
    def test_variable_length_scan_matches_individual_sequences(self):
        torch.manual_seed(924)
        memory=model().regions['L3'].hat.levels[0].memory
        lengths=torch.tensor([1,4,7])
        x=torch.randn(3,7,memory.projection.in_features,dtype=torch.float64,requires_grad=True)
        state=memory.initial_state(3,'cpu',torch.float64).requires_grad_()
        reference=copy.deepcopy(memory)
        xx=x.detach().clone().requires_grad_();ss=state.detach().clone().requires_grad_()
        actual=minkron_last(memory,x,state,fused=True,lengths=lengths)
        pieces=[minkron_last(reference,xx[i:i+1,:n],ss[i:i+1],fused=True) for i,n in enumerate(lengths.tolist())]
        expected=tuple(torch.cat([r[j] for r in pieces],0) for j in range(2))
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
        cotangents=tuple(torch.randn_like(a) for a in actual)
        ga=torch.autograd.grad(actual,(x,state,*memory.parameters()),cotangents)
        gb=torch.autograd.grad(expected,(xx,ss,*reference.parameters()),cotangents)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
        for i,n in enumerate(lengths.tolist()):self.assertEqual(float(ga[0][i,n:].abs().sum()),0.)
        portable=minkron_last(reference,xx,ss,fused=False,lengths=lengths)
        for a,b in zip(actual,portable):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)

    def test_hard_transport_values_and_first_order_gradients(self):
        torch.manual_seed(922)
        for dtype in (torch.float32,torch.float64):
            ids = torch.tensor([[3,3,1],[1,2,3]])
            p = torch.randn(2,3,8,dtype=dtype).softmax(-1).requires_grad_()
            w = torch.randn(8,7,dtype=dtype,requires_grad=True)
            pp,ww = p.detach().clone().requires_grad_(),w.detach().clone().requires_grad_()
            reference = (F.one_hot(ids,8).to(dtype)+(p-p.detach()))@w
            actual = hard_embedding(ids,pp,ww)
            torch.testing.assert_close(actual,reference,atol=0,rtol=0)
            dy = torch.randn_like(actual)
            ga = torch.autograd.grad(reference,(p,w),dy)
            gb = torch.autograd.grad(actual,(pp,ww),dy)
            for a,b in zip(ga,gb): torch.testing.assert_close(a,b)

    def test_dense_cpu_matches_previous_backend(self):
        torch.manual_seed(923)
        a = CorticalModel(small_wide()).double().set_execution_backend('fused')
        b = copy.deepcopy(a).set_execution_backend('cpu')
        x = torch.tensor([[65,66,67],[71,72,73]])
        aa,bb = a(x,targets=x),b(x,targets=x)
        torch.testing.assert_close(aa['logits'],bb['logits'],atol=1e-11,rtol=1e-10)
        aa['loss'].backward();bb['loss'].backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-9,rtol=1e-8,msg=name)

    def test_regions_really_skip_and_edges_really_exclude(self):
        m = model();choose(m,'L3')
        before = m.initial_state(2)
        actual_calls = []
        handles = [r.register_forward_pre_hook(lambda module,args,name=n:actual_calls.append(name)) for n,r in m.regions.items()]
        out = m(torch.tensor([[65,66],[71,72]]),targets=torch.tensor([[66,67],[72,73]]),state=before,trace=True)
        active = set(SPINE)|{'L3'}
        self.assertEqual(set(actual_calls),active)
        for n in set(NAMES)-active:
            self.assertTrue(torch.equal(out['state'].nodes[n].clock,before.nodes[n].clock))
            for a,b in zip(out['state'].nodes[n].blocks,before.nodes[n].blocks):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
        for entry in out['trace']:
            ordinary = {f['source'] for f in entry['inputs'] if f['source'] in NAMES}
            self.assertLessEqual(len(ordinary),2)
        out['ce'].backward()
        for n in set(NAMES)-active:
            self.assertTrue(all(p.grad is None for p in m.regions[n].parameters()),n)
        self.assertGreater(float(m.graph_router.region_bias.grad.abs().sum()),0.)
        self.assertGreater(float(m.graph_router.edge_bias['Y'].grad.abs().sum()),0.)
        for h in handles:h.remove()

    def test_per_row_publication_history_keeps_true_events(self):
        ports = {}
        for event,indices in [(0,[0]),(1,[1]),(2,[0]),(3,[1])]:
            rows = torch.tensor(indices)
            frame = Frame('L3',event,0,torch.full((len(rows),4),event,dtype=torch.long))
            ports = publish_rows(ports,'L3',{4:frame},rows,2)
        previous,latest = ports['L3:4']
        self.assertEqual(previous.event.tolist(),[0,1])
        self.assertEqual(latest.event.tolist(),[2,3])
        revision = Frame('L3',3,1,torch.full((1,4),9,dtype=torch.long))
        ports = publish_rows(ports,'L3',{4:revision},torch.tensor([1]),2)
        self.assertEqual(ports['L3:4'][0].event.tolist(),[0,1])
        self.assertEqual(ports['L3:4'][1].revision.tolist(),[0,1])

    def test_causality_chunking_batch_independence_and_threads(self):
        m = model().eval();choose(m,'L3')
        x = torch.tensor([[65,66,67,68],[71,72,73,74]])
        with torch.no_grad():
            whole = m(x)
            prefix = m(x[:,:2]);rest=m(x[:,2:],state=prefix['state'])
            alone=m(x[:1])
            changed=x.clone();changed[:,-1]=80
            other=m(changed,targets=x)
            m.set_cpu_workers(2); threaded=m(x)
        torch.testing.assert_close(whole['logits'],torch.cat((prefix['logits'],rest['logits']),1),atol=1e-12,rtol=1e-11)
        torch.testing.assert_close(whole['logits'][:1],alone['logits'],atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(whole['logits'][:,:-1],other['logits'][:,:-1],atol=0,rtol=0)
        torch.testing.assert_close(whole['logits'],threaded['logits'],atol=0,rtol=0)
        for a,b in zip(whole['routing'],threaded['routing']):
            self.assertTrue(torch.equal(a['selected'],b['selected']))

        # Diverse routes exercise the grouped variable-length regional path.
        diverse=CorticalModel(replace(small_wide(),region_topk=3,routing_noise=0.)).double().set_execution_backend('cpu').eval()
        with torch.no_grad():
            batched=diverse(x)
            individual=[diverse(row[None]) for row in x]
        torch.testing.assert_close(batched['logits'],torch.cat([r['logits'] for r in individual],0),atol=1e-10,rtol=1e-8)
        for event,record in enumerate(batched['routing']):
            self.assertTrue(torch.equal(record['selected'],torch.cat([r['routing'][event]['selected'] for r in individual],0)))

    def test_inactive_state_keeps_live_credit_and_threaded_backward(self):
        m=model();choose(m,'L3');m.set_cpu_workers(2)
        first=m(torch.tensor([[65],[71]]))
        activation=first['state'].ports[f'L3:{m.cfg.max_port_bytes}'][-1].soft
        clock=first['state'].nodes['L3'].clock.clone()
        choose(m,'R2')
        # Policy changes here are deliberate test interventions; do not request
        # its old parameter gradient after changing parameter versions.
        second=m(torch.tensor([[66],[72]]),targets=torch.tensor([[67],[73]]),state=first['state'])
        self.assertTrue(torch.equal(second['state'].nodes['L3'].clock,clock))
        gradient=torch.autograd.grad(second['ce'],activation,retain_graph=True)[0]
        self.assertGreater(float(gradient.abs().sum()),0.)

        serial=model();choose(serial,'L3')
        threaded=copy.deepcopy(serial).set_cpu_workers(2)
        x=torch.tensor([[65,66],[71,72]])
        a,b=serial(x,targets=x),threaded(x,targets=x)
        a['loss'].backward();b['loss'].backward()
        for (name,p),(_,q) in zip(serial.named_parameters(),threaded.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-9,msg=name)

    def test_ec_pair_memory_and_exact_resume(self):
        m=model();choose(m,'EC-C')
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
        x=torch.tensor([[65,66,67],[71,72,73]])
        result=m(x[:,:2],targets=x[:,1:])
        result['loss'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        state=result['state'].detach(compact=True)
        self.assertTrue(all(s.writes>0 for s in state.memory.streams))
        self.assertTrue(all(e.content.soft is None for s in state.memory.streams for e in s.episodes))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt';save(path,m,state,optimizer,step=1)
            restored,rs,opt,_=load(path,optimizer_factory=lambda p:torch.optim.AdamW(p,lr=.001))
            restored.set_execution_backend('cpu')
            a=m(x[:,2:],targets=x[:,:1],state=state)
            b=restored(x[:,2:],targets=x[:,:1],state=rs)
            torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
            a['loss'].backward();b['loss'].backward();optimizer.step();opt.step()
        for (name,p),(_,q) in zip(m.named_parameters(),restored.named_parameters()):
            torch.testing.assert_close(p,q,atol=0,rtol=0,msg=name)


if __name__=='__main__':unittest.main()
