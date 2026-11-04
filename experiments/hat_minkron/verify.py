"""Independent recurrence reference, causality, binding, dispatch and state QA."""
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from events import ByteBatch, SlotDictionary
from hat import HAT
from kron import compact_chunk, MinKron
from model import HATConfig, HATSHoE, StreamState
from references.author_recurrent import naive_recurrent_kronlstm

torch.set_num_threads(2)


def tiny_config(**kwargs):
    config=dict(width=32,experts=192,expert_hidden=4,groups=8,active_groups=2,
                experts_per_group=2,address_dim=8,laces=16,active_laces=2,
                hat_symbols=16,kron_heads=2,kron_key=8,kron_value=8,chunk_size=8)
    config.update(kwargs)
    return HATConfig(**config)


def mixed(batch=2,length=29):
    values=torch.randint(0,256,(batch,length))
    links=(torch.arange(length)%3)[None].expand(batch,-1).clone()
    if batch>1: links[1]=links[1].roll(1)
    slots=torch.randint(0,8,(batch,length))
    slots=torch.where(links==0,values,slots)
    return ByteBatch(values,links,slots)


def same_state(a,b,**tol):
    assert a.hat.clocks.keys()==b.hat.clocks.keys()
    for la,lb in zip(a.hat.levels,b.hat.levels):
        for key in la: torch.testing.assert_close(la[key],lb[key],**tol)
    for key in a.hat.clocks: torch.testing.assert_close(a.hat.clocks[key],b.hat.clocks[key])
    for ca,cb in zip(a.children,b.children):
        for x,y in zip(ca,cb): torch.testing.assert_close(x,y,**tol)
    torch.testing.assert_close(a.events_seen,b.events_seen)


class HATChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.model=HATSHoE(tiny_config()).eval()

    def test_01_core_matches_author_outputs_states_gradients(self):
        for dtype in (torch.float32,torch.float64):
            for length in (1,7,19):
                q,k,v=[F.normalize(torch.randn(2,length,2,d,dtype=dtype),dim=-1).requires_grad_() for d in (5,5,3)]
                g=(-torch.rand(2,length,2,5,dtype=dtype)*.2).requires_grad_()
                beta=torch.rand(2,length,2,dtype=dtype,requires_grad=True)
                c=torch.rand(2,length,2,dtype=dtype,requires_grad=True)
                state=torch.randn(2,2,5,3,dtype=dtype,requires_grad=True)
                args=(q,k,v,g,beta,c,state)
                out,end=compact_chunk(*args)
                ref,rs=naive_recurrent_kronlstm(q,k,v,g,beta,initial_state=state,target_gate=c,output_final_state=True)
                tol={'atol':3e-5,'rtol':3e-4} if dtype==torch.float32 else {'atol':2e-10,'rtol':2e-9}
                torch.testing.assert_close(out,ref,**tol);torch.testing.assert_close(end,rs,**tol)
                w=torch.randn_like(out);ws=torch.randn_like(end)
                ga=torch.autograd.grad((out*w).sum()+(end*ws).sum(),args,retain_graph=True)
                gb=torch.autograd.grad((ref*w).sum()+(rs*ws).sum(),args)
                for x,y in zip(ga,gb): torch.testing.assert_close(x,y,**tol)

    def test_02_zero_values_and_persistent_storage(self):
        q=k=torch.ones(1,3,1,2);v=torch.zeros(1,3,1,2)
        g=torch.zeros_like(q);beta=c=torch.ones(1,3,1)
        out,state=compact_chunk(q,k,v,g,beta,c)
        self.assertTrue(bool(torch.isfinite(out).all()));self.assertEqual(float(state.abs().sum()),0)
        memory=MinKron(32,2,8,8,8)
        _,a=memory(torch.randn(1,7,32));_,b=memory(torch.randn(1,71,32),a)
        self.assertEqual(a.numel(),b.numel());self.assertEqual(b.untyped_storage().nbytes(),b.numel()*b.element_size())

    def test_03_static_live_snapshots_and_collisions(self):
        bank=SlotDictionary();bank.publish(1,3,b'A');bank.publish(2,3,b'A')
        e,versions=bank.events([(0,65),(1,3),(2,3)])
        self.assertEqual(e.values.tolist(),[[65,65,65]])
        self.assertEqual(e.links.tolist(),[[0,1,2]])
        bank.publish(1,3,b'B');self.assertEqual(e.values[0,1].item(),65)
        self.assertEqual(bank.read(1,3).version,2)
        with self.assertRaises(ValueError): bank.publish(0,65,b'B')
        with self.assertRaises(TypeError): bank.publish(1,0,1.2)
        with self.assertRaises(KeyError): bank.read(1,99)
        a=self.model.hat(ByteBatch.stream([65],1,3))['payload']
        b=self.model.hat(ByteBatch.stream([65],2,3))['payload']
        c=self.model.hat(ByteBatch.stream([65],1,4))['payload']
        self.assertGreater(float((a-b).abs().max()),1e-4)
        self.assertGreater(float((a-c).abs().max()),1e-4)

    def test_04_source_memory_isolation(self):
        hat=self.model.hat
        a=ByteBatch.stream([3,4,5],1,7)
        alone=hat(a)
        e=ByteBatch(torch.tensor([[3,90,4,91,5]]),torch.tensor([[1,2,1,2,1]]),torch.full((1,5),7))
        combined=hat(e)
        torch.testing.assert_close(alone['payload'],combined['payload'][:,[0,2,4]])
        for x,y in zip(alone['state'].levels,combined['state'].levels): torch.testing.assert_close(x[1],y[1])
        self.assertEqual(combined['state'].clocks[1].item(),3)

    def test_05_full_chunk_step_and_uneven_split(self):
        e=mixed()
        with torch.no_grad():
            full=self.model(e);step=self.model(e,backend='step')
            torch.testing.assert_close(full['logits'],step['logits'],atol=5e-5,rtol=5e-4)
            state=None;parts=[]
            for a,b in ((0,3),(3,14),(14,15),(15,29)):
                r=self.model(e.slice(a,b),state=state);state=r['state'];parts.append(r['logits'])
            torch.testing.assert_close(full['logits'],torch.cat(parts,1),atol=5e-5,rtol=5e-4)
            same_state(full['state'],state,atol=5e-5,rtol=5e-4)

    def test_06_suffix_and_targets_do_not_change_prefix(self):
        e=mixed();changed=ByteBatch(e.values.clone(),e.links,e.slots.clone())
        changed.values[:,13:]=(changed.values[:,13:]+17)%256
        changed.slots.copy_(torch.where(changed.links==0,changed.values,changed.slots))
        a=self.model(e);b=self.model(changed)
        torch.testing.assert_close(a['logits'][:,:13],b['logits'][:,:13])
        c=self.model(e,torch.randint(0,256,e.values.shape))
        torch.testing.assert_close(a['logits'],c['logits']);same_state(a['state'],c['state'])

    def test_07_no_state_or_model_mutation(self):
        state=self.model(mixed())['state'].detach()
        before=[v.clone() for v in state.tensors()]
        buffers={k:v.clone() for k,v in self.model.named_buffers()}
        self.model(mixed(),state=state)
        for a,b in zip(before,state.tensors()): self.assertTrue(torch.equal(a,b))
        for k,v in self.model.named_buffers(): self.assertTrue(torch.equal(v,buffers[k]))

    def test_08_selective_batch_reset(self):
        e=mixed();old=self.model(e)['state']
        r=self.model(e,state=old,reset=torch.tensor([True,False]))
        fresh=self.model(ByteBatch(*(v[:1] for v in (e.values,e.links,e.slots))))
        torch.testing.assert_close(r['logits'][:1],fresh['logits'],atol=2e-5,rtol=2e-4)
        carry=self.model(e,state=old)
        torch.testing.assert_close(r['logits'][1:],carry['logits'][1:])

    def test_09_checkpoint_including_stream_continues_exactly(self):
        e=mixed();a=self.model(e.slice(0,12))
        stream=io.BytesIO();torch.save({'cfg':asdict(self.model.cfg),'model':self.model.state_dict(),
                                      'stream':a['state'].detach().to_dict()},stream)
        stream.seek(0);saved=torch.load(stream,weights_only=False)
        restored=HATSHoE(HATConfig(**saved['cfg'])).eval();restored.load_state_dict(saved['model'])
        b=self.model(e.slice(12),state=a['state']);c=restored(e.slice(12),state=StreamState.from_dict(saved['stream']))
        torch.testing.assert_close(b['logits'],c['logits'],rtol=0,atol=0)
        for l in range(3): self.assertIs(restored.children_blocks[0].blocks[l].experts.shared,restored.children_blocks[1].blocks[l].experts.shared)

    def test_10_default_counts_and_no_attention_modules(self):
        m=HATSHoE();s=m.parameter_summary()
        self.assertEqual(s['unique_expert_count'],1536);self.assertEqual(s['unique_expert_weights'],1228800)
        self.assertEqual(s['active_experts_per_event'],48)
        self.assertEqual(s['expert_recurrent_state_bytes_per_sequence'],6144)
        self.assertFalse(any('attention' in n.lower() for n,_ in m.named_modules()))

    def test_11_probes_do_not_run_or_modify_memories_or_experts(self):
        def fail(*args): raise AssertionError('Probe ran memory/expert.')
        hooks=[]
        for child in self.model.children_blocks:
            for block in child.blocks:
                hooks.extend([block.memory.register_forward_hook(fail),block.experts.register_forward_hook(fail)])
        self.model.route_only(torch.randn(2,9,32),force_discovery=True)
        for hook in hooks: hook.remove()

    def test_12_inactive_child_state_and_private_gradients(self):
        e=mixed();old=self.model.initial_state(2)
        r=self.model(e,torch.randint(0,256,e.values.shape),state=old,forced_child=0,force_discovery=True)
        r['loss'].backward()
        for v in r['state'].children[1]: self.assertEqual(float(v.abs().sum()),0)
        self.assertTrue(any(float(v.abs().sum())>0 for v in r['state'].children[0]))
        for block in self.model.children_blocks[1].blocks:
            self.assertIsNone(block.experts.private.up.grad)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in self.model.shared_banks.parameters()))

    def test_13_gradients_reach_memories_symbols_bindings_parent(self):
        e=mixed();r=self.model(e,torch.randint(0,256,e.values.shape),forced_child=0,force_discovery=True)
        r['loss'].backward()
        prefixes=['hat.byte_embedding','hat.link_embedding','hat.slot_embedding','hat.binding',
                  'hat.levels.0.memory','hat.levels.1.memory','hat.levels.0.values','hat.levels.1.keys',
                  'children_blocks.0.blocks.0.memory','parent.relation_logits']
        for prefix in prefixes:
            amount=sum(float(p.grad.abs().sum()) for n,p in self.model.named_parameters() if n.startswith(prefix) and p.grad is not None)
            self.assertGreater(amount,0,prefix)

    def test_14_memory_interventions_have_effect(self):
        e=mixed();a=self.model(e)['logits']
        b=self.model(e,hat_memory=False)['logits'];c=self.model(e,expert_memory=False)['logits']
        self.assertGreater(float((a-b).abs().max()),1e-4);self.assertGreater(float((a-c).abs().max()),1e-4)

    def test_15_feedback_is_explicit_signed_and_delayed(self):
        self.model.train();r=self.model(mixed(),torch.randint(0,256,(2,29)))
        self.assertEqual(int(self.model.parent.feedback_updates),0)
        packet=r['packet'];packet['gain']=torch.ones_like(packet['gain'])
        self.model.commit_feedback(packet)
        self.assertEqual(int(self.model.parent.feedback_updates),1)
        self.assertGreater(float(self.model.parent.route_value_sum.sum()),0)
        packet['gain']=-100*torch.ones_like(packet['gain']);self.model.commit_feedback(packet)
        self.assertLess(float(self.model.parent.route_value_sum.sum()),0)

    def test_16_long_stream_has_constant_state_size(self):
        with torch.no_grad():
            e=ByteBatch.stream(torch.randint(0,256,(1,17)),1,0)
            a=self.model(e);size=a['state'].storage_bytes()
            for _ in range(12): a=self.model(e,state=a['state'])
            self.assertEqual(size,a['state'].storage_bytes())
            self.assertGreater(int(a['state'].events_seen),self.model.cfg.context)

    def test_17_split_gradients_with_nonzero_initial_state(self):
        a=MinKron(16,2,4,3,7).double();b=copy.deepcopy(a)
        x=torch.randn(2,19,16,dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        state=torch.randn(2,2,4,3,dtype=torch.float64,requires_grad=True)
        state2=state.detach().clone().requires_grad_()
        out,end=a(x,state);left,mid=b(y[:,:5],state2);right,end2=b(y[:,5:],mid)
        merged=torch.cat((left,right),1)
        (out.square().sum()+end.square().sum()).backward()
        (merged.square().sum()+end2.square().sum()).backward()
        torch.testing.assert_close(x.grad,y.grad,atol=1e-9,rtol=1e-8)
        torch.testing.assert_close(state.grad,state2.grad,atol=1e-9,rtol=1e-8)
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=1e-9,rtol=1e-8)

    def test_18_reject_invalid_addresses(self):
        with self.assertRaises(ValueError): self.model(ByteBatch.stream([256],1,0))
        with self.assertRaises(ValueError): self.model(ByteBatch.stream([1],8,0))
        with self.assertRaises(ValueError): self.model(ByteBatch.stream([1],1,256))
        with self.assertRaises(ValueError): self.model(ByteBatch(torch.tensor([[65]]),torch.tensor([[0]]),torch.tensor([[66]])))
        with self.assertRaises(TypeError): ByteBatch.stream([1.25],1,0)

    def test_19_full_model_split_gradients(self):
        model=HATSHoE(tiny_config(exploration=0)).double().eval();other=copy.deepcopy(model)
        e=mixed(1,15)
        a=model(e,forced_child=0,force_discovery=True)
        b=other(e.slice(0,6),forced_child=0,force_discovery=True)
        c=other(e.slice(6),state=b['state'],forced_child=0,force_discovery=True)
        full=a['logits'];split=torch.cat((b['logits'],c['logits']),1)
        torch.testing.assert_close(full,split,atol=1e-9,rtol=1e-8)
        full.square().mean().backward();split.square().mean().backward()
        for (name,p),(other_name,q) in zip(model.named_parameters(),other.named_parameters()):
            self.assertEqual(name,other_name)
            if p.grad is None: self.assertIsNone(q.grad);continue
            torch.testing.assert_close(p.grad,q.grad,atol=2e-8,rtol=2e-7,msg=name)

    def test_20_existing_unselected_child_memory_is_unchanged(self):
        first=self.model(mixed(),forced_child=1,force_discovery=True)['state']
        second=self.model(mixed(),state=first,forced_child=0,force_discovery=True)['state']
        for a,b in zip(first.children[1],second.children[1]):
            self.assertTrue(torch.equal(a,b))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(HATChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path('results').mkdir(exist_ok=True)
    Path('results/verification.json').write_text(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),
        'failures':[str(x[0]) for x in result.failures],'errors':[str(x[0]) for x in result.errors]},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
