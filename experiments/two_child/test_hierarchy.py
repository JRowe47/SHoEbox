import io
import unittest
from dataclasses import replace
import torch
from torch.nn import functional as F
from model import HierarchicalSHoE,HierarchyConfig,SplitExpertBank
from semantic_core import ExpertBank


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model=HierarchicalSHoE().eval()
        self.x=torch.randint(65,(2,23))

    def test_exact_counts_and_real_aliases(self):
        m=self.model;s=m.parameter_summary()
        self.assertEqual(s['total_unique_parameters'],1636784)
        self.assertEqual(s['each_child_logical_parameters'],[1000732,1000732])
        self.assertEqual(s['unique_expert_weights'],1228800)
        self.assertEqual(s['shared_expert_weights'],409600)
        for l in range(3):
            a,b=[child.blocks[l].experts for child in m.children_blocks]
            self.assertIs(a.shared,b.shared)
            self.assertIs(a.shared,m.shared_banks[l])
            self.assertIsNot(a.private.up,b.private.up)
        self.assertEqual(len(list(m.parameters())),len({id(p) for p in m.parameters()}))

    def test_split_bank_matches_dense_reference(self):
        bank=self.model.children_blocks[0].blocks[0].experts
        x=torch.randn(1,3,100)
        ids=torch.tensor([[[0,171],[13,177],[170,341]]]);gates=torch.rand(1,3,2)
        up=torch.cat([bank.shared.up,bank.private.up]);down=torch.cat([bank.shared.down,bank.private.down])
        expected=torch.stack([sum(F.gelu(x[0,t]@up[e].T)@down[e]*gates[0,t,k]
                                  for k,e in enumerate(ids[0,t]))/2**.5 for t in range(3)])[None]
        torch.testing.assert_close(bank(x,ids,gates),expected,atol=2e-6,rtol=2e-6)

    def test_probe_does_not_execute_attention_or_experts(self):
        calls=[];handles=[]
        for child in self.model.children_blocks:
            for block in child.blocks:
                handles.append(block.attention.register_forward_hook(lambda *a:calls.append('attention')))
                handles.append(block.experts.register_forward_hook(lambda *a:calls.append('expert')))
        payload,pos=self.model.encode(self.x)
        self.model.route_only(payload,pos,force_discovery=True)
        for handle in handles:handle.remove()
        self.assertEqual(calls,[])

    def test_top_one_dispatch_and_child_isolation(self):
        r=self.model(self.x,self.x.roll(-1,1),trace=True,force_discovery=True,forced_child=0)
        r['loss'].backward()
        self.assertTrue(all(e['child']==0 for e in r['executions']))
        self.assertEqual(sum(e['source_positions'].numel() for e in r['executions']),self.x.numel())
        for block in self.model.children_blocks[1].blocks:
            self.assertIsNone(block.experts.private.up.grad)
            self.assertIsNone(block.attention.qkv.weight.grad)
        for bank in self.model.shared_banks:
            self.assertIsNotNone(bank.up.grad)
            self.assertGreater(bank.up.grad.abs().sum().item(),0)

    def test_shared_learning_visible_to_both_private_isolated(self):
        a,b=[child.blocks[0].experts for child in self.model.children_blocks]
        x=torch.randn(1,3,100);ids=torch.zeros(1,3,1,dtype=torch.long);g=torch.ones(1,3,1)
        before=b(x,ids,g).detach().clone();private=b.private.up.detach().clone()
        optimizer=torch.optim.SGD(self.model.parameters(),lr=.1)
        a(x,ids,g).square().sum().backward();optimizer.step()
        self.assertGreater((b(x,ids,g)-before).abs().sum().item(),0)
        torch.testing.assert_close(private,b.private.up,atol=0,rtol=0)

    def test_causal_prefix_and_changed_suffix(self):
        changed=self.x.clone();changed[:,13:]=torch.randint(65,changed[:,13:].shape)
        with torch.no_grad():
            full=self.model(self.x);short=self.model(self.x[:,:13]);altered=self.model(changed)
        torch.testing.assert_close(full['logits'][:,:13],short['logits'],atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(full['logits'][:,:13],altered['logits'][:,:13],atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(full['route']['choice'][:,:13],short['route']['choice'])

    def test_execution_traces_have_causal_original_positions(self):
        r=self.model(self.x,trace=True,force_discovery=True)
        seen=[[] for _ in range(2)]
        for execution in r['executions']:
            pos=execution['source_positions'];seen[execution['batch_row']]+=pos.tolist()
            for layer in execution['trace']:
                for q in range(len(pos)):
                    reads=layer['reads'][0,q][layer['valid'][0,q]]
                    self.assertTrue(bool((pos[reads]<=pos[q]).all()))
        self.assertEqual([sorted(s) for s in seen],[list(range(23)),list(range(23))])

    def test_forward_targets_cannot_change_predictions_or_state(self):
        before={k:v.clone() for k,v in self.model.named_buffers()}
        with torch.no_grad():
            a=self.model(self.x,self.x);b=self.model(self.x,self.x.roll(1,1))
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
        for name,v in self.model.named_buffers():torch.testing.assert_close(v,before[name],atol=0,rtol=0)

    def test_executed_lace_feedback_reaches_parent_semantics(self):
        r=self.model(self.x,self.x.roll(-1,1),force_discovery=True)
        r['semantic_loss'].backward()
        for p in [self.model.parent.prototypes,self.model.parent.lace.template_logits,
                  self.model.parent.relation_logits]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)

    def test_upward_signal_intervention_changes_parent_prior(self):
        payload,pos=self.model.encode(self.x)
        with torch.no_grad():
            a=self.model.route_only(payload,pos,force_discovery=True)
            b=self.model.route_only(payload,pos,force_discovery=True,scramble_upward=True)
        self.assertGreater((a['parent_prior']-b['parent_prior']).abs().max().item(),1e-5)

    def test_feedback_is_delayed_and_signed(self):
        m=self.model.train();r=m(self.x,self.x.roll(-1,1))
        self.assertEqual(m.parent.feedback_updates.item(),0)
        m.commit_feedback(r['packet']);self.assertEqual(m.parent.feedback_updates.item(),1)
        m.parent.route_mass.zero_();m.parent.route_value_sum.zero_()
        packet={'eligibility':torch.full((1,2,3,32),1/32),
                'choice':torch.tensor([[0,1]]),'gain':torch.tensor([[1.,-1.]])}
        m.commit_feedback(packet)
        values=m.parent.route_value_sum/(m.parent.route_mass+1e-8)
        self.assertGreater(values[:,:,0].mean().item(),.99)
        self.assertLess(values[:,:,1].mean().item(),-.99)
        m.eval()
        with self.assertRaises(RuntimeError):m.commit_feedback(packet)

    def test_fast_path_and_rejected_match_broadening(self):
        m=self.model;m.cfg.match_threshold=-1
        with torch.no_grad():
            m.parent.route_context.bias.copy_(torch.tensor([8.,-8.]))
            m.parent.route_mass.fill_(1);m.parent.route_value_sum.fill_(1)
            m.parent.feedback_updates.fill_(m.cfg.fast_warmup_updates)
        payload,pos=m.encode(self.x)
        fast=m.route_only(payload,pos)
        self.assertEqual(fast['probed'].sum().item(),self.x.numel())
        self.assertTrue(bool(fast['fast'].all()))
        m.cfg.match_threshold=2
        broad=m.route_only(payload,pos)
        self.assertEqual(broad['probed'].sum().item(),2*self.x.numel())
        self.assertFalse(bool(broad['fast'].any()))

    def test_continuous_payload_api(self):
        payload,pos=self.model.encode(self.x)
        with torch.no_grad():
            a=self.model(self.x)['logits'];b=self.model.forward_payload(payload,pos)['logits']
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_checkpoint_preserves_aliasing_and_predictions(self):
        stream=io.BytesIO();torch.save(self.model.state_dict(),stream);stream.seek(0)
        other=HierarchicalSHoE().eval();other.load_state_dict(torch.load(stream,weights_only=True))
        self.assertIs(other.children_blocks[0].blocks[0].experts.shared,
                      other.children_blocks[1].blocks[0].experts.shared)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.x)['logits'],other(self.x)['logits'],atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
