"""Behavioral contracts for event-clock cues and expert-input reinstatement."""
import copy
from dataclasses import replace
import unittest
import torch
from .config import GraphConfig
from .hippocampus_config import HippocampusConfig
from .model import RegionLanguageModel
from .activation_checkpoint import enable_associative_checkpointing
from .test_hippocampus import same


class HippocampusBindingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def model(self):
        cfg=replace(GraphConfig.tiny(),hippocampus=HippocampusConfig(
            enabled=True,cue_clock='event',reinstatement='expert'))
        return RegionLanguageModel(cfg)

    def test_zero_residual_preserves_legacy_logits_and_rng(self):
        torch.manual_seed(7)
        baseline=RegionLanguageModel(GraphConfig.tiny())
        rng=torch.get_rng_state().clone()
        torch.manual_seed(7)
        enabled=self.model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x=torch.tensor([[3,17,21,9],[4,17,21,9]])
        with torch.no_grad():
            a=baseline(x); b=enabled(x)
        self.assertTrue(torch.equal(a['logits'],b['logits']))
        same(self, a['state'].to_dict()|{'hippocampus':None}, b['state'].to_dict()|{'hippocampus':None})

    def test_cue_has_one_update_per_external_event(self):
        model=self.model(); x=torch.tensor([[1,2,5],[3,4,5]])
        calls=[]
        hook=model.hippocampus.interface.trace.register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad():
            one=model(x,rounds=1)
            self.assertEqual(len(calls),x.shape[1]); calls.clear()
            three=model(x,rounds=3)
            self.assertEqual(len(calls),x.shape[1])
        hook.remove()
        self.assertTrue(torch.equal(one['state'].hippocampus.cues,three['state'].hippocampus.cues))
        self.assertEqual(three['state'].hippocampus.rows[0].round,3*x.shape[1])
        self.assertEqual(one['source_occurrences'],three['source_occurrences'])

    def test_reinstatement_changes_expert_without_changing_adapter_state(self):
        model=self.model(); node=model.nodes['P']; cfg=model.cfg
        active=torch.tensor([True,False]); state=node.initial_state(2,'cpu',torch.float32)
        payload=torch.randn(2,cfg.width)
        routing=torch.zeros(2,3,cfg.groups)
        memory=torch.randn_like(payload,requires_grad=True)
        a=node(payload,routing,state,active)
        b=node(payload,routing,state,active,reinstatement=memory)
        same(self,a['state'].contextualizer,b['state'].contextualizer)
        self.assertFalse(torch.equal(a['latent'][0],b['latent'][0]))
        b['latent'][0].square().sum().backward()
        self.assertGreater(float(memory.grad[0].abs().sum()),0.)
        self.assertEqual(float(memory.grad[1].abs().sum()),0.)

    def test_sparse_recomputation_preserves_new_path_and_unused_gradients(self):
        direct=self.model().double(); torch.nn.init.normal_(direct.hippocampus.compose.weight,std=.03)
        recomputed=copy.deepcopy(direct)
        enable_associative_checkpointing(recomputed,segment=2,min_positions=0)
        x=torch.tensor([[1,9,2,8,3]])
        out1=direct(x); out2=recomputed(x)
        self.assertTrue(torch.equal(out1['logits'],out2['logits']))
        weight=torch.randn_like(out1['logits'])
        (out1['logits']*weight).sum().backward(); (out2['logits']*weight).sum().backward()
        for (name,a),(_,b) in zip(direct.named_parameters(),recomputed.named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-10,msg=name)

    def test_old_config_defaults_and_invalid_modes(self):
        old=GraphConfig.tiny().to_dict()
        old['hippocampus'].pop('cue_clock'); old['hippocampus'].pop('reinstatement')
        restored=GraphConfig.from_dict(old)
        self.assertEqual(restored.hippocampus.cue_clock,'round')
        self.assertEqual(restored.hippocampus.reinstatement,'ingress')
        for options in ({'cue_clock':'invalid'},{'reinstatement':'invalid'}):
            with self.assertRaises(ValueError): replace(HippocampusConfig(),**options).validate()

if __name__=='__main__': unittest.main()
