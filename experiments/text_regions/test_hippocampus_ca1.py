"""Bounded CA1 decisions must deliver accepted rereads and retain null exits."""
from dataclasses import replace
import unittest
import torch
from . import test_hippocampus as legacy_tests
from . import test_hippocampus_binding as binding_tests


class BoundedCA1Tests(unittest.TestCase):
    offer=legacy_tests.HippocampusTests.offer
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(71)

    def model(self,**options):
        defaults=dict(ca1_policy='bounded',cue_clock='event',reinstatement='expert')
        defaults.update(options)
        return legacy_tests.HippocampusTests.model(self,**defaults)

    def fixture(self,bias):
        model=self.model();h=model.hippocampus
        torch.nn.init.normal_(h.compose.weight,std=.1)
        with torch.no_grad():
            h.comparison[-1].weight.zero_();h.comparison[-1].bias.copy_(torch.tensor(bias))
        state=h.initial_state(1,'cpu',torch.float32)
        offer=self.offer(h,state);state=h.commit(state,((offer,),))
        return model,state,offer

    def query(self,model,state,offer):
        return model.hippocampus.query(state,offer.key[None,None],model.nodes,{'R1':'P'},
                                      torch.ones(1,1,dtype=torch.bool),regions=('R1',))

    def test_refine_then_accept_has_its_own_delivery_and_credit(self):
        model,state,offer=self.fixture([-20.,-20.,20.]);h=model.hippocampus
        bounded,receipts,work=self.query(model,state,offer)
        h.config=replace(h.config,ca1_policy='legacy')
        legacy,old_receipts,old_work=self.query(model,state,offer)
        self.assertGreater(float(bounded.detach().norm()),1e-5)
        self.assertGreater(float(bounded.detach().norm()),1000*float(legacy.detach().norm()))
        self.assertAlmostEqual(float(receipts[0][0].support.detach()),.5,places=6)
        self.assertLess(float(old_receipts[0][0].support.detach()),1e-10)
        self.assertEqual(work,old_work);self.assertEqual(state.rows[0].round,1)
        bounded.square().sum().backward()
        self.assertGreater(float(offer.anchor.capsule.grad.abs().sum()),0.)

    def test_terminal_comparison_can_reject_refined_content(self):
        model,state,offer=self.fixture([-30.,10.,30.])
        result,receipts,_=self.query(model,state,offer)
        self.assertLess(float(result.detach().abs().max()),1e-6)
        self.assertLess(float(receipts[0][0].support.detach()),1e-6)

    def test_disabled_refinement_cannot_consume_acceptance_probability(self):
        model,state,offer=self.fixture([-20.,-20.,20.]);h=model.hippocampus
        h.config=replace(h.config,control='no_refinement')
        result,receipts,work=self.query(model,state,offer)
        self.assertGreater(float(result.detach().norm()),1e-5)
        self.assertAlmostEqual(float(receipts[0][0].support.detach()),.5,places=6)
        self.assertEqual(work[0]['refinements'],0)

    def test_recomputed_logits_gradients_and_unused_owners(self):
        binding_tests.HippocampusBindingTests.test_sparse_recomputation_preserves_new_path_and_unused_gradients(self)

    def test_blocked_returns_remain_zero(self):
        model,state,offer=self.fixture([-20.,-20.,20.]);h=model.hippocampus
        h.config=replace(h.config,control='blocked')
        result,receipts,work=self.query(model,state,offer)
        self.assertEqual(float(result.detach().abs().sum()),0.)
        self.assertEqual(receipts,((),));self.assertEqual(work[0]['readbacks'],0)

if __name__=='__main__':unittest.main()
