"""Phase A contracts: causality, complete bundles, detached labels and resume."""
from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch
from .hierarchy import HierarchyModel
from .hierarchy_config import ObjectiveConfig
from .hierarchy_objectives import publication_records, future_pairs, diversity_loss, PRODUCERS
from .hierarchy_training import regional_optimizer, restore_named_optimizer, parameter_owner
from .test_hierarchy import tiny
from .checkpoint import save, load
from .train_hierarchy import train_update, evaluate


class RegionalObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randint(256, (2, 66))
        self.model = HierarchyModel(tiny(), ObjectiveConfig(mode='latent')).set_execution_backend('cpu')

    def records(self, length=48, docs=None, start=0):
        docs = torch.zeros(2, length, dtype=torch.long) if docs is None else docs
        values, positions = {}, {}
        for i, name in enumerate(PRODUCERS):
            cadence = {'A1':1, 'B1':1, 'L2':4, 'S2':4, 'L3':16}[name]
            positions[name] = [t for t in range(start, start+length) if (t+1)%cadence == 0]
            values[name] = [torch.tensor([[t+100*i], [t+100*i+1000.]], requires_grad=True)
                            for t in positions[name]]
        return publication_records(values, positions, start=start, documents=docs,
                                   valid=torch.ones_like(docs, dtype=torch.bool))

    def test_exact_temporal_source_order_and_no_window_or_row_crossing(self):
        records = self.records()
        for name, expected in [('L2', [[4],[5],[6],[7]]),
                               ('S2', [[4,104],[5,105],[6,106],[7,107]]),
                               ('L3', [[219,319],[223,323],[227,327],[231,331]])]:
            pair = future_pairs(records, name)
            self.assertEqual(pair['target'][0,:,:,0].tolist(), expected)
            self.assertFalse(pair['target'].requires_grad)
            n = pair['input'].shape[0]//2
            torch.testing.assert_close(pair['target'][n],pair['target'][0]+1000)
        self.assertEqual(future_pairs(records,'L3')['input'].shape[0],4)
        # The first L3 publication here consumes pre-window messages: reject it.
        self.assertEqual(future_pairs(self.records(32,start=7),'L3')['input'].shape[0],0)

    def test_document_boundaries_padding_and_fresh_calls_isolate_pairs(self):
        docs = torch.zeros(2,48,dtype=torch.long)
        docs[0,8:] = 1; docs[1,20:] = -1
        records = self.records(docs=docs)
        pair = future_pairs(records,'L2')
        self.assertFalse(bool(pair['mask'][0,1])) # boundary 7 -> next document
        self.assertTrue(bool(pair['mask'][0,0]))
        self.assertFalse(bool(pair['mask'][1,4:].any()))
        self.assertEqual(future_pairs(self.records(16),'L3')['input'].shape[0],0)
        other = self.records()
        other['A1'].streams = torch.tensor([1,0])
        with self.assertRaisesRegex(ValueError,'stream'):
            future_pairs(other,'L2')

    def test_future_target_branch_detached_but_producer_diversity_is_live(self):
        records = self.records()
        pair = future_pairs(records,'L2')
        loss = torch.nn.functional.smooth_l1_loss(pair['input'][:,None,None,:].expand_as(pair['target']),pair['target'])
        for r in records.values():
            r.values.retain_grad()
        loss.backward(retain_graph=True)
        self.assertIsNone(records['A1'].values.grad)
        self.assertGreater(float(records['L2'].values.grad.abs().sum()),0)
        variance, covariance, _ = diversity_loss(records,.1)
        (variance+covariance).backward()
        self.assertIsNotNone(records['A1'].values.grad)
        self.assertTrue(torch.isfinite(records['A1'].values.grad).all())
        # Near-collapse variance gradients reach the publisher values too.
        records = self.records()
        for r in records.values():
            r.values = (r.values.detach()*1e-5).requires_grad_()
        variance, _, _ = diversity_loss(records,.1)
        variance.backward()
        self.assertGreater(float(records['A1'].values.grad.abs().sum()),0)

    def test_prefix_predictions_and_inference_unchanged(self):
        legacy = HierarchyModel(tiny()).set_execution_backend('cpu')
        shared = {k:v for k,v in self.model.state_dict().items() if not k.startswith('regional_objectives.')}
        legacy.load_state_dict(shared)
        self.model.eval()
        with torch.no_grad():
            a = self.model(self.x[:,:48],self.x[:,1:49])
            changed = self.x[:,:48].clone(); changed[:,16:] = (changed[:,16:]+89)%256
            b = self.model(changed,self.x[:,1:49])
            for name in ('L2','S2','L3'):
                pa, pb = (future_pairs(o['publications'],name) for o in (a,b))
                torch.testing.assert_close(pa['input'][0],pb['input'][0],rtol=0,atol=0)
            # Changed future bytes actually change the teacher messages.
            self.assertGreater(float((future_pairs(a['publications'],'L3')['target'][0]-future_pairs(b['publications'],'L3')['target'][0]).abs().max()),1e-6)
            def forbidden(*args):
                raise AssertionError('Inference ran training objectives')
            handle = self.model.regional_objectives.register_forward_pre_hook(forbidden)
            out = self.model(self.x[:,:48])
            handle.remove()
            torch.testing.assert_close(out['logits'],legacy(self.x[:,:48])['logits'],rtol=0,atol=0)
            self.assertEqual(out['publications'],{})

    def test_named_migration_preserves_shared_moments_and_new_head_age(self):
        legacy = HierarchyModel(tiny()).set_execution_backend('cpu')
        opt = regional_optimizer(legacy,.001)
        state,_ = train_update(legacy,opt,self.x[:,:32],self.x[:,1:33],None,self.x[:,1:48])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.pt'; save(path,legacy,state,opt,step=1)
            model,state,_,record = load(path)
            record.pop('optimizer_names');record.pop('objective_config') # genuine old schema
            model.configure_objective(ObjectiveConfig(mode='latent'))
            new = regional_optimizer(model,.001)
            report = restore_named_optimizer(model,new,record)
            self.assertEqual(report['min_age'],1)
            for name,p in model.named_parameters():
                if name.startswith('regional_objectives.'):
                    self.assertNotIn(p,new.state)
                    self.assertIn(parameter_owner(name),('L2','S2','L3'))
                else:
                    for k,v in opt.state[dict(legacy.named_parameters())[name]].items():
                        torch.testing.assert_close(v,new.state[p][k],rtol=0,atol=0)
            corrupt = deepcopy(record)
            corrupt['optimizer']['param_groups'][0]['role']='unknown'
            with self.assertRaisesRegex(ValueError,'cover'):
                restore_named_optimizer(model,regional_optimizer(model,.001),corrupt)

    def test_exact_resume_statistics_adam_state_and_joint_credit(self):
        model=self.model; opt=regional_optimizer(model,.001)
        state,_=train_update(model,opt,self.x[:,:32],self.x[:,1:33],None,self.x[:,1:48])
        for name in model.regions:
            grad=model.regions[name].hat.input.weight.grad
            self.assertIsNotNone(grad); self.assertGreater(float(grad.abs().sum()),0)
        # Retained legacy heads get no gradients, decay or Adam age.
        for p in model.future_heads.parameters():
            self.assertIsNone(p.grad); self.assertNotIn(p,opt.state)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new.pt';save(path,model,state,opt,step=1)
            other,restored,_,record=load(path)
            other_opt=regional_optimizer(other,.001);restore_named_optimizer(other,other_opt,record)
            a,ra=train_update(model,opt,self.x[:,32:64],self.x[:,33:65],state,self.x[:,33:])
            b,rb=train_update(other,other_opt,self.x[:,32:64],self.x[:,33:65],restored,self.x[:,33:])
            self.assertEqual(ra['regional'],rb['regional'])
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)
            for (n,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
                for k,v in opt.state.get(p,{}).items():
                    torch.testing.assert_close(v,other_opt.state[q][k],rtol=0,atol=0)
            torch.testing.assert_close(a.latest['L3'].value,b.latest['L3'].value,rtol=0,atol=0)
        before=deepcopy(model.regional_objectives.state_dict())
        evaluate(model,[self.x[0,:48],self.x[1,:35]],context=32,span_metrics=True)
        for k,v in before.items():
            torch.testing.assert_close(v,model.regional_objectives.state_dict()[k],rtol=0,atol=0)

    def test_incomplete_window_does_not_age_or_decay_predictors(self):
        opt=regional_optimizer(self.model,.001)
        before={n:p.clone() for n,p in self.model.regional_objectives.predictors.named_parameters()}
        train_update(self.model,opt,self.x[:,:3],self.x[:,1:4],None,self.x[:,1:4])
        for n,p in self.model.regional_objectives.predictors.named_parameters():
            self.assertIsNone(p.grad)
            self.assertNotIn(p,opt.state)
            torch.testing.assert_close(p,before[n],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
