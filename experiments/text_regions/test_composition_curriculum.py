"""Curriculum boundaries, selection gates, and deterministic continuation."""
from dataclasses import asdict
import random
import tempfile
from pathlib import Path
import unittest

import torch

from ..hat_minkron.train import optimizer_for
from .composition_curriculum import Mastery, curriculum_batch, evaluate_skills, skill_pairs
from .composition_data import pairs
from .composition import encode_sources, answer_query, training_loss
from .composition_data import batch_from_keys, evaluation_keys
from .composition_train import build_model, load_checkpoint, save_checkpoint, update
from .config import GraphConfig
from .test_probe import assert_tree_equal

torch.set_num_threads(1)


class CurriculumChecks(unittest.TestCase):
    def test_training_mixture_respects_pair_split_orientation_and_balance(self):
        for stage, expected in [('lookup', (32,32,0)), ('composition',(16,16,32))]:
            batch = curriculum_batch(random.Random(58),64,stage)
            counts = [sum(a != 0 and b == 0 for a,b,q in batch.keys),
                      sum(a == 0 and b != 0 for a,b,q in batch.keys),
                      sum(a != 0 and b != 0 for a,b,q in batch.keys)]
            self.assertEqual(tuple(counts),expected)
            self.assertTrue(all((a,b) in pairs('train') for a,b,q in batch.keys))
            self.assertEqual(torch.bincount(batch.answers,minlength=256)[65:69].tolist(),[16]*4)
        for batch in (0,4,10):
            with self.assertRaises(ValueError):curriculum_batch(random.Random(1),batch,'lookup')

    def test_skill_evaluation_domains_do_not_overlap_or_include_double_identity(self):
        for split in ('train','validation','test'):
            groups = [set(skill_pairs(split,s)) for s in ('lookup_b','lookup_c','composition')]
            self.assertEqual(set.union(*groups),set(pairs(split))-{(0,0)})
            self.assertEqual(sum(map(len,groups)),len(set.union(*groups)))
            self.assertTrue(all(groups))

    def test_gate_requires_both_skills_and_consecutive_checks(self):
        gate = Mastery(.9,2)
        def metric(b,c):return {'lookup_b':{'accuracy':b},'lookup_c':{'accuracy':c}}
        self.assertFalse(gate.observe(metric(1.,.89),50))
        self.assertTrue(gate.observe(metric(.9,.9),100))
        self.assertEqual(gate.stage,'lookup')
        gate.observe(metric(.89,1.),150)
        self.assertEqual(gate.streak,0)
        gate.observe(metric(1.,1.),200)
        gate.observe(metric(1.,1.),250)
        self.assertEqual(gate.stage,'composition')
        self.assertEqual(gate.transition_step,250)
        self.assertFalse(gate.observe(metric(.5,.5),300))
        self.assertEqual(gate.transition_step,250)

    def test_resume_preserves_gate_mixture_rng_and_optimizer(self):
        torch.manual_seed(71)
        model = build_model(GraphConfig.tiny(readout_mode='concat'),.5)
        optimizer = optimizer_for(model,.001)
        rng = random.Random(71)
        gate = Mastery(.9,2,streak=1)
        protocol = dict(lr=.001,warmup=2,rounds=2,feedback=True,source_reconstruction=.5)
        update(model,optimizer,curriculum_batch(rng,8,gate.stage),0,protocol)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            save_checkpoint(path,model,optimizer,rng,1,protocol,float('inf'),{'mastery':asdict(gate)},[])
            expected_batch = curriculum_batch(rng,8,gate.stage)
            expected = update(model,optimizer,expected_batch,1,protocol)
            restored,opt,stream,saved = load_checkpoint(path,protocol)
            resumed_gate = Mastery(**saved['counters']['mastery'])
            actual_batch = curriculum_batch(stream,8,resumed_gate.stage)
            self.assertEqual(actual_batch.keys,expected_batch.keys)
            actual = update(restored,opt,actual_batch,1,protocol)
            self.assertEqual(expected,actual)
            assert_tree_equal(self,model.state_dict(),restored.state_dict())
            assert_tree_equal(self,optimizer.state_dict(),opt.state_dict())
            self.assertEqual(asdict(gate),asdict(resumed_gate))

    def test_metrics_score_every_unique_lookup_case_and_restore_mode(self):
        torch.manual_seed(32)
        model = build_model(GraphConfig.tiny(readout_mode='concat'))
        metrics = evaluate_skills(model,'validation',64)
        self.assertTrue(model.training)
        for skill, result in metrics.items():
            self.assertEqual(result['episodes'],len(skill_pairs('validation',skill))*4)
            self.assertEqual(result['episodes'],len(set(result['keys'])))
            self.assertEqual(result['correct'],sum(a==b for a,b in zip(result['predictions'],result['targets'])))

    def test_round_taps_preserve_outputs_cost_and_query_boundary(self):
        torch.manual_seed(93)
        model = build_model(GraphConfig.tiny(readout_mode='concat'))
        batch = batch_from_keys(evaluation_keys('validation',8))
        prefix = encode_sources(model,batch)
        plain = answer_query(model,batch,prefix['state'],2)
        tapped = answer_query(model,batch,prefix['state'],2,return_round_activity=True)
        torch.testing.assert_close(plain['logits'],tapped['logits'],rtol=0,atol=0)
        self.assertEqual(plain['expert_executions'],tapped['expert_executions'])
        assert_tree_equal(self,plain['state'].to_dict(),tapped['state'].to_dict())
        self.assertEqual(tapped['round_activity']['P']['payload'].shape,(8,1,2,32))
        other = batch_from_keys([(a,b,(q+1)%4) for a,b,q in batch.keys])
        changed = answer_query(model,other,prefix['state'],2,return_round_activity=True)
        torch.testing.assert_close(tapped['round_activity']['Q']['payload'][:,:,0],
                                   changed['round_activity']['Q']['payload'][:,:,0],rtol=0,atol=0)
        self.assertFalse(torch.equal(tapped['round_activity']['Q']['payload'][:,:,1],
                                     changed['round_activity']['Q']['payload'][:,:,1]))

    def test_regional_targets_are_loss_only_with_graph_gradients(self):
        torch.manual_seed(94)
        model = build_model(GraphConfig.tiny(readout_mode='concat'),.5,1.)
        batch = batch_from_keys(evaluation_keys('validation',8))
        _,_,_,query = training_loss(model,batch,source_reconstruction=.5,regional_supervision=1.)
        plain = answer_query(model,batch,encode_sources(model,batch)['state'],2)
        torch.testing.assert_close(query['logits'],plain['logits'],rtol=0,atol=0)
        query['regional_loss'].backward()
        for name in ('A','B','C'):
            self.assertGreater(float(model.connections.ingress[name].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.connections.edges['P__Q__lateral'].value.weight.grad.abs().sum()),0)
        self.assertIsNone(model.readout.weight.grad)
        with self.assertRaises(ValueError):training_loss(model,batch,rounds=1,regional_supervision=1.)

    def test_regional_head_checkpoint_continues_exactly(self):
        torch.manual_seed(95)
        model = build_model(GraphConfig.tiny(readout_mode='concat'),.5,1.)
        optimizer = optimizer_for(model,.001)
        rng = random.Random(95)
        protocol = dict(lr=.001,warmup=2,rounds=2,feedback=True,source_reconstruction=.5,regional_supervision=1.)
        update(model,optimizer,curriculum_batch(rng,8,'composition'),0,protocol)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            save_checkpoint(path,model,optimizer,rng,1,protocol,1.,{},[])
            expected = update(model,optimizer,curriculum_batch(rng,8,'composition'),1,protocol)
            restored,opt,stream,_ = load_checkpoint(path,protocol)
            actual = update(restored,opt,curriculum_batch(stream,8,'composition'),1,protocol)
            self.assertEqual(expected,actual)
            assert_tree_equal(self,model.state_dict(),restored.state_dict())
            assert_tree_equal(self,optimizer.state_dict(),opt.state_dict())


if __name__ == '__main__':
    unittest.main()
