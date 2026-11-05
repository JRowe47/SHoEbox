"""Capability protocol, information boundaries and resumable graph learning."""
import random
import tempfile
from pathlib import Path
import unittest
import torch
from ..hat_minkron.train import optimizer_for
from .composition import encode_sources, answer_query, evaluate, training_loss
from .composition_data import (PERMUTATIONS, batch_from_keys, coverage, evaluation_keys,
                               pairs, shuffled_bridge, training_batch)
from .composition_train import save_checkpoint, load_checkpoint, update, source_hashes, build_model
from .config import GraphConfig
from .model import RegionLanguageModel
from .test_probe import assert_tree_equal

torch.set_num_threads(2)


class CompositionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(551)
        self.model = RegionLanguageModel(GraphConfig.tiny(readout_mode='concat',
                                        recurrent_feedback=True,evidence_gated_ports=True))
        self.batch = batch_from_keys(evaluation_keys('validation',8))

    def test_splits_withhold_table_pairs_and_balance_all_answer_values(self):
        sets = [set(pairs(s)) for s in ('train','validation','test')]
        self.assertEqual(len(set.union(*sets)),576)
        for i in range(3):
            for j in range(i):self.assertFalse(sets[i]&sets[j])
        for name,row in coverage().items():
            self.assertEqual(row['first_tables'],24)
            self.assertEqual(row['second_tables'],24)
            batch=batch_from_keys(evaluation_keys(name))
            counts=torch.bincount(batch.answers,minlength=256)[65:69]
            self.assertTrue(bool(counts.eq(counts[0]).all()))
            self.assertEqual(len(set(batch.keys)),batch.size)
        with self.assertRaises(ValueError):evaluation_keys('test',100000)

    def test_targets_require_both_tables_and_shuffling_changes_only_b(self):
        keys=evaluation_keys('all')
        batch=batch_from_keys(keys)
        for i,(a,b,q) in enumerate(keys):
            self.assertEqual(int(batch.answers[i]),65+PERMUTATIONS[b][PERMUTATIONS[a][q]])
        # Uniform independent tables leave an exact 25% conditional prior.
        for a in range(24):
            for q in range(4):
                rows=[i for i,k in enumerate(keys) if k[0]==a and k[2]==q]
                self.assertEqual(torch.bincount(batch.answers[rows],minlength=256)[65:69].tolist(),[6]*4)
        changed=shuffled_bridge(self.batch)
        self.assertTrue(torch.equal(changed.second,self.batch.second))
        self.assertTrue(torch.equal(changed.query,self.batch.query))
        self.assertTrue(torch.equal(changed.first,self.batch.first.roll(-4,0)))

    def test_concat_head_uses_raw_publications_and_legacy_config_defaults(self):
        cfg=GraphConfig.tiny().to_dict();cfg.pop('readout_mode')
        self.assertEqual(GraphConfig.from_dict(cfg).readout_mode,'mixed')
        result=encode_sources(self.model,self.batch)
        query=answer_query(self.model,self.batch,result['state'],2)
        raw=torch.cat([query['state'].ports.latest[n].payload for n in ('P','Q')],-1)
        torch.testing.assert_close(query['logits'][:,0],self.model.readout(raw),rtol=0,atol=0)

    def test_independent_prefix_and_no_query_path_to_q_in_first_round(self):
        prefix=encode_sources(self.model,self.batch)
        self.assertFalse(bool(prefix['state'].ports.latest['P'].valid.any()))
        self.assertFalse(bool(prefix['state'].ports.latest['Q'].valid.any()))
        altered=batch_from_keys([(a,b,(q+1)%4) for a,b,q in self.batch.keys])
        a=answer_query(self.model,self.batch,prefix['state'],1)
        b=answer_query(self.model,altered,prefix['state'],1)
        torch.testing.assert_close(a['state'].ports.latest['Q'].payload,
                                   b['state'].ports.latest['Q'].payload,rtol=0,atol=0)
        a2=answer_query(self.model,self.batch,prefix['state'],2)
        b2=answer_query(self.model,altered,prefix['state'],2)
        self.assertFalse(torch.equal(a2['state'].ports.latest['Q'].payload,
                                     b2['state'].ports.latest['Q'].payload))

    def test_round_comparisons_keep_prefix_immutable_and_count_actual_work(self):
        prefix=encode_sources(self.model,self.batch)
        snapshot=prefix['state'].clone().to_dict()
        for k in (1,2,4):
            result=answer_query(self.model,self.batch,prefix['state'],k)
            assert_tree_equal(self,snapshot,prefix['state'].to_dict())
            self.assertEqual(result['source_occurrences'],3*self.batch.size)
            self.assertEqual(prefix['expert_executions'],self.batch.size*4*2*12)
            self.assertEqual(result['expert_executions'],self.batch.size*k*84)

    def test_gradients_reach_source_encoding_and_delayed_links(self):
        loss,_,_,_=training_loss(self.model,self.batch)
        loss.backward()
        for name in ('B','C'):
            grad=self.model.connections.ingress[name].weight.grad
            self.assertIsNotNone(grad);self.assertGreater(float(grad.abs().sum()),0)
        grad=self.model.connections.edges['P__Q__lateral'].value.weight.grad
        self.assertGreater(float(grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))

    def test_controls_keep_expert_budgets_and_argmax_predictions_auditable(self):
        result=evaluate(self.model,count=8,batch_size=8,controls=True)
        for name,row in result.items():
            self.assertEqual(row['correct'],sum(a==b for a,b in zip(row['predictions'],row['targets'])))
            if name.startswith('k2'):
                self.assertEqual(row['expert_executions'],result['k2']['expert_executions'])
        self.assertTrue(self.model.training)

    def test_exact_checkpoint_continuation_and_protocol_rejection(self):
        protocol=dict(lr=.003,warmup=2,rounds=2,feedback=True,source_hashes=source_hashes())
        self.assertTrue(protocol['source_hashes'])
        optimizer=optimizer_for(self.model,.003);rng=random.Random(49)
        update(self.model,optimizer,training_batch(rng,4),0,protocol)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            save_checkpoint(path,self.model,optimizer,rng,1,protocol,2.,{},[])
            batch=training_batch(rng,4)
            expected=update(self.model,optimizer,batch,1,protocol)
            restored,opt,stream,_=load_checkpoint(path,protocol)
            other=training_batch(stream,4)
            self.assertEqual(batch.keys,other.keys)
            actual=update(restored,opt,other,1,protocol)
            self.assertEqual(expected,actual)
            assert_tree_equal(self,self.model.state_dict(),restored.state_dict())
            assert_tree_equal(self,optimizer.state_dict(),opt.state_dict())
            with self.assertRaises(ValueError):load_checkpoint(path,{**protocol,'lr':.1})

    def test_lookup_curriculum_changes_examples_without_changing_the_task(self):
        batch=training_batch(random.Random(7),128,lookup_warmup=True)
        for a,b,q in batch.keys:
            self.assertTrue(a==0 or b==0)
            self.assertIn((a,b),pairs('train'))
        self.assertEqual(torch.bincount(batch.answers,minlength=256)[65:69].tolist(),[32]*4)

    def test_copy_curriculum_is_a_training_subset_with_unchanged_composition_targets(self):
        batch=training_batch(random.Random(91),64,copy_warmup=True)
        self.assertTrue(torch.equal(batch.answers,batch.query.flatten()))
        self.assertTrue(all((a,b) in pairs('train') for a,b,q in batch.keys))
        with self.assertRaises(ValueError):training_batch(random.Random(1),4,True,True)

    def test_source_reconstruction_updates_graph_and_survives_checkpoint(self):
        model=build_model(GraphConfig.tiny(readout_mode='concat'),.5)
        _,_,_,query=training_loss(model,self.batch,source_reconstruction=.5)
        query['source_loss'].backward()
        for name in ('B','C'):
            self.assertGreater(float(model.connections.ingress[name].weight.grad.abs().sum()),0)
        protocol=dict(lr=.003,warmup=2,rounds=2,feedback=True,source_reconstruction=.5)
        optimizer=optimizer_for(model,.003);rng=random.Random(92)
        update(model,optimizer,training_batch(rng,4),0,protocol)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            save_checkpoint(path,model,optimizer,rng,1,protocol,2.,{},[])
            restored,opt,stream,_=load_checkpoint(path,protocol)
            expected=update(model,optimizer,training_batch(rng,4),1,protocol)
            actual=update(restored,opt,training_batch(stream,4),1,protocol)
            self.assertEqual(expected,actual)
            assert_tree_equal(self,model.state_dict(),restored.state_dict())


if __name__=='__main__':unittest.main()
