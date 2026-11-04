"""Episode integrity, state interventions and exact optimizer continuation."""
import math
from pathlib import Path
import random
import tempfile
import unittest
import torch
from ..hat_minkron.train import optimizer_for
from .config import GraphConfig
from .episode_data import EpisodeStream, VALUES, make_batch, make_probe
from .model import RegionLanguageModel
from .probe_train import build_probe_model, evaluate_probe, load_probe, save_probe, warm_start_probe

torch.set_num_threads(2)


def assert_tree_equal(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left: assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a,b in zip(left,right): assert_tree_equal(test,a,b)
    else: test.assertEqual(left,right)


class EpisodeChecks(unittest.TestCase):
    def test_deterministic_disjoint_fresh_splits_and_rng_isolation(self):
        random.seed(151); torch.manual_seed(151)
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        splits=[]; answers=[]
        for split in ('train','validation','test'):
            episodes=[make_probe('correction',split,i,seed=71,gap=24) for i in range(32)]
            self.assertEqual(episodes,[make_probe('correction',split,i,seed=71,gap=24) for i in range(32)])
            splits.append({e.identity for e in episodes});answers.append([e.answer for e in episodes])
            self.assertEqual(len(splits[-1]),32)
            for e in episodes:
                self.assertNotEqual(e.answer,e.old_answer)
                self.assertIn(e.answer,VALUES)
                self.assertGreater(e.final_query-e.last_value_position,16)
        for i in range(3):
            for j in range(i):
                self.assertFalse(splits[i]&splits[j]);self.assertNotEqual(answers[i],answers[j])
        self.assertEqual(random.getstate(),py_state)
        torch.testing.assert_close(torch.get_rng_state(),torch_state,rtol=0,atol=0)

    def test_only_query_outputs_are_labeled_and_writes_have_local_credit(self):
        for kind in ('recall','correction','shared_private'):
            for index in range(12):
                e=make_probe(kind,'train',index,gap=8)
                text=next(iter(e.views.values()))
                markers=b'!' if kind!='shared_private' else b'SLR'
                for byte,target in zip(text,e.targets):
                    self.assertEqual(target!=-100,byte in markers)
                    if target!=-100:self.assertIn(target,VALUES)
                self.assertEqual(e.targets[e.final_query],e.answer)
                # At least one write-local query lies in the same 16-event
                # autograd segment as the value observation, before detachment.
                write_segment=e.last_value_position//16
                self.assertTrue(any(e.targets[p]==e.answer and p//16==write_segment
                                    for p in range(e.last_value_position+1,len(e.targets))))
                if kind!='shared_private':
                    observed=[b for b in text if b in VALUES]
                    self.assertEqual(len(observed),2 if kind=='correction' else 1)
                    self.assertEqual(e.final_query-e.last_value_position,10)

    def test_controls_preserve_schedule_and_change_only_declared_information(self):
        base=make_batch('correction','validation',0,8,gap=24)
        reset=make_batch('correction','validation',0,8,gap=24,control='reset_query')
        withheld=make_batch('correction','validation',0,8,gap=24,control='withhold_update')
        torch.testing.assert_close(base.inputs,reset.inputs,rtol=0,atol=0)
        self.assertTrue(bool(reset.reset[:,-1].all()))
        self.assertEqual(int(base.reset.sum()),8);self.assertEqual(int(reset.reset.sum()),16)
        self.assertTrue(torch.equal(base.final,withheld.final))
        self.assertTrue(torch.equal(base.targets!=-100,withheld.targets!=-100))
        for row in range(8):
            e=make_probe('correction','validation',row,gap=24);a,b=e.update_span
            self.assertTrue(torch.equal(base.inputs[row,:a],withheld.inputs[row,:a]))
            self.assertTrue(torch.equal(base.inputs[row,b:],withheld.inputs[row,b:]))
            self.assertTrue(bool(withheld.inputs[row,a:b].eq(ord('.')).all()))
            self.assertEqual(int(withheld.targets[row,-1]),e.old_answer)
            self.assertEqual(int(withheld.answers[row]),e.answer)

    def test_private_value_mask_and_source_only_controls(self):
        base=make_batch('shared_private','test',0,12,gap=3)
        masked=make_batch('shared_private','test',0,12,gap=3,control='no_private')
        for name in ('A','B','C'):
            self.assertTrue(torch.equal(base.inputs[name],masked.inputs[name]))
            changes=(base.availability[name]!=masked.availability[name]).nonzero().tolist()
            self.assertEqual(changes,[[row,14] for row in range(12)] if name in ('A','C') else [])
        self.assertTrue(bool(base.inputs['A'][:,14].unsqueeze(1).eq(torch.tensor(list(VALUES))).any(1).all()))
        self.assertTrue(bool(base.inputs['C'][:,14].unsqueeze(1).eq(torch.tensor(list(VALUES))).any(1).all()))
        self.assertFalse(bool(base.availability['B'][:,12:16].any()))
        for name in ('a','b','c'):
            only=make_batch('shared_private','test',0,12,gap=3,control=name+'_only')
            for view in ('A','B','C'):
                self.assertTrue(torch.equal(only.inputs[view],base.inputs[view]))
                if view!=name.upper():self.assertFalse(bool(only.availability[view].any()))
                else:self.assertTrue(torch.equal(only.availability[view],base.availability[view]))

    def test_cursor_reconstructs_partial_batch_and_episode_reset(self):
        stream=EpisodeStream('correction',3,gap=7,seed=3)
        first=stream.next_chunk(16)
        self.assertTrue(bool(first.reset[:,0].all()))
        clone=EpisodeStream('correction',3,gap=7,seed=3);clone.load_state_dict(stream.state_dict())
        for _ in range(8):
            a,b=stream.next_chunk(16),clone.next_chunk(16)
            assert_tree_equal(self,vars(a),vars(b))
            self.assertLessEqual(a.length,16)
            self.assertEqual(int(a.reset.sum()),3 if a.identities!=first.identities and a.reset[:,0].all() else 0)
        with self.assertRaises(ValueError):EpisodeStream('recall',3,gap=7).load_state_dict(stream.state_dict())


class ProbeExecutionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.model=RegionLanguageModel(GraphConfig.tiny()).eval()

    @torch.no_grad()
    def test_chunked_logits_and_state_match_whole_episode(self):
        batch=make_batch('correction','test',0,2,gap=4)
        whole=self.model(batch.inputs,reset=batch.reset)
        state=None;parts=[];calls=0
        for pos in range(0,batch.length,7):
            chunk=batch.slice(pos,pos+7)
            result=self.model(chunk.inputs,state=state,reset=chunk.reset)
            state=result['state'];parts.append(result['logits']);calls+=result['expert_executions']
        torch.testing.assert_close(torch.cat(parts,1),whole['logits'],rtol=0,atol=0)
        assert_tree_equal(self,state.to_dict(),whole['state'].to_dict())
        self.assertEqual(calls,whole['expert_executions'])

    @torch.no_grad()
    def test_reset_final_query_removes_prefix_and_labels_do_not_change_state(self):
        batch=make_batch('recall','test',0,2,gap=2)
        alternate=batch.targets.clone();alternate[alternate!=-100]=ord('Z')
        a=self.model(batch.inputs,batch.targets,reset=batch.reset)
        b=self.model(batch.inputs,alternate,reset=batch.reset)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        assert_tree_equal(self,a['state'].to_dict(),b['state'].to_dict())
        reset=make_batch('recall','test',0,2,gap=2,control='reset_query')
        controlled=self.model(reset.inputs,reset=reset.reset)
        fresh=self.model(reset.inputs[:,-1:])
        torch.testing.assert_close(controlled['logits'][:,-1:],fresh['logits'],rtol=0,atol=0)
        self.assertEqual(controlled['state'].events_seen.tolist(),[1,1])

    def test_evaluator_exact_denominator_costs_chunking_and_training_mode(self):
        self.model.train()
        result=evaluate_probe(self.model,'correction',gap=2,count=3,batch_size=2,segment=7)
        self.assertTrue(self.model.training)
        rows=list(result['controls'].values())
        self.assertEqual(len({r['expert_executions'] for r in rows}),1)
        self.assertEqual(len({tuple(r['episode_ids']) for r in rows}),1)
        for row in rows:
            answers=make_batch('correction','validation',0,3,gap=2).answers.tolist()
            correct=sum(a==b for a,b in zip(row['predictions'],answers))
            self.assertEqual(row['episodes'],3)
            self.assertEqual(row['exact_accuracy'],correct/3)
            self.assertAlmostEqual(row['answer_bits_per_byte'],row['answer_nats_per_byte']/math.log(2))
        all_at_once=evaluate_probe(self.model,'correction',gap=2,count=3,batch_size=2,segment=100,controls=('carried',))
        self.assertEqual(result['controls']['carried']['predictions'],all_at_once['controls']['carried']['predictions'])

    def test_shared_private_controls_execute_equal_expert_cost(self):
        result=evaluate_probe(self.model,'shared_private',gap=0,count=2,batch_size=2,segment=16)
        rows=list(result['controls'].values())
        self.assertEqual(len({r['expert_executions'] for r in rows}),1)
        self.assertEqual(len({tuple(r['episode_ids']) for r in rows}),1)
        for row in rows:self.assertEqual(sum(t['episodes'] for t in row['by_query'].values()),2)

    def test_checkpoint_resumes_rng_cursor_state_and_next_optimizer_step_exactly(self):
        stream=EpisodeStream('correction',2,gap=8,seed=31)
        model=self.model.train();optimizer=optimizer_for(model,.001)
        def step(model,optimizer,state,stream):
            batch=stream.next_chunk(16);optimizer.zero_grad(set_to_none=True)
            result=model(batch.inputs,batch.targets,state=state,reset=batch.reset)
            result['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            return result['state'].detach(),result['logits'].detach()
        state,_=step(model,optimizer,None,stream)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'checkpoint.pt'
            save_probe(checkpoint,model,optimizer,state,stream,1,{'lr':.001})
            torch_draw,python_draw=torch.rand(4),random.random()
            expected_state,expected_logits=step(model,optimizer,state,stream)
            resumed_stream=EpisodeStream('correction',2,gap=8,seed=31)
            resumed,resumed_optimizer,resumed_state,saved=load_probe(checkpoint,resumed_stream)
            torch.testing.assert_close(torch.rand(4),torch_draw,rtol=0,atol=0)
            self.assertEqual(random.random(),python_draw);self.assertEqual(saved['step'],1)
            resumed_state,resumed_logits=step(resumed,resumed_optimizer,resumed_state,resumed_stream)
            torch.testing.assert_close(resumed_logits,expected_logits,rtol=0,atol=0)
            assert_tree_equal(self,resumed.state_dict(),model.state_dict())
            assert_tree_equal(self,resumed_optimizer.state_dict(),optimizer.state_dict())
            assert_tree_equal(self,resumed_state.to_dict(),expected_state.to_dict())
            self.assertEqual(resumed_stream.state_dict(),stream.state_dict())

    def test_auxiliary_training_checkpoint_commits_and_eval_freezes_utility(self):
        model=build_probe_model(GraphConfig.tiny(),'shared_private',view_objective=True,utility=True)
        optimizer=optimizer_for(model,.001);stream=EpisodeStream('shared_private',2,gap=2,seed=31)
        def step(model,optimizer,state,stream):
            chunk=stream.next_chunk(16);optimizer.zero_grad(set_to_none=True)
            result=model(chunk.inputs,chunk.targets,state=state,reset=chunk.reset,availability=chunk.availability)
            before=model.utility.feedback_updates.clone()
            self.assertIsNotNone(result['utility_packet'])
            result['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            self.assertTrue(torch.equal(model.utility.feedback_updates,before))
            model.commit_feedback(result['utility_packet'])
            self.assertGreater(int(model.utility.feedback_updates),int(before))
            return result['state'].detach(),result['logits'].detach()
        state,_=step(model,optimizer,None,stream)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'learner.pt'
            save_probe(checkpoint,model,optimizer,state,stream,1,{'lr':.001})
            expected_state,expected_logits=step(model,optimizer,state,stream)
            other_stream=EpisodeStream('shared_private',2,gap=2,seed=31)
            other,other_optimizer,other_state,saved=load_probe(checkpoint,other_stream)
            self.assertEqual(saved['schema_version'],2)
            other_state,other_logits=step(other,other_optimizer,other_state,other_stream)
            torch.testing.assert_close(other_logits,expected_logits,rtol=0,atol=0)
            assert_tree_equal(self,model.state_dict(),other.state_dict())
            assert_tree_equal(self,optimizer.state_dict(),other_optimizer.state_dict())
            assert_tree_equal(self,expected_state.to_dict(),other_state.to_dict())
            before={key:value.clone() for key,value in other.state_dict().items()}
            evaluate_probe(other,'shared_private',gap=0,count=2,batch_size=2,controls=('carried',))
            assert_tree_equal(self,before,other.state_dict())

    def test_legacy_bare_checkpoint_still_loads(self):
        optimizer=optimizer_for(self.model,.001);stream=EpisodeStream()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.pt'
            save_probe(path,self.model,optimizer,None,stream,0,{'lr':.001})
            saved=torch.load(path,weights_only=True);saved['schema_version']=1;del saved['learner_config']
            torch.save(saved,path)
            loaded,_,state,_=load_probe(path)
            self.assertIsInstance(loaded,RegionLanguageModel);self.assertIsNone(state)
            assert_tree_equal(self,loaded.state_dict(),self.model.state_dict())

    def test_warm_start_uses_only_matching_weights_and_keeps_fresh_rng_and_cursor(self):
        optimizer=optimizer_for(self.model,.001);stream=EpisodeStream()
        stream.next_chunk(16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pt'
            save_probe(path,self.model,optimizer,None,stream,25,{'lr':.001})
            torch.manual_seed(20261105)
            fresh=build_probe_model(GraphConfig.tiny(seed=20261105),'correction')
            fresh_optimizer=optimizer_for(fresh,.001);fresh_stream=EpisodeStream(seed=20261105)
            torch_rng=torch.get_rng_state().clone();python_rng=random.getstate()
            metadata=warm_start_probe(path,fresh)
            self.assertEqual(metadata['checkpoint_step'],25)
            assert_tree_equal(self,fresh.state_dict(),self.model.state_dict())
            torch.testing.assert_close(torch.get_rng_state(),torch_rng,rtol=0,atol=0)
            self.assertEqual(random.getstate(),python_rng)
            self.assertFalse(fresh_optimizer.state)
            self.assertEqual(fresh_stream.position,0);self.assertEqual(fresh_stream.batch_start,0)
            changed=build_probe_model(GraphConfig.tiny(rounds=2),'correction')
            with self.assertRaises(ValueError):warm_start_probe(path,changed)


if __name__=='__main__':unittest.main()
