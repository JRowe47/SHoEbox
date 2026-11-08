"""Conformance gates for causality, live credit, identity and bounded work."""
import io
import unittest
from dataclasses import replace
import torch
from .config import GraphConfig
from .model import RegionLanguageModel
from .state import GraphState
from .hippocampus_config import HippocampusConfig
from .hippocampus import Hippocampus
from .hippocampus_interface import EngramHandle,LocalAnchor,BindingOffer,CueRequest,CreditReceipt
from .hippocampus_index import WorkBudget
from .hippocampus_runtime import RecallFabric,FabricState,Envelope
from .hippocampus_checkpoint import migrate_optimizer,load_parent
from .paired_recall import fork_state


def same(test,left,right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left,right),str((left.shape,right.shape)))
    elif isinstance(left,dict):
        test.assertEqual(left.keys(),right.keys())
        for k in left: same(test,left[k],right[k])
    elif isinstance(left,(tuple,list)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right): same(test,a,b)
    else: test.assertEqual(left,right)


class HippocampusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self,**kwargs):
        torch.manual_seed(17)
        return RegionLanguageModel(GraphConfig.tiny(hippocampus=HippocampusConfig(enabled=True,**kwargs)))

    def offer(self,service,state,*,owner='A',region='R0',value=None,action=2,write=None):
        row=state.rows[0]
        key=torch.randn(service.config.address_width)
        key=torch.nn.functional.normalize(key,dim=-1)
        code,_=service.separate(key)
        capsule=torch.randn(service.width,requires_grad=True) if value is None else value
        h=EngramHandle(row.namespace,region,owner,-1,0,0,1)
        anchor=LocalAnchor(h,capsule,key,key,'external',row.round+1,row.round+1)
        actions=torch.full((4,),-3.); actions[action]=3.; actions=actions.softmax(-1).requires_grad_()
        return BindingOffer((row.namespace,row.round,owner),row.namespace,owner,anchor,key,code,key,
            torch.tensor(.8,requires_grad=True) if write is None else write,torch.tensor(.7,requires_grad=True),actions,row.round+1)

    def test_disabled_rng_state_logits_and_zero_path(self):
        torch.manual_seed(123); baseline=RegionLanguageModel(GraphConfig.tiny()); rng=torch.get_rng_state()
        torch.manual_seed(123); enabled=RegionLanguageModel(GraphConfig.tiny(hippocampus=HippocampusConfig(enabled=True)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in baseline.state_dict().items(): self.assertTrue(torch.equal(value,enabled.state_dict()[name]))
        x=torch.tensor([[1,2,3,4]])
        a=baseline(x); b=enabled(x)
        self.assertTrue(torch.equal(a['logits'],b['logits']))
        same(self,a['state'].to_dict()|{'hippocampus':None},b['state'].to_dict()|{'hippocampus':None})
        self.assertIsNone(a['state'].hippocampus); self.assertEqual(a['hippocampus_work'],[])
        self.assertEqual(a['source_occurrences'],b['source_occurrences'])

    def test_causal_read_and_generation_validation(self):
        model=self.model(anchors_per_region=1); h=model.hippocampus
        state=h.initial_state(1,'cpu',torch.float32); before=state
        offer=self.offer(h,state); state=h.commit(state,((offer,),)); row=state.rows[0]
        anchor=row.anchors['R0'][0]
        self.assertIsNone(before.rows[0].anchor(anchor.handle,0))
        self.assertIsNone(row.anchor(anchor.handle,0))
        self.assertIsNotNone(row.anchor(anchor.handle,1))
        next_state=h.commit(state,((self.offer(h,state),),))
        self.assertIsNone(next_state.rows[0].anchor(anchor.handle,2))
        self.assertIsNotNone(row.anchor(anchor.handle,1))
        self.assertNotEqual(anchor.handle.generation,next_state.rows[0].anchors['R0'][0].handle.generation)

    def test_duplicate_write_and_receipt_apply_once(self):
        model=self.model(); h=model.hippocampus; state=h.initial_state(1,'cpu',torch.float32)
        offer=self.offer(h,state); row=h.offer(state.rows[0],offer)
        repeated=h.offer(row,offer)
        self.assertIs(row,repeated)
        state=replace(state,rows=(replace(row,round=1),))
        binding=next(iter(next(iter(row.bindings.values())).values())); anchor=binding.links[0].handle
        receipt=CreditReceipt((0,1,'R1','credit'),(0,1,'R1'),((binding.handle,anchor),),torch.tensor(.9),0)
        once=h.feedback(state.rows[0],(receipt,),0); twice=h.feedback(once,(receipt,),0)
        self.assertTrue(torch.equal(once.bindings[binding.handle.shard][binding.handle.slot].utility,
                                    twice.bindings[binding.handle.shard][binding.handle.slot].utility))

    def test_query_write_reader_and_selected_link_gradients(self):
        model=self.model(); h=model.hippocampus
        torch.nn.init.normal_(h.compose.weight,std=.1)
        state=h.initial_state(1,'cpu',torch.float32)
        first=self.offer(h,state); state=h.commit(state,((first,),))
        second=self.offer(h,state); state=h.commit(state,((second,),))
        keys=first.key[None,None].clone().requires_grad_()
        values,_,work=h.query(state,keys,model.nodes,{'R1':'P'},torch.ones(1,1,dtype=torch.bool),regions=('R1',))
        values.square().sum().backward()
        self.assertGreater(float(keys.grad.abs().sum()),0)
        self.assertGreater(float(first.anchor.capsule.grad.abs().sum()),0)
        self.assertGreater(float(model.nodes['A'].anchor_reader.weight.grad.abs().sum()),0)
        self.assertGreater(float(first.actions.grad.abs().sum()),0)
        self.assertGreater(float(second.write.grad.abs().sum()),0)
        self.assertGreater(float(h.dg.weight.grad.abs().sum()),0)
        self.assertLessEqual(work[0]['readbacks'],8)

    def test_hold_preserves_content_but_trains_write_gate(self):
        model=self.model(); h=model.hippocampus; state=h.initial_state(1,'cpu',torch.float32)
        state=h.commit(state,((self.offer(h,state),),)); old=state.rows[0].anchors['R0'][0]
        write=torch.tensor(.2,requires_grad=True); offer=self.offer(h,state,write=write)
        updated=h.commit(state,((offer,),)); current=updated.rows[0].anchors['R0'][0]
        self.assertTrue(torch.equal(old.capsule,current.capsule)); self.assertEqual(old.handle,current.handle)
        current.capsule.sum().backward(); self.assertGreater(float(write.grad.abs()),0)

    def test_revision_retains_snapshot_and_learns_erase(self):
        model=self.model(); h=model.hippocampus; state=h.initial_state(1,'cpu',torch.float32)
        state=h.commit(state,((self.offer(h,state),),)); old=state
        offer=self.offer(h,state,action=1); state=h.commit(state,((offer,),))
        records=[r for shard in state.rows[0].bindings.values() for r in shard.values()]
        self.assertTrue(any(r.handle.version==2 for r in records))
        latest=max(state.rows[0].anchors['R0'].values(),key=lambda a:a.ready_round)
        latest.capsule.square().sum().backward(); self.assertGreater(float(offer.erase.grad.abs()),0)
        self.assertEqual(len(old.rows[0].anchors['R0']),1)

    def test_reader_query_switches_same_capsule_and_st_sparse_estimator(self):
        model=self.model(); node=model.nodes['A']; h=model.hippocampus
        capsule=torch.randn(2,32); capsule[1]=capsule[0]
        output=node.read_anchor(capsule,torch.randn(2,32))
        self.assertFalse(torch.equal(output[0],output[1]))
        x=torch.randn(2,32,requires_grad=True); code,_=h.separate(x)
        self.assertTrue(torch.equal(code.sum(-1),torch.full((2,),8.)))
        target=torch.randn_like(code); (code*target).sum().backward(); actual=x.grad.clone()
        x.grad=None; soft=h.dg(x).tanh().softmax(-1)*8
        (soft*target).sum().backward(); self.assertTrue(torch.equal(actual,x.grad))

    def test_fork_reset_checkpoint_and_exact_continuation(self):
        model=self.model(); x=torch.tensor([[3,4,5],[6,7,8]])
        with torch.no_grad(): state=model(x)['state']
        stream=io.BytesIO(); torch.save(state.to_dict(),stream); stream.seek(0)
        restored=GraphState.from_dict(torch.load(stream,weights_only=True))
        with torch.no_grad():
            left=model(x,state=state); right=model(x,state=restored)
        self.assertTrue(torch.equal(left['logits'],right['logits'])); same(self,left['state'].to_dict(),right['state'].to_dict())
        branch=fork_state(state,torch.tensor([0,0,1,1]))
        cleared=branch.reset(torch.tensor([True,False,False,False]))
        self.assertEqual(cleared.hippocampus.rows[0].anchors,{})
        self.assertEqual(len(cleared.hippocampus.rows[1].anchors),3)
        self.assertIs(branch.hippocampus.rows[0],branch.hippocampus.rows[1])
        self.assertEqual(state.hippocampus.rows[0].namespace,0)
        same(self,state.to_dict(),restored.to_dict())
        same(self,restored.to_dict(),restored.clone().to('cpu').to_dict())

    def test_budgets_and_stale_null(self):
        model=self.model(anchors_per_region=1,bindings_per_shard=8,candidates_per_shard=4,posting_reads=8)
        h=model.hippocampus; state=h.initial_state(1,'cpu',torch.float32)
        for _ in range(12): state=h.commit(state,((self.offer(h,state),),))
        q=torch.randn(1,1,32)
        with torch.no_grad(): result,receipts,work=h.query(state,q,model.nodes,{'R1':'P'},torch.ones(1,1,dtype=torch.bool),regions=('R1',))
        self.assertLessEqual(work[0]['candidates'],8); self.assertLessEqual(work[0]['posting_reads'],16)
        self.assertLessEqual(work[0]['shards'],2); self.assertLessEqual(work[0]['readbacks'],8)
        empty=replace(state,rows=(replace(state.rows[0],anchors={}),))
        with torch.no_grad(): result,receipts,work=h.query(empty,q,model.nodes,{'R1':'P'},torch.ones(1,1,dtype=torch.bool),regions=('R1',))
        self.assertEqual(float(result.abs().sum()),0.); self.assertEqual(receipts,((),))

    def test_provenance_and_masked_owner(self):
        model=self.model(); h=model.hippocampus; state=h.initial_state(1,'cpu',torch.float32)
        offer=self.offer(h,state)
        self.assertIs(h.offer(state.rows[0],replace(offer,provenance='reinstated')),state.rows[0])
        state=h.commit(state,((offer,),)); torch.nn.init.normal_(h.compose.weight,std=.1)
        result,receipts,_=h.query(state,offer.key[None,None],model.nodes,{'R2':'Q'},torch.ones(1,1,dtype=torch.bool),
            available_owners={'A':torch.tensor([False])},regions=('R2',))
        self.assertEqual(float(result.detach().abs().sum()),0.)

    def test_name_migration_preserves_adam_moments(self):
        torch.manual_seed(4); old=RegionLanguageModel(GraphConfig.tiny())
        opt=torch.optim.AdamW(old.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
        old(torch.tensor([[2,3]]))['logits'].sum().backward(); opt.step()
        new=self.model(); load_parent(new,{'weights':old.state_dict()})
        migrated=migrate_optimizer(old,opt.state_dict(),new,.003)
        old_named=dict(old.named_parameters()); new_named=dict(new.named_parameters())
        for name,param in old_named.items():
            same(self,opt.state[param],migrated.state[new_named[name]])
        for name,param in new_named.items():
            if name not in old_named: self.assertNotIn(param,migrated.state)

    def test_fabric_overload_expiry_fairness_and_exact_restore(self):
        c=HippocampusConfig(queue_capacity=3,service_budget=2)
        fabric=RecallFabric(c)
        for name in ('R0','R1','R2'): fabric.register(name,lambda payload,snapshot:payload+snapshot)
        state=FabricState()
        for i in range(3): state,status=fabric.submit(state,Envelope((i,0),f'R{i}',0,1,i))
        state,status=fabric.submit(state,Envelope((4,0),'R0',0,1,4)); self.assertEqual(status,'overload')
        state,status=fabric.submit(state,Envelope((0,0),'R0',0,1,0)); self.assertEqual(status,'duplicate')
        snapshot=FabricState.from_dict(state.to_dict())
        state,out=fabric.service(state,0,10); self.assertEqual(len(out),2)
        restored,restored_out=fabric.service(snapshot,0,10)
        self.assertEqual(out,restored_out); self.assertEqual(state,restored)
        state,out=fabric.service(state,2,10); self.assertEqual(out[0][1],'expired')
        self.assertEqual(state.expired,1); self.assertEqual(state.completed,2)




class HippocampusIntegrationTests(unittest.TestCase):
    model=HippocampusTests.model
    offer=HippocampusTests.offer

    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_third_region_two_producers_and_permuted_owner_names(self):
        model=self.model(); hub=model.hippocampus
        nn=torch.nn
        nn.init.normal_(hub.compose.weight,std=.1)
        state=hub.initial_state(1,'cpu',torch.float32)
        first=self.offer(hub,state,owner='U',region='R1')
        second=self.offer(hub,state,owner='V',region='R2')
        second=replace(second,key=first.key,code=first.code,context=first.context)
        state=hub.commit(state,((first,second),))
        result,receipts,_=hub.query(state,first.key[None,None],model.nodes,{'R0':'C'},torch.ones(1,1,dtype=torch.bool),regions=('R0',))
        self.assertEqual({h.region for _,h in receipts[0][0].contributors},{'R1','R2'})
        from .ports import map_tensors
        from .hippocampus_interface import EngramHandle
        from dataclasses import fields,is_dataclass
        def rename(value):
            if isinstance(value,EngramHandle): return replace(value,node={'U':'P','V':'Q'}.get(value.node,value.node))
            if is_dataclass(value): return replace(value,**{f.name:rename(getattr(value,f.name)) for f in fields(value)})
            if isinstance(value,dict): return {k:rename(v) for k,v in value.items()}
            if isinstance(value,tuple): return tuple(rename(x) for x in value)
            return value
        changed=rename(state)
        output,_,_=hub.query(changed,first.key[None,None],{'P':model.nodes['U'],'Q':model.nodes['V']},
            {'R0':'C'},torch.ones(1,1,dtype=torch.bool),regions=('R0',))
        self.assertTrue(torch.equal(result,output))

    def test_invalid_descriptors_are_masked_and_cues_ignore_candidates(self):
        model=self.model(); hub=model.hippocampus; torch.nn.init.normal_(hub.compose.weight,std=.1)
        state=hub.initial_state(1,'cpu',torch.float32)
        state=hub.commit(state,((self.offer(hub,state),),))
        row=state.rows[0]; absent=replace(state,rows=(replace(row,anchors={}),))
        modified={s:{k:replace(v,key=v.key*1000,context=v.context*1000) for k,v in bank.items()}
                  for s,bank in row.bindings.items()}
        other=replace(absent,rows=(replace(absent.rows[0],bindings=modified),))
        ingress=torch.randn(1,3,32); active=torch.ones(1,3,dtype=torch.bool)
        a,qa=hub.advance_cues(absent,ingress,active); b,qb=hub.advance_cues(other,ingress,active)
        self.assertTrue(torch.equal(qa,qb))
        for current in (a,b):
            result,receipts,_=hub.query(current,qa[:,:1],model.nodes,{'R0':'C'},active[:,:1],regions=('R0',))
            self.assertEqual(float(result.abs().sum()),0.); self.assertEqual(receipts,((),))

    def test_live_fork_matches_duplicated_history_and_gradients(self):
        from .activation_checkpoint import checkpoint_tree
        model=self.model().double(); torch.nn.init.normal_(model.hippocampus.compose.weight,std=.05)
        history=torch.tensor([[2,3,4]])
        questions=torch.tensor([[5,6],[7,8]])
        results=[]
        for fork in (False,True):
            model.zero_grad(set_to_none=True)
            if fork:
                prefix=model(history); state=fork_state(prefix['state'],torch.tensor([0,0]))
            else:
                state=model(history.expand(2,-1))['state']
            result=model(questions,state=state)
            result['logits'].square().mean().backward()
            results.append((result,[None if p.grad is None else p.grad.clone() for p in model.parameters()]))
        torch.testing.assert_close(results[0][0]['logits'],results[1][0]['logits'],atol=1e-10,rtol=1e-9)
        for a,b in zip(results[0][1],results[1][1]):
            if a is None: self.assertIsNone(b)
            else: torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-6)

    def test_activation_checkpoint_keeps_sparse_credit(self):
        from .activation_checkpoint import checkpoint_tree
        model=self.model(); torch.nn.init.normal_(model.hippocampus.compose.weight,std=.05)
        inputs=torch.tensor([[2,3,4]])
        results=[]
        for checkpointed in (False,True):
            model.zero_grad(set_to_none=True)
            function=lambda x,state:model(x,state=state)
            result=checkpoint_tree(function,inputs,model.initial_state(1)) if checkpointed else function(inputs,model.initial_state(1))
            result['logits'].square().mean().backward()
            results.append((result,[None if p.grad is None else p.grad.clone() for p in model.parameters()]))
        torch.testing.assert_close(results[0][0]['logits'],results[1][0]['logits'],atol=0,rtol=0)
        for a,b in zip(results[0][1],results[1][1]):
            if a is None: self.assertIsNone(b)
            else: torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_owner_encoder_source_and_live_boundary_gradients(self):
        model=self.model(); hub=model.hippocampus
        torch.nn.init.normal_(hub.compose.weight,std=.1)
        source=torch.randn(1,32,requires_grad=True)
        capsule=model.nodes['U'].encode_anchor(source)[0]
        state=hub.initial_state(1,'cpu',torch.float32)
        offer=self.offer(hub,state,owner='U',region='R1',value=capsule)
        state=hub.commit(state,((offer,),))
        current=torch.randn(1,3,32,requires_grad=True)
        advanced,keys=hub.advance_cues(state,current,torch.ones(1,3,dtype=torch.bool))
        result,_,_=hub.query(advanced,keys[:,:1],model.nodes,{'R0':'C'},torch.ones(1,1,dtype=torch.bool),regions=('R0',))
        result.square().sum().backward()
        self.assertGreater(float(source.grad.abs().sum()),0)
        self.assertGreater(float(current.grad.abs().sum()),0)
        self.assertGreater(float(model.nodes['U'].anchor_encoder.weight.grad.abs().sum()),0)
        self.assertGreater(float(hub.interface.address.weight.grad.abs().sum()),0)
        from .ports import map_tensors
        boundary=map_tensors(advanced,lambda x:x.detach())
        self.assertTrue(torch.equal(boundary.rows[0].anchors['R1'][0].capsule,advanced.rows[0].anchors['R1'][0].capsule))
        self.assertFalse(boundary.rows[0].anchors['R1'][0].capsule.requires_grad)

    def test_directory_zero_cue_has_logarithmic_work(self):
        import math
        from .hippocampus_index import SparseDirectory
        for shards in (2,4,8,16,32):
            config=HippocampusConfig(enabled=True,shards=shards)
            directory=SparseDirectory(config)
            for key in (torch.zeros(32),torch.ones(32),torch.randn(32)):
                budget=WorkBudget(config); selected=directory.route(key,budget,explore=False)
                self.assertEqual(len(selected),2)
                self.assertLessEqual(budget.directory_nodes,2*int(math.log2(shards))+1)

    def test_future_actor_head_does_not_block_independent_region(self):
        fabric=RecallFabric(HippocampusConfig(service_budget=1))
        fabric.register('future',lambda x,s:x); fabric.register('ready',lambda x,s:x)
        state=FabricState()
        state,_=fabric.submit(state,Envelope(('future',0),'future',5,8,10))
        state,_=fabric.submit(state,Envelope(('ready',0),'ready',0,2,20))
        state,results=fabric.service(state,0,None)
        self.assertEqual(results,((('ready',0),'ok',20),))
        self.assertIn('future',state.queues)
        state,results=fabric.service(state,5,None)
        self.assertEqual(results,((('future',0),'ok',10),))

    def test_segmented_recomputation_preserves_masked_objective_and_state(self):
        from .activation_checkpoint import enable_associative_checkpointing
        model=self.model(); torch.nn.init.normal_(model.hippocampus.compose.weight,std=.05)
        x=torch.tensor([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
        valid=torch.ones_like(x,dtype=torch.bool); valid[0,:2]=False
        reset=torch.zeros_like(valid); reset[0,2]=True; reset[1,0]=True
        expected=None
        for checkpoint in (False,True):
            model.zero_grad(set_to_none=True)
            if checkpoint: enable_associative_checkpointing(model,segment=3,min_positions=0)
            result=model(x,targets=x,reset=reset,availability={v:valid for v in model.cfg.views},node_masks={n:valid for n in model.nodes})
            result['loss'].backward()
            gradients=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
            if expected is None: expected=(result,gradients)
            else:
                torch.testing.assert_close(result['logits'],expected[0]['logits'],atol=0,rtol=0)
                torch.testing.assert_close(result['loss'],expected[0]['loss'],atol=1e-7,rtol=1e-6)
                same(self,result['state'].to_dict(),expected[0]['state'].to_dict())
                self.assertEqual(result['external_events'],expected[0]['external_events'])
                actual_flat=[]; expected_flat=[]
                for a,b in zip(gradients,expected[1]):
                    if a is None: self.assertIsNone(b)
                    else:
                        actual_flat.append(a.flatten()); expected_flat.append(b.flatten())
                actual_flat=torch.cat(actual_flat); expected_flat=torch.cat(expected_flat)
                # Chunking changes FP32 accumulation order; the FP64 probe
                # agrees to 1.2e-15 relative L2. Check the full gradient norm.
                relative=(actual_flat-expected_flat).norm()/expected_flat.norm()
                self.assertLess(float(relative),2e-6)

    def test_training_resume_names_moments_fast_state_and_rng(self):
        import hashlib
        from .data import OrderedByteStreams,split_bytes
        from .task_stream import TaskStreamConfig
        from scripts.run_hippocampus import HippocampusRun
        raw=b'Ordinary words and context.\n'*100
        cfg=GraphConfig.tiny(task_publication=('A','B','C','U','V'),publication_state_commit='publication',recall_slots=2,recurrent_feedback=True)
        torch.manual_seed(4); model=RegionLanguageModel(cfg)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
        streams=OrderedByteStreams.from_bytes(split_bytes(raw)[0],1)
        parent=dict(kind='selective_recall',config=cfg.to_dict(),data_config=TaskStreamConfig(entities='ABCD',acquisition_queries=True).to_dict(),
            raw_sha256=hashlib.sha256(raw).hexdigest(),weights=model.state_dict(),optimizer=optimizer.state_dict(),
            language_stream=streams.state_dict(),language_state=model.initial_state(1).to_dict(),rng=torch.get_rng_state(),
            step=48,next_document=660,protocol=dict(data_seed=683,language_batch=1,language_context=2))
        run=HippocampusRun(parent,raw,arm='C',validation_cases=1); run.update()
        stream=io.BytesIO(); torch.save(run.state_dict(),stream); stream.seek(0); saved=torch.load(stream,weights_only=True)
        expected=run.update(); final=io.BytesIO(); torch.save(run.state_dict(),final); final.seek(0)
        restored=HippocampusRun(parent,raw,arm='C',validation_cases=1); restored.restore(saved); got=restored.update()
        for key in ('seconds','peak_mib'): expected.pop(key); got.pop(key)
        self.assertEqual(expected,got); same(self,restored.state_dict(),torch.load(final,weights_only=True))


if __name__=='__main__': unittest.main()
