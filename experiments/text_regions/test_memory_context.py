"""Integrated acceptance for explicit evidence and event-clock ownership."""
import copy
from dataclasses import replace
import unittest
import torch
from .config import GraphConfig
from .hippocampus_config import HippocampusConfig
from .memory_context import MemoryContextConfig
from .model import RegionLanguageModel
from .state import GraphState
from .paired_recall import fork_state
from .activation_checkpoint import enable_associative_checkpointing
from .test_hippocampus import same


class MemoryContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(61)

    def model(self, **options):
        config = GraphConfig.tiny(rounds=2,hippocampus=HippocampusConfig(enabled=True,
            cue_clock='event',ca1_policy='bounded',admission=False),
            memory_context=MemoryContextConfig(enabled=True,**options))
        return RegionLanguageModel(config)

    def test_ordinary_loss_trains_query_reader_and_owner(self):
        model = self.model()
        x = torch.tensor([[1,9,2,8],[4,6,5,7]])
        result = model(x,targets=x.roll(-1,1)); result['loss'].backward()
        for name in ('request.weight','question.weight','hold.weight','reader.query.weight','composer.weight'):
            grad = dict(model.memory_interface.named_parameters())[name].grad
            self.assertIsNotNone(grad,name)
            self.assertGreater(float(grad.abs().sum()),0,name)
        self.assertTrue(any(n.anchor_content.weight.grad is not None and float(n.anchor_content.weight.grad.abs().sum()) > 0
                            for n in model.nodes.values()))
        self.assertLessEqual(max(int(c.valid.sum(1).max()) for c in result['state'].memory_context.values()),8)

    def test_direct_reader_keeps_content_and_separates_query_key(self):
        model = self.model(); node = model.nodes['A']
        with torch.no_grad():
            node.anchor_adjustment.weight.zero_(); node.anchor_adjustment.bias.zero_()
        capsule = torch.randn(2,model.cfg.width)
        cue = torch.randn(2,model.cfg.hippocampus.address_width)
        torch.testing.assert_close(node.read_anchor(capsule,cue,stored_key=-cue),capsule,rtol=0,atol=0)
        with torch.no_grad():
            node.anchor_adjustment.weight.normal_(std=.1)
        a = node.read_anchor(capsule,cue,stored_key=-cue)
        b = node.read_anchor(capsule,-cue,stored_key=cue)
        self.assertFalse(torch.equal(a,b))

    def test_event_links_dedup_rounds_and_keep_provenance(self):
        model = self.model()
        with torch.no_grad():
            one = model(torch.tensor([[1]]),rounds=1)['state']
            many = model(torch.tensor([[1]]),rounds=4)['state']
        self.assertEqual(one.hippocampus.rows[0].anchor_counters,many.hippocampus.rows[0].anchor_counters)
        row = many.hippocampus.rows[0]
        for region,bank in row.anchors.items():
            self.assertEqual({a.detail for a in bank.values()},{'source','abstract'})
            self.assertEqual(len({a.event_id for a in bank.values()}),1)
            direct = next(a for a in bank.values() if a.detail == 'source')
            self.assertEqual(direct.source_event,0)
            self.assertTrue(direct.sources)
            binding = row.binding(row.event_bindings[direct.event_id],row.round)
            self.assertEqual(len(binding.links),2)

    def test_bounded_requests_reuse_cache_and_distinct_requests(self):
        model = self.model(requests_per_round=2,request_change=2.,refresh_interval=100)
        with torch.no_grad():
            result = model(torch.tensor([[1,2,3,4,5]]))
        counts = [w['context_requests'] for w in result['hippocampus_work'] if 'context_requests' in w]
        self.assertLessEqual(max(counts),2)
        self.assertGreaterEqual(sum(counts),len(model.nodes))
        self.assertLess(sum(counts),len(counts)*2)
        contexts = result['state'].memory_context
        self.assertTrue(all(int(c.fetched[0]) >= 0 for c in contexts.values()))
        self.assertFalse(torch.equal(contexts['A'].cue,contexts['B'].cue))

    def test_slot_null_does_not_erase_stored_value(self):
        model = self.model(); interface = model.memory_interface
        context = interface.empty(1,'cpu',torch.float32)
        context = replace(context,values=torch.randn_like(context.values),valid=torch.ones_like(context.valid))
        before = context.values.clone()
        with torch.no_grad():
            interface.reader.null.bias.fill_(1000)
            output = interface.reader(context.query,context,torch.tensor([0]))
        self.assertTrue(torch.equal(output,torch.zeros_like(output)))
        self.assertTrue(torch.equal(context.values,before))

    def test_query_switch_and_one_source_content_intervention(self):
        interface = self.model().memory_interface
        context = interface.empty(2,'cpu',torch.float32)
        w = context.query.shape[-1]
        values = torch.zeros_like(context.values); keys = values.clone()
        values[:,0,0] = 1.; values[:,1,1] = 1.
        keys[:,0,0] = 1.; keys[:,1,1] = 1.
        query = torch.zeros_like(context.query); query[0,0] = 100.; query[1,1] = 100.
        valid = torch.zeros_like(context.valid); valid[:,:2] = True
        context = replace(context,values=values,keys=keys,query=query,valid=valid,
                          strength=torch.ones_like(context.strength))
        with torch.no_grad():
            interface.reader.query.weight.zero_()
            interface.reader.query.weight[:,:w].copy_(torch.eye(w))
            interface.reader.key.weight.copy_(torch.eye(w)); interface.reader.null.bias.fill_(-100)
            intact = interface.reader(torch.zeros_like(query),context,torch.zeros(2,dtype=torch.long))
            changed = values.clone(); changed[:,0] = 0.; changed[:,0,2] = 1.
            intervention = interface.reader(torch.zeros_like(query),replace(context,values=changed),torch.zeros(2,dtype=torch.long))
        self.assertEqual(intact.argmax(-1).tolist(),[0,1])
        self.assertEqual(intervention.argmax(-1).tolist(),[2,1])

    def test_cached_refinement_delivers_the_accepted_reread(self):
        import types
        model = self.model().eval(); interface = model.memory_interface
        with torch.no_grad():
            state = model(torch.tensor([[1,2]]))['state']
        row = state.hippocampus.rows[0]
        anchor = next(iter(row.anchors['R0'].values())); h = anchor.handle
        context = interface.empty(1,'cpu',torch.float32)
        ident = [2,interface.indices[h.node],h.namespace,h.slot,h.generation,h.version,h.semantic_version,anchor.source_event]
        context = interface.merge(context,[[(anchor.capsule,interface.key(anchor.key),interface.role(anchor.role),
            ident,anchor.capsule.new_ones(()),False)]],state.events_seen)
        context = replace(context,cue=torch.randn_like(context.cue))
        reads=[]
        def read(this,capsule,cue,working=None,stored_key=None):
            reads.append(cue.clone())
            return capsule+cue[:,:1]
        def compare(this,query,expected,returned,valid,age,terminal=False):
            return query.new_tensor([[1.,0.,0.] if terminal else [0.,0.,1.]]).expand(len(query),-1)
        model.nodes[h.node].read_anchor=types.MethodType(read,model.nodes[h.node])
        model.hippocampus.compare=types.MethodType(compare,model.hippocampus)
        result=interface.interpret(context,state,model,torch.tensor([True]))
        self.assertEqual(len(reads),2)
        torch.testing.assert_close(result.values[:,0],context.values[:,0]+reads[1][:,:1])
        self.assertFalse(torch.equal(reads[0],reads[1]))
        torch.testing.assert_close(context.values[0,0],anchor.capsule,rtol=0,atol=0)

    def test_working_state_advances_while_publication_holds(self):
        config = replace(self.model().cfg,task_publication=('A',),publication_state_commit='publication')
        model = RegionLanguageModel(config)
        before = model(torch.tensor([[1]]))['state']
        after = model(torch.tensor([[7]]),state=before,publication_overrides={'A':'hold'})['state']
        torch.testing.assert_close(before.ports.latest['A'].payload,after.ports.latest['A'].payload,rtol=0,atol=0)
        self.assertFalse(torch.equal(before.nodes['A'].working,after.nodes['A'].working))
        self.assertFalse(torch.equal(before.nodes['A'].blocks[0],after.nodes['A'].blocks[0]))

    def test_resume_fork_reset_and_causal_chunks(self):
        model = self.model().eval()
        x = torch.tensor([[1,2,3,4],[4,3,2,1]])
        with torch.no_grad():
            whole = model(x)
            prefix = model(x[:,:2])
            saved = GraphState.from_dict(prefix['state'].to_dict())
            suffix = model(x[:,2:],state=saved)
            torch.testing.assert_close(whole['logits'][:,2:],suffix['logits'],rtol=0,atol=0)
            fork = fork_state(saved,torch.tensor([1,0,1]))
            output = model(x[[1,0,1],2:],state=fork)
            torch.testing.assert_close(output['logits'],suffix['logits'][[1,0,1]],rtol=1e-5,atol=1e-6)
            reset = model(x[:,2:],state=saved,reset=torch.ones(2,dtype=torch.bool))
            fresh = model(x[:,2:])
            torch.testing.assert_close(reset['logits'],fresh['logits'],rtol=0,atol=0)

    def test_recomputation_preserves_logits_and_gradients(self):
        direct = self.model().double()
        recompute = copy.deepcopy(direct)
        enable_associative_checkpointing(recompute,segment=2,min_positions=0)
        x = torch.tensor([[1,4,2,3]])
        a,b = direct(x),recompute(x)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        weight = torch.randn_like(a['logits'])
        (a['logits']*weight).sum().backward(); (b['logits']*weight).sum().backward()
        for (name,p),(_,q) in zip(direct.named_parameters(),recompute.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-9,msg=name)

    def test_masked_owner_stale_handle_and_source_intervention(self):
        model = self.model().eval()
        with torch.no_grad():
            state = model(torch.tensor([[1,2,3,4]]))['state']
            intact = model(torch.tensor([[5]]),state=state)
            model.memory_interface.blocked_sources = ('A',)
            blocked = model(torch.tensor([[5]]),state=state)
            self.assertFalse(torch.equal(intact['logits'],blocked['logits']))
            model.memory_interface.blocked_sources = ()
            active = {n:torch.ones(1,dtype=torch.bool) for n in model.nodes}; active['A'].zero_()
            for context in state.memory_context.values():
                valid = model.memory_interface.valid_slots(context,state,active)
                self.assertFalse(bool((valid.valid & valid.identities[...,1].eq(0)).any()))
            row = state.hippocampus.rows[0]
            empty = replace(state,hippocampus=replace(state.hippocampus,rows=(replace(row,anchors={}),)))
            for context in state.memory_context.values():
                checked = model.memory_interface.valid_slots(context,empty,active)
                self.assertFalse(bool((checked.valid & checked.identities[...,0].eq(2)).any()))

    def test_reference_runtime_and_stage_order(self):
        model = self.model().eval()
        x = torch.tensor([[2,4,6]])
        with torch.no_grad():
            a = model(x)
            b = model(x,reverse_independent=True)
            model.set_execution_backend('reference')
            c = model(x)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        torch.testing.assert_close(a['logits'],c['logits'],rtol=2e-5,atol=2e-6)

    def test_default_off_config_roundtrip(self):
        old = GraphConfig.tiny().to_dict(); old.pop('memory_context')
        self.assertFalse(GraphConfig.from_dict(old).memory_context.enabled)
        config = self.model().cfg
        self.assertEqual(GraphConfig.from_dict(config.to_dict()),config)


if __name__ == '__main__':
    unittest.main()
