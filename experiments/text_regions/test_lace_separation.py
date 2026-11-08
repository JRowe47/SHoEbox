"""Contracts for query bias, first-order credit, isolation and recovery."""
import copy,hashlib,io,unittest
from dataclasses import replace
import torch
from torch.nn import functional as F
from .config import GraphConfig
from .model import RegionLanguageModel
from .hippocampus_config import HippocampusConfig
from .lace_separation import LaceBiasHead,LaceCredit,LaceSeparationConfig,extra_score_gradient,local_derivative
from .activation_checkpoint import enable_associative_checkpointing
from .runtime import forward_scope
from ..hat_minkron.semantic_core import LaceController
from .runtime import RuntimeLaceController


class LaceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(43)

    def model(self,installed=True,enabled=True):
        return RegionLanguageModel(GraphConfig.tiny(rounds=2,
            hippocampus=HippocampusConfig(enabled=True,cue_clock='event',reinstatement='expert',ca1_policy='bounded'),
            lace_separation=LaceSeparationConfig(installed=installed,enabled=enabled)))

    def test_disabled_zero_and_rng_parity(self):
        torch.manual_seed(43)
        a=self.model(False,False);rng=torch.get_rng_state().clone()
        torch.manual_seed(43)
        b=self.model(True,False);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.manual_seed(43)
        c=self.model();self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for model in (b,c):
            missing,unexpected=model.load_state_dict(a.state_dict(),strict=False)
            self.assertFalse(unexpected);self.assertTrue(all('.lace_head.' in n for n in missing))
        with torch.no_grad():
            x=torch.tensor([[1,2,3],[4,5,3]])
            expected=a(x)['logits']
            self.assertTrue(torch.equal(expected,b(x)['logits']))
            self.assertTrue(torch.equal(expected,c(x)['logits']))

    def test_credit_signs_layer_alignment_and_same_target(self):
        ids=torch.tensor([[[[0,1],[0,1]],[[0,2],[0,2]]]])
        g=torch.tensor([[[[-1.,-1.],[-1.,-1.]],[[1.,-1.],[-1.,-1.]]]])
        extra,shared,conflict=extra_score_gradient(ids,g,torch.tensor([True]),.1,.05)
        self.assertEqual(float(extra[0,0,0,0]),0.)
        self.assertGreater(float(extra[0,1,0,0]),0.)
        self.assertLess(float(extra[0,0,0,1]),0.)
        self.assertEqual(float(extra[0,1,1,0]),0.) # useful to both; other layer independent
        self.assertEqual(int(conflict.sum()),2)
        suppressed,_,_=extra_score_gradient(ids,g,torch.tensor([False]),.1,.05)
        self.assertEqual(float(suppressed.abs().sum()),0.)
        zero,_,_=extra_score_gradient(ids,torch.zeros_like(g),torch.tensor([True]),.1,.05)
        self.assertEqual(float(zero.abs().sum()),0.)

    def test_identical_sets_can_get_opposing_credit(self):
        ids=torch.tensor([[[[0,1]],[[0,1]]]])
        g=torch.tensor([[[[-1.,1.]],[[1.,-1.]]]])
        e,s,c=extra_score_gradient(ids,g,torch.tensor([True]),.1,.05)
        self.assertTrue(bool(s.all()));self.assertTrue(bool(c.all()))
        self.assertGreater(float(e[0,1,0,0]),0.);self.assertGreater(float(e[0,0,0,1]),0.)

    def test_analytic_derivative_matches_local_autograd(self):
        head=LaceBiasHead(6,3,5,LaceSeparationConfig(),1).double()
        torch.nn.init.normal_(head.V,std=.2)
        z=torch.randn(4,6,dtype=torch.double);e=torch.randn(4,15,dtype=torch.double)
        bias,cache=head(z,True);(bias.flatten(1)*e).sum().backward()
        z,h,b,v=cache;du,dv=local_derivative(e*(1-(b/.5).square()),z,h,v)
        torch.testing.assert_close(du,head.U.grad,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(dv,head.V.grad,atol=1e-12,rtol=1e-12)

    def test_runtime_reference_and_fresh_score_bias(self):
        c=GraphConfig.tiny().hat_config().child_config(0)
        ref=LaceController(c);fast=RuntimeLaceController(c);fast.load_state_dict(ref.state_dict())
        evidence=torch.randn(2,1,3,c.groups);context=torch.randn(2,1,c.width)
        bias=torch.randn(2,3,c.laces)*.2
        expected,records=ref(evidence,context,query_bias=bias,capture_scores=True)
        class Host:
            execution_backend='optimized'
            _large_expert_banks=False
            @forward_scope
            def run(self):return fast(evidence,context,query_bias=bias,capture_scores=True)
        for _ in range(2):
            actual,got=Host().run()
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
            torch.testing.assert_close(got['selected_scores'],records['selected_scores'])
        # Direct balance does not reach the captured score tensor.
        records['selected_scores'].retain_grad();records['balance'].backward()
        self.assertIsNone(records['selected_scores'].grad)

    def test_recompute_capture_once_and_same_gradients(self):
        direct=self.model();recomputed=copy.deepcopy(direct)
        enable_associative_checkpointing(recomputed,segment=2,min_positions=0)
        x=torch.tensor([[1,7,3],[2,7,3]]);y=torch.tensor([48,49]);outputs=[]
        for model in (direct,recomputed):
            capture=LaceCredit(2,model.cfg.lace_separation);model._lace_capture=capture
            result=model(x);capture.supervise(result['logits'][:,-1],y)
            F.cross_entropy(result['logits'][:,-1],y).backward()
            self.assertEqual(len(capture.records),14)
            stats=capture.apply();self.assertFalse(capture.records)
            self.assertLessEqual(stats['extra_head_norm'],stats['ordinary_head_norm']*.250001)
            model._lace_capture=None
            outputs.append((result['logits'],{n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}))
        torch.testing.assert_close(outputs[0][0],outputs[1][0])
        for name,g in outputs[0][1].items():
            other=outputs[1][1][name]
            if g is None:self.assertIsNone(other,name)
            else:torch.testing.assert_close(g,other,rtol=2e-4,atol=2e-6,msg=name)

    def test_inactive_pair_exclusion_and_gate(self):
        head=LaceBiasHead(4,1,4,LaceSeparationConfig(),1)
        with torch.no_grad():head.V.fill_(.1)
        capture=LaceCredit(0,head.config)
        z=torch.randn(3,4);bias,cache=head(z,True)
        scores=bias.gather(-1,torch.tensor([[[0,1]],[[0,2]],[[0,3]]]))
        lace=dict(selected_scores=scores,lace_ids=torch.tensor([[[[0,1]]],[[[0,2]]],[[[0,3]]]]),lace_weights=scores.softmax(-1)[:,None])
        capture.record(('node',0),head,torch.tensor([0,1,2]),torch.tensor([True,True,True]),lace,cache)
        logits=torch.zeros(4,256);capture.supervise(logits,torch.tensor([48,49,48,49]))
        (scores*torch.tensor([[[-1.,1.]],[[1.,-1.]],[[1.,-1.]]])).sum().backward()
        stats=capture.apply();self.assertEqual(stats['pair_decisions'],1)
        capture=LaceCredit(0,head.config);logits[0,48]=2;logits[1,49]=2
        capture.supervise(logits,torch.tensor([48,49,48,48]));self.assertFalse(bool(capture.eligible.any()))

    def test_bias_reaches_routes_experts_logits_and_cue_gradients(self):
        model=self.model();x=torch.tensor([[1,2,3],[7,2,3]])
        with torch.no_grad():baseline=model(x)['logits']
        for node in model.nodes.values():torch.nn.init.normal_(node.lace_head.V,std=.4)
        result=model(x);self.assertFalse(torch.equal(baseline,result['logits']))
        F.cross_entropy(result['logits'][:,-1],torch.tensor([48,49])).backward()
        self.assertGreater(sum(float(n.lace_head.U.grad.abs().sum()) for n in model.nodes.values()),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.hippocampus.interface.trace.parameters() if p.grad is not None),0.)

    def test_exact_optimizer_resume_and_inherited_settings(self):
        from scripts.run_lace_separation import LaceRun
        from .data import OrderedByteStreams,split_bytes
        from .task_stream import TaskStreamConfig
        from .test_probe import assert_tree_equal
        raw=b'Ordinary language for continuation checks. '*80
        model=self.model(False,False)
        opt=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
        streams=OrderedByteStreams.from_bytes(split_bytes(raw)[0],1)
        parent=dict(kind='hippocampus_v1',config=model.cfg.to_dict(),weights=model.state_dict(),
            optimizer=opt.state_dict(),data_config=TaskStreamConfig(entities='AB',values='01',acquisition_queries=True).to_dict(),
            raw_sha256=hashlib.sha256(raw).hexdigest(),language_stream=streams.state_dict(),
            language_state=model.initial_state(1).to_dict(),rng=torch.get_rng_state(),step=16,next_document=800,
            protocol=dict(lr=.003,language_weight=.25,syntax_weight=.1,global_clip=1.,gaps=[0],
                data_seed=683,language_batch=1,language_context=2))
        run=LaceRun(parent,raw,'D');run.update()
        buf=io.BytesIO();torch.save(run.state_dict(),buf);buf.seek(0);saved=torch.load(buf,weights_only=True)
        run.update();expected=copy.deepcopy(run.state_dict())
        resumed=LaceRun(parent,raw,'D');resumed.restore(saved);resumed.update()
        assert_tree_equal(self,expected,resumed.state_dict())
        self.assertEqual(resumed.next_document,802)
        self.assertEqual(resumed.optimizer.param_groups[0]['betas'],(.9,.95))
        self.assertTrue(all(p.requires_grad for p in resumed.model.parameters()))

    def test_config_migration_and_validation(self):
        old=GraphConfig.tiny().to_dict();old.pop('lace_separation')
        self.assertFalse(GraphConfig.from_dict(old).lace_separation.installed)
        for change in ({'rank':0},{'cap':float('nan')},{'alpha':-1},{'enabled':True}):
            with self.assertRaises(ValueError):replace(LaceSeparationConfig(),**change).validate()

if __name__=='__main__':unittest.main()
