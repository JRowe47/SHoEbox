"""Normal-pass activity taps, utility interventions and state provenance."""
import unittest
import torch
from .config import GraphConfig
from .model import RegionLanguageModel
from .graph import EdgeSubstitution
from .state import GraphState

torch.set_num_threads(2)

class ActivityChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model=RegionLanguageModel(GraphConfig.tiny(rounds=2,recurrent_feedback=True)).eval()
        self.x=torch.tensor([[65,66],[70,71]])

    def test_taps_preserve_predictions_state_and_gradients(self):
        plain=self.model(self.x)
        tapped=self.model(self.x,return_view_activity=True,return_edge_activity=True)
        torch.testing.assert_close(plain['logits'],tapped['logits'],rtol=0,atol=0)
        for a,b in zip(plain['state'].tensors(),tapped['state'].tensors()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        payload=tapped['view_activity']['text']['payload']
        self.assertEqual(payload.shape,(2,2,32))
        payload.square().sum().backward()
        self.assertGreater(float(self.model.hat.byte_embedding.weight.grad.abs().sum()),0)

    def test_source_taps_stay_independent_after_feedback(self):
        a={'A':self.x,'B':self.x+2,'C':self.x+3}
        b={**a,'B':self.x+30}
        sa=self.model(a)['state'];sb=self.model(b)['state']
        ra=self.model(a,state=sa,return_view_activity=True)
        rb=self.model(b,state=sb,return_view_activity=True)
        torch.testing.assert_close(ra['view_activity']['A']['payload'],rb['view_activity']['A']['payload'],rtol=0,atol=0)
        self.assertFalse(torch.equal(ra['state'].ports.latest['A'].payload,rb['state'].ports.latest['A'].payload))

    def test_usage_counts_only_consumed_dependencies(self):
        mask={'A__U__forward':torch.zeros_like(self.x,dtype=torch.bool)}
        r=self.model(self.x,return_edge_activity=True,edge_masks=mask)
        self.assertEqual(int(r['edge_activity'][:,:,r['edge_names'].index('A__U__forward')].sum()),0)
        self.assertTrue(bool((r['edge_activity'][:,:,r['edge_names'].index('B__U__forward')]==2).all()))
        base=self.model(self.x)
        altered=self.model(self.x,edge_bias={'A__U__forward':-4.})
        self.assertEqual(base['expert_executions'],altered['expert_executions'])
        self.assertFalse(torch.equal(base['logits'],altered['logits']))
        with self.assertRaises(ValueError):self.model(self.x,edge_bias={'bogus':1.})

    def test_row_substitution_identity_survives_inactive_update(self):
        sub=EdgeSubstitution(torch.randn(2,2,32),torch.zeros(2,2,3,4))
        state=self.model(self.x,substitutions={'A__U__forward':sub})['state'].detach()
        mask={'U':torch.tensor([[False,False],[True,True]])}
        result=self.model(self.x,state=state,node_masks=mask)
        dep=next(d for d in result['state'].ports.latest['U'].dependencies if d.edge=='A__U__forward')
        self.assertEqual(dep.substituted.tolist(),[True,False])
        self.assertEqual(dep.intervention,'mixed')
        loaded=GraphState.from_dict(result['state'].to_dict())
        self.assertEqual(next(d for d in loaded.ports.latest['U'].dependencies if d.edge=='A__U__forward').substituted.tolist(),[True,False])
        cleared=loaded.reset(torch.tensor([True,False]))
        dep=next(d for d in cleared.ports.latest['U'].dependencies if d.edge=='A__U__forward')
        self.assertEqual(dep.intervention,'none')
        self.assertEqual(dep.substituted.tolist(),[False,False])

    def test_lace_bias_controls_executed_selection_without_extra_experts(self):
        base=self.model(self.x,trace=True)
        zero=self.model(self.x,lace_bias={'A':torch.zeros(3,16)})
        torch.testing.assert_close(base['logits'],zero['logits'],rtol=0,atol=0)
        bias=torch.full((3,16),-100.)
        bias[:,:2]=100.
        changed=self.model(self.x,trace=True,lace_bias={'A':bias})
        for record in changed['trace']:
            if record['node']=='A':
                self.assertTrue(bool((record['routing']['laces']<2).all()))
        self.assertFalse(torch.equal(base['logits'],changed['logits']))
        self.assertEqual(base['expert_executions'],changed['expert_executions'])
        with self.assertRaises(ValueError):self.model(self.x,lace_bias={'A':torch.zeros(16)})

    def test_generate_prefill_state_and_reset_are_not_reapplied(self):
        state=self.model(self.x)['state'].detach()
        output,end=self.model.generate(self.x[:,:1],count=3,state=state,reset=torch.tensor([True,False]))
        self.assertEqual(output.shape,(2,4))
        self.assertEqual(end.events_seen.tolist(),[4,6])
        self.assertEqual(end.epochs.tolist(),[1,0])

if __name__=='__main__':unittest.main()
