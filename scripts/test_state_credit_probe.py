"""Validate fact intervention provenance, padding and isolated state swaps."""
import unittest
import torch
from torch.nn import functional as F

from scripts.evaluate_state_credit import counterfactual_pairs, counterfactual_probe, swap_producer_contents
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents
from experiments.text_regions import test_state_credit as state_checks

torch.set_num_threads(1)


class RetentionProbeChecks(unittest.TestCase):
    def pairs(self):
        config=TaskStreamConfig(entities='AB',cycles=1,queries_per_cycle=2,delays=(2,4),semantic_fraction=0.)
        return counterfactual_pairs(make_documents(config,split='long_delay',seed=483,count=2),minimum_delay=4)

    def test_only_observed_source_changes_and_query_suffix_is_identical(self):
        for pair in self.pairs():
            a,b=pair['original'],pair['changed']; q=pair['query']
            self.assertEqual([i for i in range(len(a)) if a[i]!=b[i]],[q['source_position']])
            self.assertEqual(a[q['source_position']],q['answer'])
            self.assertEqual(b[q['source_position']],pair['alternative'])
            self.assertEqual(a[-1:],b'?')

    def test_left_padded_probe_matches_each_real_document(self):
        model=state_checks.StateCreditChecks().model().double(); model.eval()
        pairs=self.pairs()
        expected=[]
        with torch.no_grad():
            for pair in pairs:
                for key,target in (('original',pair['query']['answer']),('changed',pair['alternative'])):
                    logits=model(torch.tensor([list(pair[key])]))['logits'][0,-1]
                    expected.append(float(F.cross_entropy(logits[None],torch.tensor([target]))))
            result=counterfactual_probe(model,pairs,context=7)
        self.assertAlmostEqual(result['answer_nll'],sum(expected)/len(expected),places=9)
        self.assertEqual(result['pairs'],len(pairs))

    def test_swaps_only_producer_contents_and_retains_identity(self):
        model=state_checks.StateCreditChecks().model()
        with torch.no_grad():
            state=model(torch.tensor([[2,3],[5,6]]))['state']
        donors=torch.tensor([1,0])
        swapped=swap_producer_contents(state,model.cfg.task_publication,donors)
        self.assertIs(swapped.ports,state.ports)
        self.assertIs(swapped.ingress,state.ingress)
        self.assertIs(swapped.recall_feedback,state.recall_feedback)
        self.assertIs(swapped.publication_demand,state.publication_demand)
        for name in model.nodes:
            self.assertIs(swapped.nodes[name].clock,state.nodes[name].clock)
            if name in model.cfg.task_publication:
                torch.testing.assert_close(swapped.nodes[name].blocks[0],state.nodes[name].blocks[0][donors])
            else:
                self.assertIs(swapped.nodes[name],state.nodes[name])


if __name__=='__main__':
    unittest.main()
