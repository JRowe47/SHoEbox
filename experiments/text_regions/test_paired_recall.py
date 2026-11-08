"""A shared history must preserve both forward results and source credit."""
import hashlib
import io
import unittest
import torch
from torch.nn import functional as F

from .config import GraphConfig
from .model import RegionLanguageModel
from .task_stream import TaskStreamConfig
from .task_memory_train import byte_loss
from .paired_recall import paired_batch, paired_forward, fork_state
from .test_probe import assert_tree_equal
from .data import OrderedByteStreams, split_bytes
from scripts.run_paired_recall import PairedRecallRun
from scripts.diagnose_paired_recall import features, ridge_readout, query_readout

torch.set_num_threads(1)


class PairedRecallChecks(unittest.TestCase):
    def batch(self, **options):
        args = dict(split='train', seed=683, start=660, count=1, gap=0,
                    distractor_bytes=b'Natural text between events. ')
        args.update(options)
        return paired_batch(TaskStreamConfig(entities='ABCD', acquisition_queries=True), **args)

    def model(self):
        torch.manual_seed(37)
        return RegionLanguageModel(GraphConfig.tiny(task_publication=('A','B','C','U','V'),
            publication_state_commit='publication', recall_slots=2, recurrent_feedback=True)).double()

    def test_balance_causality_and_shared_history(self):
        batch = self.batch(gap=4)
        self.assertEqual(batch['history'].shape, (16,16))
        self.assertEqual(batch['query'].shape, (32,4))
        pairs = batch['answers'].reshape(-1,2).tolist()
        self.assertEqual(set(map(tuple,pairs)), {(a,b) for a in b'0123' for b in b'0123'})
        self.assertTrue(bool((batch['syntax'][:, -1] == -100).all()))
        self.assertTrue(bool((batch['full'][:, -1] == ord('?')).all()))
        for i, (a,b) in enumerate(pairs):
            self.assertEqual(int(batch['history'][i,4]), a)
            self.assertEqual(int(batch['history'][i,10]), b)
            self.assertEqual(int(batch['query'][2*i,0]), int(batch['history'][i,0]))
            self.assertEqual(int(batch['query'][2*i+1,0]), int(batch['history'][i,6]))

    def test_fork_matches_duplicate_prefix_logits_gradients_and_state(self):
        model, batch = self.model(), self.batch()
        actual = []
        for fork in (False,True):
            model.zero_grad(set_to_none=True)
            result = paired_forward(model,batch) if fork else model(batch['full'])
            loss = (F.cross_entropy(result['logits'][:,-1],batch['answers'])
                    +.1*byte_loss(result['logits'],batch['syntax'])
                    +model.cfg.auxiliary_loss_weight*result['aux']
                    +model.cfg.hat_balance_weight*result['hat_aux'])
            loss.backward()
            actual.append((result, [None if p.grad is None else p.grad.clone() for p in model.parameters()]))
        expected, got = actual[0][0], actual[1][0]
        for key in ('logits','aux','hat_aux'):
            torch.testing.assert_close(got[key],expected[key],rtol=1e-9,atol=1e-10)
        for a,b in zip(got['state'].tensors(),expected['state'].tensors()):
            torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
        for a,b in zip(actual[1][1],actual[0][1]):
            if b is None: self.assertIsNone(a)
            else: torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8)
        self.assertLess(got['executed_positions'],got['duplicated_positions'])

    def test_both_branches_backpropagate_to_shared_history(self):
        model, batch = self.model(), self.batch()
        first = model(batch['history'])
        memory = first['state'].ingress['text'].levels[0][0]
        memory.retain_grad()
        branch = model(batch['query'],state=fork_state(first['state'],batch['rows']))
        gradients=[]
        for offset in (0,1):
            loss=F.cross_entropy(branch['logits'][offset::2,-1],batch['answers'][offset::2])
            gradient, = torch.autograd.grad(loss,memory,retain_graph=True)
            self.assertGreater(float(gradient.norm()),0.)
            gradients.append(gradient)
        loss=F.cross_entropy(branch['logits'][:,-1],batch['answers'])
        combined, = torch.autograd.grad(loss,memory)
        torch.testing.assert_close(combined,(gradients[0]+gradients[1])/2,rtol=1e-8,atol=1e-10)

    def test_exact_resume_preserves_branches_optimizer_and_language_cursor(self):
        raw=b'Ordinary language alongside remembered observations.\n'*100
        model=self.model().float()
        streams=OrderedByteStreams.from_bytes(split_bytes(raw)[0],1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
        parent=dict(kind='selective_recall',config=model.cfg.to_dict(),
            data_config=TaskStreamConfig(entities='ABCD',acquisition_queries=True).to_dict(),
            raw_sha256=hashlib.sha256(raw).hexdigest(),weights=model.state_dict(),
            optimizer=optimizer.state_dict(),language_stream=streams.state_dict(),
            language_state=model.initial_state(1).to_dict(),rng=torch.get_rng_state(),
            step=48,next_document=660,protocol=dict(data_seed=683,language_batch=1,language_context=2))
        run=PairedRecallRun(parent,raw,validation_cases=1); run.update()
        saved=io.BytesIO(); torch.save(run.state_dict(),saved); saved.seek(0)
        checkpoint=torch.load(saved,weights_only=True)
        expected_row=run.update()
        expected=io.BytesIO(); torch.save(run.state_dict(),expected); expected.seek(0)
        restored=PairedRecallRun(parent,raw,validation_cases=1); restored.restore(checkpoint)
        actual_row=restored.update()
        for key in ('seconds','peak_mib'):
            expected_row.pop(key); actual_row.pop(key)
        self.assertEqual(actual_row,expected_row)
        assert_tree_equal(self,restored.state_dict(),torch.load(expected,weights_only=True))
        self.assertEqual(restored.next_document,662)
        self.assertEqual(restored.totals['memory_targets'],64)
        changed=PairedRecallRun(parent,raw,lr=.001,validation_cases=1)
        with self.assertRaises(ValueError): changed.restore(checkpoint)

    def test_diagnostic_probes_recover_signal_and_fail_without_content(self):
        labels=torch.cartesian_prod(torch.arange(4),torch.arange(4))
        values=F.one_hot(labels,num_classes=4).flatten(1).double()
        keys=torch.tensor([[65,66]]*len(labels))
        for x,expected in ((values,1.),(torch.zeros_like(values),.25)):
            probe=ridge_readout(x,x,labels,labels,4)
            binding=query_readout(x,x,labels,labels,keys,keys,4)
            self.assertEqual(probe['older_accuracy'],expected)
            self.assertEqual(probe['newer_accuracy'],expected)
            self.assertEqual(binding['older_accuracy'],expected)
            self.assertEqual(binding['newer_accuracy'],expected)
            self.assertEqual(binding['conflicting_both_accuracy'],1. if expected==1. else 0.)
        # Integer provenance does not become a probe feature.
        state=self.model().initial_state(2)
        from dataclasses import replace
        changed=replace(state,events_seen=torch.tensor([999,1000]),epochs=torch.tensor([87,92]))
        for key,x in features(state).items():
            torch.testing.assert_close(x,features(changed)[key],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
