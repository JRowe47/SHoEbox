"""Shared nonlinear byte decoding, matched controls, and artifact integrity."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from .emission_adapter import (fit_adapter, fit_adapters, load_adapters, save_adapters,
                               score_adapters, compact_capture, same_protocol)
from .readout_diagnostic import fit_byte_decoder
from .emission_compare import audit_rows

torch.set_num_threads(2)


def fixture(split):
    x=torch.eye(12);y=torch.tensor(list(b'ABCD')*3)
    queries=[q for q in ('shared','left','right') for _ in range(4)]
    return {'features':{'final/raw':x,'final/mixed':x},'answers':y,
        'queries':queries,'controls':['all']*12,'identities':[f'{split}/{i}' for i in range(12)],
        'protocol':{'split':split,'model_tensor_sha256':'fixture','policy':{},'gap':0,'query_every':4},
        'direct_predictions':y.clone(),'direct_nll_per_case':torch.zeros(12),
        'groups':[{'control':'all','query':q,'start':4*i,'stop':4*i+4,'information_present':True,
                   'external_events':4,'expert_executions':4} for i,q in enumerate(('shared','left','right'))]}


class NonlinearChecks(unittest.TestCase):
    def test_protocol_comparison_accepts_json_arrays_but_rejects_real_changes(self):
        self.assertTrue(same_protocol({'nodes':({'name':'P'},),'width':32},
                                      {'width':32,'nodes':[{'name':'P'}]}))
        self.assertFalse(same_protocol({'width':32},{'width':64}))

    def test_gelu_learns_xor_with_same_parameters_and_initialization(self):
        x=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        y=torch.tensor([0,255,255,0])
        linear=fit_adapter(x,y,activation='identity',hidden=16,seed=6,max_iter=150)
        nonlinear=fit_adapter(x,y,activation='gelu',hidden=16,seed=6,max_iter=150)
        self.assertLessEqual(float((linear.predict(x)==y).float().mean()),.75)
        self.assertTrue(torch.equal(nonlinear.predict(x),y))
        self.assertEqual(linear.fit['parameters'],nonlinear.fit['parameters'])
        self.assertEqual(linear.fit['initial_tensor_sha256'],nonlinear.fit['initial_tensor_sha256'])

    def test_identity_head_collapses_to_one_affine_map(self):
        x=torch.eye(4);y=torch.arange(65,69)
        head=fit_adapter(x,y,activation='identity',hidden=8,max_iter=40)
        collapsed=((x.double()-head.mean)/head.scale) @ (head.w1 @ head.w2)+head.b1 @ head.w2+head.b2
        torch.testing.assert_close(head.logits(x),collapsed,atol=1e-12,rtol=1e-12)

    def test_repeatable_fit_preserves_global_rng_and_input_gradients(self):
        x=torch.eye(4,requires_grad=True);y=torch.arange(65,69)
        before=x.detach().clone();rng=torch.get_rng_state().clone()
        first=fit_adapter(x,y,hidden=8,max_iter=40)
        second=fit_adapter(x,y,hidden=8,max_iter=40)
        self.assertTrue(torch.equal(first.logits(x),second.logits(x)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(x,before));self.assertIsNone(x.grad)

    def test_normalization_uses_only_weighted_training_features(self):
        x=torch.tensor([[0.,2.],[2.,4.],[4.,8.],[8.,16.]])
        weights=torch.tensor([1.,1.,1.,3.]);y=torch.arange(65,69)
        head=fit_adapter(x,y,hidden=4,max_iter=5,sample_weights=weights)
        expected=(x.double()*weights.double()[:,None]).sum(0)/weights.sum()
        torch.testing.assert_close(head.mean,expected)
        mean=head.mean.clone();head.logits(torch.full((2,2),10000.))
        self.assertTrue(torch.equal(mean,head.mean))

    def test_output_is_not_restricted_to_four_answers(self):
        head=fit_adapter(torch.eye(4),torch.arange(65,69),hidden=4,max_iter=10)
        bias=torch.zeros(256,dtype=torch.double);bias[233]=20
        forced=replace(head,w2=torch.zeros_like(head.w2),b2=bias)
        self.assertTrue(bool((forced.predict(torch.eye(4))==233).all()))

    def test_invalid_fits_are_rejected(self):
        x=torch.eye(4);y=torch.arange(65,69)
        for args in ({'activation':'unknown'},{'hidden':0},{'l2':0},{'max_iter':0},
                     {'sample_weights':torch.tensor([1.,1.,0.,1.])}):
            with self.assertRaises(ValueError):fit_adapter(x,y,**args)
        with self.assertRaises(ValueError):fit_adapter(x,torch.tensor([65,66,67,256]))


class SharedHeadChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=fixture('train')
        cls.fitted=fit_adapters(cls.training,seeds=(7,),hidden=8,max_iter=40)
        cls.reference={'answer_heads':{r:fit_byte_decoder(torch.eye(12),cls.training['answers'],max_iter=40)
                                       for r in ('raw','mixed')}}

    def test_roundtrip_preserves_logits_and_rejects_corrupt_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapters.pt';save_adapters(path,self.fitted)
            restored=load_adapters(path)
            self.assertEqual(len(restored['heads']),4)
            for name,head in restored['heads'].items():
                self.assertTrue(torch.equal(head.logits(torch.eye(12)),self.fitted['heads'][name].logits(torch.eye(12))))
            payload=torch.load(path,weights_only=True)
            payload['heads']['raw/gelu/7']['w2']=torch.zeros(8,4)
            torch.save(payload,path)
            with self.assertRaises(ValueError):load_adapters(path)

    def test_scoring_is_shared_and_query_metadata_cannot_change_predictions(self):
        data=fixture('test')
        _,original=score_adapters(data,self.fitted,self.reference)
        data['queries']=['right']*12;data['controls']=['b_only']*12
        _,changed=score_adapters(data,self.fitted,self.reference)
        self.assertEqual(original,changed)
        self.assertEqual(set(original[0]['predictions']),set(self.fitted['heads'])|{'direct','linear/raw','linear/mixed'})

    def test_reject_evaluation_fitting_unsupported_sources_and_leakage(self):
        with self.assertRaises(ValueError):fit_adapters(fixture('test'))
        bad=fixture('train');bad['controls'][4]='b_only'
        with self.assertRaises(ValueError):fit_adapters(bad)
        bad=fixture('test');bad['identities'][0]=self.training['identities'][0]
        with self.assertRaises(ValueError):score_adapters(bad,self.fitted,self.reference)
        bad=fixture('test');bad['protocol']['model_tensor_sha256']='other'
        with self.assertRaises(ValueError):score_adapters(bad,self.fitted,self.reference)

    def test_compact_capture_preserves_direct_predictions_and_losses(self):
        data=fixture('test');data['logits']=torch.eye(256)[data['answers']]*10
        data['features']['prequery/raw']=torch.zeros(12,100)
        compact=compact_capture(data)
        self.assertNotIn('logits',compact)
        self.assertEqual(set(compact['features']),{'final/raw','final/mixed'})
        self.assertTrue(torch.equal(compact['direct_predictions'],data['answers']))
        self.assertEqual(compact['direct_nll_per_case'].shape,(12,))

    def test_raw_prediction_audit_rejects_changed_answers(self):
        result,rows=score_adapters(fixture('test'),self.fitted,self.reference)
        report={'fits':self.fitted['heads'],'evaluations':{'test':result}}
        self.assertEqual(audit_rows(report,rows)[('test','all','shared','direct')],1.)
        bad=copy.deepcopy(rows);bad[0]['predictions']['direct'][0]=233
        with self.assertRaises(ValueError):audit_rows(report,bad)
        bad=copy.deepcopy(rows);bad[0]['answers'][0]=66
        with self.assertRaises(ValueError):audit_rows(report,bad)

    def test_raw_prediction_audit_rejects_task_specific_head_or_duplicate_case(self):
        result,rows=score_adapters(fixture('test'),self.fitted,self.reference)
        report={'fits':self.fitted['heads'],'evaluations':{'test':result}}
        bad=copy.deepcopy(rows);bad[0]['predictions']['shared']=[65]*4
        with self.assertRaises(ValueError):audit_rows(report,bad)
        bad=copy.deepcopy(rows);bad[0]['episode_ids'][0]=bad[0]['episode_ids'][1]
        with self.assertRaises(ValueError):audit_rows(report,bad)

    def test_raw_prediction_audit_rejects_altered_metrics_or_conditions(self):
        result,rows=score_adapters(fixture('test'),self.fitted,self.reference)
        report={'fits':self.fitted['heads'],'evaluations':{'test':result}}
        bad=copy.deepcopy(report);bad['evaluations']['test']['groups'][0]['decoders']['direct']['exact_accuracy']=.5
        with self.assertRaises(ValueError):audit_rows(bad,rows)
        bad=copy.deepcopy(rows);bad[0]['control']='b_only'
        with self.assertRaises(ValueError):audit_rows(report,bad)


if __name__ == '__main__':
    unittest.main()
