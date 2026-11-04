"""Frozen-base residual fitting, held-out selection, and portable artifact checks."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .readout_diagnostic import ByteDecoder, fit_byte_decoder
from .residual_adapter import (balanced_weights, base_fingerprint, capture_validation,
    fit_and_select, fit_residual, initialize_residual, load_residuals, save_residuals,
    score_residuals, select_strength, write_bytes)
from .test_emission_adapter import fixture as emission_fixture
from .residual_compare import audit_rows, expected_validation

torch.set_num_threads(2)


def fixture(split):
    data=emission_fixture(split)
    if split=='validation':
        data['features']['final/raw']=torch.cat((torch.eye(12),torch.eye(12)[:4]))
        data['answers']=torch.cat((data['answers'],data['answers'][:4]))
        data['queries']+=['shared']*4;data['controls']+=['b_only']*4
        data['identities'] += [f'{split}/{i}' for i in range(12,16)]
        data['groups'] += [{'control':'b_only','query':'shared','start':12,'stop':16}]
    data['protocol']['episodes']=len(data['answers'])
    return data


class FittingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        cls.y=torch.tensor([0,255,255,0])
        cls.base=fit_byte_decoder(cls.x,cls.y,max_iter=80)

    def test_exact_zero_initial_output_and_zero_strength_fallback(self):
        a=initialize_residual(self.base,hidden=16,activation='gelu')
        b=initialize_residual(self.base,hidden=16,activation='identity')
        self.assertTrue(torch.equal(a.logits(self.x),self.base.logits(self.x)))
        self.assertEqual(a.correction.fit['initial_tensor_sha256'],b.correction.fit['initial_tensor_sha256'])
        self.assertEqual(a.correction.fit['parameters'],(2+1)*16+17*256)
        fitted=fit_residual(self.x,self.y,self.base,max_iter=100)
        self.assertTrue(torch.equal(replace(fitted,strength=0).logits(self.x),self.base.logits(self.x)))

    def test_nonlinear_residual_learns_xor_while_identity_remains_affine(self):
        identity=fit_residual(self.x,self.y,self.base,activation='identity',seed=6,max_iter=150)
        gelu=fit_residual(self.x,self.y,self.base,activation='gelu',seed=6,max_iter=150)
        self.assertLessEqual(float((identity.predict(self.x)==self.y).float().mean()),.75)
        self.assertTrue(torch.equal(gelu.predict(self.x),self.y))
        self.assertEqual(identity.correction.fit['parameters'],gelu.correction.fit['parameters'])

    def test_base_inputs_and_rng_remain_unchanged_with_repeatable_fit(self):
        x=self.x.clone().requires_grad_(True);before=base_fingerprint(self.base)
        rng=torch.get_rng_state().clone()
        a=fit_residual(x,self.y,self.base,max_iter=60)
        b=fit_residual(x,self.y,self.base,max_iter=60)
        self.assertTrue(torch.equal(a.logits(x),b.logits(x)))
        self.assertEqual(before,base_fingerprint(self.base))
        self.assertIsNone(x.grad);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(x,self.x))

    def test_full_byte_output_and_invalid_fitting_inputs(self):
        h=initialize_residual(self.base)
        bias=torch.zeros(256,dtype=torch.double);bias[233]=100
        h=replace(h,correction=replace(h.correction,b2=bias))
        self.assertTrue(bool((h.predict(self.x)==233).all()))
        for args in ({'hidden':0},{'activation':'other'},{'l2':0},{'max_iter':0},
                     {'sample_weights':torch.zeros(4)}):
            with self.assertRaises(ValueError):fit_residual(self.x,self.y,self.base,**args)
        with self.assertRaises(ValueError):replace(h,strength=1.5).predict(self.x)


class SelectionChecks(unittest.TestCase):
    def setUp(self):
        self.data=fixture('validation')
        weight=torch.zeros(13,256,dtype=torch.double);weight[-1,65]=1
        base=ByteDecoder(torch.zeros(12,dtype=torch.double),torch.ones(12,dtype=torch.double),weight,{})
        head=initialize_residual(base)
        bias=torch.zeros(256,dtype=torch.double);bias[66]=10
        self.head=replace(head,correction=replace(head.correction,b2=bias))

    def test_guards_reject_shared_regression_and_choose_exact_fallback(self):
        self.data['answers'].fill_(66);self.data['answers'][:4]=65
        h=select_strength(self.head,self.data,fit_identities=['training'])
        self.assertEqual(h.strength,0)
        self.assertTrue(all(not row['eligible'] for row in h.selection['candidates'][1:]))

    def test_b_only_guard_is_independent_of_all_source_guard(self):
        self.data['answers'].fill_(66);self.data['answers'][-4:]=65
        h=select_strength(self.head,self.data,fit_identities=['training'])
        self.assertEqual(h.strength,0)

    def test_eligible_validation_improvement_selects_nonzero_strength(self):
        self.data['answers'].fill_(66)
        h=select_strength(self.head,self.data,fit_identities=['training'])
        self.assertEqual(h.strength,1)
        self.assertLess(h.selection['selected_weighted_nll'],h.selection['candidates'][0]['weighted_nll'])

    def test_ties_choose_zero_and_test_or_overlapping_cases_are_rejected(self):
        h=replace(self.head,correction=replace(self.head.correction,b2=torch.zeros(256,dtype=torch.double)))
        self.assertEqual(select_strength(h,self.data,fit_identities=['training']).strength,0)
        with self.assertRaises(ValueError):select_strength(h,self.data,fit_identities=self.data['identities'])
        self.data['protocol']['split']='test'
        with self.assertRaises(ValueError):select_strength(h,self.data,fit_identities=['training'])


class ArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=fixture('train');cls.validation=fixture('validation')
        cls.base=fit_byte_decoder(cls.training['features']['final/raw'],cls.training['answers'],
                                 sample_weights=balanced_weights(cls.training),max_iter=40)
        cls.fitted=fit_and_select(cls.training,cls.validation,cls.base,seeds=(7,),hidden=8,max_iter=30)

    def test_roundtrip_preserves_predictions_and_rejects_changed_base_or_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'residuals.pt';save_residuals(path,self.fitted)
            restored=load_residuals(path)
            x=self.training['features']['final/raw']
            for name,h in restored['heads'].items():
                self.assertTrue(torch.equal(h.logits(x),self.fitted['heads'][name].logits(x)))
            payload=torch.load(path,weights_only=True);bad=copy.deepcopy(payload)
            bad['base']['weight'][0,0]+=1;torch.save(bad,path)
            with self.assertRaises(ValueError):load_residuals(path)
            bad=copy.deepcopy(payload);bad['heads']['gelu/7']['correction']['w2']=torch.zeros(8,4)
            torch.save(bad,path)
            with self.assertRaises(ValueError):load_residuals(path)

    def test_selection_and_evaluation_reject_split_and_graph_leakage(self):
        with self.assertRaises(ValueError):fit_and_select(fixture('test'),self.validation,self.base)
        bad=fixture('validation');bad['identities'][0]=self.training['identities'][0]
        with self.assertRaises(ValueError):fit_and_select(self.training,bad,self.base)
        bad=fixture('validation');bad['protocol']['model_tensor_sha256']='changed'
        with self.assertRaises(ValueError):fit_and_select(self.training,bad,self.base)
        for identity in (self.training['identities'][0],self.validation['identities'][0]):
            bad=fixture('test');bad['identities'][0]=identity
            with self.assertRaises(ValueError):score_residuals(bad,self.fitted)
        with self.assertRaises(ValueError):score_residuals(self.validation,self.fitted)

    def test_predictions_do_not_use_query_control_or_answer_metadata(self):
        data=fixture('test');_,rows=score_residuals(data,self.fitted)
        data['queries']=['right']*12;data['controls']=['b_only']*12;data['answers']=data['answers'].roll(1)
        _,changed=score_residuals(data,self.fitted)
        self.assertEqual([r['predictions'] for r in rows],[r['predictions'] for r in changed])
        self.assertEqual(len(rows[0]['predictions']),5)

    def test_atomic_prediction_export_writes_and_verifies_complete_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.jsonl';data=b'{"complete":true}\n'*30
            write_bytes(path,data)
            self.assertEqual(path.read_bytes(),data)
            self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_validation_capture_uses_supported_cells_and_is_batch_invariant(self):
        # Isolate scheduling and hashing; graph execution is tested by the existing capture suite.
        def capture(model,batch,segment):
            n=len(batch.identities);t=batch.inputs['A'].shape[1]
            return {'features':{'final/raw':batch.answers[:,None].float().expand(n,64).clone()},
                    'expert_executions':n*t*84,'external_events':n*t}
        with patch('experiments.text_regions.residual_adapter.tensor_fingerprint',return_value='fixture'), \
             patch('experiments.text_regions.residual_adapter._policy',return_value={}), \
             patch('experiments.text_regions.residual_adapter.capture_readout_features',side_effect=capture):
            a=capture_validation(None,count=16,batch_size=16)
            b=capture_validation(None,count=16,batch_size=7)
        self.assertEqual(len(a['groups']),15);self.assertEqual(len(set(a['identities'])),48)
        self.assertEqual(a['protocol']['inputs_sha256'],b['protocol']['inputs_sha256'])
        self.assertEqual(a['protocol']['case_sha256'],b['protocol']['case_sha256'])
        self.assertEqual(a['protocol']['expert_executions'],240*25*84)
        self.assertTrue(torch.equal(a['features']['final/raw'],b['features']['final/raw']))
        expected=expected_validation(a['protocol'])
        self.assertEqual(expected['inputs_sha256'],a['protocol']['inputs_sha256'])
        self.assertEqual(expected['case_sha256'],a['protocol']['case_sha256'])

    def test_prediction_audit_rejects_truncation_changed_counts_or_extra_heads(self):
        evaluation,rows=score_residuals(fixture('test'),self.fitted)
        report={'fits':self.fitted['heads'],'evaluations':{'test':evaluation}}
        self.assertEqual(len(audit_rows(report,rows)),15)
        with self.assertRaises(ValueError):audit_rows(report,rows[:-1])
        changed=copy.deepcopy(rows);changed[0]['predictions']['linear/raw'][0]=233
        with self.assertRaises(ValueError):audit_rows(report,changed)
        changed=copy.deepcopy(rows);changed[0]['predictions']['task_specific']=[65]*4
        with self.assertRaises(ValueError):audit_rows(report,changed)

    def test_prediction_audit_rejects_changed_metric_or_duplicate_identity(self):
        evaluation,rows=score_residuals(fixture('test'),self.fitted)
        report={'fits':self.fitted['heads'],'evaluations':{'test':evaluation}}
        changed=copy.deepcopy(report)
        changed['evaluations']['test']['groups'][0]['decoders']['linear/raw']['exact_accuracy']=-1
        with self.assertRaises(ValueError):audit_rows(changed,rows)
        changed=copy.deepcopy(rows);changed[0]['episode_ids'][0]=changed[0]['episode_ids'][1]
        with self.assertRaises(ValueError):audit_rows(report,changed)


if __name__=='__main__':unittest.main()
