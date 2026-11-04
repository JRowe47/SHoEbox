"""Frozen representations, a common byte interface, and diagnostic controls."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .config import GraphConfig
from .model import RegionLanguageModel
from .readout_diagnostic import (SUBSETS, ByteDecoder, capture_readout_features,
    collect_features, controlled_batch, fit_byte_decoder, fit_diagnostic, load_diagnostic, save_diagnostic,
    score_diagnostic, visible)
from .view_data import make_view_batch
from .view_eval import evaluation_mode
from .view_train import tensor_fingerprint
from .readout_compare import audit_predictions, expected_cohort

torch.set_num_threads(2)


def synthetic_data(split):
    """Factor/query conjunctions represent observed content, not label metadata."""
    torch.manual_seed(81)
    features, answers, queries, controls, factors, groups, ids = [], [], [], [], [], [], []
    for control in SUBSETS:
        for axis,query in enumerate(('shared','left','right')):
            if split == 'train' and not visible(control,query):
                continue
            start = len(features)
            for n in range(64):
                values = (n % 4, (n//4) % 4, n//16)
                x = torch.zeros(12)
                x[axis*4+values[axis]] = 1
                features.append(x); answers.append(65+values[axis]); factors.append([65+v for v in values])
                queries.append(query); controls.append(control); ids.append(f'{split}/{query}/{n}')
            groups.append({'control':control,'query':query,'start':start,'stop':len(features),
                           'expert_executions':640,'external_events':640,'information_present':visible(control,query)})
    x,y = torch.stack(features),torch.tensor(answers)
    return {'features':{moment+'/'+name:x.clone() for moment in ('prequery','final') for name in ('raw','mixed')},
            'valid':{moment+'/'+name:torch.ones(len(y),dtype=torch.bool) for moment in ('prequery','final') for name in ('raw','mixed')},
            'answers':y,'logits':torch.zeros(len(y),256),'factors':{q:torch.tensor(factors)[:,i] for i,q in enumerate(('shared','left','right'))},
            'queries':queries,'controls':controls,'identities':ids,'groups':groups,
            'protocol':{'split':split,'model_tensor_sha256':'fixture','policy':{},'gap':0,'query_every':4}}


class CaptureChecks(unittest.TestCase):
    def test_cohort_coverage_and_hashes_are_invariant_to_batching(self):
        model = RegionLanguageModel(GraphConfig.tiny())
        def fake_capture(model,batch,segment):
            count = len(batch.answers)
            x = batch.inputs['A'][:,:2].float()
            return {'features':{'final/raw':x},'valid':{'final/raw':torch.ones(count,dtype=torch.bool)},
                    'logits':torch.zeros(count,256),'expert_executions':count*batch.length*84,
                    'external_events':count*batch.length}
        with patch('experiments.text_regions.readout_diagnostic.capture_readout_features',fake_capture):
            a = collect_features(model,split='train',count_per_cell=32,batch_size=13,gap=0)
            b = collect_features(model,split='train',count_per_cell=32,batch_size=32,gap=0)
            test = collect_features(model,split='independent',count_per_cell=64,batch_size=64,gap=0)
        self.assertEqual(len(a['groups']),15)
        self.assertEqual(len(a['answers']),480)
        self.assertEqual(len(test['groups']),27)
        self.assertEqual(a['protocol']['inputs_sha256'],b['protocol']['inputs_sha256'])
        self.assertEqual(a['protocol']['case_sha256'],b['protocol']['case_sha256'])
        for group in a['groups']:
            self.assertTrue(group['information_present'])
            y = a['answers'][group['start']:group['stop']]
            self.assertEqual([int((y == v).sum()) for v in b'ABCD'],[8]*4)
        self.assertFalse(set(a['identities']) & set(test['identities']))

    def test_whole_source_controls_preserve_bytes_and_never_use_answers(self):
        for control,sources in SUBSETS.items():
            batch = controlled_batch('train',0,32,control=control,query='left',seed=91,gap=0,query_every=4)
            reference = make_view_batch('train',0,32,query_type='left',seed=91,gap=0)
            for source in 'ABC':
                self.assertTrue(torch.equal(batch.inputs[source],reference.inputs[source]))
                expected = reference.availability[source] & (source in sources)
                self.assertTrue(torch.equal(batch.availability[source],expected))
            self.assertTrue(torch.equal(batch.answers,reference.answers))
        self.assertFalse(visible('b_only','left'))
        self.assertFalse(visible('reset_query','shared'))
        self.assertTrue(visible('ac_only','right'))

    def test_raw_mixed_exact_readout_and_label_free_capture_preserve_graph(self):
        torch.manual_seed(73)
        model = RegionLanguageModel(GraphConfig.tiny(evidence_gated_ports=True)).train()
        model.readout.eval()
        batch = controlled_batch('test',0,2,control='b_only',query='shared',seed=13,gap=0,query_every=4)
        before,rng = tensor_fingerprint(model),torch.get_rng_state().clone()
        capture = capture_readout_features(model,batch,segment=5)
        changed = replace(batch,answers=torch.full_like(batch.answers,255),
                          query_types=('right','left'),factors={q:torch.full_like(x,255) for q,x in batch.factors.items()},
                          targets=torch.full_like(batch.targets,255))
        again = capture_readout_features(model,changed,segment=9)
        with evaluation_mode(model):
            direct = model(batch.inputs,reset=batch.reset,availability=batch.availability)
        torch.testing.assert_close(capture['logits'],direct['logits'][:,-1],atol=0,rtol=0)
        self.assertEqual(capture['expert_executions'],direct['expert_executions'])
        for key,feature in capture['features'].items():
            torch.testing.assert_close(feature,again['features'][key],atol=0,rtol=0)
        self.assertEqual(capture['features']['final/raw'].shape[1],2*model.cfg.width)
        self.assertEqual(capture['features']['final/mixed'].shape[1],model.cfg.width)
        self.assertFalse(bool(capture['valid']['final/A'].any()))
        self.assertEqual(float(capture['features']['final/A'].abs().sum()),0.)
        self.assertEqual(tensor_fingerprint(model),before)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training);self.assertFalse(model.readout.training)


class DecoderChecks(unittest.TestCase):
    def test_one_decoder_learns_all_queries_and_can_emit_any_byte(self):
        data = synthetic_data('train')
        rng = torch.get_rng_state().clone()
        head = fit_byte_decoder(data['features']['final/raw'],data['answers'],max_iter=60)
        self.assertEqual(head.weight.shape[1],256)
        self.assertTrue(torch.equal(head.predict(data['features']['final/raw']),data['answers']))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        forced = head.weight.clone();forced.zero_();forced[-1,233] = 10
        arbitrary = ByteDecoder(head.mean,head.scale,forced,head.fit)
        self.assertTrue(bool((arbitrary.predict(data['features']['final/raw']) == 233).all()))

    def test_common_heads_and_fixed_factor_probes_roundtrip_and_score(self):
        training = synthetic_data('train')
        fitted = fit_diagnostic(training,max_iter=60)
        self.assertEqual(set(fitted['answer_heads']),{'raw','mixed'})
        self.assertEqual(fitted['fit_query_counts'],{'shared':448,'left':256,'right':256})
        data = synthetic_data('test')
        result,predictions = score_diagnostic(data,fitted)
        self.assertTrue(all(g['decoders']['raw']['exact_accuracy'] == 1 for g in result['groups']))
        self.assertTrue(all(g['decoders']['raw']['outside_answer_vocabulary'] == 0 for g in result['groups']))
        self.assertEqual(len(predictions),21)
        self.assertIn('prequery->final/raw/left',result['factor_decodability'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'heads.pt';save_diagnostic(path,fitted)
            restored = load_diagnostic(path)
            self.assertTrue(torch.equal(restored['answer_heads']['raw'].predict(data['features']['final/raw']),data['answers']))
            payload = torch.load(path,weights_only=True)
            payload['answer_heads']['raw']['scale'][0] = 0
            torch.save(payload,path)
            with self.assertRaises(ValueError):load_diagnostic(path)

    def test_fit_and_evaluation_reject_leakage_and_mismatched_model(self):
        training = synthetic_data('train')
        with self.assertRaises(ValueError):fit_diagnostic(synthetic_data('test'))
        bad = copy.deepcopy(training);bad['controls'][0] = 'reset_query'
        with self.assertRaises(ValueError):fit_diagnostic(bad)
        fitted = fit_diagnostic(training,max_iter=30)
        test = synthetic_data('test');test['identities'][0] = training['identities'][0]
        with self.assertRaises(ValueError):score_diagnostic(test,fitted)
        test = synthetic_data('test');test['protocol']['model_tensor_sha256'] = 'other'
        with self.assertRaises(ValueError):score_diagnostic(test,fitted)

    def test_invalid_solver_inputs_fail(self):
        x = torch.eye(4);y = torch.arange(65,69)
        for kwargs in ({'l2':0},{'max_iter':0},{'sample_weights':torch.tensor([1.,-1,1,1])}):
            with self.assertRaises(ValueError):fit_byte_decoder(x,y,**kwargs)
        with self.assertRaises(ValueError):fit_byte_decoder(x,torch.tensor([65,66,67,256]))


class ComparisonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fitted = fit_diagnostic(synthetic_data('train'),max_iter=60)
        result,cls.predictions = score_diagnostic(synthetic_data('test'),fitted)
        cls.report = {'evaluations':{'test':result}}

    def test_prediction_audit_recomputes_scores_and_rejects_tampering(self):
        scores = audit_predictions(self.report,self.predictions)
        self.assertEqual(scores['test/all/left/raw'],1.)
        changed = copy.deepcopy(self.predictions)
        changed[0]['predictions']['raw'][0] = 233
        with self.assertRaises(ValueError):audit_predictions(self.report,changed)
        changed_report = copy.deepcopy(self.report)
        changed_report['evaluations']['test']['groups'][0]['decoders']['raw']['exact_accuracy'] = .5
        with self.assertRaises(ValueError):audit_predictions(changed_report,self.predictions)

    def test_prediction_audit_rejects_mismatched_cases_or_extra_heads(self):
        changed = copy.deepcopy(self.predictions);changed[3]['episode_ids'][0] = 'different'
        with self.assertRaises(ValueError):audit_predictions(self.report,changed)
        changed = copy.deepcopy(self.predictions);changed[0]['predictions']['left'] = changed[0]['predictions']['raw']
        with self.assertRaises(ValueError):audit_predictions(self.report,changed)

    def test_independent_input_reconstruction_covers_all_supported_subsets(self):
        protocol = {'split':'train','count_per_cell':32,'seed':17,'gap':0,'query_every':4}
        rebuilt = expected_cohort(protocol)
        self.assertEqual(rebuilt['episodes'],480)
        self.assertEqual(len(rebuilt['fit_identities']),96)
        self.assertEqual(len(rebuilt['groups']),15)
        for control,query in rebuilt['groups']:
            self.assertTrue(visible(control,query))
        alternative = expected_cohort({**protocol,'seed':18})
        self.assertNotEqual(rebuilt['inputs_sha256'],alternative['inputs_sha256'])


if __name__ == '__main__':
    unittest.main()
