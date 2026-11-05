"""Capability-qualified selection and exact optimizer-preserving continuation."""
import copy
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch

from ..hat_minkron.train import optimizer_for
from .composition import evaluate
from .composition_continue import completed_updates, evaluate_progress, gate_failures, main, record_validation
from .composition_data import training_batch
from .composition_train import build_model, save_checkpoint, load_checkpoint, update
from .config import GraphConfig
from .test_probe import assert_tree_equal

torch.set_num_threads(1)


def progress(accuracy=.95,nats=.2):
    metrics={n:{'accuracy':accuracy,'correct':95,'count':100} for n in
             ('overall','lookup_b','lookup_c','composition','p_first','q_second')}
    metrics.update({n:{'accuracy':1.,'correct':100,'count':100} for n in
                    ('prefix_b','prefix_c','query_b','query_c')})
    return {'metrics':metrics,'nats_per_answer':nats,'unique_cases':100}


class ContinuationChecks(unittest.TestCase):
    def test_each_capability_is_required_and_sources_must_be_exact(self):
        original=progress();self.assertEqual(gate_failures(original),[])
        for name in original['metrics']:
            other=copy.deepcopy(original)
            other['metrics'][name]['accuracy']=.999 if name.startswith(('query_','prefix_')) else .899
            self.assertEqual(gate_failures(other),[name])
        for bad in (None,float('nan')):
            other=copy.deepcopy(original);other['metrics']['q_second']['accuracy']=bad
            self.assertIn('q_second',gate_failures(other))
        with self.assertRaises(ValueError):gate_failures(original,float('nan'))

    def test_gate_requires_consecutive_checks_and_records_first_mastery(self):
        counters={'gate_streak':0,'mastered_step':None}
        self.assertFalse(record_validation(progress(),100,counters,.9,2))
        self.assertIsNone(counters['mastered_step'])
        self.assertTrue(record_validation(progress(.8),200,counters,.9,2))
        self.assertEqual(counters['gate_streak'],0)
        record_validation(progress(),300,counters,.9,2)
        record_validation(progress(),400,counters,.9,2)
        self.assertEqual(counters['mastered_step'],400)

    def test_update_offset_accumulates_across_phases(self):
        self.assertEqual(completed_updates({'protocol':{},'step':600}),600)
        self.assertEqual(completed_updates({'protocol':{'optimizer_step_offset':600},'step':200}),800)

    def test_progress_matches_primary_evaluator_and_unique_skill_denominators(self):
        torch.manual_seed(506)
        model=build_model(GraphConfig.tiny(readout_mode='concat'),.5,1.)
        p=evaluate_progress(model)
        expected=evaluate(model,count=None,rounds=(2,))['k2']
        self.assertTrue(model.training)
        self.assertEqual(p['unique_cases'],528)
        self.assertEqual(p['metrics']['overall']['correct'],expected['correct'])
        self.assertEqual(p['nats_per_answer'],expected['nats_per_answer'])
        self.assertEqual([p['metrics'][n]['count'] for n in ('lookup_b','lookup_c','composition')],[32,16,480])

    def make_parent(self,path):
        torch.manual_seed(507)
        model=build_model(GraphConfig.tiny(readout_mode='concat'),.5,1.)
        protocol=dict(lr=.001,warmup=20,rounds=2,feedback=True,
                      source_reconstruction=.5,regional_supervision=1.,batch=8)
        opt=optimizer_for(model,.001);rng=random.Random(507)
        update(model,opt,training_batch(rng,8),0,protocol)
        save_checkpoint(path,model,opt,rng,1,protocol,2.,{},[])

    def run_cli(self,parent,out,extras=(),scores=None):
        argv=['continue','--parent',str(parent),'--out',str(out),'--steps','2',
              '--validate-every','1','--save-every','1',*extras]
        with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
            with patch('experiments.text_regions.composition_continue.evaluate_progress',
                       side_effect=scores or (lambda *a,**kw:progress(.8,.5))):
                main()

    def test_full_cli_resume_matches_uninterrupted_weights_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent.pt';self.make_parent(parent)
            self.run_cli(parent,root/'full')
            self.run_cli(parent,root/'split',('--stop-after','1'))
            self.run_cli(parent,root/'split',('--resume',))
            a=torch.load(root/'full/last.pt',weights_only=True)
            b=torch.load(root/'split/last.pt',weights_only=True)
            for key in ('weights','optimizer','rng','torch_rng'):assert_tree_equal(self,a[key],b[key])
            self.assertEqual(a['step'],2)
            self.assertEqual(completed_updates(a),3)
            parent_saved=torch.load(parent,weights_only=True)
            initial=torch.load(root/'full/initial.pt',weights_only=True)
            for key in ('weights','optimizer','rng','torch_rng'):
                assert_tree_equal(self,parent_saved[key],initial[key])
            self.assertEqual(a['history'][1]['lr'],.001*2/20)

    def test_better_nll_cannot_replace_a_qualified_checkpoint_after_regression(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent.pt';self.make_parent(parent)
            self.run_cli(parent,root/'run',scores=[progress(.8,.8),progress(.95,.3),progress(.8,.1)])
            best=torch.load(root/'run/best.pt',weights_only=True)
            candidate=torch.load(root/'run/best_candidate.pt',weights_only=True)
            self.assertEqual(best['step'],1);self.assertEqual(candidate['step'],2)
            self.assertFalse(gate_failures(best['history'][-1]['validation']))
            self.assertTrue(gate_failures(candidate['history'][-1]['validation']))


if __name__=='__main__':unittest.main()
