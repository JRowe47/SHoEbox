"""Producer ownership, local gradients, bounded history and exact continuation."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from .config import GraphConfig, NodeSpec
from .model import RegionLanguageModel
from .composition_train import atomic_save
from .publication_data import make_batch
from .retained_composition import build, tables, ingest, query
from .joint_retained import allowed, configure, optimizer_for, producer_segment, update
from .joint_retained_train import load

torch.set_num_threads(1)


def fixture(folder):
    torch.manual_seed(712)
    cfg=GraphConfig.tiny(rounds=2,recurrent_feedback=True,readout_mode='concat',evidence_gated_ports=True)
    composer=RegionLanguageModel(cfg)
    pc=replace(cfg,nodes=(NodeSpec('A','R0','A'),),edges=(),readout=('A',),rounds=1,
               recurrent_feedback=False,selective_publication=('A',))
    producer=RegionLanguageModel(pc)
    for name,model in (('composer',composer),('producer',producer)):
        atomic_save(Path(folder)/(name+'.pt'),dict(config=model.cfg.to_dict(),weights=model.state_dict()))
    return build(Path(folder)/'composer.pt',Path(folder)/'producer.pt',713)


class JointRetained(unittest.TestCase):
    def test_owned_parameters_train_and_composer_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            m=fixture(folder);opt=optimizer_for(m,'joint')
            before={k:v.clone() for k,v in m.state_dict().items()}
            b=make_batch('train',0,2,715,0);tb=tables('train',0,2,715,b.answers)
            row=update(m,opt,b,tb)
            changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
            self.assertTrue(all(allowed(k,'joint') for k in changed))
            for prefix in ('nodes.F.', 'independent_hats.F.', 'independent_banks.F.',
                           'connections.ingress.F.', 'connections.edges.F__A__forward.value.'):
                self.assertTrue(any(k.startswith(prefix) for k in changed),prefix)
            self.assertEqual(row['expert_executions'],12*b.inputs.numel()+264*len(b.answers))
            self.assertEqual(row['external_events'],b.inputs.numel()+5*len(b.answers))
            self.assertEqual(row['source_occurrences'],b.inputs.numel()+10*len(b.answers))

    def test_edge_control_only_changes_edge(self):
        with tempfile.TemporaryDirectory() as folder:
            m=fixture(folder);opt=optimizer_for(m,'edge')
            before={k:v.clone() for k,v in m.state_dict().items()}
            b=make_batch('train',0,2,716,0);tb=tables('train',0,2,716,b.answers)
            update(m,opt,b,tb)
            changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
            self.assertEqual(changed,['connections.edges.F__A__forward.value.weight'])

    def test_training_capture_matches_inference_and_labels_cannot_enter(self):
        with tempfile.TemporaryDirectory() as folder, torch.no_grad():
            m=fixture(folder);b=make_batch('test',0,2,717,0)
            tampered=replace(b,answers=b.answers.roll(1),retained=torch.zeros_like(b.retained),writes=~b.writes)
            a=producer_segment(m,b,0,b.inputs.shape[1]);z=producer_segment(m,tampered,0,b.inputs.shape[1])
            s,_=ingest(m,b)
            for one,two,three in zip(a['state'].tensors(),z['state'].tensors(),s.tensors()):
                torch.testing.assert_close(one,two,rtol=0,atol=0)
                torch.testing.assert_close(one,three,rtol=0,atol=0)

    def test_detach_severs_history_and_last_segment_reaches_query(self):
        with tempfile.TemporaryDirectory() as folder:
            m=fixture(folder);configure(m,'joint');b=make_batch('train',0,2,718,0)
            first=producer_segment(m,b,0,16);s=first['state'].detach()
            self.assertTrue(all(t.grad_fn is None for t in s.tensors()))
            final=producer_segment(m,b,16,b.inputs.shape[1],s)
            payload=final['state'].ports.latest['F'].payload;payload.retain_grad()
            tb=tables('train',0,2,718,b.answers);r=query(m,tb.first,tb.second,final['state'])
            r['logits'].square().mean().backward()
            self.assertIsNotNone(payload.grad);self.assertGreater(float(payload.grad.abs().sum()),0.)

    def test_optimizer_resume_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            m=fixture(folder);opt=optimizer_for(m,'joint')
            b=make_batch('train',0,2,719,0);tb=tables('train',0,2,719,b.answers)
            update(m,opt,b,tb)
            path=Path(folder)/'resume.pt'
            atomic_save(path,dict(schema='shoebox-joint-retained-v1',config=m.cfg.to_dict(),
                weights=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
                protocol=dict(mode='joint',producer_lr=.0003,edge_lr=.001)))
            restored,other,_=load(path)
            self.assertEqual(update(m,opt,b,tb),update(restored,other,b,tb))
            for k,v in m.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            for k,values in opt.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,other.state_dict()['state'][k][name],rtol=0,atol=0)

    def test_invalid_modes_and_segment_fail_early(self):
        with tempfile.TemporaryDirectory() as folder:
            m=fixture(folder)
            with self.assertRaises(ValueError):configure(m,'unknown')
            opt=optimizer_for(m,'joint');b=make_batch('train',0,1,720,0)
            with self.assertRaises(ValueError):update(m,opt,b,tables('train',0,1,720,b.answers),segment=0)


if __name__=='__main__':unittest.main()
