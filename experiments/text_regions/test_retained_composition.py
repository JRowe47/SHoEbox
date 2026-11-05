"""Independent source ownership and causal retained-query integration."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from .config import GraphConfig,NodeSpec,EdgeSpec
from .model import RegionLanguageModel
from .ports import tensor_list
from .publication_data import make_batch
from .retained_composition import EDGE,build,freeze,ingest,query,tables,teacher_inputs,update
from .retained_composition_train import optimizer_for,load
from .composition_train import atomic_save
from .publication_train import forward_batch

torch.set_num_threads(1)


class IndependentSources(unittest.TestCase):
    def components(self,folder):
        torch.manual_seed(66)
        cfg=GraphConfig.tiny(rounds=2,recurrent_feedback=True,readout_mode='concat',evidence_gated_ports=True)
        composer=RegionLanguageModel(cfg)
        pc=replace(cfg,nodes=(NodeSpec('A','R0','A'),),edges=(),readout=('A',),rounds=1,
                   recurrent_feedback=False,selective_publication=('A',))
        producer=RegionLanguageModel(pc)
        for name,m in (('composer',composer),('producer',producer)):
            atomic_save(Path(folder)/(name+'.pt'),dict(config=m.cfg.to_dict(),weights=m.state_dict()))
        model=build(Path(folder)/'composer.pt',Path(folder)/'producer.pt',321)
        return composer,producer,model

    def test_config_rejects_ambiguous_ownership_and_preserves_old_defaults(self):
        c=GraphConfig.tiny();data=c.to_dict();data.pop('independent_sources')
        self.assertEqual(GraphConfig.from_dict(data).independent_sources,())
        for names in (('Z',),('U',),('A','A')):
            with self.assertRaises(ValueError):RegionLanguageModel(replace(c,independent_sources=names))
        with self.assertRaises(ValueError):RegionLanguageModel(replace(c,independent_sources=('A',),
            nodes=c.nodes+(NodeSpec('X','R3','A'),)))

    def test_component_weights_exact_and_banks_do_not_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            c,p,m=self.components(folder)
            for key,value in c.state_dict().items():torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(sum(x.numel() for x in m.parameters() if x.requires_grad),1024)
            self.assertNotEqual(m.shared_banks[0].up.data_ptr(),m.independent_banks['F'][0].up.data_ptr())
            with self.assertRaises(ValueError):m(torch.tensor([[65]]))

    def test_transplanted_producer_is_exact_and_chunk_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            c,p,m=self.components(folder);b=make_batch('validation',0,2,17,0)
            with torch.no_grad():
                original=forward_batch(p,b,'learned')['state'];a,_=ingest(m,b);z,_=ingest(m,b,segment=7)
            for x,y in zip(tensor_list(original.nodes['A']),tensor_list(a.nodes['F'])):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(original.ports.latest['A'].payload,a.ports.latest['F'].payload,rtol=0,atol=0)
            for x,y in zip(a.tensors(),z.tensors()):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_multiple_queries_preserve_memory_state_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,m=self.components(folder);b=make_batch('validation',0,2,17,0)
            with torch.no_grad():
                s,_=ingest(m,b);before=[x.clone() for x in tensor_list((s.nodes['F'],s.ingress['F'],s.ports.latest['F']))]
                epoch=s.epochs.clone()
                for i in range(3):
                    tb=tables('validation',0,2,17,b.answers,i);r=query(m,tb.first,tb.second,s,trace=True);s=r['state']
                    for x,y in zip(before,tensor_list((s.nodes['F'],s.ingress['F'],s.ports.latest['F']))):
                        torch.testing.assert_close(x,y,rtol=0,atol=0)
                    self.assertEqual(r['expert_executions'],2*(8+14)*12)
                    self.assertEqual(r['source_occurrences'],2*10)
                self.assertTrue(bool(s.epochs.eq(epoch+3).all()))

    def test_labels_do_not_enter_ingest_or_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,m=self.components(folder);b=make_batch('validation',0,2,17,0)
            changed=replace(b,answers=b.answers.roll(1),writes=~b.writes,retained=torch.zeros_like(b.retained))
            with torch.no_grad():
                a,_=ingest(m,b);z,_=ingest(m,changed)
                tb=tables('validation',0,2,17,b.answers)
                ra=query(m,tb.first,tb.second,a);rz=query(m,tb.first,tb.second,z)
            torch.testing.assert_close(ra['logits'],rz['logits'],rtol=0,atol=0)

    def test_edge_ablation_and_memory_reset_keep_execution_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,m=self.components(folder);b=make_batch('validation',0,4,17,0)
            with torch.no_grad():
                s,_=ingest(m,b);tb=tables('validation',0,4,17,b.answers)
                results=[query(m,tb.first,tb.second,s,control=control) for control in
                         ('intact','no_edge','reset_memory','shuffle_memory')]
            for r in results[1:]:self.assertEqual(r['expert_executions'],results[0]['expert_executions'])
            self.assertFalse(bool(results[2]['state'].ports.latest['F'].valid.any()))
            self.assertFalse(torch.equal(results[0]['logits'],results[1]['logits']))

    def test_tables_are_split_disjoint_and_query_inputs_independent_of_fact(self):
        answers=torch.tensor([65,66,67,68]*8)
        a=tables('train',0,32,5,answers);b=tables('train',0,32,5,answers.roll(1))
        self.assertEqual([k[:2] for k in a.keys],[k[:2] for k in b.keys])
        self.assertFalse(set(k[:2] for k in a.keys)&set(k[:2] for k in tables('test',0,32,5,answers).keys))

    def test_only_edge_trains_and_optimizer_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,m=self.components(folder);before={k:v.clone() for k,v in m.state_dict().items()}
            opt=optimizer_for(m,.01);b=make_batch('train',0,4,7,0);tb=tables('train',0,4,7,b.answers)
            update(m,opt,b,tb)
            changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
            self.assertEqual(changed,['connections.edges.'+EDGE+'.value.weight'])
            path=Path(folder)/'resume.pt'
            atomic_save(path,dict(schema='shoebox-retained-composition-v1',config=m.cfg.to_dict(),
                weights=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),protocol={'lr':.01}))
            restored,other,_=load(path)
            x=update(m,opt,b,tb);y=update(restored,other,b,tb);self.assertEqual(x,y)
            for k,v in m.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            for key,values in opt.state_dict()['state'].items():
                for name,v in values.items():torch.testing.assert_close(v,other.state_dict()['state'][key][name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
