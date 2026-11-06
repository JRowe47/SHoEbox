"""Text causality, source ownership, sampling and joint optimizer continuation."""
from pathlib import Path
import tempfile
import unittest
import torch
from .test_multifact import fixture
from . import byte_stream
from .composition_train import atomic_save
from .data import OrderedByteStreams
from .joint_text import optimizer_for,update,load,selection
from .multifact import ingest,table_batches
from .multifact_data import make_batch
from .multifact_eval import memory_tensors
from .retained_composition import teacher_inputs

torch.set_num_threads(1)


class JointText(unittest.TestCase):
    def assert_tensors(self,left,right):
        self.assertEqual(len(left),len(right))
        for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_stream_chunking_causality_and_compute(self):
        with tempfile.TemporaryDirectory() as folder,torch.no_grad():
            model,_=fixture(folder);x=torch.tensor([[82,79,77,69]])
            whole=byte_stream.forward(model,x)
            first=byte_stream.forward(model,x[:,:2]);last=byte_stream.forward(model,x[:,2:],state=first['state'])
            torch.testing.assert_close(whole['logits'],torch.cat((first['logits'],last['logits']),1),atol=0,rtol=0)
            self.assert_tensors(whole['state'].tensors(),last['state'].tensors())
            altered=x.clone();altered[:,2:]=255
            other=byte_stream.forward(model,altered)
            torch.testing.assert_close(whole['logits'][:,:2],other['logits'][:,:2],atol=0,rtol=0)
            self.assertEqual(whole['external_events'],4)
            self.assertEqual(whole['source_occurrences'],12)
            self.assertEqual(whole['expert_executions'],4*7*3*2*2*2)

    def test_text_keeps_memory_owners_and_cannot_read_parked_memories(self):
        with tempfile.TemporaryDirectory() as folder,torch.no_grad():
            model,_=fixture(folder);b=make_batch('test',0,1,881,0)
            state,_=ingest(model,b);snapshot=[x.clone() for x in memory_tensors(state)]
            x=torch.tensor([[65,66,67]])
            result=byte_stream.forward(model,x,state=state)
            self.assert_tensors(snapshot,memory_tensors(result['state']))
            # Core starts cold after acquisition. Clearing memories must leave
            # text logits unchanged because their outgoing edges are masked.
            cold=byte_stream.forward(model,x)
            torch.testing.assert_close(result['logits'],cold['logits'],atol=0,rtol=0)

    def test_generation_resume_exactly_preserves_rng_logits_and_all_state(self):
        with tempfile.TemporaryDirectory() as folder:
            model,_=fixture(folder);model.train();global_rng=torch.get_rng_state().clone()
            initial=byte_stream.start(model,b'RO',seed=882)
            whole,whole_session,cost=byte_stream.generate(model,initial,6)
            left,mid,left_cost=byte_stream.generate(model,initial,2)
            path=Path(folder)/'session.pt';torch.save(mid.to_dict(),path)
            restored=byte_stream.ByteSession.from_dict(torch.load(path,weights_only=True))
            right,final,right_cost=byte_stream.generate(model,restored,4)
            torch.testing.assert_close(whole,torch.cat((left,right),1),atol=0,rtol=0)
            self.assert_tensors(whole_session.state.tensors(),final.state.tensors())
            torch.testing.assert_close(whole_session.next_logits,final.next_logits,atol=0,rtol=0)
            torch.testing.assert_close(whole_session.rng_state,final.rng_state,atol=0,rtol=0)
            torch.testing.assert_close(global_rng,torch.get_rng_state(),atol=0,rtol=0)
            self.assertEqual(cost,left_cost+right_cost);self.assertTrue(model.training)

    def test_zero_length_generation_does_not_advance_state(self):
        with tempfile.TemporaryDirectory() as folder:
            model,_=fixture(folder);initial=byte_stream.start(model,b'x')
            tokens,final,cost=byte_stream.generate(model,initial,0)
            self.assertEqual(tokens.shape,(1,0));self.assertEqual(cost,0)
            self.assert_tensors(initial.state.tensors(),final.state.tensors())
            torch.testing.assert_close(initial.rng_state,final.rng_state,atol=0,rtol=0)
            for temperature in (0,float('nan'),float('inf')):
                with self.assertRaises(ValueError):byte_stream.start(model,'x',temperature=temperature)
            with self.assertRaises(ValueError):byte_stream.start(model,b'')

    def test_unicode_prompt_is_ingested_as_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            model,_=fixture(folder)
            text=byte_stream.start(model,'caf\u00e9',seed=884)
            raw=byte_stream.start(model,'caf\u00e9'.encode('utf-8'),seed=884)
            torch.testing.assert_close(text.next_logits,raw.next_logits,atol=0,rtol=0)
            self.assert_tensors(text.state.tensors(),raw.state.tensors())
            self.assertEqual(text.state.events_seen.tolist(),[5])

    def test_joint_step_and_checkpoint_resume_weights_optimizer_and_cursor(self):
        with tempfile.TemporaryDirectory() as folder:
            model,_=fixture(folder);opt=optimizer_for(model)
            streams=OrderedByteStreams.from_bytes(b'Romeo says hello. '*12,2)
            reference=teacher_inputs(model,torch.arange(65,69))
            before={k:v.clone() for k,v in model.state_dict().items()}
            b=make_batch('train',0,2,883,0);tb=table_batches(b,'train',0,883)
            state,metrics=update(model,opt,streams.next_chunk(2),None,b,tb,reference)
            for prefix in ('readout.','nodes.P.','nodes.F.'):
                self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith(prefix)))
            self.assertGreater(metrics['text_nats'],0)
            saved=dict(schema='shoebox-joint-text-v1',config=model.cfg.to_dict(),weights=model.state_dict(),
                optimizer=opt.state_dict(),protocol=dict(core_lr=.0003,readout_lr=.002,producer_lr=.0001),
                graph_state=state.to_dict(),streams=streams.state_dict(),torch_rng=torch.get_rng_state())
            path=Path(folder)/'last.pt';atomic_save(path,saved)
            streams2=OrderedByteStreams.from_bytes(b'Romeo says hello. '*12,2)
            other,opt2,state2,_=load(path,streams2)
            b=make_batch('train',2,2,883,0,'shared_left');tb=table_batches(b,'train',2,883)
            state,a=update(model,opt,streams.next_chunk(2),state,b,tb,reference)
            state2,z=update(other,opt2,streams2.next_chunk(2),state2,b,tb,reference)
            self.assertEqual(a,z);self.assertEqual(streams.state_dict(),streams2.state_dict())
            self.assert_tensors(list(model.state_dict().values()),list(other.state_dict().values()))
            self.assert_tensors(state.tensors(),state2.tensors())
            for a,z in zip(opt.state.values(),opt2.state.values()):
                for k in a:torch.testing.assert_close(a[k],z[k],atol=0,rtol=0)

    def test_checkpoint_selection_uses_memory_guard_then_language(self):
        def scores(value):return [dict(fields=[dict(accuracy=value)]*3)]*5
        self.assertGreater(selection(dict(nats=4),scores(.91)),selection(dict(nats=2),scores(.89)))
        self.assertGreater(selection(dict(nats=3),scores(.91)),selection(dict(nats=4),scores(.99)))


if __name__=='__main__':unittest.main()
